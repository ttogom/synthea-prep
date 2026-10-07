#!/usr/bin/env python3
"""
extract_presenting.py — Extract presenting symptoms for condition-positive patients.

For each positive in a scrub output, finds the raw symptoms.csv episode whose
AGE_BEGIN is closest to the patient's diagnosis age (within a 7-year lookback
window per SCRUBBING.md), and writes symptom names and severities to a JSON file.

Also produces:
  - Leak review: symptom names that share words with the target description.
  - History-only probe: TF-IDF + LogReg on pre-diagnosis CSV text only (no symptoms).
  - Symptoms-only probe: bag-of-symptom-names + LogReg vs age-matched negatives.

Usage:
    python3 scripts/extract_presenting.py <scrub_dir> <src_dir> <labels_json>
        [--out <presenting_json>]   default: data/presenting/<scrub_stem>.json
        [--report <path>]           default: docs/PRESENTING_<scrub_stem>.md
        [--probe]                   run both classifier probes (requires sklearn)
"""

import argparse
import collections
import csv
import json
import math
import sys
from datetime import date
from pathlib import Path


MAX_AGE_WINDOW = 7   # AGE_BEGIN may lag up to this many years below diagnosis age

# Whitelisted vital sign LOINC codes.
# Only these six standard physiological measurements are included.
# Excluded from consideration even if recorded as vital-signs in Synthea:
#   - Body Weight (29463-7), Body Height (8302-2), BMI (39156-5) — not on six-item list
#   - Pain severity (72514-3) — numeric scale, not a physiological measurement
#   - FEV1/FVC (19926-5) — pulmonary function ratio; < 0.70 is the diagnostic criterion
#     for COPD, so including it would directly name the answer
VITAL_SIGN_WHITELIST = {
    "8310-5": "Body temperature",
    "8480-6": "Systolic Blood Pressure",
    "8462-4": "Diastolic Blood Pressure",
    "8867-4": "Heart rate",
    "9279-1": "Respiratory rate",
    "2708-6": "Oxygen saturation",
}


# ── Low-level utilities ───────────────────────────────────────────────────────

def _d(s: str):
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _age(birth: date, ref: date) -> int:
    return ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))


def _iter_csv(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_patients(patients_csv: Path) -> dict:
    """Returns {pid: {"birth": date, "gender": str}}."""
    result = {}
    for row in _iter_csv(patients_csv):
        pid = row.get("Id", "")
        birth = _d(row.get("BIRTHDATE", ""))
        if pid and birth:
            result[pid] = {"birth": birth, "gender": row.get("GENDER", "")}
    return result


def target_descriptions(src_dir: Path, codes: list) -> list:
    """All DESCRIPTION strings for the target codes, from source conditions.csv."""
    code_set = set(codes)
    descs = set()
    for row in _iter_csv(src_dir / "csv" / "conditions.csv"):
        if row.get("CODE", "") in code_set:
            d = row.get("DESCRIPTION", "").strip()
            if d:
                descs.add(d)
    return sorted(descs)


# ── Symptom parsing ───────────────────────────────────────────────────────────

def parse_symptoms(raw: str) -> list:
    """'Name:Severity:0;Name2:Severity2:0' → [{"name": ..., "severity": ...}]."""
    if not raw.strip():
        return []
    result = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(":")
        name = pieces[0].strip()
        try:
            severity = int(pieces[1]) if len(pieces) > 1 else 0
        except ValueError:
            severity = 0
        if name:
            result.append({"name": name, "severity": severity})
    return result


# ── Episode selection ─────────────────────────────────────────────────────────

def pick_episode(rows_for_patient: list, diagnosis_age: int) -> dict | None:
    """
    rows_for_patient: list of rows (dicts) from symptoms.csv for one patient,
                      all with matching PATHOLOGY.
    diagnosis_age: patient's age at cutoff (whole years).

    Selects the episode with the highest AGE_BEGIN that is:
      - ≤ diagnosis_age
      - within MAX_AGE_WINDOW years of diagnosis_age

    If no row satisfies both, falls back to the closest row at or below
    diagnosis_age (distance > MAX_AGE_WINDOW). Returns None if no row is at or
    below diagnosis_age.

    Returns dict with age_begin, parsed symptoms, distance, and selection_note.
    """
    # Parse AGE_BEGIN for each row
    parsed = []
    for row in rows_for_patient:
        try:
            ab = int(row.get("AGE_BEGIN", -1))
        except (ValueError, TypeError):
            continue
        if ab < 0:
            continue
        syms = parse_symptoms(row.get("SYMPTOMS", ""))
        parsed.append((ab, syms))

    if not parsed:
        return None

    # Rows at or below diagnosis_age, within window
    in_window = [(ab, s) for ab, s in parsed
                 if ab <= diagnosis_age and (diagnosis_age - ab) <= MAX_AGE_WINDOW]
    if in_window:
        ab, syms = max(in_window, key=lambda x: x[0])
        note = f"within {MAX_AGE_WINDOW}-year window"
        if len(rows_for_patient) > 1:
            note += f"; {len(rows_for_patient)} episodes total, picked age {ab}"
        return {"age_begin": ab, "symptoms": syms,
                "distance": diagnosis_age - ab, "selection_note": note}

    # Fallback: closest below (distance > MAX_AGE_WINDOW)
    below = [(ab, s) for ab, s in parsed if ab <= diagnosis_age]
    if below:
        ab, syms = max(below, key=lambda x: x[0])
        return {"age_begin": ab, "symptoms": syms,
                "distance": diagnosis_age - ab,
                "selection_note": f"fallback: best below {MAX_AGE_WINDOW}-year window"}

    return None


# ── Diagnosing-encounter vitals ───────────────────────────────────────────────

def load_diagnosing_vitals(labels: dict, src_dir: Path, codes: list) -> dict:
    """
    For each positive in labels, finds its diagnosing encounter (the first row in
    conditions.csv where PATIENT matches and CODE is in codes), then returns whitelisted
    vital sign readings from that encounter's observations.

    Returns {pid: [{"name": ..., "value": ..., "unit": ...}]}.
    Patients whose diagnosing encounter has no whitelisted vitals are included with [].
    """
    code_set = set(codes)
    pos_pids = set(labels)

    # Map pid → diagnosing encounter id (first hit only)
    diag_enc: dict = {}
    for row in _iter_csv(src_dir / "csv" / "conditions.csv"):
        pid = row.get("PATIENT", "")
        if pid not in pos_pids or pid in diag_enc:
            continue
        if row.get("CODE", "") in code_set:
            enc = row.get("ENCOUNTER", "")
            if enc:
                diag_enc[pid] = enc

    enc_set = set(diag_enc.values())

    # Collect whitelisted vitals at those encounters
    enc_vitals: dict = collections.defaultdict(list)
    for row in _iter_csv(src_dir / "csv" / "observations.csv"):
        enc = row.get("ENCOUNTER", "")
        if enc not in enc_set:
            continue
        loinc = row.get("CODE", "")
        if loinc not in VITAL_SIGN_WHITELIST:
            continue
        try:
            value = float(row.get("VALUE", ""))
        except (ValueError, TypeError):
            continue
        unit = row.get("UNITS", "").strip()
        name = VITAL_SIGN_WHITELIST[loinc]
        enc_vitals[enc].append({"name": name, "value": value, "unit": unit})

    return {pid: enc_vitals.get(enc, []) for pid, enc in diag_enc.items()}


# ── Main presenting extraction ────────────────────────────────────────────────

def extract_presenting(labels: dict, patients: dict, src_dir: Path,
                       target_descs: list) -> dict:
    """
    Returns {pid: {age_begin, diagnosis_age, distance, symptoms, selection_note}}.
    Patients with no matching symptom episode are omitted.
    """
    target_lower = {d.lower() for d in target_descs}
    # Short forms: strip Synthea tag "(disorder)" etc.
    for d in list(target_lower):
        short = d.rsplit(" (", 1)[0]
        target_lower.add(short)

    # Group raw symptom rows by (patient, pathology) — only for our positives
    pos_pids = set(labels)
    episodes: dict = collections.defaultdict(list)  # pid → [row, ...]
    for row in _iter_csv(src_dir / "symptoms" / "csv" / "symptoms.csv"):
        pid = row.get("PATIENT", "")
        if pid not in pos_pids:
            continue
        path_lower = row.get("PATHOLOGY", "").lower()
        if any(t in path_lower for t in target_lower):
            episodes[pid].append(row)

    result = {}
    for pid, label_info in labels.items():
        p = patients.get(pid)
        if not p:
            continue
        cutoff = _d(label_info["cutoff"])
        if not cutoff:
            continue
        diag_age = _age(p["birth"], cutoff)

        ep = pick_episode(episodes.get(pid, []), diag_age)
        if ep is None or not ep["symptoms"]:
            continue  # skip empty-symptom rows (e.g. hypertension has 0 symptoms)
        result[pid] = {
            "diagnosis_age": diag_age,
            **ep,
        }

    return result


# ── Leak review ───────────────────────────────────────────────────────────────

def leak_review(presenting: dict, target_descs: list,
                src_dir: Path, target_codes: list) -> dict:
    """
    Lists symptom names that share a word (≥4 chars) with any target description.
    For each overlapping symptom, also checks how many other conditions share it
    (multi-condition sharing = shared evidence, not a giveaway).
    """
    # Words in target descriptions (≥4 chars, lower)
    desc_words: set = set()
    for d in target_descs:
        base = d.lower().rsplit(" (", 1)[0]
        for w in base.replace("-", " ").split():
            w = w.strip(".,;:()[]")
            if len(w) >= 4:
                desc_words.add(w)

    # Collect all symptom names from presenting.json
    sym_counts: collections.Counter = collections.Counter()
    for ep in presenting.values():
        for s in ep["symptoms"]:
            sym_counts[s["name"]] += 1

    # Find overlapping symptom names
    overlapping = []
    for name, count in sym_counts.most_common():
        name_words = set(w.strip(".,;:()[]").lower()
                         for w in name.replace("-", " ").split() if len(w) >= 4)
        shared_words = name_words & desc_words
        if shared_words:
            overlapping.append({
                "symptom": name,
                "n_patients": count,
                "overlapping_words": sorted(shared_words),
            })

    # For each overlapping symptom, count other conditions that share it
    # by scanning full symptoms.csv
    if overlapping:
        overlap_names_lower = {r["symptom"].lower() for r in overlapping}
        other_cond_counts: dict = collections.defaultdict(set)
        target_code_set = set(target_codes)
        target_descs_lower = {d.lower() for d in target_descs}

        for row in _iter_csv(src_dir / "symptoms" / "csv" / "symptoms.csv"):
            path_lower = row.get("PATHOLOGY", "").lower()
            if path_lower in target_descs_lower:
                continue  # skip target itself
            for part in row.get("SYMPTOMS", "").split(";"):
                name = part.split(":")[0].strip().lower()
                if name in overlap_names_lower:
                    other_cond_counts[name].add(path_lower)

        for r in overlapping:
            others = other_cond_counts.get(r["symptom"].lower(), set())
            r["n_other_conditions"] = len(others)
            r["verdict"] = ("shared" if len(others) >= 3
                            else "possible giveaway" if len(others) == 0
                            else "ambiguous")

    return {
        "desc_words_checked": sorted(desc_words),
        "n_overlapping_symptoms": len(overlapping),
        "overlapping": overlapping,
    }


# ── Probes ────────────────────────────────────────────────────────────────────

def _match_negatives(labels: dict, patients: dict, ref_date: date,
                     max_per_pos: int = 3) -> dict:
    """Exact birth year + sex matching. Returns {neg_pid: pseudo_cutoff}."""
    pos_pids = set(labels)
    by_yr_sex: dict = collections.defaultdict(list)
    for pid, p in patients.items():
        if pid not in pos_pids:
            by_yr_sex[(p["birth"].year, p["gender"])].append((pid, p["birth"]))

    neg_cutoffs: dict = {}
    used: set = set()
    for pos_pid, info in sorted(labels.items()):
        p = patients.get(pos_pid)
        if not p:
            continue
        cutoff = _d(info["cutoff"])
        if not cutoff:
            continue
        pos_age = _age(p["birth"], cutoff)
        candidates = by_yr_sex.get((p["birth"].year, p["gender"]), [])
        count = 0
        for neg_pid, neg_birth in candidates:
            if neg_pid in used:
                continue
            try:
                pseudo = neg_birth.replace(year=neg_birth.year + pos_age)
            except ValueError:
                pseudo = date(neg_birth.year + pos_age, neg_birth.month,
                              min(neg_birth.day, 28))
            neg_cutoffs[neg_pid] = min(pseudo, ref_date)
            used.add(neg_pid)
            count += 1
            if count >= max_per_pos:
                break

    return neg_cutoffs


def _build_history_texts(pids_cutoffs: dict, csv_dir: Path) -> dict:
    """Pre-cutoff text per patient from conditions, medications, observations."""
    texts: dict = {pid: [] for pid in pids_cutoffs}
    for stem, col in [("conditions", "START"), ("medications", "START"),
                      ("observations", "DATE"), ("procedures", "START"),
                      ("encounters", "START")]:
        path = csv_dir / f"{stem}.csv"
        if not path.exists():
            continue
        for row in _iter_csv(path):
            pid = row.get("PATIENT", "")
            if pid not in texts:
                continue
            dt = _d(row.get(col, ""))
            if dt is not None and dt < pids_cutoffs[pid]:
                desc = row.get("DESCRIPTION", "")
                if desc:
                    texts[pid].append(desc)
                reason = row.get("REASONDESCRIPTION", "")
                if reason:
                    texts[pid].append(reason)
    return {pid: " ".join(words) for pid, words in texts.items()}


def run_history_probe(labels: dict, neg_cutoffs: dict,
                      scrub_dir: Path, src_dir: Path) -> dict:
    """TF-IDF + LogReg on pre-diagnosis history text, 5-fold CV."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        import numpy as np
    except ImportError:
        return {"error": "scikit-learn not available"}

    pos_cutoffs = {pid: _d(info["cutoff"]) for pid, info in labels.items()}
    pos_texts = _build_history_texts(pos_cutoffs, scrub_dir / "csv")
    neg_texts = _build_history_texts(neg_cutoffs, src_dir / "csv")

    texts = [pos_texts.get(p, "") for p in pos_cutoffs] + \
            [neg_texts.get(p, "") for p in neg_cutoffs]
    y = [1] * len(pos_cutoffs) + [0] * len(neg_cutoffs)
    if len(set(y)) < 2:
        return {"error": "need both classes"}

    vec = TfidfVectorizer(min_df=2, max_features=5000, ngram_range=(1, 2),
                          sublinear_tf=True)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                             random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    baseline = max(sum(y) / len(y), 1.0 - sum(y) / len(y))

    # Top features
    vec.fit(texts)
    clf.fit(vec.transform(texts), y)
    names = vec.get_feature_names_out()
    coef = clf.coef_[0]
    top_pos = [names[i] for i in coef.argsort()[-10:][::-1]]
    top_neg = [names[i] for i in coef.argsort()[:10]]

    return {
        "n_pos": len(pos_cutoffs),
        "n_neg": len(neg_cutoffs),
        "accuracy": round(float(np.mean(scores)), 4),
        "accuracy_std": round(float(np.std(scores)), 4),
        "baseline": round(baseline, 4),
        "lift": round(float(np.mean(scores)) - baseline, 4),
        "top_positive_features": top_pos,
        "top_negative_features": top_neg,
    }


def run_symptoms_probe(presenting: dict, neg_cutoffs: dict) -> dict:
    """Bag-of-symptom-names + LogReg vs matched negatives (no symptoms), 5-fold CV."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import MultiLabelBinarizer
        import numpy as np
    except ImportError:
        return {"error": "scikit-learn not available"}

    pos_pids = sorted(presenting)
    neg_pids = sorted(neg_cutoffs)

    # Feature: set of symptom names for each patient
    pos_features = [[s["name"] for s in presenting[pid]["symptoms"]]
                    for pid in pos_pids]
    neg_features = [[] for _ in neg_pids]

    mlb = MultiLabelBinarizer()
    all_features = pos_features + neg_features
    X = mlb.fit_transform(all_features)
    y = [1] * len(pos_pids) + [0] * len(neg_pids)

    if len(set(y)) < 2 or X.shape[1] == 0:
        return {"error": "insufficient data"}

    # Minimum 5 samples per fold
    n_splits = min(5, min(sum(y), len(y) - sum(y)))
    if n_splits < 2:
        return {"error": "too few samples for CV"}

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                             random_state=42)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    baseline = max(sum(y) / len(y), 1.0 - sum(y) / len(y))

    # Top features
    clf.fit(X, y)
    names = mlb.classes_
    coef = clf.coef_[0]
    top_sym = [names[i] for i in coef.argsort()[-10:][::-1]]

    return {
        "n_pos": len(pos_pids),
        "n_neg": len(neg_pids),
        "accuracy": round(float(np.mean(scores)), 4),
        "accuracy_std": round(float(np.std(scores)), 4),
        "baseline": round(baseline, 4),
        "lift": round(float(np.mean(scores)) - baseline, 4),
        "top_symptom_features": top_sym,
    }


# ── Markdown report ───────────────────────────────────────────────────────────

def write_report(out_path: Path, stem: str, target_descs: list, n_pos: int,
                 n_with_ep: int, presenting: dict,
                 leak: dict, history_probe: dict | None,
                 symptoms_probe: dict | None):
    L = []
    L.append(f"# Presenting Symptoms: {stem}")
    L.append("")
    L.append(f"**Target descriptions:** {', '.join(target_descs)}")
    L.append(f"**Positives:** {n_pos}  **With presenting episode:** {n_with_ep} "
             f"({n_with_ep/n_pos*100:.1f}%)")
    L.append("")

    # Episode distance distribution (skip vitals-only entries which have distance=None)
    dists = [ep["distance"] for ep in presenting.values() if ep.get("distance") is not None]
    if dists:
        from collections import Counter
        dc = Counter(dists)
        dist_str = ", ".join(f"{k}yr: {v}" for k, v in sorted(dc.items())[:8])
        L.append(f"**Episode distances (diagnosis_age − AGE_BEGIN):** {dist_str}")
        L.append("")

    # Leak review
    L.append("## Symptom overlap review")
    L.append("")
    L.append(f"Description words checked (≥4 chars): "
             f"`{', '.join(leak['desc_words_checked'])}`")
    L.append("")
    if not leak["overlapping"]:
        L.append("No symptom names share words with the target description.")
    else:
        L.append("| Symptom | Patients | Overlapping words | Other conditions | Verdict |")
        L.append("|---------|--------:|-------------------|-----------------|---------|")
        for r in leak["overlapping"]:
            L.append(f"| {r['symptom']} | {r['n_patients']} | "
                     f"`{', '.join(r['overlapping_words'])}` | "
                     f"{r.get('n_other_conditions', '?')} | {r.get('verdict', '?')} |")
    L.append("")

    # Probes
    if history_probe is not None:
        L.append("## History-only probe")
        L.append("")
        if "error" in history_probe:
            L.append(f"**Error:** {history_probe['error']}")
        else:
            acc = history_probe["accuracy"]
            base = history_probe["baseline"]
            lift = history_probe["lift"]
            clean = "CLEAN" if abs(lift) < 0.05 else "SIGNAL PRESENT"
            L.append(f"**Accuracy:** {acc:.1%} ± {history_probe['accuracy_std']:.1%}  "
                     f"**Baseline:** {base:.1%}  **Lift:** {lift:+.1%}  → **{clean}**")
            L.append("")
            L.append(f"Top history features (positive direction): "
                     f"`{'`, `'.join(history_probe['top_positive_features'][:5])}`")
        L.append("")

    if symptoms_probe is not None:
        L.append("## Symptoms-only probe")
        L.append("")
        if "error" in symptoms_probe:
            L.append(f"**Error:** {symptoms_probe['error']}")
        else:
            acc = symptoms_probe["accuracy"]
            base = symptoms_probe["baseline"]
            lift = symptoms_probe["lift"]
            L.append(f"**Accuracy:** {acc:.1%} ± {symptoms_probe['accuracy_std']:.1%}  "
                     f"**Baseline:** {base:.1%}  **Lift:** {lift:+.1%}  "
                     f"→ high score expected and not a leak")
            L.append("")
            L.append(f"Top symptom features: "
                     f"`{'`, `'.join(symptoms_probe['top_symptom_features'][:5])}`")
        L.append("")

    # Example record
    L.append("## Example record")
    L.append("")
    if presenting:
        ex_pid = next(iter(presenting))
        ep = presenting[ex_pid]
        dist_str = f"{ep['distance']}yr" if ep.get("distance") is not None else "N/A"
        L.append(f"**Patient:** `{ex_pid}`  "
                 f"**Diagnosis age:** {ep['diagnosis_age']}  "
                 f"**AGE_BEGIN:** {ep.get('age_begin', 'N/A')}  "
                 f"**Distance:** {dist_str}")
        L.append("")
        L.append("```")
        L.append("PRESENTING COMPLAINT")
        for s in ep.get("symptoms", []):
            L.append(f"- {s['name']} (severity: {s['severity']})")
        vitals = ep.get("vitals", [])
        if vitals:
            vital_str = ", ".join(
                f"{v['name'].lower()} {v['value']} {v['unit']}" for v in vitals
            )
            L.append(f"Vital signs at this visit: {vital_str}")
        L.append("```")
    L.append("")

    out_path.write_text("\n".join(L) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scrub_dir")
    ap.add_argument("src_dir")
    ap.add_argument("labels_json")
    ap.add_argument("--out", default=None,
                    help="presenting.json output path")
    ap.add_argument("--report", default=None,
                    help="markdown report path")
    ap.add_argument("--probe", action="store_true",
                    help="run history-only and symptoms-only probes")
    args = ap.parse_args()

    scrub_dir = Path(args.scrub_dir).resolve()
    src_dir = Path(args.src_dir).resolve()
    labels_path = Path(args.labels_json)

    # Load manifest for codes
    manifest = json.loads((scrub_dir / "manifest.json").read_text())
    codes = manifest["codes"]
    stem = scrub_dir.name.split("__scrub-")[1].rsplit("-", 1)[0] if "__scrub-" in scrub_dir.name else scrub_dir.name

    # Determine output paths
    presenting_dir = scrub_dir.parent.parent / "data" / "presenting"
    out_path = Path(args.out) if args.out else presenting_dir / f"{stem}.json"
    report_path = Path(args.report) if args.report else \
        scrub_dir.parent.parent / "docs" / f"PRESENTING_{stem.upper()}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    labels = json.loads(labels_path.read_text())
    n_pos = len(labels)
    print(f"  {stem}: {n_pos} positives", end="", flush=True)

    # Patient birthdates: scrub dir has positives; source run has everyone
    patients = load_patients(src_dir / "csv" / "patients.csv")

    # All target descriptions (match against PATHOLOGY)
    target_descs = target_descriptions(src_dir, codes)

    # Extract presenting symptom episodes
    presenting = extract_presenting(labels, patients, src_dir, target_descs)

    # Load diagnosing-encounter vitals for all positives
    print(" loading vitals ...", end="", flush=True)
    diagnosing_vitals = load_diagnosing_vitals(labels, src_dir, codes)

    # Merge vitals into presenting; also create entries for vitals-only patients
    for pid, label_info in labels.items():
        p = patients.get(pid)
        if not p:
            continue
        cutoff = _d(label_info["cutoff"])
        if not cutoff:
            continue
        vitals = diagnosing_vitals.get(pid, [])
        if pid in presenting:
            presenting[pid]["vitals"] = vitals
        elif vitals:
            diag_age = _age(p["birth"], cutoff)
            presenting[pid] = {
                "diagnosis_age": diag_age,
                "age_begin": None,
                "symptoms": [],
                "vitals": vitals,
                "distance": None,
                "selection_note": "vitals only (no symptom episode)",
            }

    n_with_symptoms = sum(1 for ep in presenting.values() if ep.get("symptoms"))
    n_with_vitals = sum(1 for ep in presenting.values() if ep.get("vitals"))
    n_with = len(presenting)
    print(f" → {n_with} with evidence ({n_with/n_pos*100:.1f}%): "
          f"{n_with_symptoms} symptoms, {n_with_vitals} vitals", flush=True)

    # Probes
    src_manifest = manifest.get("source_manifest", {})
    ref_date = date.fromisoformat(
        src_manifest.get("reference_date", "2099-01-01")[:10]
    )
    neg_cutoffs = _match_negatives(labels, patients, ref_date)

    history_probe = None
    symptoms_probe = None
    if args.probe:
        print("    running history probe ...", end="", flush=True)
        history_probe = run_history_probe(labels, neg_cutoffs, scrub_dir, src_dir)
        acc = history_probe.get("accuracy", "?")
        base = history_probe.get("baseline", "?")
        print(f" acc={acc:.1%} baseline={base:.1%}" if isinstance(acc, float) else f" {history_probe.get('error')}")

        print("    running symptoms probe ...", end="", flush=True)
        symptoms_probe = run_symptoms_probe(presenting, neg_cutoffs)
        acc2 = symptoms_probe.get("accuracy", "?")
        base2 = symptoms_probe.get("baseline", "?")
        print(f" acc={acc2:.1%} baseline={base2:.1%}" if isinstance(acc2, float) else f" {symptoms_probe.get('error')}")

    # Leak review
    leak = leak_review(presenting, target_descs, src_dir, codes)

    # Write outputs
    result = {
        "stem": stem,
        "codes": codes,
        "descriptions": target_descs,
        "n_positives": n_pos,
        "n_with_episode": n_with,
        "n_with_symptoms": n_with_symptoms,
        "n_with_vitals": n_with_vitals,
        "pct_with_episode": round(n_with / n_pos * 100, 1) if n_pos else 0.0,
        "vital_whitelist": VITAL_SIGN_WHITELIST,
        "leak_review": leak,
        "history_probe": history_probe,
        "symptoms_probe": symptoms_probe,
        "presenting": presenting,
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    write_report(report_path, stem, target_descs, n_pos, n_with,
                 presenting, leak, history_probe, symptoms_probe)

    print(f"    wrote {out_path.name}  report {report_path.name}")


if __name__ == "__main__":
    main()

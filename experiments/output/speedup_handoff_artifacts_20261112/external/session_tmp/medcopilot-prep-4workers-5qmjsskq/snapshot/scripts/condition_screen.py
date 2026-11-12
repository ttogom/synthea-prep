#!/usr/bin/env python3
"""
condition_screen.py — antecedent feature screen, classifier probe, and symptom
availability for a single scrubbed condition.

All CSV reading is streaming (no full-file materialisation) to handle the
8.6M-row observations.csv without exhausting RAM.

Usage:
    python3 scripts/condition_screen.py <scrub_dir> <source_dir> <labels_json>
        --symptoms <symptoms_csv> [--out <json_out>]
"""

import argparse
import collections
import csv
import json
import math
import sys
from datetime import date
from pathlib import Path


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


def _load_csv(path: Path) -> list:
    return list(_iter_csv(path))


# ── Negative matching ─────────────────────────────────────────────────────────

def match_negatives(labels: dict, src_patients: list, ref_date: date,
                    max_per_pos: int = 3) -> dict:
    """
    Birth year (±10) + same sex. Returns {neg_pid: pseudo_cutoff_date}.
    """
    pos_pids = set(labels)
    src_map = {r["Id"]: r for r in src_patients}

    by_birth_sex: dict = collections.defaultdict(list)
    for row in src_patients:
        pid = row["Id"]
        if pid in pos_pids:
            continue
        birth = _d(row.get("BIRTHDATE", ""))
        if birth:
            sex = row.get("GENDER", "")
            by_birth_sex[(birth.year, sex)].append((pid, birth))

    neg_cutoffs: dict = {}
    used: set = set()
    for pos_pid, info in sorted(labels.items()):
        pos_row = src_map.get(pos_pid)
        if not pos_row:
            continue
        pos_birth = _d(pos_row.get("BIRTHDATE", ""))
        pos_sex = pos_row.get("GENDER", "")
        if not pos_birth:
            continue
        pos_cutoff = date.fromisoformat(info["cutoff"])
        pos_age = _age(pos_birth, pos_cutoff)

        candidates = []
        for dy in range(11):
            for sign in ([-1, 1] if dy else [0]):
                for neg_pid, neg_birth in by_birth_sex.get(
                        (pos_birth.year + sign * dy, pos_sex), []):
                    if neg_pid not in used:
                        candidates.append((neg_pid, neg_birth, dy))
        candidates.sort(key=lambda x: x[2])

        count = 0
        for neg_pid, neg_birth, _ in candidates:
            if neg_pid in used:
                continue
            try:
                pseudo = neg_birth.replace(year=neg_birth.year + pos_age)
            except ValueError:
                pseudo = date(neg_birth.year + pos_age, neg_birth.month,
                              min(neg_birth.day, 28))
            pseudo = min(pseudo, ref_date)
            neg_cutoffs[neg_pid] = pseudo
            used.add(neg_pid)
            count += 1
            if count >= max_per_pos:
                break

    return neg_cutoffs


# ── Streaming feature extraction ──────────────────────────────────────────────

SMOKING_LOINC = "72166-2"


def extract_features(pids_cutoffs: dict, src_dir: Path) -> dict:
    """
    Binary feature sets per patient, streaming each CSV once.
    Features: cond:<SNOMED>, med:<desc>, obs:<desc>, obs:smoking_ever.
    Returns {pid: frozenset}.
    """
    feats: dict = {pid: set() for pid in pids_cutoffs}

    cond_path = src_dir / "csv" / "conditions.csv"
    if cond_path.exists():
        for row in _iter_csv(cond_path):
            pid = row.get("PATIENT", "")
            if pid not in feats:
                continue
            dt = _d(row.get("START", ""))
            if dt and dt < pids_cutoffs[pid]:
                feats[pid].add(f"cond:{row['CODE']}")

    med_path = src_dir / "csv" / "medications.csv"
    if med_path.exists():
        for row in _iter_csv(med_path):
            pid = row.get("PATIENT", "")
            if pid not in feats:
                continue
            dt = _d(row.get("START", ""))
            if dt and dt < pids_cutoffs[pid]:
                desc = row.get("DESCRIPTION", "").lower().strip()
                if desc:
                    feats[pid].add(f"med:{desc[:80]}")

    obs_path = src_dir / "csv" / "observations.csv"
    if obs_path.exists():
        for row in _iter_csv(obs_path):
            pid = row.get("PATIENT", "")
            if pid not in feats:
                continue
            dt = _d(row.get("DATE", ""))
            if not dt or dt >= pids_cutoffs[pid]:
                continue
            code = row.get("CODE", "")
            desc = row.get("DESCRIPTION", "").lower().strip()
            if code == SMOKING_LOINC:
                val = row.get("VALUE", "").lower()
                if val and "never" not in val:
                    feats[pid].add("obs:smoking_ever")
            if desc:
                feats[pid].add(f"obs:{desc[:80]}")

    return {pid: frozenset(s) for pid, s in feats.items()}


# ── Antecedent screen ─────────────────────────────────────────────────────────

def _z_prop(p1: float, p2: float, n1: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return 0.0
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    denom = math.sqrt(p * (1.0 - p) * (1.0 / n1 + 1.0 / n2))
    return 0.0 if denom == 0 else (p1 - p2) / denom


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(abs(z) / math.sqrt(2))


def antecedent_screen(pos_features: dict, neg_features: dict,
                      code_desc: dict) -> dict:
    all_feats: set = set()
    for s in pos_features.values():
        all_feats.update(s)
    for s in neg_features.values():
        all_feats.update(s)

    n_pos = len(pos_features)
    n_neg = len(neg_features)
    n_feats = max(len(all_feats), 1)
    bonf_threshold = 0.05 / n_feats

    results = []
    for feat in all_feats:
        c_pos = sum(1 for s in pos_features.values() if feat in s)
        c_neg = sum(1 for s in neg_features.values() if feat in s)
        p_pos = c_pos / n_pos if n_pos else 0.0
        p_neg = c_neg / n_neg if n_neg else 0.0
        z = _z_prop(p_pos, p_neg, n_pos, n_neg)
        pval = 2.0 * _norm_sf(z)
        if feat.startswith("cond:"):
            label = code_desc.get(feat[5:], feat)
        elif feat.startswith("med:"):
            label = feat[4:]
        elif feat.startswith("obs:"):
            label = feat[4:]
        else:
            label = feat
        results.append({
            "feature": feat,
            "description": label,
            "p_pos": round(p_pos, 4),
            "p_neg": round(p_neg, 4),
            "n_pos_with": c_pos,
            "n_neg_with": c_neg,
            "z": round(z, 2),
            "pval_two_sided": pval,
            "significant": pval < bonf_threshold,
        })

    results.sort(key=lambda r: -abs(r["z"]))
    sig = [r for r in results if r["significant"]]
    return {
        "n_features_tested": n_feats,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "bonf_threshold": bonf_threshold,
        "n_significant": len(sig),
        "strongest": results[0] if results else None,
        "top_10": results[:10],
    }


# ── Classifier probe ──────────────────────────────────────────────────────────

def _build_probe_texts(pids_cutoffs: dict, src_dir: Path) -> dict:
    """
    Stream each CSV once, keeping only rows for patients in pids_cutoffs.
    Returns {pid: "word word ..."}. O(n_csv_rows) time, O(n_patients) space.
    """
    texts: dict = {pid: [] for pid in pids_cutoffs}

    for stem, col in [("conditions", "START"), ("medications", "START"),
                      ("procedures", "START"), ("encounters", "START"),
                      ("careplans", "START")]:
        path = src_dir / "csv" / f"{stem}.csv"
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

    obs_path = src_dir / "csv" / "observations.csv"
    if obs_path.exists():
        for row in _iter_csv(obs_path):
            pid = row.get("PATIENT", "")
            if pid not in texts:
                continue
            dt = _d(row.get("DATE", ""))
            if dt is not None and dt < pids_cutoffs[pid]:
                desc = row.get("DESCRIPTION", "")
                if desc:
                    texts[pid].append(desc)

    return {pid: " ".join(words) for pid, words in texts.items()}


def run_probe(pos_cutoffs: dict, neg_cutoffs: dict,
              scrub_dir: Path, src_dir: Path) -> dict:
    """TF-IDF + LogReg, class_weight=balanced, 5-fold CV."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        import numpy as np
    except ImportError:
        return {"error": "scikit-learn not available"}

    pos_texts_map = _build_probe_texts(pos_cutoffs, scrub_dir)
    neg_texts_map = _build_probe_texts(neg_cutoffs, src_dir)

    pos_texts = [pos_texts_map.get(pid, "") for pid in pos_cutoffs]
    neg_texts = [neg_texts_map.get(pid, "") for pid in neg_cutoffs]

    texts = pos_texts + neg_texts
    y = [1] * len(pos_texts) + [0] * len(neg_texts)
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

    return {
        "n_pos": len(pos_texts),
        "n_neg": len(neg_texts),
        "accuracy": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "baseline": float(baseline),
        "lift": float(np.mean(scores)) - float(baseline),
    }


# ── Symptom availability ──────────────────────────────────────────────────────

def symptom_availability(pos_pids: set, symptoms_csv: Path,
                          condition_names: list) -> dict:
    if not symptoms_csv or not Path(symptoms_csv).exists():
        return {"error": "symptoms.csv not found"}

    target_lower = {n.lower() for n in condition_names}
    patients_with_symptoms: set = set()
    symptom_names: collections.Counter = collections.Counter()
    total_records = 0

    for row in _iter_csv(symptoms_csv):
        pid = row.get("PATIENT", "")
        if pid not in pos_pids:
            continue
        path_lower = row.get("PATHOLOGY", "").lower()
        if not any(t in path_lower for t in target_lower):
            continue
        syms_raw = row.get("SYMPTOMS", "").strip()
        if not syms_raw:
            continue
        patients_with_symptoms.add(pid)
        total_records += 1
        for sym in syms_raw.split(";"):
            sym = sym.strip()
            if sym:
                name = sym.split(":")[0].strip() if ":" in sym else sym
                symptom_names[name] += 1

    top_symptoms = [s for s, _ in symptom_names.most_common(10)]
    return {
        "n_positives": len(pos_pids),
        "n_with_symptoms": len(patients_with_symptoms),
        "pct_with_symptoms": round(len(patients_with_symptoms) / len(pos_pids) * 100, 1)
                             if pos_pids else 0.0,
        "total_records": total_records,
        "top_symptoms": top_symptoms,
    }


# ── Code description lookup ───────────────────────────────────────────────────

def build_code_desc(src_dir: Path) -> dict:
    d: dict = {}
    path = src_dir / "csv" / "conditions.csv"
    if path.exists():
        for row in _iter_csv(path):
            d[row.get("CODE", "")] = row.get("DESCRIPTION", "")
    return d


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Antecedent feature screen + probe + symptom availability."
    )
    ap.add_argument("scrub_dir")
    ap.add_argument("source_dir")
    ap.add_argument("labels_json")
    ap.add_argument("--symptoms", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    scrub_dir = Path(args.scrub_dir).resolve()
    src_dir = Path(args.source_dir).resolve()
    labels_path = Path(args.labels_json)

    labels = json.loads(labels_path.read_text())
    codes = sorted({info["code"] for info in labels.values()})
    descs = sorted({info["description"] for info in labels.values()})
    pos_cutoffs = {pid: date.fromisoformat(info["cutoff"])
                   for pid, info in labels.items()}

    print(f"Condition : {descs}", file=sys.stderr)
    print(f"Positives : {len(labels)}", file=sys.stderr)

    manifest_path = src_dir / "manifest.json"
    ref_date_str = json.loads(manifest_path.read_text()).get("reference_date", "20260921")
    ref_date = date(int(ref_date_str[:4]), int(ref_date_str[4:6]), int(ref_date_str[6:8]))

    src_patients = _load_csv(src_dir / "csv" / "patients.csv")

    print("Matching negatives...", file=sys.stderr)
    neg_cutoffs = match_negatives(labels, src_patients, ref_date)
    print(f"Negatives : {len(neg_cutoffs)}", file=sys.stderr)

    code_desc = build_code_desc(src_dir)

    print("Building feature vectors (positives)...", file=sys.stderr)
    pos_features = extract_features(pos_cutoffs, scrub_dir)
    print("Building feature vectors (negatives)...", file=sys.stderr)
    neg_features = extract_features(neg_cutoffs, src_dir)

    print("Running antecedent screen...", file=sys.stderr)
    screen = antecedent_screen(pos_features, neg_features, code_desc)

    probe = {"skipped": True}
    if not args.skip_probe:
        print("Running classifier probe...", file=sys.stderr)
        probe = run_probe(pos_cutoffs, neg_cutoffs, scrub_dir, src_dir)
        if "error" not in probe:
            print(f"  Probe: {probe['accuracy']:.1%} ± {probe['accuracy_std']:.1%} "
                  f"(baseline {probe['baseline']:.1%}, lift {probe['lift']:+.1%})",
                  file=sys.stderr)

    symptoms = {"skipped": True}
    if args.symptoms:
        print("Checking symptom availability...", file=sys.stderr)
        symptoms = symptom_availability(set(pos_cutoffs), Path(args.symptoms), descs)
        print(f"  Symptoms: {symptoms['n_with_symptoms']}/{symptoms['n_positives']} "
              f"({symptoms['pct_with_symptoms']}%)", file=sys.stderr)

    result = {
        "codes": codes,
        "descriptions": descs,
        "n_pos": len(labels),
        "n_neg": len(neg_cutoffs),
        "screen": screen,
        "probe": probe,
        "symptoms": symptoms,
    }

    out_str = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(out_str)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(out_str)


if __name__ == "__main__":
    main()

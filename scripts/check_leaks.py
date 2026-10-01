#!/usr/bin/env python3
"""
check_leaks.py — Independent post-scrub leak-check suite.

Four checks against a scrub output directory:

  1. Date assertion  — every date value in every date column of every CSV
                       must be before the patient's cutoff. End dates that
                       survived must also be before it. No row may reference
                       an encounter that does not appear in the scrubbed
                       encounters.csv. No note entry may be dated on or after
                       the cutoff.

  2. Text/code scan  — the target SNOMED code and description must not appear
                       in any CSV field or note. Checks: exact code, full
                       description (tag stripped, case-insensitive), word-
                       order-independent (all distinctive words present), and
                       distinctive partial phrases.

  3. Classifier probe — TF-IDF + logistic regression on scrubbed positive
                        history vs age-matched source-run negatives truncated
                        at a matched age. Reports 5-fold cross-validated
                        accuracy, majority-class baseline, and top 30 features
                        in both directions.

  4. Probe validation — shifts each positive's cutoff one encounter forward
                        (so the diagnosing visit is included) and reruns the
                        probe on source-run data. If accuracy does not rise
                        clearly, the probe is too weak to use.

Implemented from SCRUBBING.md only; no code from scrub.py is read or
imported. Checks 1-2 use the standard library; checks 3-4 require
scikit-learn (pip install scikit-learn).

Usage:
    python3 scripts/check_leaks.py <scrub-dir> <source-dir> <labels.json>
        [--report <path>]   default: docs/LEAK_REPORT.md
        [--skip-probe]      skip checks 3-4 (no sklearn required)
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


# ── Date parsing (independent of scrub.py) ────────────────────────────────────

def _d(s: str):
    """UTC local date from a Synthea date or datetime string. None if empty."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        if len(s) == 10:
            return date.fromisoformat(s)
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo("UTC")).date()
    except ValueError:
        return None


def _age(birth: date, on: date) -> int:
    return on.year - birth.year - ((on.month, on.day) < (birth.month, birth.day))


# ── CSV loading ────────────────────────────────────────────────────────────────

def _load(path: Path) -> list:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _by_pid(path: Path, col: str = "PATIENT") -> dict:
    """Index CSV by patient column. Returns defaultdict(list)."""
    out = defaultdict(list)
    for row in _load(path):
        pid = row.get(col)
        if pid:
            out[pid].append(row)
    return out


# ── Note parsing ──────────────────────────────────────────────────────────────

_NOTE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _note_entries(path: Path) -> list:
    """Parse note file → [(entry_date_or_None, full_entry_text), ...]."""
    lines = path.read_text(errors="replace").splitlines()
    starts = [i for i, ln in enumerate(lines)
              if _NOTE_DATE.match(ln.strip()) and (i == 0 or not lines[i - 1].strip())]
    entries = []
    for n, s in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        entries.append((_d(lines[s].strip()), "\n".join(lines[s:end])))
    return entries


def _pid_from_note(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


# ── Date-column map from SCRUBBING.md ─────────────────────────────────────────
# (stem, filter_column, end_column_or_None)
_DATED = [
    ("conditions",        "START",      "STOP"),
    ("encounters",        "START",      "STOP"),
    ("medications",       "START",      "STOP"),
    ("observations",      "DATE",       None),
    ("procedures",        "START",      "STOP"),
    ("allergies",         "START",      "STOP"),
    ("careplans",         "START",      "STOP"),
    ("devices",           "START",      "STOP"),
    ("imaging_studies",   "DATE",       None),
    ("immunizations",     "DATE",       None),
    ("supplies",          "DATE",       None),
    ("payer_transitions", "START_DATE", "END_DATE"),
]


# ── Check 1: Date assertion ───────────────────────────────────────────────────

def check_dates(scrub_dir: Path, labels: dict) -> list:
    """No post-cutoff dates anywhere in the scrub output. Returns violations."""
    cutoffs = {pid: date.fromisoformat(info["cutoff"]) for pid, info in labels.items()}
    violations = []

    def flag(file, pid, col, val, cutoff, note=""):
        violations.append({"file": file, "patient": pid, "col": col,
                            "value": val, "cutoff": str(cutoff), "note": note})

    # All encounter IDs in the scrubbed output are pre-cutoff.
    # Any ENCOUNTER reference in other CSVs must be in this set.
    valid_enc_ids = set()
    for row in _load(scrub_dir / "csv" / "encounters.csv"):
        pid = row.get("PATIENT", "")
        cutoff = cutoffs.get(pid)
        if cutoff is None:
            continue
        start = _d(row.get("START", ""))
        stop = _d(row.get("STOP", ""))
        if start is not None and start >= cutoff:
            flag("encounters.csv", pid, "START", row["START"], cutoff,
                 "encounter START on/after cutoff")
        if stop is not None and stop >= cutoff:
            flag("encounters.csv", pid, "STOP", row["STOP"], cutoff,
                 "encounter STOP on/after cutoff (should be blank)")
        valid_enc_ids.add(row["Id"])

    # All other dated CSVs.
    for stem, fcol, ecol in _DATED:
        if stem == "encounters":
            continue
        path = scrub_dir / "csv" / f"{stem}.csv"
        for row in _load(path):
            pid = row.get("PATIENT", "")
            cutoff = cutoffs.get(pid)
            if cutoff is None:
                continue

            # Filter column must be strictly before cutoff.
            fval = row.get(fcol, "")
            fd = _d(fval)
            if fd is not None and fd >= cutoff:
                flag(f"{stem}.csv", pid, fcol, fval, cutoff, "filter date on/after cutoff")

            # End column must be absent or strictly before cutoff.
            if ecol:
                eval_ = row.get(ecol, "")
                if eval_:
                    ed = _d(eval_)
                    if ed is not None and ed >= cutoff:
                        flag(f"{stem}.csv", pid, ecol, eval_, cutoff,
                             "end date on/after cutoff (should be blank)")

            # Encounter reference must be in the scrubbed encounters.
            enc_ref = row.get("ENCOUNTER", "")
            if enc_ref and enc_ref not in valid_enc_ids:
                flag(f"{stem}.csv", pid, "ENCOUNTER", enc_ref, cutoff,
                     "row references an encounter absent from scrubbed encounters.csv")

    # Symptoms CSV: check AGE_END (must be < age at cutoff for scrubbed patients).
    sym_path = scrub_dir / "symptoms" / "csv" / "symptoms.csv"
    patients_rows = {r["Id"]: r for r in _load(scrub_dir / "csv" / "patients.csv")}
    for row in _load(sym_path):
        pid = row.get("PATIENT", "")
        cutoff = cutoffs.get(pid)
        if cutoff is None:
            continue
        birth_row = patients_rows.get(pid, {})
        birth = _d(birth_row.get("BIRTHDATE", ""))
        if birth is None:
            continue
        age_at_cutoff = _age(birth, cutoff)
        # AGE_BEGIN must be < age_at_cutoff
        ab = row.get("AGE_BEGIN", "")
        if ab:
            try:
                if int(ab) >= age_at_cutoff:
                    flag("symptoms.csv", pid, "AGE_BEGIN", ab, cutoff,
                         f"AGE_BEGIN {ab} >= cutoff age {age_at_cutoff}")
            except ValueError:
                pass
        # AGE_END if present must be < age_at_cutoff
        ae = row.get("AGE_END", "")
        if ae:
            try:
                if int(ae) >= age_at_cutoff:
                    flag("symptoms.csv", pid, "AGE_END", ae, cutoff,
                         f"AGE_END {ae} >= cutoff age {age_at_cutoff} (should be blank)")
            except ValueError:
                pass

    # Notes: every entry must be dated before cutoff.
    notes_dir = scrub_dir / "notes"
    if notes_dir.is_dir():
        for note_path in sorted(notes_dir.glob("*.txt")):
            pid = _pid_from_note(note_path)
            cutoff = cutoffs.get(pid)
            if cutoff is None:
                continue
            for entry_date, _ in _note_entries(note_path):
                if entry_date is not None and entry_date >= cutoff:
                    flag(f"notes/{note_path.name}", pid, "entry_date",
                         str(entry_date), cutoff, "note entry on/after cutoff")

    return violations


# ── Check 2: Text and code scan ───────────────────────────────────────────────

def check_text(scrub_dir: Path, codes: list, descriptions: list) -> list:
    """Scan every CSV field and note for target code/description. Returns hits."""
    hits = []

    # Patterns
    code_pats = [(c, re.compile(r"\b" + re.escape(c) + r"\b")) for c in codes]

    # Description: strip SNOMED tag, case-insensitive full match.
    desc_base = []
    for desc in descriptions:
        b = re.sub(r"\s*\([^)]+\)\s*$", "", desc).strip()
        desc_base.append((b, re.compile(re.escape(b), re.IGNORECASE)))

    # Word-order-independent: all words present regardless of order.
    word_sets = []
    for desc in descriptions:
        b = re.sub(r"\s*\([^)]+\)\s*$", "", desc).strip()
        words = [w.lower() for w in re.split(r"\W+", b) if len(w) > 2]
        if words:
            word_sets.append((b, words))

    # Partial: distinctive sub-phrases and long individual words.
    partial_pats = []
    for desc in descriptions:
        b = re.sub(r"\s*\([^)]+\)\s*$", "", desc).strip().lower()
        toks = b.split()
        # All n-grams of length 2 up to len-1
        for n in range(2, len(toks)):
            for i in range(len(toks) - n + 1):
                phrase = " ".join(toks[i:i + n])
                partial_pats.append((f"partial:{phrase}",
                                     re.compile(re.escape(phrase), re.IGNORECASE)))
        # Individual long words (> 6 chars)
        for w in toks:
            if len(w) > 6:
                partial_pats.append((f"word:{w}",
                                     re.compile(r"\b" + re.escape(w) + r"\b",
                                                re.IGNORECASE)))

    def _scan(text: str, location: dict):
        t_lower = text.lower()
        for code, pat in code_pats:
            if pat.search(text):
                hits.append({**location, "match": f"code:{code}", "fragment": text[:120]})
                return  # one hit per field is enough
        for base, pat in desc_base:
            if pat.search(text):
                hits.append({**location, "match": f"full_desc:{base}", "fragment": text[:120]})
                return
        for base, words in word_sets:
            if all(w in t_lower for w in words):
                hits.append({**location, "match": f"word_order_ignored:{base}",
                             "fragment": text[:120]})
                return
        for label, pat in partial_pats:
            if pat.search(text):
                hits.append({**location, "match": label, "fragment": text[:120]})
                return

    # Scan CSVs
    for csv_path in sorted((scrub_dir / "csv").glob("*.csv")):
        for rownum, row in enumerate(_load(csv_path), 2):
            for col, val in row.items():
                if val:
                    _scan(val, {"file": f"csv/{csv_path.name}",
                                "row": rownum, "col": col})

    sym = scrub_dir / "symptoms" / "csv" / "symptoms.csv"
    if sym.is_file():
        for rownum, row in enumerate(_load(sym), 2):
            for col, val in row.items():
                if val:
                    _scan(val, {"file": "symptoms/csv/symptoms.csv",
                                "row": rownum, "col": col})

    # Scan notes line by line
    notes_dir = scrub_dir / "notes"
    if notes_dir.is_dir():
        for note_path in sorted(notes_dir.glob("*.txt")):
            pid = _pid_from_note(note_path)
            text = note_path.read_text(errors="replace")
            for lineno, line in enumerate(text.splitlines(), 1):
                line = line.rstrip()
                if line:
                    _scan(line, {"file": f"notes/{note_path.name}",
                                 "row": lineno, "col": "text", "patient": pid})

    return hits


# ── Classifier helpers ────────────────────────────────────────────────────────

def _src_index(src_dir: Path, stems: list) -> dict:
    """Load source CSVs into {stem: defaultdict(pid→[rows])}."""
    idx = {}
    for stem in stems:
        path = src_dir / "csv" / f"{stem}.csv"
        if path.is_file():
            idx[stem] = _by_pid(path)
    return idx


def _patient_text(pid: str, idx: dict, cutoff: date) -> str:
    """Bag-of-words text for a patient from source index, cut at cutoff."""
    parts = []
    for stem, col in [("conditions", "START"), ("medications", "START"),
                      ("procedures", "START"), ("encounters", "START"),
                      ("careplans", "START")]:
        for row in idx.get(stem, {}).get(pid, []):
            d = _d(row.get(col, ""))
            if d is not None and d < cutoff:
                desc = row.get("DESCRIPTION", "")
                if desc:
                    parts.append(desc)
                reason = row.get("REASONDESCRIPTION", "")
                if reason:
                    parts.append(reason)
    for row in idx.get("observations", {}).get(pid, []):
        d = _d(row.get("DATE", ""))
        if d is not None and d < cutoff:
            desc = row.get("DESCRIPTION", "")
            if desc:
                parts.append(desc)
    return " ".join(parts)


def _note_text(pid: str, notes_dir: Path, cutoff: date) -> str:
    """Concatenate note entry text for a patient before the cutoff."""
    if not notes_dir.is_dir():
        return ""
    for note_path in notes_dir.glob("*.txt"):
        if _pid_from_note(note_path) == pid:
            return " ".join(
                text for d, text in _note_entries(note_path)
                if d is not None and d < cutoff
            )
    return ""


def _match_negatives(labels: dict, src_patients: list, max_per_pos: int = 3) -> dict:
    """Return {neg_pid: pseudo_cutoff_date} for age-matched negatives.

    For each positive patient (age A at their cutoff), find source patients
    without the target condition and cut their records at the same age A.
    This controls for record length so the classifier can't simply detect
    'this record ends recently' (the D9 shortcut).
    """
    pos_pids = set(labels)
    by_birthyear = defaultdict(list)
    for row in src_patients:
        pid = row["Id"]
        if pid in pos_pids:
            continue
        birth = _d(row.get("BIRTHDATE", ""))
        if birth:
            by_birthyear[birth.year].append((pid, birth))

    neg_cutoffs = {}
    used = set()
    for pos_pid, info in sorted(labels.items()):
        pos_row = next((r for r in src_patients if r["Id"] == pos_pid), None)
        if pos_row is None:
            continue
        pos_birth = _d(pos_row.get("BIRTHDATE", ""))
        if pos_birth is None:
            continue
        pos_cutoff = date.fromisoformat(info["cutoff"])
        pos_age = _age(pos_birth, pos_cutoff)

        # Gather candidates within ±10 birth years, sorted by proximity.
        candidates = []
        for dy in range(11):
            for sign in ([-1, 1] if dy else [0]):
                for neg_pid, neg_birth in by_birthyear[pos_birth.year + sign * dy]:
                    if neg_pid not in used:
                        candidates.append((neg_pid, neg_birth, dy))
        candidates.sort(key=lambda x: x[2])

        count = 0
        for neg_pid, neg_birth, _ in candidates:
            if neg_pid in used:
                continue
            try:
                pseudo = neg_birth.replace(year=neg_birth.year + pos_age)
            except ValueError:  # Feb 29
                pseudo = date(neg_birth.year + pos_age, neg_birth.month,
                              min(neg_birth.day, 28))
            # Clamp to reference date
            pseudo = min(pseudo, date(2026, 9, 21))
            neg_cutoffs[neg_pid] = pseudo
            used.add(neg_pid)
            count += 1
            if count >= max_per_pos:
                break

    return neg_cutoffs


def _run_probe(pos_texts: list, neg_texts: list, label: str) -> dict:
    """Fit TF-IDF + LogReg and return cross-validated accuracy + top features."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        import numpy as np
    except ImportError:
        return {"error": "scikit-learn not installed"}

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
    baseline = max(sum(y) / len(y), 1 - sum(y) / len(y))

    clf.fit(X, y)
    names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    top_p = [(names[i], float(coefs[i])) for i in coefs.argsort()[-30:][::-1]]
    top_n = [(names[i], float(coefs[i])) for i in coefs.argsort()[:30]]

    return {
        "label": label,
        "n_pos": len(pos_texts), "n_neg": len(neg_texts),
        "accuracy_mean": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "cv_scores": scores.tolist(),
        "baseline": float(baseline),
        "top_pos": top_p,
        "top_neg": top_n,
    }


# ── Check 3: Classifier probe ─────────────────────────────────────────────────

def check_probe(scrub_dir: Path, src_dir: Path, labels: dict) -> dict:
    """Bag-of-words classifier: scrubbed positives vs age-matched source negatives."""
    idx = _src_index(src_dir, ["conditions", "medications", "procedures",
                                "encounters", "careplans", "observations"])
    src_patients = _load(src_dir / "csv" / "patients.csv")
    neg_cutoffs = _match_negatives(labels, src_patients)
    print(f"  {len(labels)} positives, {len(neg_cutoffs)} negatives", file=sys.stderr)

    # Positives: text from the scrubbed output (already cut).
    # Build from scrub output CSVs (no cutoff needed — everything there is pre-dx).
    scrub_idx = _src_index(scrub_dir, ["conditions", "medications", "procedures",
                                        "encounters", "careplans", "observations"])
    pos_texts = [_patient_text(pid, scrub_idx, date(9999, 1, 1))
                 for pid in sorted(labels)]

    neg_texts = [_patient_text(pid, idx, cutoff)
                 for pid, cutoff in sorted(neg_cutoffs.items())]

    return _run_probe(pos_texts, neg_texts, "clean_scrubbed")


# ── Check 4: Probe validation ─────────────────────────────────────────────────

def check_validation(src_dir: Path, labels: dict, probe_result: dict) -> dict:
    """Shift each positive's cutoff one encounter forward; probe should score higher."""
    if "error" in probe_result:
        return {"error": f"skipped — probe failed: {probe_result['error']}"}

    idx = _src_index(src_dir, ["conditions", "medications", "procedures",
                                "encounters", "careplans", "observations"])
    src_patients = _load(src_dir / "csv" / "patients.csv")
    enc_by_pid = _by_pid(src_dir / "csv" / "encounters.csv")
    notes_dir = src_dir / "notes"
    neg_cutoffs = _match_negatives(labels, src_patients)

    # Shifted cutoffs: include the first post-cutoff encounter.
    shifted = {}
    n_shifted = 0
    for pid, info in labels.items():
        original = date.fromisoformat(info["cutoff"])
        post = sorted(
            [e for e in enc_by_pid.get(pid, [])
             if _d(e.get("START", "")) is not None and _d(e["START"]) >= original],
            key=lambda e: _d(e["START"])
        )
        if post:
            shifted[pid] = _d(post[0]["START"]) + timedelta(days=1)
            n_shifted += 1
        else:
            shifted[pid] = original

    print(f"  {n_shifted}/{len(labels)} positives shifted by ≥1 encounter",
          file=sys.stderr)

    # Positives: source run with shifted cutoffs + note text.
    pos_texts = [
        _patient_text(pid, idx, shifted[pid]) + " " +
        _note_text(pid, notes_dir, shifted[pid])
        for pid in sorted(labels)
    ]
    neg_texts = [_patient_text(pid, idx, cutoff)
                 for pid, cutoff in sorted(neg_cutoffs.items())]

    result = _run_probe(pos_texts, neg_texts, "shifted_cutoff")
    result["original_accuracy"] = probe_result["accuracy_mean"]
    result["improvement"] = result["accuracy_mean"] - probe_result["accuracy_mean"]
    result["n_shifted"] = n_shifted
    result["probe_valid"] = result["improvement"] > 0.05
    return result


# ── Report writing ────────────────────────────────────────────────────────────

# For each feature, heuristically classify it for interpretation.
# Judgment call; listed here so the report is explicit about the category.
_CLINICAL_SIGNALS = {
    "ischemic", "coronary", "hypertension", "kidney", "renal",
    "diabetes", "obesity", "anemia", "atrial", "fibrillation",
    "echocardiogram", "cardiomegaly", "infarction", "angina",
    "cardiac", "myocardial", "artery", "ventricular", "aortic",
    "cholesterol", "lipid", "statin", "beta-blocker",
    "lisinopril", "enalapril", "losartan",  # ACE/ARB: pre-dx HTN treatment
}
_LEAK_SIGNALS = {
    "failure", "congestive", "furosemide", "lasix", "digoxin",
    "natriuretic", "bnp", "ejection", "cardiomyopathy",
}


def _classify_feature(feat: str) -> str:
    f = feat.lower()
    for w in _LEAK_SIGNALS:
        if w in f:
            return "POSSIBLE_LEAK"
    for w in _CLINICAL_SIGNALS:
        if w in f:
            return "clinical"
    return "other"


def _feature_table(features: list, direction: str) -> list:
    lines = [f"\n### Top 30 features — {direction}\n",
             "| Feature | Weight | Category |",
             "|---------|--------|----------|"]
    for feat, weight in features:
        cat = _classify_feature(feat)
        lines.append(f"| `{feat}` | {weight:+.3f} | {cat} |")
    return lines


def write_report(path: Path, scrub_dir: Path, labels: dict,
                 violations: list, text_hits: list,
                 probe: dict, validation: dict) -> None:
    L = []
    L.append(f"# Leak Check Report: {scrub_dir.name}\n")
    L.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    codes = sorted({info['code'] for info in labels.values()})
    descs = sorted({info['description'] for info in labels.values()})
    L.append(f"Target codes: {codes}  \nDescriptions: {descs}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    L.append("## Summary\n")
    L.append("| Check | Result |")
    L.append("|-------|--------|")
    L.append(f"| 1. Date assertion | "
             f"{'PASS — 0 violations' if not violations else f'FAIL — {len(violations)} violations'} |")

    # Classify text hits
    code_hits = [h for h in text_hits if h.get("match", "").startswith("code:")]
    full_desc_hits = [h for h in text_hits
                      if h.get("match", "").startswith("full_desc:") or
                         h.get("match", "").startswith("word_order")]
    partial_hits = [h for h in text_hits
                    if h not in code_hits and h not in full_desc_hits]
    scan_result = (f"PASS (0 code/full-desc hits, "
                   f"{len(partial_hits)} partial match(es))"
                   if not code_hits and not full_desc_hits
                   else f"FAIL — {len(code_hits)} code, "
                        f"{len(full_desc_hits)} full-desc hits")
    L.append(f"| 2. Text/code scan | {scan_result} |")

    if "error" not in probe:
        acc = probe["accuracy_mean"]
        base = probe["baseline"]
        lift = acc - base
        L.append(f"| 3. Classifier probe | acc {acc:.1%} ± {probe['accuracy_std']:.1%}, "
                 f"baseline {base:.1%}, lift {lift:+.1%} |")
    else:
        L.append(f"| 3. Classifier probe | ERROR: {probe['error']} |")

    if "error" not in validation:
        valid_str = "✓ probe valid" if validation.get("probe_valid") else "✗ probe too weak"
        L.append(f"| 4. Probe validation | shifted acc {validation['accuracy_mean']:.1%}, "
                 f"improvement {validation['improvement']:+.1%} — {valid_str} |")
    else:
        L.append(f"| 4. Probe validation | {validation['error']} |")

    # ── Check 1 ──────────────────────────────────────────────────────────────
    L.append("\n## Check 1: Date assertion\n")
    L.append("Every filter date, end date, and note entry date in the scrubbed output must "
             "be before the patient's cutoff. ENCOUNTER references must appear in the "
             "scrubbed encounters.csv (which contains only pre-cutoff encounters).\n")
    if violations:
        L.append(f"**{len(violations)} violation(s):**\n")
        L.append("| File | Patient | Col | Value | Cutoff | Note |")
        L.append("|------|---------|-----|-------|--------|------|")
        for v in violations[:50]:
            pid_short = v['patient'][:8] + "…"
            L.append(f"| {v['file']} | {pid_short} | {v['col']} | "
                     f"`{v['value']}` | {v['cutoff']} | {v['note']} |")
        if len(violations) > 50:
            L.append(f"\n*(and {len(violations) - 50} more)*")
    else:
        L.append("**PASS.** No violations found.")

    # ── Check 2 ──────────────────────────────────────────────────────────────
    L.append("\n## Check 2: Text and code scan\n")
    L.append("Checks: exact SNOMED code; full description (tag stripped, case-insensitive); "
             "word-order-independent (all description words present); and partial matches "
             "(distinctive sub-phrases and long individual words).\n")
    if not text_hits:
        L.append("**PASS.** No hits of any kind.")
    else:
        L.append(f"**{len(text_hits)} hit(s)** — "
                 f"{len(code_hits)} exact code, "
                 f"{len(full_desc_hits)} full/word-order, "
                 f"{len(partial_hits)} partial.\n")
        L.append("Partial matches are expected before the cutoff (see SCRUBBING.md "
                 "§Known limitations). Only code and full-description hits are hard failures.\n")
        L.append("| File | Row | Col | Match type | Fragment |")
        L.append("|------|-----|-----|------------|---------|")
        shown = 0
        for h in text_hits[:60]:
            L.append(f"| {h['file']} | {h.get('row','')} | {h.get('col','')} | "
                     f"`{h['match']}` | {h['fragment'][:80]} |")
            shown += 1
        if len(text_hits) > shown:
            L.append(f"\n*(and {len(text_hits) - shown} more — see full output)*")

    # ── Check 3 ──────────────────────────────────────────────────────────────
    L.append("\n## Check 3: Classifier probe\n")
    L.append("TF-IDF (1–2 grams) + logistic regression, 5-fold stratified CV. "
             "Positives: scrubbed CHF patients. Negatives: source-run patients without "
             "CHF, age-matched (birth year ±10) and truncated at the same age to prevent "
             "the D9 record-length shortcut.\n")
    L.append("> **Interpretation note:** CHF is a progressive condition. A high accuracy "
             "is expected from legitimate clinical features (prior cardiac disease, hypertension, "
             "kidney disease). What matters is **what the classifier keys on**. Features "
             "labelled `POSSIBLE_LEAK` (see table) name the target directly or are specific "
             "to CHF treatment; they warrant manual review. The final call is yours.\n")
    if "error" in probe:
        L.append(f"ERROR: {probe['error']}")
    else:
        L.append(f"**Accuracy: {probe['accuracy_mean']:.1%} ± {probe['accuracy_std']:.1%}** "
                 f"(majority-class baseline: {probe['baseline']:.1%}, "
                 f"lift: {probe['accuracy_mean'] - probe['baseline']:+.1%})")
        L.append(f"\nn = {probe['n_pos']} positive, {probe['n_neg']} negative. "
                 f"CV scores: {', '.join(f'{s:.2f}' for s in probe['cv_scores'])}\n")
        L.extend(_feature_table(probe["top_pos"], "CHF-positive (positive coefficient)"))
        L.extend(_feature_table(probe["top_neg"], "CHF-negative (negative coefficient)"))

    # ── Check 4 ──────────────────────────────────────────────────────────────
    L.append("\n## Check 4: Probe validation\n")
    L.append("Shifts each positive's cutoff one encounter forward so the diagnosing visit "
             "(and its note) is included. This dataset deliberately contains the leak. "
             "The shifted probe must score clearly higher (>5 pp) than the clean probe "
             "to confirm the probe can detect a leak of this kind.\n")
    if "error" in validation:
        L.append(f"ERROR: {validation['error']}")
    else:
        L.append(f"**Shifted accuracy: {validation['accuracy_mean']:.1%} ± "
                 f"{validation['accuracy_std']:.1%}** "
                 f"(clean: {validation['original_accuracy']:.1%}, "
                 f"improvement: {validation['improvement']:+.1%})\n")
        L.append(f"{validation['n_shifted']}/{len(labels)} positives shifted by ≥1 encounter.\n")
        if validation.get("probe_valid"):
            L.append("**PROBE VALID.** Shifted accuracy is clearly higher; "
                     "the probe can detect this class of leak.")
        else:
            L.append("**PROBE TOO WEAK.** Shifted accuracy did not improve by >5 pp. "
                     "The probe result in check 3 should be treated as inconclusive.")
        L.extend(_feature_table(validation["top_pos"],
                                "shifted CHF-positive (positive coefficient)"))

    # ── Limitations ──────────────────────────────────────────────────────────
    L.append("\n## What this suite cannot detect\n")
    L.append("""
- **Related codes and text surviving before the cutoff** (SCRUBBING.md §Known
  limitations): conditions like childhood asthma under asthma, or sinusitis
  variants. CHF has no such related Synthea codes, but text like "heart failure"
  may appear in notes before diagnosis (monitoring, family history). The partial
  scan flags these; they fall under known limitations, not scrubbing failures.
- **Numerical signals**: blood pressure trends, BNP levels, and ejection fractions
  in observations.csv are not scanned by the text check. The probe can detect them
  as features (look for observation DESCRIPTION names in the feature table).
- **Implicit information**: a record that stops updating in 2005 may be from a
  patient who died then; the fact of stopping is not a text leak but may be
  clinically informative. The age-matching in check 3 mitigates this for the probe.
- **Probe sensitivity**: the probe uses only condition/medication/procedure/encounter
  descriptions, not note text for the clean dataset. It may miss leaks that only
  appear in notes.
""".strip())

    path.write_text("\n".join(L) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Independent leak-check suite for a scrub output directory."
    )
    ap.add_argument("scrub_dir", help="Path to scrub output directory")
    ap.add_argument("source_dir", help="Path to source Synthea run directory")
    ap.add_argument("labels", help="Path to labels.json from extract_labels.py")
    ap.add_argument("--report", default="docs/LEAK_REPORT.md",
                    help="Path to write the report (default: docs/LEAK_REPORT.md)")
    ap.add_argument("--skip-probe", action="store_true",
                    help="Skip checks 3-4 (no scikit-learn required)")
    args = ap.parse_args()

    scrub_dir = Path(args.scrub_dir).resolve()
    src_dir = Path(args.source_dir).resolve()
    labels_path = Path(args.labels)
    report_path = Path(args.report)

    if not (scrub_dir / "csv").is_dir():
        sys.exit(f"ERROR: {scrub_dir}/csv not found")
    if not (src_dir / "csv").is_dir():
        sys.exit(f"ERROR: {src_dir}/csv not found")
    if not labels_path.is_file():
        sys.exit(f"ERROR: {labels_path} not found")

    labels = json.loads(labels_path.read_text())
    codes = sorted({info["code"] for info in labels.values()})
    descs = sorted({info["description"] for info in labels.values()})

    print(f"Target       : {codes}  —  {descs}")
    print(f"Patients     : {len(labels)}")

    # Check 1
    print("\nCheck 1: Date assertion ...", end=" ", flush=True)
    violations = check_dates(scrub_dir, labels)
    print(f"{'PASS' if not violations else f'FAIL ({len(violations)} violations)'}")

    # Check 2
    print("Check 2: Text/code scan ...", end=" ", flush=True)
    text_hits = check_text(scrub_dir, codes, descs)
    code_hits = [h for h in text_hits if h.get("match", "").startswith("code:")]
    full_hits = [h for h in text_hits if "full_desc" in h.get("match", "") or
                 "word_order" in h.get("match", "")]
    partial_hits = [h for h in text_hits if h not in code_hits and h not in full_hits]
    status = ("PASS" if not code_hits and not full_hits else
              f"FAIL ({len(code_hits)} code, {len(full_hits)} full-desc hits)")
    print(f"{status}, {len(partial_hits)} partial match(es)")

    probe = {"error": "skipped (--skip-probe)"}
    validation = {"error": "skipped (--skip-probe)"}

    if not args.skip_probe:
        # Check 3
        print("Check 3: Classifier probe ...")
        probe = check_probe(scrub_dir, src_dir, labels)
        if "error" not in probe:
            print(f"  Accuracy {probe['accuracy_mean']:.1%} ± {probe['accuracy_std']:.1%} "
                  f"(baseline {probe['baseline']:.1%})")
        else:
            print(f"  ERROR: {probe['error']}")

        # Check 4
        print("Check 4: Probe validation ...")
        validation = check_validation(src_dir, labels, probe)
        if "error" not in validation:
            valid = validation.get("probe_valid")
            print(f"  Shifted {validation['accuracy_mean']:.1%}, "
                  f"improvement {validation['improvement']:+.1%} — "
                  f"{'probe valid' if valid else 'PROBE TOO WEAK'}")
        else:
            print(f"  {validation['error']}")

    # Report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(report_path, scrub_dir, labels, violations, text_hits, probe, validation)
    print(f"\nReport       : {report_path}")


if __name__ == "__main__":
    main()

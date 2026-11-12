#!/usr/bin/env python3
"""
quick_screen.py — lightweight data screen for all candidate conditions.

Reads the source run's CSV files once, then for each candidate computes:
  1. Case count and diagnosis dates (same cutoff rule as scrub.py)
  2. Antecedent comparison: positives vs negatives matched on exact birth year
     and sex, truncated at the same age. Top 10 features by z-score with
     Bonferroni correction.
  3. Symptom availability from raw symptoms.csv.

No scrub, no probe, no leak checks. Those run later on final picks only.

Usage:
    python3 scripts/quick_screen.py [--src <run-dir>] [--out-dir <dir>]

Outputs:
    data/screen/all_results.json
    docs/CONDITION_SCREEN.md
"""

import argparse
import collections
import csv
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / "data"

SMOKING_LOINC = "72166-2"
MAX_NEG_PER_POS = 3

# ── Candidate list ─────────────────────────────────────────────────────────────
CANDIDATES = [
    # (stem, group, codes)
    ("copd",                "history", ["87433001", "185086009"]),
    ("polyp_colon",         "history", ["68496003"]),
    ("hypertension",        "history", ["59621000"]),
    ("strep_throat",        "acute",   ["43878008"]),
    ("viral_pharyngitis",   "acute",   ["195662009"]),
    ("bacterial_sinusitis", "acute",   ["75498004"]),
    ("cystitis",            "acute",   ["307426000"]),
    ("heart_failure",       "demo",    ["88805009"]),
    ("ischemic_heart",      "demo",    ["414545008"]),
    ("sleep_apnea",         "demo",    ["78275009"]),
    ("alzheimers",          "demo",    ["26929004"]),
]


# ── Utilities ──────────────────────────────────────────────────────────────────

def _d(s: str):
    try:
        return date.fromisoformat(s[:10]) if s else None
    except Exception:
        return None


def _age(birth: date, ref: date) -> int:
    return ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))


def _iter_csv(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)


# ── Load source data once ──────────────────────────────────────────────────────

def load_source(src_dir: Path, verbose: bool = True) -> dict:
    """
    Returns a dict with:
      patients     : {pid: {birth, sex}}
      cond_first   : {pid: {code: earliest_start_date}}
      code_desc    : {code: description}
      med_first    : {pid: {desc_lower: earliest_start_date}}
      obs_first    : {pid: {desc_lower_or_smoking_key: earliest_date}}
      ref_date     : date (from manifest.json)
    """
    t0 = time.time()

    manifest = json.loads((src_dir / "manifest.json").read_text())
    ref_str = manifest["reference_date"]
    ref_date = date(int(ref_str[:4]), int(ref_str[4:6]), int(ref_str[6:8]))

    if verbose:
        print("Loading patients ...", file=sys.stderr)
    patients: dict = {}
    for row in _iter_csv(src_dir / "csv" / "patients.csv"):
        pid = row["Id"]
        birth = _d(row.get("BIRTHDATE", ""))
        if birth:
            patients[pid] = {"birth": birth, "sex": row.get("GENDER", "")}

    if verbose:
        print(f"  {len(patients)} patients. Loading conditions ...", file=sys.stderr)
    cond_first: dict = collections.defaultdict(dict)
    code_desc: dict = {}
    n_rows = 0
    for row in _iter_csv(src_dir / "csv" / "conditions.csv"):
        pid, code = row["PATIENT"], row["CODE"]
        dt = _d(row.get("START", ""))
        if not dt:
            continue
        if code not in cond_first[pid] or dt < cond_first[pid][code]:
            cond_first[pid][code] = dt
        code_desc.setdefault(code, row.get("DESCRIPTION", ""))
        n_rows += 1

    if verbose:
        print(f"  {n_rows} condition rows. Loading medications ...", file=sys.stderr)
    med_first: dict = collections.defaultdict(dict)
    n_rows = 0
    for row in _iter_csv(src_dir / "csv" / "medications.csv"):
        pid = row["PATIENT"]
        dt = _d(row.get("START", ""))
        if not dt:
            continue
        desc = row.get("DESCRIPTION", "").lower().strip()[:80]
        if desc:
            if desc not in med_first[pid] or dt < med_first[pid][desc]:
                med_first[pid][desc] = dt
        n_rows += 1

    if verbose:
        print(f"  {n_rows} medication rows. Loading observations (streaming) ...",
              file=sys.stderr)
    obs_first: dict = collections.defaultdict(dict)
    n_rows = 0
    for row in _iter_csv(src_dir / "csv" / "observations.csv"):
        pid = row["PATIENT"]
        dt = _d(row.get("DATE", ""))
        if not dt:
            continue
        code = row.get("CODE", "")
        desc = row.get("DESCRIPTION", "").lower().strip()[:80]
        # Smoking: track earliest non-"never" record
        if code == SMOKING_LOINC:
            val = row.get("VALUE", "").lower()
            if val and "never" not in val:
                k = "__smoking__"
                if k not in obs_first[pid] or dt < obs_first[pid][k]:
                    obs_first[pid][k] = dt
        # Observation type (binary presence)
        if desc:
            if desc not in obs_first[pid] or dt < obs_first[pid][desc]:
                obs_first[pid][desc] = dt
        n_rows += 1

    if verbose:
        print(f"  {n_rows} observation rows in {time.time()-t0:.1f}s.", file=sys.stderr)

    return {
        "patients": patients,
        "cond_first": dict(cond_first),
        "code_desc": code_desc,
        "med_first": dict(med_first),
        "obs_first": dict(obs_first),
        "ref_date": ref_date,
    }


def load_symptoms(sym_path: Path) -> dict:
    """
    Returns {pid: {pathology_lower: [symptom_name, ...]}}
    from raw symptoms.csv (no time filter).
    """
    by_pid: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    if not sym_path.exists():
        return {}
    for row in _iter_csv(sym_path):
        pid = row.get("PATIENT", "")
        path_lower = row.get("PATHOLOGY", "").lower()
        syms_raw = row.get("SYMPTOMS", "").strip()
        if not pid or not path_lower or not syms_raw:
            continue
        for sym in syms_raw.split(";"):
            sym = sym.strip()
            if sym:
                name = sym.split(":")[0].strip() if ":" in sym else sym
                if name:
                    by_pid[pid][path_lower].append(name)
    return {pid: dict(d) for pid, d in by_pid.items()}


# ── Per-candidate helpers ──────────────────────────────────────────────────────

def find_cases(codes: list, cond_first: dict) -> dict:
    """Return {pid: earliest_diagnosis_date} for patients with any of the codes."""
    code_set = set(codes)
    cases: dict = {}
    for pid, cond_map in cond_first.items():
        for code in code_set:
            if code in cond_map:
                dt = cond_map[code]
                if pid not in cases or dt < cases[pid]:
                    cases[pid] = dt
    return cases


def match_negatives(pos_cases: dict, patients: dict, ref_date: date,
                    max_per_pos: int = MAX_NEG_PER_POS) -> dict:
    """
    Exact birth year + sex. Returns {neg_pid: pseudo_cutoff_date}.
    """
    pos_pids = set(pos_cases)
    neg_by_group: dict = collections.defaultdict(list)
    for pid, info in patients.items():
        if pid not in pos_pids:
            neg_by_group[(info["birth"].year, info["sex"])].append(pid)

    neg_cutoffs: dict = {}
    used: set = set()

    for pos_pid, cutoff in sorted(pos_cases.items()):
        pos_info = patients.get(pos_pid)
        if not pos_info:
            continue
        pos_age = _age(pos_info["birth"], cutoff)
        key = (pos_info["birth"].year, pos_info["sex"])

        count = 0
        for neg_pid in neg_by_group.get(key, []):
            if neg_pid in used:
                continue
            neg_birth = patients[neg_pid]["birth"]
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


def build_features(pids_cutoffs: dict, cond_first: dict, med_first: dict,
                   obs_first: dict) -> dict:
    """
    {pid: frozenset_of_feature_strings}
    Features: cond:<CODE>, med:<desc>, obs:<desc>, obs:smoking_ever.
    Only events strictly before the cutoff date.
    """
    result: dict = {}
    for pid, cutoff in pids_cutoffs.items():
        s: set = set()
        for code, first_dt in cond_first.get(pid, {}).items():
            if first_dt < cutoff:
                s.add(f"cond:{code}")
        for desc, first_dt in med_first.get(pid, {}).items():
            if first_dt < cutoff:
                s.add(f"med:{desc}")
        for key, first_dt in obs_first.get(pid, {}).items():
            if first_dt < cutoff:
                if key == "__smoking__":
                    s.add("obs:smoking_ever")
                else:
                    s.add(f"obs:{key}")
        result[pid] = frozenset(s)
    return result


# ── Antecedent screen ──────────────────────────────────────────────────────────

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
    bonf = 0.05 / n_feats

    results = []
    for feat in all_feats:
        c_pos = sum(1 for s in pos_features.values() if feat in s)
        c_neg = sum(1 for s in neg_features.values() if feat in s)
        p_pos = c_pos / n_pos if n_pos else 0.0
        p_neg = c_neg / n_neg if n_neg else 0.0
        z = _z_prop(p_pos, p_neg, n_pos, n_neg)
        pval = 2.0 * _norm_sf(z)

        if feat.startswith("cond:"):
            label = code_desc.get(feat[5:], feat[5:])
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
            "pval": pval,
            "significant": pval < bonf,
        })

    results.sort(key=lambda r: -abs(r["z"]))
    sig = [r for r in results if r["significant"]]
    return {
        "n_features": n_feats,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "bonf_threshold": round(bonf, 8),
        "n_significant": len(sig),
        "top_10": results[:10],
    }


# ── Symptom availability ───────────────────────────────────────────────────────

def symptom_check(pos_pids: set, descriptions: list, sym_data: dict) -> dict:
    """
    Counts positives with symptom records whose PATHOLOGY matches any target
    description. Returns counts and top symptom names.
    """
    target_lower = {d.lower() for d in descriptions}
    # Also try short name (strip SNOMED tag)
    for d in list(target_lower):
        short = d.split(" (")[0]
        target_lower.add(short)

    patients_with: set = set()
    sym_counter: collections.Counter = collections.Counter()

    for pid in pos_pids:
        if pid not in sym_data:
            continue
        for path_lower, syms in sym_data[pid].items():
            if any(t in path_lower for t in target_lower):
                patients_with.add(pid)
                sym_counter.update(syms)

    top = [s for s, _ in sym_counter.most_common(10)]
    return {
        "n_positives": len(pos_pids),
        "n_with_symptoms": len(patients_with),
        "pct_with_symptoms": round(len(patients_with) / len(pos_pids) * 100, 1)
                             if pos_pids else 0.0,
        "top_symptoms": top,
    }


# ── Per-candidate computation ──────────────────────────────────────────────────

def run_candidate(stem: str, group: str, codes: list,
                  src_data: dict, sym_data: dict) -> dict:
    patients  = src_data["patients"]
    cond_first = src_data["cond_first"]
    med_first  = src_data["med_first"]
    obs_first  = src_data["obs_first"]
    code_desc  = src_data["code_desc"]
    ref_date   = src_data["ref_date"]

    # 1. Find cases
    pos_cases = find_cases(codes, cond_first)
    n_pos = len(pos_cases)

    # Descriptions from code_desc
    descriptions = sorted({code_desc.get(c, c) for c in codes})

    # 2. Match negatives
    neg_cutoffs = match_negatives(pos_cases, patients, ref_date)
    n_neg = len(neg_cutoffs)

    # 3. Build feature vectors
    pos_features = build_features(pos_cases, cond_first, med_first, obs_first)
    neg_features = build_features(neg_cutoffs, cond_first, med_first, obs_first)

    # 4. Antecedent screen
    screen = antecedent_screen(pos_features, neg_features, code_desc)

    # 5. Symptom availability
    sym = symptom_check(set(pos_cases), descriptions, sym_data)

    return {
        "stem": stem,
        "group": group,
        "codes": codes,
        "descriptions": descriptions,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "screen": screen,
        "symptoms": sym,
    }


# ── Markdown output ────────────────────────────────────────────────────────────

def _fmt_strongest(r: dict) -> str:
    top = r.get("screen", {}).get("top_10", [])
    if not top:
        return "—"
    best = top[0]
    sig = "\\*" if best.get("significant") else ""
    desc = best["description"][:45]
    return f"{desc}{sig} (z={best['z']:.1f})"


def _fmt_syms(r: dict) -> str:
    s = r.get("symptoms", {})
    n = s.get("n_with_symptoms", 0)
    t = s.get("n_positives", 0)
    pct = s.get("pct_with_symptoms", 0)
    top = s.get("top_symptoms", [])[:3]
    return f"{n}/{t} ({pct}%) — {', '.join(top)}" if top else f"{n}/{t} ({pct}%)"


def write_markdown(results: dict, out_path: Path):
    L = []
    L.append("# Condition Data Screen")
    L.append("")
    L.append("Generated by `scripts/quick_screen.py` from the 10 k UTC run. "
             "For each candidate: cases, antecedent feature comparison "
             "(positives vs exact-birth-year+sex-matched negatives, truncated "
             "at same age), and symptom availability from raw `symptoms.csv`. "
             "No scrub or classifier probe — those run on final picks only.")
    L.append("")
    L.append("**Methodology note:** z-scores use the two-proportion test. "
             "Bonferroni threshold = 0.05 / N_features. "
             "\\* = survives Bonferroni correction.")
    L.append("")

    # Summary table
    group_order = ["history", "acute", "demo"]
    stem_order = [s for g in group_order for s, grp, _ in CANDIDATES if grp == g]

    L.append("## Summary")
    L.append("")
    L.append("| Group | Condition | Cases | Matched neg | "
             "Strongest antecedent | Symptom availability |")
    L.append("|-------|-----------|------:|------------:|"
             "---------------------|---------------------|")
    for stem in stem_order:
        r = results.get(stem, {})
        if not r:
            continue
        name = " + ".join(d.split(" (")[0] for d in r.get("descriptions", [stem]))
        L.append(f"| {r['group']} | {name} | {r['n_pos']} | {r['n_neg']} | "
                 f"{_fmt_strongest(r)} | {_fmt_syms(r)} |")

    L.append("")
    L.append("\\* Bonferroni-corrected at α = 0.05 / N_features.")
    L.append("")

    # Per-condition sections
    for stem in stem_order:
        r = results.get(stem)
        if not r:
            continue
        name = " + ".join(d.split(" (")[0] for d in r.get("descriptions", []))
        L.append(f"## {name}")
        L.append("")
        L.append(f"**Group:** {r['group']}  "
                 f"**Code(s):** {', '.join(r['codes'])}  "
                 f"**Cases:** {r['n_pos']}  "
                 f"**Matched negatives:** {r['n_neg']}")
        L.append("")

        screen = r.get("screen", {})
        n_feats = screen.get("n_features", 0)
        n_sig   = screen.get("n_significant", 0)
        bonf    = screen.get("bonf_threshold", 0)
        L.append(f"**Antecedent screen:** {n_feats} features, "
                 f"Bonferroni threshold p < {bonf:.2e}, "
                 f"{n_sig} significant features.")
        L.append("")
        top10 = screen.get("top_10", [])
        if top10:
            L.append("| Rank | Feature (type:detail) | p pos | p neg | z | Sig |")
            L.append("|-----:|-----------------------|------:|------:|--:|-----|")
            for i, f in enumerate(top10, 1):
                sig_mark = "\\*" if f.get("significant") else ""
                desc = f.get("description", f["feature"])[:60]
                L.append(f"| {i} | {desc} | "
                         f"{f['p_pos']:.3f} | {f['p_neg']:.3f} | "
                         f"{f['z']:.1f} | {sig_mark} |")
        L.append("")

        sym = r.get("symptoms", {})
        n_sym = sym.get("n_with_symptoms", 0)
        pct   = sym.get("pct_with_symptoms", 0)
        tops  = sym.get("top_symptoms", [])
        L.append(f"**Symptom availability:** {n_sym}/{r['n_pos']} positives "
                 f"({pct}%) have at least one symptom record with this pathology "
                 f"in the raw `symptoms.csv`.")
        if tops:
            L.append(f"Top symptoms: {', '.join(tops[:8])}")
        L.append("")

    out_path.write_text("\n".join(L) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Quick data screen for all candidates.")
    ap.add_argument("--src", default=str(DATA / "pop10000-seed20260916"))
    ap.add_argument("--out-dir", default=str(DATA / "screen"))
    args = ap.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Load all source data once
    print("=== Loading source data ===", file=sys.stderr)
    src_data = load_source(src_dir)
    print("=== Loading symptoms ===", file=sys.stderr)
    sym_data = load_symptoms(src_dir / "symptoms" / "csv" / "symptoms.csv")
    load_time = time.time() - t_start
    print(f"Data loaded in {load_time:.1f}s", file=sys.stderr)

    # Run each candidate
    results: dict = {}
    for stem, group, codes in CANDIDATES:
        t_c = time.time()
        print(f"\n--- {stem} ({group}, codes={codes}) ---", file=sys.stderr)
        r = run_candidate(stem, group, codes, src_data, sym_data)
        elapsed = time.time() - t_c
        top = r["screen"]["top_10"][0] if r["screen"]["top_10"] else {}
        print(f"  n_pos={r['n_pos']}  n_neg={r['n_neg']}  "
              f"n_features={r['screen']['n_features']}  "
              f"n_sig={r['screen']['n_significant']}  "
              f"({elapsed:.1f}s)", file=sys.stderr)
        if top:
            print(f"  top: {top['description'][:50]}  z={top['z']:.1f}  "
                  f"({top['p_pos']:.0%} pos / {top['p_neg']:.0%} neg)  "
                  f"sig={top['significant']}", file=sys.stderr)
        sym = r["symptoms"]
        print(f"  symptoms: {sym['n_with_symptoms']}/{sym['n_positives']} "
              f"({sym['pct_with_symptoms']}%)", file=sys.stderr)
        results[stem] = r
        (out_dir / f"{stem}.json").write_text(json.dumps(r, indent=2, default=str))

    (out_dir / "all_results.json").write_text(
        json.dumps(results, indent=2, default=str))

    write_markdown(results, ROOT / "docs" / "CONDITION_SCREEN.md")

    total = time.time() - t_start
    print(f"\n=== Done in {total:.1f}s ===", file=sys.stderr)
    print(f"Wrote docs/CONDITION_SCREEN.md and {out_dir}/all_results.json", file=sys.stderr)


if __name__ == "__main__":
    main()

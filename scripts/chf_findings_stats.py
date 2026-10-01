"""
Compute all statistics cited in docs/CHF_FINDINGS.md.

Usage:
    python3 scripts/chf_findings_stats.py \\
        data/pop10000-seed20260916__scrub-heart_failure-9e8474 \\
        data/pop10000-seed20260916 \\
        data/labels-9e8474.json \\
        [--leak-report docs/LEAK_REPORT_10k.md] \\
        [--out docs/chf_findings_stats.json]

Keyword definitions mirror scripts/review_cohort.py exactly.
"No cardiac antecedent" = none of: 7 drug classes, 5 cardiac conditions, kidney disease.

Outputs a JSON file and prints a human-readable summary.
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


# ── Shared keyword definitions (mirror review_cohort.py) ──────────────────────

DRUG_CLASSES = {
    "beta_blocker": [
        "metoprolol", "atenolol", "carvedilol", "bisoprolol", "propranolol",
        "nadolol", "nebivolol", "acebutolol", "betaxolol", "labetalol",
    ],
    "ace_inhibitor": [
        "lisinopril", "enalapril", "ramipril", "captopril", "fosinopril",
        "perindopril", "quinapril", "benazepril", "trandolapril", "moexipril",
    ],
    "arb": [
        "losartan", "valsartan", "olmesartan", "irbesartan", "candesartan",
        "telmisartan", "azilsartan", "eprosartan",
    ],
    "diuretic": [
        "furosemide", "hydrochlorothiazide", "spironolactone", "torsemide",
        "chlorthalidone", "bumetanide", "indapamide", "metolazone",
        "triamterene", "eplerenone", "amiloride",
    ],
    "nitrate": ["nitroglycerin", "isosorbide mononitrate", "isosorbide dinitrate"],
    "antiplatelet": [
        "clopidogrel", "ticagrelor", "prasugrel", "dipyridamole",
        "aspirin 81", "aspirin 325",
    ],
    "statin": [
        "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
        "lovastatin", "fluvastatin", "pitavastatin",
    ],
}

CARDIAC_GROUPS = {
    "ischemic_heart": [
        "ischemic heart disease", "coronary artery disease", "coronary heart",
        "angina pectoris", "stable angina", "unstable angina",
        "coronary atherosclerosis",
    ],
    "mi": [
        "myocardial infarction", "heart attack", "stemi", "nstemi",
        "acute coronary syndrome",
    ],
    "afib": ["atrial fibrillation", "atrial flutter"],
    "hypertension": [
        "hypertension (", "essential hypertension", "hypertensive disorder",
        "hypertensive disease",
    ],
    "hyperlipidemia": [
        "hyperlipidemia", "hypercholesterolemia", "dyslipidemia",
        "hyperlipoproteinemia",
    ],
}

KIDNEY_KWS = [
    "chronic kidney disease", "renal failure", "renal insufficiency",
    "end-stage renal", "end stage renal", "nephropathy",
]

DRUG_ORDER = ["beta_blocker", "ace_inhibitor", "arb", "diuretic",
              "nitrate", "antiplatelet", "statin"]
COND_ORDER = ["ischemic_heart", "mi", "afib", "hypertension", "hyperlipidemia"]

HBP_REASON_KW = ["hypertension", "hypertensive", "blood pressure"]
CHF_REASON_KW  = ["heart failure", "chf", "congestive", "cardiomyopathy",
                   "fluid overload", "cardiac decompensation"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path):
    p = Path(path)
    if not p.is_file():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def by_pid(rows, col="PATIENT"):
    out = defaultdict(list)
    for r in rows:
        out[r[col]].append(r)
    return out


def matches_any(desc, keywords):
    d = desc.lower()
    return any(k in d for k in keywords)


def has_drug(rows, keywords):
    return any(matches_any(r.get("DESCRIPTION", ""), keywords) for r in rows)


def has_cond(rows, keywords):
    return any(matches_any(r.get("DESCRIPTION", ""), keywords) for r in rows)


def has_kidney(rows):
    return has_cond(rows, KIDNEY_KWS)


def z2(p1, n1, p2, n2):
    """Two-proportion z-test. Returns (z, ci_lo, ci_hi)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool <= 0 or p_pool >= 1:
        return 0.0, p1 - p2, p1 - p2
    se_z = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_z if se_z > 0 else 0.0
    se_ci = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return z, (p1 - p2) - 1.96 * se_ci, (p1 - p2) + 1.96 * se_ci


def birth_year(pat):
    bd = pat.get("BIRTHDATE", "") or pat.get("BIRTH_DATE", "")
    return bd[:4]


def age_at(pat, date_str):
    return int(date_str[:4]) - int(birth_year(pat))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scrub_dir")
    ap.add_argument("src_dir")
    ap.add_argument("labels")
    ap.add_argument("--leak-report", default="docs/LEAK_REPORT_10k.md")
    ap.add_argument("--out", default="docs/chf_findings_stats.json")
    args = ap.parse_args()

    scrub = Path(args.scrub_dir)
    src   = Path(args.src_dir)

    labels_raw = json.loads(Path(args.labels).read_text())
    pos_ids = set(labels_raw.keys())
    cutoffs = {pid: v["cutoff"] for pid, v in labels_raw.items()}

    print("Loading patients…", file=sys.stderr)
    scrub_pats = {r["Id"]: r for r in load_csv(scrub / "csv/patients.csv")}
    src_pats   = {r["Id"]: r for r in load_csv(src   / "csv/patients.csv")}

    # ── Cohort description ────────────────────────────────────────────────────

    ages = []
    sexes = {"M": 0, "F": 0}
    for pid in sorted(pos_ids):
        pat = scrub_pats.get(pid) or src_pats.get(pid)
        if pat is None:
            continue
        ages.append(age_at(pat, cutoffs[pid]))
        sexes[pat.get("GENDER", "?")] = sexes.get(pat.get("GENDER", "?"), 0) + 1

    ages.sort()
    n_pos = len(ages)
    age_bins = {
        "<50":   sum(1 for a in ages if a < 50),
        "50-59": sum(1 for a in ages if 50 <= a < 60),
        "60-69": sum(1 for a in ages if 60 <= a < 70),
        "70-79": sum(1 for a in ages if 70 <= a < 80),
        "80+":   sum(1 for a in ages if a >= 80),
    }

    # ── Zero cardiac antecedent ───────────────────────────────────────────────
    # "No cardiac antecedent" = none of:
    #   7 drug classes (DRUG_ORDER), 5 cardiac conditions (COND_ORDER), kidney disease
    # This mirrors the "None of the above" row in review_cohort.py SUMMARY_TABLE.

    print("Loading scrub CSVs…", file=sys.stderr)
    pos_conds = by_pid(load_csv(scrub / "csv/conditions.csv"))
    pos_meds  = by_pid(load_csv(scrub / "csv/medications.csv"))

    def any_drug(pid):
        rows = pos_meds.get(pid, [])
        return any(has_drug(rows, kws) for kws in DRUG_CLASSES.values())

    def any_cardiac_cond(pid):
        rows = pos_conds.get(pid, [])
        return any(has_cond(rows, kws) for kws in CARDIAC_GROUPS.values())

    n_zero = sum(
        1 for pid in pos_ids
        if not any_cardiac_cond(pid) and
           not any_drug(pid) and
           not has_kidney(pos_conds.get(pid, []))
    )

    # ── Matching pool ─────────────────────────────────────────────────────────
    # Negatives: exact birth year + sex, sorted by patient ID for determinism,
    # up to 3 per positive.

    print("Matching negatives…", file=sys.stderr)
    neg_pool = {pid: p for pid, p in src_pats.items() if pid not in pos_ids}
    neg_pool_sorted = sorted(neg_pool.items())  # deterministic order

    neg_cutoff = {}  # neg_id -> pseudo_cutoff
    used = set()

    for pos_pid in sorted(pos_ids):
        pos_pat = scrub_pats.get(pos_pid) or src_pats.get(pos_pid)
        if pos_pat is None:
            continue
        pos_by = birth_year(pos_pat)
        pos_sex = pos_pat.get("GENDER", "")
        pos_age = age_at(pos_pat, cutoffs[pos_pid])

        found = 0
        for nid, np_ in neg_pool_sorted:
            if nid in used:
                continue
            if birth_year(np_) == pos_by and np_.get("GENDER") == pos_sex:
                neg_by = int(birth_year(np_))
                pseudo = f"{neg_by + pos_age}-{cutoffs[pos_pid][5:]}"
                neg_cutoff[nid] = pseudo
                used.add(nid)
                found += 1
                if found == 3:
                    break

    n_neg = len(neg_cutoff)
    print(f"  n_pos={n_pos}, n_neg={n_neg}, unmatched={(n_pos - (n_neg // 3))}",
          file=sys.stderr)

    # ── Load neg source CSVs filtered by pseudo-cutoff ────────────────────────

    print("Loading source CSVs for negatives…", file=sys.stderr)
    neg_conds = defaultdict(list)
    neg_meds  = defaultdict(list)
    for row in load_csv(src / "csv/conditions.csv"):
        pid = row["PATIENT"]
        if pid in neg_cutoff and row.get("START", "") < neg_cutoff[pid]:
            neg_conds[pid].append(row)
    for row in load_csv(src / "csv/medications.csv"):
        pid = row["PATIENT"]
        if pid in neg_cutoff and row.get("START", "") < neg_cutoff[pid]:
            neg_meds[pid].append(row)

    pos_ids_list = sorted(pos_ids)
    neg_ids_list = sorted(neg_cutoff.keys())

    # ── Antecedent comparison table ───────────────────────────────────────────

    ALL_ANTECEDENTS = [
        ("ischemic_heart",  "cond", CARDIAC_GROUPS["ischemic_heart"]),
        ("mi",              "cond", CARDIAC_GROUPS["mi"]),
        ("afib",            "cond", CARDIAC_GROUPS["afib"]),
        ("hypertension",    "cond", CARDIAC_GROUPS["hypertension"]),
        ("hyperlipidemia",  "cond", CARDIAC_GROUPS["hyperlipidemia"]),
        ("ckd",             "cond", KIDNEY_KWS),
        ("diuretic",        "drug", DRUG_CLASSES["diuretic"]),
        ("ace_inhibitor",   "drug", DRUG_CLASSES["ace_inhibitor"]),
        ("beta_blocker",    "drug", DRUG_CLASSES["beta_blocker"]),
        ("arb",             "drug", DRUG_CLASSES["arb"]),
        ("nitrate",         "drug", DRUG_CLASSES["nitrate"]),
        ("antiplatelet",    "drug", DRUG_CLASSES["antiplatelet"]),
        ("statin",          "drug", DRUG_CLASSES["statin"]),
    ]

    antecedent_rows = []
    bonferroni_z = 2.576 * math.sqrt(13)  # approx; proper: scipy.stats.norm.ppf(1-0.05/26)
    # Correct: alpha/2 per tail, 13 comparisons -> alpha_per = 0.05/13, z = norm.ppf(1-0.05/26)
    # z ≈ 2.897 (we compute it below without scipy)
    # From standard table: P(Z>2.897) ≈ 0.00192, two-tailed p = 0.00385 = 0.05/13
    bonferroni_z = 2.897

    for name, kind, kws in ALL_ANTECEDENTS:
        if kind == "drug":
            np_h = sum(1 for pid in pos_ids_list if has_drug(pos_meds.get(pid, []), kws))
            nn_h = sum(1 for pid in neg_ids_list if has_drug(neg_meds.get(pid, []), kws))
        else:
            np_h = sum(1 for pid in pos_ids_list if has_cond(pos_conds.get(pid, []), kws))
            nn_h = sum(1 for pid in neg_ids_list if has_cond(neg_conds.get(pid, []), kws))
        p1 = np_h / n_pos
        p2 = nn_h / n_neg
        z, ci_lo, ci_hi = z2(p1, n_pos, p2, n_neg)
        antecedent_rows.append({
            "name": name, "kind": kind,
            "pos_n": np_h, "pos_pct": round(p1 * 100, 1),
            "neg_n": nn_h, "neg_pct": round(p2 * 100, 1),
            "diff_pp": round((p1 - p2) * 100, 1),
            "z": round(z, 2),
            "ci_lo": round(ci_lo * 100, 1),
            "ci_hi": round(ci_hi * 100, 1),
            "bonferroni": abs(z) >= bonferroni_z,
            "nominal": abs(z) >= 1.96,
        })

    # ── Drug reason check (diuretics + ACE) ──────────────────────────────────

    diur_rows = [r for pid in pos_ids_list for r in pos_meds.get(pid, [])
                 if has_drug([r], DRUG_CLASSES["diuretic"])]
    ace_rows  = [r for pid in pos_ids_list for r in pos_meds.get(pid, [])
                 if has_drug([r], DRUG_CLASSES["ace_inhibitor"])]

    def chf_reason(rows):
        return sum(1 for r in rows
                   if matches_any(r.get("REASONDESCRIPTION", ""), CHF_REASON_KW))

    diur_chf_reason = chf_reason(diur_rows)
    ace_chf_reason  = chf_reason(ace_rows)

    # ── Hypertensive-only comparison ─────────────────────────────────────────

    htn_kws = CARDIAC_GROUPS["hypertension"]
    pos_htn = [pid for pid in pos_ids_list if has_cond(pos_conds.get(pid, []), htn_kws)]
    neg_htn = [pid for pid in neg_ids_list if has_cond(neg_conds.get(pid, []), htn_kws)]

    n_ph, n_nh = len(pos_htn), len(neg_htn)

    def htn_drug_rate(pids, meds_dict, kws):
        n = sum(1 for pid in pids if has_drug(meds_dict.get(pid, []), kws))
        return n, n / len(pids) if pids else 0.0

    hctz_kws = ["hydrochlorothiazide"]
    lisi_kws = ["lisinopril"]

    ph_hctz_n, ph_hctz_p = htn_drug_rate(pos_htn, pos_meds,  hctz_kws)
    nh_hctz_n, nh_hctz_p = htn_drug_rate(neg_htn, neg_meds,  hctz_kws)
    ph_lisi_n, ph_lisi_p = htn_drug_rate(pos_htn, pos_meds,  lisi_kws)
    nh_lisi_n, nh_lisi_p = htn_drug_rate(neg_htn, neg_meds,  lisi_kws)

    hctz_z, hctz_cilo, hctz_cihi = z2(ph_hctz_p, n_ph, nh_hctz_p, n_nh)
    lisi_z, lisi_cilo, lisi_cihi = z2(ph_lisi_p, n_ph, nh_lisi_p, n_nh)

    # ── HCTZ first-start timing ───────────────────────────────────────────────

    def first_hctz_years_before(pid, cutoff_str, meds_dict):
        rows = [r for r in meds_dict.get(pid, [])
                if matches_any(r.get("DESCRIPTION", ""), hctz_kws)]
        if not rows:
            return None
        earliest = min(r["START"] for r in rows)
        years = (int(cutoff_str[:4]) - int(earliest[:4]) +
                 (int(cutoff_str[5:7]) - int(earliest[5:7])) / 12)
        return years

    def bin_years(y):
        if y > 10:  return ">10"
        if y > 5:   return "5-10"
        if y > 2:   return "2-5"
        if y > 1:   return "1-2"
        return "<1"

    BIN_ORDER = [">10", "5-10", "2-5", "1-2", "<1"]

    pos_hctz_bins = defaultdict(int)
    for pid in pos_htn:
        y = first_hctz_years_before(pid, cutoffs[pid], pos_meds)
        if y is not None:
            pos_hctz_bins[bin_years(y)] += 1
    n_pos_hctz = sum(pos_hctz_bins.values())

    neg_hctz_bins = defaultdict(int)
    for pid in neg_htn:
        y = first_hctz_years_before(pid, neg_cutoff[pid], neg_meds)
        if y is not None:
            neg_hctz_bins[bin_years(y)] += 1
    n_neg_hctz = sum(neg_hctz_bins.values())

    # ── Parse probe stats from leak report ───────────────────────────────────

    probe_acc = probe_sd = probe_baseline = probe_lift = probe_shifted = None
    leak_path = Path(args.leak_report)
    if leak_path.is_file():
        text = leak_path.read_text()
        m = re.search(r"Accuracy:\s*([\d.]+)%\s*[±]\s*([\d.]+)%.*baseline:\s*([\d.]+)%.*lift:\s*([+-][\d.]+)%", text)
        if m:
            probe_acc, probe_sd, probe_baseline, probe_lift = (
                float(m.group(1)), float(m.group(2)),
                float(m.group(3)), float(m.group(4)))
        m2 = re.search(r"Shifted accuracy:\s*([\d.]+)%", text)
        if m2:
            probe_shifted = float(m2.group(1))

    # ── Assemble output ───────────────────────────────────────────────────────

    stats = {
        "source": {
            "scrub_dir": str(scrub),
            "src_dir":   str(src),
            "labels":    args.labels,
        },
        "cohort": {
            "n_pos": n_pos,
            "n_neg_matched": n_neg,
            "age_min": ages[0],
            "age_median": ages[len(ages) // 2],
            "age_max": ages[-1],
            "age_bins": {k: {"n": v, "pct": round(v / n_pos * 100, 1)}
                         for k, v in age_bins.items()},
            "sex_M": {"n": sexes.get("M", 0), "pct": round(sexes.get("M", 0) / n_pos * 100, 1)},
            "sex_F": {"n": sexes.get("F", 0), "pct": round(sexes.get("F", 0) / n_pos * 100, 1)},
            "n_zero_antecedent": n_zero,
            "pct_zero_antecedent": round(n_zero / n_pos * 100, 1),
        },
        "probe": {
            "acc": probe_acc, "sd": probe_sd,
            "baseline": probe_baseline, "lift": probe_lift,
            "shifted_acc": probe_shifted,
            "source": str(leak_path),
        },
        "antecedents": antecedent_rows,
        "drug_reason_check": {
            "diuretic_total_rows": len(diur_rows),
            "diuretic_chf_reason_rows": diur_chf_reason,
            "ace_total_rows": len(ace_rows),
            "ace_chf_reason_rows": ace_chf_reason,
        },
        "hypertensives": {
            "pos_n": n_ph, "neg_n": n_nh,
            "hctz": {
                "pos_n": ph_hctz_n, "pos_pct": round(ph_hctz_p * 100, 1),
                "neg_n": nh_hctz_n, "neg_pct": round(nh_hctz_p * 100, 1),
                "diff_pp": round((ph_hctz_p - nh_hctz_p) * 100, 1),
                "z": round(hctz_z, 2),
                "ci_lo": round(hctz_cilo * 100, 1),
                "ci_hi": round(hctz_cihi * 100, 1),
            },
            "lisinopril": {
                "pos_n": ph_lisi_n, "pos_pct": round(ph_lisi_p * 100, 1),
                "neg_n": nh_lisi_n, "neg_pct": round(nh_lisi_p * 100, 1),
                "diff_pp": round((ph_lisi_p - nh_lisi_p) * 100, 1),
                "z": round(lisi_z, 2),
                "ci_lo": round(lisi_cilo * 100, 1),
                "ci_hi": round(lisi_cihi * 100, 1),
            },
        },
        "hctz_timing": {
            "pos_n_on_hctz": n_pos_hctz,
            "neg_n_on_hctz": n_neg_hctz,
            "pos_bins": {b: {"n": pos_hctz_bins[b],
                             "pct": round(pos_hctz_bins[b] / n_pos_hctz * 100, 1)
                             if n_pos_hctz else 0}
                         for b in BIN_ORDER},
            "neg_bins": {b: {"n": neg_hctz_bins[b],
                             "pct": round(neg_hctz_bins[b] / n_neg_hctz * 100, 1)
                             if n_neg_hctz else 0}
                         for b in BIN_ORDER},
        },
    }

    # ── Print summary ─────────────────────────────────────────────────────────

    print()
    print("=== Cohort ===")
    print(f"n_pos={n_pos}, n_neg={n_neg}")
    print(f"Age: min={ages[0]}, median={ages[len(ages)//2]}, max={ages[-1]}")
    for b, v in age_bins.items():
        print(f"  {b}: {v} ({v/n_pos*100:.1f}%)")
    print(f"Sex: M={sexes.get('M',0)} ({sexes.get('M',0)/n_pos*100:.1f}%), "
          f"F={sexes.get('F',0)} ({sexes.get('F',0)/n_pos*100:.1f}%)")
    print(f"Zero antecedent (no drugs/conditions/kidney): {n_zero}/{n_pos} ({n_zero/n_pos*100:.1f}%)")

    if probe_acc is not None:
        print(f"\n=== Probe (from {leak_path.name}) ===")
        print(f"Acc={probe_acc}% ± {probe_sd}%, baseline={probe_baseline}%, "
              f"lift={probe_lift}%, shifted={probe_shifted}%")

    print("\n=== Antecedent comparison ===")
    BONF = 2.897
    print(f"{'Name':<22} {'POS%':>6} {'NEG%':>6} {'Diff':>7} {'z':>7}  {'95% CI':>21}")
    for r in antecedent_rows:
        sig = "**" if r["bonferroni"] else ("*" if r["nominal"] else "")
        print(f"{r['name']:<22} {r['pos_pct']:>5.1f}% {r['neg_pct']:>5.1f}% "
              f"{r['diff_pp']:>+6.1f}pp {r['z']:>+7.2f}  "
              f"[{r['ci_lo']:>+5.1f}%, {r['ci_hi']:>+5.1f}%]  {sig}")
    print(f"** = Bonferroni (|z|≥{BONF}), * = p<0.05 only")

    print(f"\n=== Drug reason check ===")
    print(f"Diuretic rows: {len(diur_rows)}, CHF-reason rows: {diur_chf_reason}")
    print(f"ACE rows:      {len(ace_rows)}, CHF-reason rows: {ace_chf_reason}")

    print(f"\n=== Hypertensives only (pos n={n_ph}, neg n={n_nh}) ===")
    print(f"HCTZ:      pos {ph_hctz_n}/{n_ph} ({ph_hctz_p*100:.1f}%)  "
          f"neg {nh_hctz_n}/{n_nh} ({nh_hctz_p*100:.1f}%)  "
          f"diff {(ph_hctz_p-nh_hctz_p)*100:+.1f}pp  z={hctz_z:+.2f}  "
          f"CI[{hctz_cilo*100:+.1f}%, {hctz_cihi*100:+.1f}%]")
    print(f"Lisinopril: pos {ph_lisi_n}/{n_ph} ({ph_lisi_p*100:.1f}%)  "
          f"neg {nh_lisi_n}/{n_nh} ({nh_lisi_p*100:.1f}%)  "
          f"diff {(ph_lisi_p-nh_lisi_p)*100:+.1f}pp  z={lisi_z:+.2f}  "
          f"CI[{lisi_cilo*100:+.1f}%, {lisi_cihi*100:+.1f}%]")

    print(f"\n=== HCTZ first-start timing ===")
    print(f"{'Bin':<8} {'CHF+ n':>7} {'CHF+ %':>7}  {'NEG n':>7} {'NEG %':>7}")
    for b in BIN_ORDER:
        pn = pos_hctz_bins[b]; pp = pn/n_pos_hctz*100 if n_pos_hctz else 0
        nn = neg_hctz_bins[b]; np2 = nn/n_neg_hctz*100 if n_neg_hctz else 0
        print(f"{b:<8} {pn:>7} {pp:>6.1f}%  {nn:>7} {np2:>6.1f}%")
    print(f"(pos n={n_pos_hctz}, neg n={n_neg_hctz})")

    # ── Write JSON ────────────────────────────────────────────────────────────

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

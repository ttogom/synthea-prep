#!/usr/bin/env python3
"""
review_cohort.py — Generate per-patient review files for a scrubbed cohort.

Produces:
  <out-dir>/patient_NN.md      — per-patient timeline, richest records first
  <out-dir>/SUMMARY_TABLE.md   — clinical indicators, one row per patient
  <out-dir>/MAPPINGS.md        — drug/condition classification used

Usage:
    python3 scripts/review_cohort.py <scrub-dir> <labels.json>
        [--out-dir docs/review]
        [--source-dir <src>]    only needed for observations (if not in scrub dir)

Standard library only. All classification is rule-based keyword matching on
DESCRIPTION strings — MAPPINGS.md lists every match for verification.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path


# ── Clinical classification tables ───────────────────────────────────────────

# Drug classification: case-insensitive substring match on DESCRIPTION
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
        "aspirin 81", "aspirin 325",   # only low-dose/antiplatelet aspirin
    ],
    "statin": [
        "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
        "lovastatin", "fluvastatin", "pitavastatin",
    ],
}

# Cardiac condition classification: case-insensitive substring
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

CARDIAC_OBS_KWS = [
    "bnp", "brain natriuretic", "nt-probnp", "ejection fraction",
    "troponin", "creatine kinase-mb", "left ventricular ejection",
    "cardiac output",
]

BP_SYS_KWS = ["systolic blood pressure", "systolic bp", "bp systolic"]
BP_DIA_KWS = ["diastolic blood pressure", "diastolic bp", "bp diastolic"]


def _classify_drug(desc: str) -> list:
    d = desc.lower()
    return [cls for cls, terms in DRUG_CLASSES.items() if any(t in d for t in terms)]


def _classify_cond(desc: str) -> list:
    d = desc.lower()
    return [g for g, terms in CARDIAC_GROUPS.items() if any(t in d for t in terms)]


def _is_kidney(desc: str) -> bool:
    d = desc.lower()
    return any(k in d for k in KIDNEY_KWS)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _load(path: Path) -> list:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _by_pid(path: Path, pid_col: str = "PATIENT") -> dict:
    out = defaultdict(list)
    for row in _load(path):
        p = row.get(pid_col)
        if p:
            out[p].append(row)
    return out


def _d(s: str):
    s = (s or "").strip()[:10]
    if len(s) == 10:
        try:
            return date.fromisoformat(s)
        except ValueError:
            pass
    return None


def _age(birth: date, on: date) -> int:
    return on.year - birth.year - ((on.month, on.day) < (birth.month, birth.day))


# ── Observation helpers ───────────────────────────────────────────────────────

def _obs_groups(rows: list) -> dict:
    """Group numeric observations by DESCRIPTION. Returns {desc: [(date, val, units)]}."""
    groups = defaultdict(list)
    for r in rows:
        d = _d(r.get("DATE", ""))
        val_s = (r.get("VALUE") or "").strip()
        desc = r.get("DESCRIPTION", "")
        if not d or not desc:
            continue
        try:
            val = float(val_s)
        except ValueError:
            continue
        groups[desc].append((d, val, r.get("UNITS", "")))
    for v in groups.values():
        v.sort(key=lambda x: x[0])
    return groups


def _slope_per_year(pairs):
    """(date, float) → linear slope per year. Returns 0 if < 2 points."""
    if len(pairs) < 2:
        return 0.0
    xs = [p[0].toordinal() for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope_per_day = num / den if den else 0.0
    return slope_per_day * 365.25


def _bp_summary(obs: dict, cutoff: date) -> tuple:
    """Returns (last_sys_str, last_dia_str, trend_label)."""
    five_ago = date(cutoff.year - 5, cutoff.month, cutoff.day)

    def _latest(kws):
        for desc, vals in obs.items():
            if any(k in desc.lower() for k in kws):
                before = [(d, v, u) for d, v, u in vals if d < cutoff]
                if before:
                    return before[-1]
        return None

    def _recent(kws):
        result = []
        for desc, vals in obs.items():
            if any(k in desc.lower() for k in kws):
                result += [(d, v) for d, v, _ in vals if five_ago <= d < cutoff]
        return sorted(result)

    ls = _latest(BP_SYS_KWS)
    ld = _latest(BP_DIA_KWS)
    last_sys = f"{ls[1]:.0f} {ls[2]} ({ls[0]})" if ls else "—"
    last_dia = f"{ld[1]:.0f} {ld[2]} ({ld[0]})" if ld else "—"

    recent_sys = _recent(BP_SYS_KWS)
    if len(recent_sys) >= 2:
        sp = _slope_per_year(recent_sys)
        if sp > 2:
            trend = f"rising (+{sp:.1f}/yr over 5y, n={len(recent_sys)})"
        elif sp < -2:
            trend = f"falling ({sp:.1f}/yr over 5y, n={len(recent_sys)})"
        else:
            trend = f"flat ({sp:+.1f}/yr over 5y, n={len(recent_sys)})"
    else:
        trend = f"insufficient ({len(recent_sys)} readings in 5y)"

    return last_sys, last_dia, trend


# ── Timeline builder ─────────────────────────────────────────────────────────

def _build_timeline(enc_rows, cond_rows, med_rows, proc_rows, care_rows, cutoff):
    """Returns list of encounter dicts sorted by date, each with linked events."""
    enc_by_id = {r["Id"]: r for r in enc_rows}
    linked = defaultdict(lambda: {"cond": [], "med": [], "proc": [], "care": []})

    for r in cond_rows:
        e = r.get("ENCOUNTER", "")
        if e in enc_by_id:
            linked[e]["cond"].append(r)
    for r in med_rows:
        e = r.get("ENCOUNTER", "")
        if e in enc_by_id:
            linked[e]["med"].append(r)
    for r in proc_rows:
        e = r.get("ENCOUNTER", "")
        if e in enc_by_id:
            linked[e]["proc"].append(r)
    for r in care_rows:
        e = r.get("ENCOUNTER", "")
        if e in enc_by_id:
            linked[e]["care"].append(r)

    events = []
    for r in enc_rows:
        d = _d(r.get("START", ""))
        if d and d < cutoff:
            events.append({"date": d, "enc": r, "linked": linked[r["Id"]]})

    events.sort(key=lambda e: e["date"])
    return events


def _render_patient_md(pid, info, pat_row, events, obs, sym_rows, cutoff) -> str:
    birth = _d(pat_row.get("BIRTHDATE", ""))
    age_at = _age(birth, cutoff) if birth else "?"
    sex = pat_row.get("GENDER", "?")
    race = pat_row.get("RACE", "?")
    eth = pat_row.get("ETHNICITY", "?")

    first_date = events[0]["date"] if events else cutoff
    years = (cutoff - first_date).days / 365.25 if events else 0

    # Distinct counts
    distinct_conds = {r["DESCRIPTION"] for e in events for r in e["linked"]["cond"]}
    distinct_meds = {r["DESCRIPTION"] for e in events for r in e["linked"]["med"]}

    lines = [
        f"# Patient — {pid[:8]}…",
        f"",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Sex | {sex} |",
        f"| Race/ethnicity | {race} / {eth} |",
        f"| Born | {pat_row.get('BIRTHDATE', '?')} |",
        f"| Age at cutoff | {age_at} |",
        f"| Cutoff | {cutoff} |",
        f"| Code | {info['code']} |",
        f"",
        f"**Encounters:** {len(events)}  "
        f"**Years of history:** {years:.1f}  "
        f"**Distinct conditions:** {len(distinct_conds)}  "
        f"**Distinct medications:** {len(distinct_meds)}",
        f"",
        f"---",
        f"",
        f"## Pre-diagnosis timeline (oldest → newest)",
        f"",
    ]

    for e in events:
        enc = e["enc"]
        d = e["date"]
        cls = enc.get("ENCOUNTERCLASS", "")
        reason = enc.get("REASONDESCRIPTION") or enc.get("REASONCODE") or ""
        desc = enc.get("DESCRIPTION", "")

        hdr = f"### {d} — `{cls}`"
        if desc:
            hdr += f" — {desc}"
        if reason:
            hdr += f"\n> Reason: *{reason}*"
        lines.append(hdr)
        lines.append("")

        for r in e["linked"]["cond"]:
            stop = r.get("STOP", "")
            tag = f" _(resolved {stop[:10]})_" if stop and stop[:10] < str(cutoff) else ""
            lines.append(f"- **Condition** {r['DESCRIPTION']}{tag}")
        for r in e["linked"]["med"]:
            stop = r.get("STOP", "")
            tag = f" _(stopped {stop[:10]})_" if stop else ""
            lines.append(f"- **Medication** {r['DESCRIPTION']}{tag}")
        for r in e["linked"]["proc"]:
            lines.append(f"- Procedure: {r['DESCRIPTION']}")
        for r in e["linked"]["care"]:
            stop = r.get("STOP", "")
            tag = f" _(to {stop[:10]})_" if stop else ""
            lines.append(f"- Careplan: {r['DESCRIPTION']}{tag}")
        lines.append("")

    # Observation summary
    lines += ["---", "", "## Observations (numeric, summarized)", ""]
    if obs:
        lines.append("| Description | Count | Range | Last 3 (date: value) |")
        lines.append("|-------------|-------|-------|----------------------|")
        for desc, vals in sorted(obs.items()):
            nums = [(d, v) for d, v, _ in vals if d < cutoff]
            if not nums:
                continue
            units = next((u for _, _, u in vals if u), "")
            lo, hi = min(v for _, v in nums), max(v for _, v in nums)
            last3 = nums[-3:]
            last3_str = "  ".join(f"{d}: {v:.1f}" for d, v in last3)
            rng = f"{lo:.1f}–{hi:.1f} {units}".strip()
            lines.append(f"| {desc} | {len(nums)} | {rng} | {last3_str} |")
    else:
        lines.append("*(no numeric observations)*")
    lines.append("")

    # BP section
    last_sys, last_dia, trend = _bp_summary(obs, cutoff)
    lines += [
        "### Blood pressure",
        f"- Last systolic: {last_sys}",
        f"- Last diastolic: {last_dia}",
        f"- Systolic trend (final 5y before cutoff): {trend}",
        "",
    ]

    # Symptoms
    lines += ["---", "", "## Symptoms", ""]
    if sym_rows:
        lines.append("| Pathology | Symptom | Age begin | Age end |")
        lines.append("|-----------|---------|-----------|---------|")
        for r in sym_rows:
            lines.append(f"| {r.get('PATHOLOGY','')} | {r.get('SYMPTOM','')} | "
                         f"{r.get('AGE_BEGIN','')} | {r.get('AGE_END','')} |")
    else:
        lines.append("*(none)*")
    lines.append("")

    return "\n".join(lines)


# ── Summary table ────────────────────────────────────────────────────────────

def _patient_indicators(pid, cutoff, events, obs):
    """Returns a dict of clinical indicators for the summary table."""
    # Collect all pre-cutoff descriptions
    all_cond_descs = [r["DESCRIPTION"] for e in events for r in e["linked"]["cond"]]
    all_med_descs = [r["DESCRIPTION"] for e in events for r in e["linked"]["med"]]

    # Drug classes
    drug_hits = defaultdict(list)
    for desc in all_med_descs:
        for cls in _classify_drug(desc):
            if desc not in drug_hits[cls]:
                drug_hits[cls].append(desc)

    # Cardiac conditions
    cond_hits = defaultdict(list)
    for desc in all_cond_descs:
        for g in _classify_cond(desc):
            if desc not in cond_hits[g]:
                cond_hits[g].append(desc)

    # Kidney disease
    kidney = [d for d in all_cond_descs if _is_kidney(d)]
    kidney = list(dict.fromkeys(kidney))

    # Cardiac observations
    cardiac_obs = [desc for desc in obs
                   if any(k in desc.lower() for k in CARDIAC_OBS_KWS)]

    # BP
    last_sys, last_dia, trend = _bp_summary(obs, cutoff)

    return {
        "drug_classes": dict(drug_hits),
        "cardiac_conds": dict(cond_hits),
        "kidney": kidney,
        "cardiac_obs": cardiac_obs,
        "last_sys": last_sys,
        "bp_trend": trend,
        "n_enc": len(events),
        "years": round((cutoff - events[0]["date"]).days / 365.25, 1) if events else 0,
        "n_cond": len({r["DESCRIPTION"] for e in events for r in e["linked"]["cond"]}),
        "n_med": len({r["DESCRIPTION"] for e in events for r in e["linked"]["med"]}),
    }


def _render_summary_table(patients_data: list) -> str:
    """patients_data: list of (rank, pid, cutoff, age, sex, ind) dicts."""
    DRUG_ORDER = ["beta_blocker", "ace_inhibitor", "arb", "diuretic",
                  "nitrate", "antiplatelet", "statin"]
    COND_ORDER = ["ischemic_heart", "mi", "afib", "hypertension", "hyperlipidemia"]

    lines = [
        "# Cohort Summary Table",
        "",
        "Cardiac medications and conditions present **before the diagnosis cutoff**.",
        "✓ = at least one matching record found; — = none found.",
        "",
        "## Drug classes",
        "",
    ]

    # Drug class table
    hdr = "| # | PID | Cutoff | Age | Sex | Enc | Yrs | beta_blocker | ace_inhibitor | arb | diuretic | nitrate | antiplatelet | statin |"
    lines.append(hdr)
    lines.append("|---|-----|--------|-----|-----|-----|-----|---|---|---|---|---|---|---|")
    for p in patients_data:
        ind = p["ind"]
        drug_cells = []
        for cls in DRUG_ORDER:
            hits = ind["drug_classes"].get(cls, [])
            drug_cells.append("✓ " + ", ".join(h.split()[0] for h in hits[:2]) if hits else "—")
        row = (f"| {p['rank']} | {p['pid'][:8]} | {p['cutoff']} | {p['age']} | {p['sex']} | "
               f"{ind['n_enc']} | {ind['years']} | " +
               " | ".join(drug_cells) + " |")
        lines.append(row)

    lines += [
        "",
        "## Cardiac/vascular conditions",
        "",
    ]
    hdr2 = "| # | PID | ischemic_heart | MI | afib | hypertension | hyperlipidemia | kidney_disease | cardiac_obs |"
    lines.append(hdr2)
    lines.append("|---|-----|---|---|---|---|---|---|---|")
    for p in patients_data:
        ind = p["ind"]
        cond_cells = []
        for g in COND_ORDER:
            hits = ind["cardiac_conds"].get(g, [])
            cond_cells.append("✓ " + hits[0][:30] if hits else "—")
        kidney_cell = "✓ " + ind["kidney"][0][:30] if ind["kidney"] else "—"
        obs_cell = "✓ " + ", ".join(ind["cardiac_obs"][:2]) if ind["cardiac_obs"] else "—"
        row = (f"| {p['rank']} | {p['pid'][:8]} | " +
               " | ".join(cond_cells) + f" | {kidney_cell} | {obs_cell} |")
        lines.append(row)

    lines += [
        "",
        "## Blood pressure",
        "",
        "| # | PID | Last systolic | 5y trend |",
        "|---|-----|---------------|----------|",
    ]
    for p in patients_data:
        ind = p["ind"]
        lines.append(f"| {p['rank']} | {p['pid'][:8]} | {ind['last_sys']} | {ind['bp_trend']} |")

    # Tallies
    lines += ["", "## Tallies (n = 34)", ""]
    lines.append("| Indicator | Count | % |")
    lines.append("|-----------|-------|---|")
    for cls in DRUG_ORDER:
        n = sum(1 for p in patients_data if p["ind"]["drug_classes"].get(cls))
        lines.append(f"| Drug: {cls} | {n} | {n*100//34}% |")
    for g in COND_ORDER:
        n = sum(1 for p in patients_data if p["ind"]["cardiac_conds"].get(g))
        lines.append(f"| Cond: {g} | {n} | {n*100//34}% |")
    n = sum(1 for p in patients_data if p["ind"]["kidney"])
    lines.append(f"| Kidney disease | {n} | {n*100//34}% |")
    n = sum(1 for p in patients_data if p["ind"]["cardiac_obs"])
    lines.append(f"| Cardiac observations | {n} | {n*100//34}% |")
    n = sum(1 for p in patients_data
            if not any(p["ind"]["drug_classes"].get(cls) for cls in DRUG_ORDER) and
               not any(p["ind"]["cardiac_conds"].get(g) for g in COND_ORDER) and
               not p["ind"]["kidney"])
    lines.append(f"| **None of the above** | **{n}** | **{n*100//34}%** |")

    return "\n".join(lines) + "\n"


def _render_mappings(patients_data: list) -> str:
    """All distinct medication and condition descriptions that were classified."""
    drug_map = defaultdict(set)
    cond_map = defaultdict(set)

    for p in patients_data:
        ind = p["ind"]
        for cls, descs in ind["drug_classes"].items():
            drug_map[cls].update(descs)
        for g, descs in ind["cardiac_conds"].items():
            cond_map[g].update(descs)

    lines = [
        "# Classification mappings",
        "",
        "Every distinct description that matched a classification rule.",
        "",
        "## Drug classes",
        "",
    ]
    for cls in ["beta_blocker", "ace_inhibitor", "arb", "diuretic",
                "nitrate", "antiplatelet", "statin"]:
        lines.append(f"### {cls}")
        for d in sorted(drug_map.get(cls, [])):
            lines.append(f"- {d}")
        if not drug_map.get(cls):
            lines.append("*(no matches)*")
        lines.append("")

    lines += ["## Cardiac conditions", ""]
    for g in ["ischemic_heart", "mi", "afib", "hypertension", "hyperlipidemia"]:
        lines.append(f"### {g}")
        for d in sorted(cond_map.get(g, [])):
            lines.append(f"- {d}")
        if not cond_map.get(g):
            lines.append("*(no matches)*")
        lines.append("")

    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scrub_dir")
    ap.add_argument("labels")
    ap.add_argument("--out-dir", default="docs/review")
    args = ap.parse_args()

    scrub_dir = Path(args.scrub_dir).resolve()
    labels = json.loads(Path(args.labels).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = scrub_dir / "csv"
    sym_dir = scrub_dir / "symptoms" / "csv"

    print(f"Loading CSVs from {csv_dir} …", file=sys.stderr)
    pat_map = {r["Id"]: r for r in _load(csv_dir / "patients.csv")}
    enc_by_pid = _by_pid(csv_dir / "encounters.csv")
    cond_by_pid = _by_pid(csv_dir / "conditions.csv")
    med_by_pid = _by_pid(csv_dir / "medications.csv")
    proc_by_pid = _by_pid(csv_dir / "procedures.csv")
    care_by_pid = _by_pid(csv_dir / "careplans.csv")

    print("Loading observations (large) …", file=sys.stderr)
    obs_by_pid = _by_pid(csv_dir / "observations.csv")

    sym_by_pid = _by_pid(sym_dir / "symptoms.csv")

    # Build per-patient data
    all_patients = []
    for pid, info in labels.items():
        cutoff = date.fromisoformat(info["cutoff"])
        pat_row = pat_map.get(pid, {})
        birth = _d(pat_row.get("BIRTHDATE", ""))
        age_at = _age(birth, cutoff) if birth else "?"
        sex = pat_row.get("GENDER", "?")

        events = _build_timeline(
            enc_by_pid.get(pid, []),
            cond_by_pid.get(pid, []),
            med_by_pid.get(pid, []),
            proc_by_pid.get(pid, []),
            care_by_pid.get(pid, []),
            cutoff,
        )
        obs = _obs_groups([r for r in obs_by_pid.get(pid, [])
                           if _d(r.get("DATE", "")) is not None and
                              _d(r.get("DATE", "")) < cutoff])
        sym_rows = [r for r in sym_by_pid.get(pid, [])
                    if r.get("PATIENT") == pid]

        ind = _patient_indicators(pid, cutoff, events, obs)
        all_patients.append({
            "pid": pid, "info": info, "cutoff": cutoff,
            "age": age_at, "sex": sex, "pat_row": pat_row,
            "events": events, "obs": obs, "sym_rows": sym_rows,
            "ind": ind,
        })

    # Sort richest first (most encounters)
    all_patients.sort(key=lambda p: -p["ind"]["n_enc"])

    # Write per-patient files
    print("Writing patient files …", file=sys.stderr)
    patients_data = []
    for n, p in enumerate(all_patients, 1):
        p["rank"] = n
        md = _render_patient_md(
            p["pid"], p["info"], p["pat_row"],
            p["events"], p["obs"], p["sym_rows"], p["cutoff"],
        )
        path = out_dir / f"patient_{n:02d}.md"
        path.write_text(md)
        patients_data.append(p)

    # Write summary table
    st = _render_summary_table(patients_data)
    (out_dir / "SUMMARY_TABLE.md").write_text(st)

    # Write mappings
    mp = _render_mappings(patients_data)
    (out_dir / "MAPPINGS.md").write_text(mp)

    # Print tally to stdout
    print("\nTally (n=34):")
    DRUG_ORDER = ["beta_blocker", "ace_inhibitor", "arb", "diuretic",
                  "nitrate", "antiplatelet", "statin"]
    COND_ORDER = ["ischemic_heart", "mi", "afib", "hypertension", "hyperlipidemia"]
    for cls in DRUG_ORDER:
        n = sum(1 for p in patients_data if p["ind"]["drug_classes"].get(cls))
        print(f"  Drug: {cls:<18} {n:>2}/34")
    for g in COND_ORDER:
        n = sum(1 for p in patients_data if p["ind"]["cardiac_conds"].get(g))
        print(f"  Cond: {g:<18} {n:>2}/34")
    n_kidney = sum(1 for p in patients_data if p["ind"]["kidney"])
    print(f"  Kidney disease        {n_kidney:>2}/34")
    n_obs = sum(1 for p in patients_data if p["ind"]["cardiac_obs"])
    print(f"  Cardiac obs           {n_obs:>2}/34")
    n_none = sum(1 for p in patients_data
                 if not any(p["ind"]["drug_classes"].get(c) for c in DRUG_ORDER) and
                    not any(p["ind"]["cardiac_conds"].get(g) for g in COND_ORDER) and
                    not p["ind"]["kidney"])
    print(f"  None of the above     {n_none:>2}/34")

    print(f"\nWrote {len(patients_data)} patient files + SUMMARY_TABLE.md + MAPPINGS.md → {out_dir}/")


if __name__ == "__main__":
    main()

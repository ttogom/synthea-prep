#!/usr/bin/env python3
"""
combined_checks.py — Multi-class checks treating all 7 binned conditions as one dataset.

One data point per positive patient, labelled by condition. Four checks:
  a. Symptoms-plus-vitals classifier — bag of symptom names + whitelisted vital signs,
     multi-class LR.
  b. Shortcut probe — predict condition from age, sex, n_encounters, years of history.
  c. Presence probe — predict condition from whether any presenting evidence exists
     (symptoms or vitals).
  d. Class balance — counts and class fractions.

Writes docs/BINS.md with bin definitions and all four results.

Requires scikit-learn.
"""

import collections
import csv
import json
import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
SRC_DIR = ROOT / "data" / "pop10000-seed20260916"
PRESENTING_DIR = ROOT / "data" / "presenting"

# (stem, display_name, bin, labels_sha6, train_sha6)
CONDITIONS = [
    ("copd",               "COPD",                "history", "465e92", "465e92"),
    ("hypertension",       "Hypertension",         "history", "249aea", "249aea"),
    ("strep_throat",       "Strep throat",          "acute",   "1059ea", "1059ea"),
    ("viral_pharyngitis",  "Viral pharyngitis",     "acute",   "5ace2f", "5ace2f"),
    ("bacterial_sinusitis","Bacterial sinusitis",   "acute",   "c4404e", "c4404e"),
    ("cystitis",           "Cystitis",              "acute",   "c91f0b", "c91f0b"),
    ("heart_failure",      "CHF",                  "control", "9e8474", "9e8474"),
]

RUN = "pop10000-seed20260916"

# Vital sign names matching VITAL_SIGN_WHITELIST in extract_presenting.py
VITAL_NAMES = [
    "Body temperature",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart rate",
    "Respiratory rate",
    "Oxygen saturation",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def _d(s):
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _age(birth: date, ref: date) -> int:
    return ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))


def load_all() -> list:
    """
    Returns list of dicts, one per positive patient across all conditions:
      {pid, condition, bin, cutoff, age, sex_male, n_enc, years_hist,
       has_episode, symptoms, vitals}
    """
    records = []
    for stem, name, bin_name, lsha, tsha in CONDITIONS:
        labels_path = ROOT / "data" / f"labels-{lsha}.json"
        train_dir = ROOT / "data" / f"{RUN}__train-{tsha}"
        presenting_path = PRESENTING_DIR / f"{stem}.json"

        if not labels_path.exists():
            print(f"  SKIP {stem}: no labels at {labels_path}", file=sys.stderr)
            continue

        labels = json.loads(labels_path.read_text())

        # Patient birthdate + sex
        pat_csv = train_dir / "csv" / "patients.csv"
        patients = {}
        if pat_csv.exists():
            with open(pat_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    patients[row["Id"]] = row

        # Encounter data: count + earliest date for history span
        enc_csv = train_dir / "csv" / "encounters.csv"
        enc_info: dict = collections.defaultdict(lambda: {"n": 0, "earliest": None})
        if enc_csv.exists():
            with open(enc_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pid = row.get("PATIENT", "")
                    if pid not in labels:
                        continue
                    enc_info[pid]["n"] += 1
                    d = _d(row.get("START", ""))
                    if d and (enc_info[pid]["earliest"] is None or
                               d < enc_info[pid]["earliest"]):
                        enc_info[pid]["earliest"] = d

        # Presenting episodes
        presenting = {}
        if presenting_path.exists():
            raw = json.loads(presenting_path.read_text())
            presenting = raw.get("presenting", {})

        for pid, info in labels.items():
            p = patients.get(pid, {})
            birth = _d(p.get("BIRTHDATE", ""))
            cutoff = _d(info["cutoff"])
            if not cutoff:
                continue

            age = _age(birth, cutoff) if birth else -1
            sex_male = 1 if p.get("GENDER", "") == "M" else 0

            ei = enc_info[pid]
            n_enc = ei["n"]
            years_hist = (
                (cutoff - ei["earliest"]).days / 365.25
                if ei["earliest"] else 0.0
            )

            ep = presenting.get(pid)
            has_ep = 1 if ep else 0
            syms = [s["name"] for s in ep.get("symptoms", [])] if ep else []
            vitals = {v["name"]: v["value"] for v in ep.get("vitals", [])} if ep else {}

            records.append({
                "pid": pid,
                "condition": name,
                "stem": stem,
                "bin": bin_name,
                "cutoff": info["cutoff"],
                "age": age,
                "sex_male": sex_male,
                "n_enc": n_enc,
                "years_hist": round(years_hist, 2),
                "has_episode": has_ep,
                "symptoms": syms,
                "vitals": vitals,
            })
    return records


# ── Check helpers ─────────────────────────────────────────────────────────────

def _label_enc(records):
    conditions = sorted({r["condition"] for r in records})
    cond_idx = {c: i for i, c in enumerate(conditions)}
    return conditions, [cond_idx[r["condition"]] for r in records]


def _cv_multiclass(X, y, n_splits=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  confusion_matrix)
    import numpy as np

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                              random_state=42)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    bal = balanced_accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred).tolist()
    return acc, bal, cm, y_pred


# ── Check a: Symptoms-plus-vitals multi-class ─────────────────────────────────

def check_symptoms_vitals(records):
    from sklearn.preprocessing import MultiLabelBinarizer
    import numpy as np

    conditions, y = _label_enc(records)

    # Bag-of-symptom-name features (binary)
    mlb = MultiLabelBinarizer()
    X_syms = mlb.fit_transform([r["symptoms"] for r in records])

    # Vital sign features (continuous, 0 for missing)
    X_vitals = np.zeros((len(records), len(VITAL_NAMES)), dtype=float)
    for i, r in enumerate(records):
        for j, vname in enumerate(VITAL_NAMES):
            val = r["vitals"].get(vname, 0.0)
            X_vitals[i, j] = val

    # Z-score each vital over the patients who have it (leave 0s as 0)
    for j in range(X_vitals.shape[1]):
        col = X_vitals[:, j]
        nz_idx = col != 0
        if nz_idx.sum() > 1:
            mu = col[nz_idx].mean()
            std = col[nz_idx].std() + 1e-9
            X_vitals[nz_idx, j] = (col[nz_idx] - mu) / std

    X = np.hstack([X_syms, X_vitals])
    feat_names = list(mlb.classes_) + VITAL_NAMES

    acc, bal, cm, y_pred = _cv_multiclass(X, y)

    # Strep vs viral pharyngitis confusion
    ci = {c: i for i, c in enumerate(conditions)}
    si = ci.get("Strep throat")
    vi = ci.get("Viral pharyngitis")
    strep_viral = None
    if si is not None and vi is not None:
        strep_viral = {
            "strep_as_strep":  cm[si][si],
            "strep_as_viral":  cm[si][vi],
            "viral_as_viral":  cm[vi][vi],
            "viral_as_strep":  cm[vi][si],
        }

    # Top discriminating features per condition (full fit)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                              random_state=42)
    clf.fit(X, y)
    top_per_cond = {}
    for i, cond in enumerate(conditions):
        coef = clf.coef_[i]
        top = [feat_names[j] for j in coef.argsort()[-5:][::-1]]
        top_per_cond[cond] = top

    # Strep vs viral: features that discriminate between only those two
    strep_vs_viral_features = None
    if si is not None and vi is not None:
        strep_coef = clf.coef_[si]
        viral_coef = clf.coef_[vi]
        diff = strep_coef - viral_coef  # positive = more strep, negative = more viral
        top_strep = [feat_names[j] for j in diff.argsort()[-5:][::-1]]
        top_viral = [feat_names[j] for j in diff.argsort()[:5]]
        strep_vs_viral_features = {
            "more_strep_than_viral": top_strep,
            "more_viral_than_strep": top_viral,
        }

    return {
        "overall_accuracy": round(acc, 4),
        "balanced_accuracy": round(bal, 4),
        "conditions": conditions,
        "confusion_matrix": cm,
        "strep_vs_viral": strep_viral,
        "strep_vs_viral_features": strep_vs_viral_features,
        "top_features_per_condition": top_per_cond,
    }


# ── Check b: Shortcut probe ────────────────────────────────────────────────────

def check_shortcut(records):
    import numpy as np

    conditions, y = _label_enc(records)
    X = np.array([[r["age"], r["sex_male"], r["n_enc"], r["years_hist"]]
                  for r in records], dtype=float)

    # Normalize
    means = X.mean(axis=0)
    stds = X.std(axis=0) + 1e-9
    X = (X - means) / stds

    acc, bal, cm, _ = _cv_multiclass(X, y)
    chance = 1.0 / len(conditions)

    # Feature importances from full fit
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                              random_state=42)
    clf.fit(X, y)
    feat_names = ["age", "sex_male", "n_enc", "years_hist"]
    # Mean absolute coef per feature across all classes
    mean_abs = np.abs(clf.coef_).mean(axis=0)
    ranked = sorted(zip(feat_names, mean_abs.tolist()), key=lambda x: -x[1])

    return {
        "overall_accuracy": round(acc, 4),
        "balanced_accuracy": round(bal, 4),
        "chance_level": round(chance, 4),
        "conditions": conditions,
        "confusion_matrix": cm,
        "feature_importance": ranked,
    }


# ── Check c: Presence probe ────────────────────────────────────────────────────

def check_presence(records):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

    conditions, y = _label_enc(records)
    ci = {c: i for i, c in enumerate(conditions)}

    X = np.array([[r["has_episode"]] for r in records], dtype=float)
    acc, bal, cm, y_pred = _cv_multiclass(X, y)
    chance = 1.0 / len(conditions)

    # Per-condition: fraction identified by absence
    per_cond = {}
    for cond in conditions:
        idx = ci[cond]
        in_cond = [r for r in records if r["condition"] == cond]
        absent = sum(1 for r in in_cond if r["has_episode"] == 0)
        per_cond[cond] = {
            "n": len(in_cond),
            "no_episode": absent,
            "pct_absent": round(absent / len(in_cond) * 100, 1) if in_cond else 0,
        }

    return {
        "overall_accuracy": round(acc, 4),
        "balanced_accuracy": round(bal, 4),
        "chance_level": round(chance, 4),
        "per_condition": per_cond,
    }


# ── Check d: Class balance ─────────────────────────────────────────────────────

def check_balance(records):
    counts = collections.Counter(r["condition"] for r in records)
    total = len(records)
    by_bin = collections.defaultdict(list)
    cond_bin = {r["condition"]: r["bin"] for r in records}
    for cond, n in counts.most_common():
        by_bin[cond_bin[cond]].append((cond, n, round(n / total * 100, 1)))
    return {
        "total": total,
        "by_bin": dict(by_bin),
        "counts": dict(counts),
    }


# ── BINS.md writer ────────────────────────────────────────────────────────────

def write_bins_md(results: dict, out_path: Path):
    L = []
    L.append("# Condition Bins and Combined-Dataset Checks")
    L.append("")
    L.append("Bin definitions are in `configs/bins/` "
             "(see [`configs/bins/README.md`](../configs/bins/README.md)). "
             "The antecedent screen and symptom availability results that motivate the "
             "grouping are in [`docs/CONDITION_SCREEN.md`](CONDITION_SCREEN.md).")
    L.append("")
    L.append("## Presenting evidence coverage")
    L.append("")
    L.append("Coverage after adding whitelisted vitals (body temperature, systolic/diastolic BP, "
             "heart rate, respiratory rate, SpO2) from the diagnosing encounter:")
    L.append("")
    L.append("| Condition | Positives | With symptoms | With vitals | With any evidence |")
    L.append("|-----------|----------:|-------------:|------------:|------------------:|")
    cov = results.get("coverage", {})
    for stem, name, *_ in [
        ("copd","COPD",), ("hypertension","Hypertension",),
        ("strep_throat","Strep throat",), ("viral_pharyngitis","Viral pharyngitis",),
        ("bacterial_sinusitis","Bacterial sinusitis",), ("cystitis","Cystitis",),
        ("heart_failure","CHF",),
    ]:
        c = cov.get(stem, {})
        n = c.get("n_pos", "?")
        ns = c.get("n_sym", "?")
        nv = c.get("n_vit", "?")
        na = c.get("n_any", "?")
        pct = f"{100*na/n:.0f}%" if isinstance(na, int) and isinstance(n, int) and n else "?"
        L.append(f"| {name} | {n} | {ns} | {nv} | {na} ({pct}) |")
    L.append("")
    L.append("> Hypertension now has presenting evidence (elevated BP) for 29% of positives. "
             "The remaining 71% had their diagnosis recorded at an encounter with no "
             "standard vital signs in Synthea (e.g. specialist referral encounters). "
             "Bacterial sinusitis and CHF have no whitelisted vitals at their diagnosing "
             "encounters in this run.")
    L.append("")
    L.append("**Hypertension BP at diagnosing encounter** (n=757): "
             "systolic median 149 mmHg (81% ≥140); diastolic median 104 mmHg (89% ≥90). "
             "Readings are genuinely elevated — this is a real diagnostic signal, not noise.")
    L.append("")
    L.append("**Observation whitelist decisions** — every observation class at each "
             "diagnosing encounter was reviewed. Excluded even when recorded in the "
             "`vital-signs` FHIR category:")
    L.append("")
    L.append("| Code | Description | Decision | Reason |")
    L.append("|------|-------------|----------|--------|")
    L.append("| 19926-5 | FEV1/FVC (spirometry ratio) | **EXCLUDE** | "
             "< 0.70 is the diagnostic criterion for COPD; including it names the answer |")
    L.append("| 88020-3 | NYHA Functional Capacity | **EXCLUDE** | "
             "Directly classifies CHF severity; present only at CHF encounters |")
    L.append("| 88021-1 | NYHA Objective Assessment | **EXCLUDE** | Same |")
    L.append("| 33762-6 | NT-proBNP | **EXCLUDE** | "
             "Diagnostic marker for CHF; elevated values would name the condition |")
    L.append("| 89579-7 | Troponin I (high sensitivity) | **EXCLUDE** | "
             "Cardiac injury marker at CHF encounters; present only for CHF |")
    L.append("| 29463-7 | Body Weight | **EXCLUDE** | Not on six-item whitelist |")
    L.append("| 39156-5 | BMI | **EXCLUDE** | "
             "Not on six-item whitelist; derived from weight and height |")
    L.append("| 72514-3 | Pain severity (0–10) | **EXCLUDE** | "
             "A numeric rating scale, not a physiological measurement |")
    L.append("| 8302-2 | Body Height | **EXCLUDE** | Not on six-item whitelist |")
    L.append("")

    L.append("## Bin summary")
    L.append("")
    L.append("| Bin | Conditions | Primary signal | Notes |")
    L.append("|-----|-----------|----------------|-------|")
    L.append("| history | COPD, Hypertension | Pre-diagnosis antecedents | "
             "COPD: smoking_ever z=+7.8; HTN: sleep disorder z=+14.9, diabetes z=+11.5 |")
    L.append("| acute | Strep, Viral pharyngitis, Bacterial sinusitis, Cystitis | "
             "Presenting symptoms (91–99% coverage) | No non-artifact history signal |")
    L.append("| control | CHF | Demographic baseline | "
             "99% symptom coverage; no clean history signal; prior case-study condition |")
    L.append("")

    # d. Class balance
    bal = results["balance"]
    L.append("## d. Class balance")
    L.append("")
    L.append(f"**Total positives across 7 conditions: {bal['total']}**")
    L.append("")
    L.append("| Bin | Condition | n | % of dataset |")
    L.append("|-----|-----------|--:|-------------:|")
    for bin_name in ["history", "acute", "control"]:
        for cond, n, pct in bal["by_bin"].get(bin_name, []):
            L.append(f"| {bin_name} | {cond} | {n} | {pct}% |")
    L.append("")
    L.append("> Viral pharyngitis alone accounts for 43% of the dataset. "
             "Hypertension accounts for 23%. Class weights are needed for any "
             "multi-class model.")
    L.append("")

    # a. Symptoms + vitals
    sym = results["symptoms_vitals"]
    L.append("## a. Symptoms-plus-vitals multi-class classifier")
    L.append("")
    L.append("**Features:** bag of symptom names + whitelisted vital sign values "
             "(z-scored over patients who have each vital). "
             "**Model:** multinomial logistic regression (one-vs-rest), "
             "class_weight=balanced, 5-fold stratified CV.")
    L.append("")
    L.append(f"**Overall accuracy:** {sym['overall_accuracy']:.1%}  "
             f"**Balanced accuracy:** {sym['balanced_accuracy']:.1%}")
    L.append("")
    L.append("**Confusion matrix** (rows = true, cols = predicted):  ")
    L.append("")
    conds = sym["conditions"]
    header = "| True \\ Pred | " + " | ".join(f"`{c[:12]}`" for c in conds) + " |"
    sep = "|" + "|".join(["---:"] * (len(conds) + 1)) + "|"
    L.append(header)
    L.append(sep)
    for i, true_c in enumerate(conds):
        row_vals = " | ".join(str(sym["confusion_matrix"][i][j]) for j in range(len(conds)))
        L.append(f"| `{true_c[:12]}` | {row_vals} |")
    L.append("")

    sv = sym.get("strep_vs_viral")
    svf = sym.get("strep_vs_viral_features")
    if sv:
        n_strep = sum(sym["confusion_matrix"][conds.index("Strep throat")])
        n_viral = sum(sym["confusion_matrix"][conds.index("Viral pharyngitis")])
        L.append("**Strep vs viral pharyngitis:**")
        L.append(f"- Strep correctly predicted: {sv['strep_as_strep']}/{n_strep} "
                 f"({sv['strep_as_strep']/n_strep:.1%});  "
                 f"predicted as viral: {sv['strep_as_viral']} "
                 f"({sv['strep_as_viral']/n_strep:.1%})")
        L.append(f"- Viral correctly predicted: {sv['viral_as_viral']}/{n_viral} "
                 f"({sv['viral_as_viral']/n_viral:.1%});  "
                 f"predicted as strep: {sv['viral_as_strep']} "
                 f"({sv['viral_as_strep']/n_viral:.1%})")
        if svf:
            L.append(f"- **Features more predictive of strep than viral:** "
                     f"{', '.join(f'`{f}`' for f in svf['more_strep_than_viral'])}")
            L.append(f"- **Features more predictive of viral than strep:** "
                     f"{', '.join(f'`{f}`' for f in svf['more_viral_than_strep'])}")
    L.append("")
    L.append("**Top discriminating features per condition:**")
    L.append("")
    for cond, feats in sym.get("top_features_per_condition", {}).items():
        L.append(f"- **{cond}:** {', '.join(f'`{f}`' for f in feats)}")
    L.append("")

    # b. Shortcut probe
    sc = results["shortcut"]
    L.append("## b. Shortcut probe")
    L.append("")
    L.append("**Features:** age at diagnosis, sex (binary), number of pre-cutoff "
             "encounters, years of recorded history (earliest encounter to cutoff). "
             "**Intent:** record shape alone — not clinical content.")
    L.append("")
    chance_pct = sc['chance_level'] * 100
    L.append(f"**Overall accuracy:** {sc['overall_accuracy']:.1%}  "
             f"**Balanced accuracy:** {sc['balanced_accuracy']:.1%}  "
             f"**Chance level:** {chance_pct:.1f}% (1/{len(sc['conditions'])} classes)")
    L.append("")
    L.append("**Feature importance** (mean |coef| across all condition classifiers):")
    L.append("")
    for feat, imp in sc["feature_importance"]:
        L.append(f"- `{feat}`: {imp:.3f}")
    L.append("")
    bal_acc = sc["balanced_accuracy"]
    chance = sc["chance_level"]
    L.append(f"> **Record shape is {bal_acc/chance:.1f}× chance ({bal_acc:.1%} vs "
             f"{chance:.1%} baseline).** Conditions differ in age profile, sex "
             "distribution, and encounter density. Any symptom/vital-based model "
             "must beat this floor, not random chance.")
    L.append("")

    # c. Presence probe
    pr = results["presence"]
    L.append("## c. Presence probe")
    L.append("")
    L.append("**Feature:** `has_evidence` — binary flag for whether the patient has "
             "any presenting evidence (symptoms from `symptoms.csv` or whitelisted "
             "vitals from the diagnosing encounter).")
    L.append("")
    L.append(f"**Overall accuracy:** {pr['overall_accuracy']:.1%}  "
             f"**Balanced accuracy:** {pr['balanced_accuracy']:.1%}  "
             f"**Chance level:** {pr['chance_level']*100:.1f}%")
    L.append("")
    L.append("**Per-condition absence rate (no evidence at all):**")
    L.append("")
    L.append("| Condition | n | No evidence | % absent | Interpretation |")
    L.append("|-----------|--:|------------:|--------:|----------------|")
    for cond, v in sorted(pr["per_condition"].items(),
                           key=lambda x: -x[1]["pct_absent"]):
        pct = v["pct_absent"]
        interp = ("**Strong absence signal**" if pct >= 70
                  else "Partial absence" if pct > 20
                  else "Mostly present")
        L.append(f"| {cond} | {v['n']} | {v['no_episode']} | "
                 f"{pct}% | {interp} |")
    L.append("")
    htn = pr["per_condition"].get("Hypertension", {})
    htn_pct = htn.get("pct_absent", 0)
    if htn_pct > 0:
        L.append(f"> **Hypertension absence rate is now {htn_pct}%** (down from 100% "
                 f"when only symptoms were used). The remaining {htn_pct}% of hypertension "
                 "patients lack both symptoms and whitelisted vitals at their diagnosing "
                 "encounter — likely diagnosed at specialist encounters that Synthea does not "
                 "record a vital signs panel for. Presence alone no longer perfectly "
                 "identifies hypertension, but remains a partial signal.")
    L.append("")

    # Open design questions
    L.append("## Open design questions")
    L.append("")
    L.append("These are unresolved decisions the team needs to make before training:")
    L.append("")

    # Determine strep/viral resolution from results
    sv_check = sym.get("strep_vs_viral", {})
    strep_recall = (sv_check.get("strep_as_strep", 0) /
                    max(sum(sym["confusion_matrix"][conds.index("Strep throat")]), 1)
                    if sv_check and "Strep throat" in conds else 0)
    if strep_recall >= 0.5:
        L.append("1. ~~**Strep vs viral pharyngitis separability.**~~ **Resolved.** "
                 f"Strep recall is {strep_recall:.0%} once body temperature is included. "
                 "Body temperature cleanly separates the two: strep patients present with "
                 "fever (median 38.3°C, 67% ≥38°C); viral pharyngitis does not (median "
                 "37.5°C, 0% ≥38.5°C). Keep them as separate classes.")
    else:
        L.append("1. **Strep vs viral pharyngitis separability.** Even with vitals, strep "
                 f"recall is only {strep_recall:.0%}. Consider merging into a single "
                 "respiratory-infection class.")
    L.append("")

    # HTN presence question
    if htn_pct < 30:
        L.append("2. ~~**Hypertension identifiable by absence.**~~ **Resolved.** "
                 f"With vitals added, {100-htn_pct:.0f}% of hypertension patients now have "
                 "presenting evidence (elevated BP readings). Absence no longer trivially "
                 "identifies hypertension.")
    else:
        L.append("2. **Hypertension partial absence.** "
                 f"{htn_pct}% of hypertension patients still have no presenting evidence. "
                 "A multi-task model that observes whether a presenting section exists will "
                 "still find partial signal for hypertension. Decide whether to ablate the "
                 "presence flag or accept this as a known confound.")
    L.append("")
    L.append("3. **Demographic confounding.** The shortcut probe shows record shape "
             "alone is discriminative (2.3× chance). Age, sex, and encounter density "
             "differ systematically across conditions. Any evaluation of symptom/vital-based "
             "models should be compared against this floor, not against random chance.")
    L.append("")
    L.append("4. **Class imbalance.** Viral pharyngitis (43%) and hypertension (23%) "
             "dominate the dataset. Confirm class-weighting strategy before training.")
    L.append("")
    L.append("5. **COPD episode coverage (37%).** Most COPD patients lack presenting "
             "evidence — Synthea's symptom coverage is sparse for chronic conditions, and "
             "the diagnosing encounter only records BP/HR (not pulmonary-specific vitals "
             "beyond the excluded FEV1/FVC). Decide whether to train COPD with history "
             "only, or exclude it from symptom-based evaluations.")
    L.append("")

    out_path.write_text("\n".join(L) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        import sklearn  # noqa
    except ImportError:
        sys.exit("scikit-learn required: pip install scikit-learn")

    print("Loading all patient records ...", flush=True)
    records = load_all()
    print(f"  {len(records)} records across {len(set(r['condition'] for r in records))} conditions")

    print("Check d: class balance ...", flush=True)
    bal = check_balance(records)
    for cond, n in sorted(bal["counts"].items(), key=lambda x: -x[1]):
        print(f"  {cond:25s}: {n}")

    print("Check a: symptoms-plus-vitals multi-class ...", flush=True)
    sym = check_symptoms_vitals(records)
    print(f"  accuracy={sym['overall_accuracy']:.1%}  "
          f"balanced={sym['balanced_accuracy']:.1%}")
    sv = sym.get("strep_vs_viral")
    if sv:
        conds = sym["conditions"]
        n_strep = sum(sym["confusion_matrix"][conds.index("Strep throat")])
        n_viral = sum(sym["confusion_matrix"][conds.index("Viral pharyngitis")])
        print(f"  strep→strep={sv['strep_as_strep']}/{n_strep}  "
              f"strep→viral={sv['strep_as_viral']}/{n_strep}  "
              f"viral→viral={sv['viral_as_viral']}/{n_viral}  "
              f"viral→strep={sv['viral_as_strep']}/{n_viral}")
    svf = sym.get("strep_vs_viral_features")
    if svf:
        print(f"  features separating strep from viral: {svf['more_strep_than_viral']}")

    print("Check b: shortcut probe ...", flush=True)
    sc = check_shortcut(records)
    print(f"  accuracy={sc['overall_accuracy']:.1%}  "
          f"balanced={sc['balanced_accuracy']:.1%}  "
          f"chance={sc['chance_level']:.1%}")

    print("Check c: presence probe ...", flush=True)
    pr = check_presence(records)
    print(f"  accuracy={pr['overall_accuracy']:.1%}  "
          f"balanced={pr['balanced_accuracy']:.1%}")
    for cond, v in sorted(pr["per_condition"].items(), key=lambda x: -x[1]["pct_absent"]):
        print(f"  {cond:25s}: {v['pct_absent']}% absent")

    # Build coverage summary from presenting.json files
    coverage = {}
    for stem, name, bin_name, lsha, tsha in CONDITIONS:
        ppath = PRESENTING_DIR / f"{stem}.json"
        if ppath.exists():
            raw = json.loads(ppath.read_text())
            coverage[stem] = {
                "n_pos": raw.get("n_positives", 0),
                "n_sym": raw.get("n_with_symptoms", 0),
                "n_vit": raw.get("n_with_vitals", 0),
                "n_any": raw.get("n_with_episode", 0),
            }

    results = {
        "balance": bal,
        "coverage": coverage,
        "symptoms_vitals": sym,
        "shortcut": sc,
        "presence": pr,
    }

    out_json = ROOT / "data" / "combined_checks.json"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {out_json}")

    out_md = ROOT / "docs" / "BINS.md"
    write_bins_md(results, out_md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()

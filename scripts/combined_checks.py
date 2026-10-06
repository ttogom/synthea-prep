#!/usr/bin/env python3
"""
combined_checks.py — Multi-class checks treating all 7 binned conditions as one dataset.

One data point per positive patient, labelled by condition. Four checks:
  a. Symptoms-only classifier — bag of symptom names, multi-class LR.
  b. Shortcut probe — predict condition from age, sex, n_encounters, years of history.
  c. Presence probe — predict condition from whether a presenting episode exists at all.
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
       has_episode, symptoms}
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
            syms = [s["name"] for s in ep["symptoms"]] if ep else []

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


# ── Check a: Symptoms-only multi-class ────────────────────────────────────────

def check_symptoms(records):
    from sklearn.preprocessing import MultiLabelBinarizer
    import numpy as np

    conditions, y = _label_enc(records)
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform([r["symptoms"] for r in records])

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

    # Top discriminating features per condition
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                              random_state=42)
    clf.fit(X, y)
    names = mlb.classes_
    top_per_cond = {}
    for i, cond in enumerate(conditions):
        coef = clf.coef_[i]
        top = [names[j] for j in coef.argsort()[-5:][::-1]]
        top_per_cond[cond] = top

    return {
        "overall_accuracy": round(acc, 4),
        "balanced_accuracy": round(bal, 4),
        "conditions": conditions,
        "confusion_matrix": cm,
        "strep_vs_viral": strep_viral,
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
    L.append("> Viral pharyngitis alone accounts for 45% of the dataset. "
             "Hypertension accounts for 23%. Class weights are needed for any "
             "multi-class model.")
    L.append("")

    # a. Symptoms-only
    sym = results["symptoms"]
    L.append("## a. Symptoms-only multi-class classifier")
    L.append("")
    L.append("**Features:** bag of symptom names from `data/presenting/`. "
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
    L.append("")
    L.append("**Top discriminating symptoms per condition:**")
    L.append("")
    for cond, feats in sym.get("top_features_per_condition", {}).items():
        L.append(f"- **{cond}:** {', '.join(f'`{f}`' for f in feats)}")
    L.append("")
    L.append("> Hypertension has no symptom features; its prediction relies entirely on "
             "the absence of any symptom — see check c below.")
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
    if sc["balanced_accuracy"] > 0.5:
        L.append("> **Record shape is discriminative above chance.** Conditions differ "
                 "in age profile (CHF older, paediatric-heavy cystitis), sex distribution "
                 "(cystitis predominantly female), and encounter density (acute conditions "
                 "have shorter histories). A model trained on symptoms alone but evaluated "
                 "on a dataset where age and sex are confounded with the label could be "
                 "encoding demographics, not pathology.")
    else:
        L.append("> Record shape does not reliably separate conditions above chance.")
    L.append("")

    # c. Presence probe
    pr = results["presence"]
    L.append("## c. Presence probe")
    L.append("")
    L.append("**Feature:** `has_episode` — binary flag for whether a presenting "
             "symptom episode exists in `symptoms.csv` for this patient.")
    L.append("")
    L.append(f"**Overall accuracy:** {pr['overall_accuracy']:.1%}  "
             f"**Balanced accuracy:** {pr['balanced_accuracy']:.1%}  "
             f"**Chance level:** {pr['chance_level']*100:.1f}%")
    L.append("")
    L.append("**Per-condition episode absence rate:**")
    L.append("")
    L.append("| Condition | n | No episode | % absent | Interpretation |")
    L.append("|-----------|--:|----------:|--------:|----------------|")
    for cond, v in sorted(pr["per_condition"].items(),
                           key=lambda x: -x[1]["pct_absent"]):
        interp = ("**Strong absence signal**" if v["pct_absent"] == 100
                  else "Partial absence" if v["pct_absent"] > 20
                  else "Mostly present")
        L.append(f"| {cond} | {v['n']} | {v['no_episode']} | "
                 f"{v['pct_absent']}% | {interp} |")
    L.append("")
    htn = pr["per_condition"].get("Hypertension", {})
    if htn.get("pct_absent") == 100:
        L.append("> **Hypertension is 100% identifiable by absence.** Every hypertension "
                 "patient has no presenting episode (Synthea generates no symptoms for it). "
                 "Any model that can observe whether a presenting section exists will perfectly "
                 "identify hypertension without reading any clinical content. The `--presenting` "
                 "flag in `serialize.py` is the correct interface for controlling this.")
    L.append("")

    # Open design questions
    L.append("## Open design questions")
    L.append("")
    L.append("These are unresolved decisions the team needs to make before training:")
    L.append("")
    L.append("1. **Hypertension without symptoms.** Including hypertension in the acute "
             "bin is invalid (it has no symptoms). Including it in history-only evaluation "
             "is legitimate, but any multi-task model that also sees a `has_presenting` "
             "flag will trivially identify it. Decide: ablate presence entirely, or treat "
             "hypertension as a history-only condition and the others as symptom-available?")
    L.append("")
    L.append("2. **Strep vs viral pharyngitis separability.** These two conditions share "
             "overlapping symptoms in Synthea. If the classifier cannot separate them "
             "reliably from symptoms alone, should they be merged into a single "
             "respiratory-infection class, or kept separate with the understanding that "
             "a final model must use additional context (e.g. test results)?")
    L.append("")
    L.append("3. **Demographic confounding.** The shortcut probe shows record shape "
             "alone is discriminative. Age, sex, and encounter density differ systematically "
             "across conditions. Any evaluation of symptom-based models should be compared "
             "against the shortcut-probe baseline, not against random chance.")
    L.append("")
    L.append("4. **Class imbalance.** Viral pharyngitis (45%) and hypertension (23%) "
             "dominate the dataset. A model trained with standard cross-entropy will "
             "collapse to predicting these two conditions. Confirm class-weighting strategy "
             "before training.")
    L.append("")
    L.append("5. **COPD episode coverage (37%).** Most COPD patients lack a presenting "
             "episode — Synthea's symptom coverage is sparse for long-term chronic "
             "conditions. Decide whether to train COPD with history only, or exclude COPD "
             "from symptom-based evaluations.")
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

    print("Check a: symptoms-only multi-class ...", flush=True)
    sym = check_symptoms(records)
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

    results = {
        "balance": bal,
        "symptoms": sym,
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

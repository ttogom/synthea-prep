#!/usr/bin/env python3
"""
probe_text.py — How well does the serialized text separate the target conditions?

A bag-of-words classifier (TF-IDF + logistic regression) is trained on the
serialized `text` of every record to predict which condition it was built
for. It is run separately for each variant, so the variants can be compared:

  history     pre-diagnosis record only
  full        history + presenting complaint
  no-history  demographics + presenting complaint

Two probes per variant:
  text   TF-IDF over 1-2 word n-grams of the text
  shape  text length and which sections are present, nothing else. A high
         score means the record's shape alone points to the condition.

Settings follow scripts/combined_checks.py (logistic regression, C=1,
class_weight=balanced, 5 folds, seed 42), except a higher max_iter so the
solver converges, and folds grouped by
patient: a patient can be a positive for several conditions, and their
records share the same history.

Usage:
    python3 serialization/probe_text.py data/serialized/seven \
        [--variants history full no-history] [--report <path.md>]

Input files: <dir>/<condition>-<variant>.jsonl. Requires scikit-learn.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

SECTIONS = ("PRESENTING COMPLAINT", "CHIEF COMPLAINTS", "ACTIVE PROBLEMS", "PAST PROBLEMS",
            "OTHER FINDINGS", "ALLERGIES", "CURRENT MEDICATIONS", "PAST MEDICATIONS",
            "ACTIVE CARE PLANS", "DEVICES IN USE", "VITALS AND LABS", "PROCEDURES",
            "IMMUNIZATIONS", "RECENT VISITS")
# Randy's baselines (CONDITION_SELECTION.md), same 7 conditions, balanced accuracy
BASELINES = [("Chance (1 of 7)", 14.3), ("Record shape: age, sex, visits, years (Randy)", 33.6),
             ("Symptoms + vitals classifier (Randy)", 79.5)]


def load(directory, variant):
    texts, labels, groups = [], [], []
    for path in sorted(Path(directory).glob(f"*-{variant}.jsonl")):
        condition = path.name[: -len(f"-{variant}.jsonl")]
        if variant == "history" and condition.endswith("-no"):
            continue  # "<c>-no-history.jsonl" also ends in "-history.jsonl"
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(condition)
            groups.append(r["patient_id"])
    return texts, np.array(labels), np.array(groups)


def shape_features(texts):
    rows = []
    for t in texts:
        rows.append([np.log1p(len(t)), t.count("\n- ")] + [float(s in t) for s in SECTIONS])
    return csr_matrix(StandardScaler().fit_transform(np.array(rows)))


def classifier():
    # max_iter above combined_checks.py's 2000: lbfgs did not converge on the text features
    return LogisticRegression(max_iter=10000, C=1.0, class_weight="balanced", random_state=42)


def evaluate(X, y, groups):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(classifier(), X, y, cv=cv, groups=groups)
    classes = sorted(set(y))
    recall = recall_score(y, pred, labels=classes, average=None)
    return {"accuracy": accuracy_score(y, pred), "balanced": balanced_accuracy_score(y, pred),
            "recall": dict(zip(classes, recall))}


def top_features(vectorizer, X, y, n=8):
    clf = classifier().fit(X, y)
    names = np.array(vectorizer.get_feature_names_out())
    return {c: list(names[np.argsort(clf.coef_[i])[-n:][::-1]]) for i, c in enumerate(clf.classes_)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory", type=Path)
    ap.add_argument("--variants", nargs="+", default=["history", "full", "no-history"])
    ap.add_argument("--report", type=Path, help="default: <directory>/probe_report.md")
    args = ap.parse_args()

    results, tops, counts = {}, {}, {}
    for variant in args.variants:
        texts, y, groups = load(args.directory, variant)
        if not texts:
            raise SystemExit(f"ERROR: no {variant} files in {args.directory}")
        counts[variant] = (len(texts), len(set(groups)))
        print(f"{variant}: {len(texts)} records, {len(set(groups))} patients ...", flush=True)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=100_000,
                              sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b")
        X = vec.fit_transform(texts)
        results[(variant, "text")] = evaluate(X, y, groups)
        results[(variant, "shape")] = evaluate(shape_features(texts), y, groups)
        tops[variant] = top_features(vec, X, y)
        for probe in ("text", "shape"):
            r = results[(variant, probe)]
            print(f"  {probe:5}: balanced {100 * r['balanced']:.1f}%  accuracy {100 * r['accuracy']:.1f}%")

    classes = sorted(results[(args.variants[0], "text")]["recall"])
    L = ["# Text probe: which condition does the serialized text point to?", "",
         f"Input: `{args.directory}`. TF-IDF (1-2 grams) + logistic regression, "
         "class_weight=balanced, 5-fold cross-validation grouped by patient.", "",
         "## Balanced accuracy", "",
         "| Variant | Records (patients) | Text probe | Shape-only probe |", "|---|---|---|---|"]
    for v in args.variants:
        n, p = counts[v]
        L.append(f"| {v} | {n:,} ({p:,}) | {100 * results[(v, 'text')]['balanced']:.1f}% "
                 f"| {100 * results[(v, 'shape')]['balanced']:.1f}% |")
    L += ["", "Baselines from CONDITION_SELECTION.md (same conditions):", ""]
    L += [f"- {name}: {score}%" for name, score in BASELINES]
    L += ["", "## Recall per condition (text probe)", "",
          "| Condition | " + " | ".join(args.variants) + " |", "|---" * (len(args.variants) + 1) + "|"]
    for c in classes:
        L.append(f"| {c} | " + " | ".join(f"{100 * results[(v, 'text')]['recall'][c]:.1f}%"
                                         for v in args.variants) + " |")
    for v in args.variants:
        L += ["", f"## Strongest words per condition: {v}", "", "| Condition | Top features |", "|---|---|"]
        L += [f"| {c} | {', '.join(tops[v][c])} |" for c in classes]
    report = args.report or args.directory / "probe_report.md"
    report.write_text("\n".join(L) + "\n")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
check_text_leaks.py — Leak test on serialized text (the output of serialize.py).

scripts/check_leaks.py checks the CSVs. This checks what the model will
actually read, including note-derived text, after serialization.

Checks, per patient, against that patient's own label:

  FAIL  (hard leak, exit code 1)
    * target SNOMED code appears in the text
    * target description (tag like "(disorder)" stripped) appears as a phrase
    * all distinctive words of the description appear on one line, any order
      (catches "Microalbuminuria due to type 2 diabetes mellitus")
    * Synthea simulation scores (DALY, QALY, QOLS) appear
    * the patient's name appears (names come from the unscrubbed notes/CSVs)

  REVIEW  (possible hint, reported but not failing)
    * a two-word piece of the description appears ("heart failure")
    * a term from --terms appears (condition-specific hints, e.g. symptoms
      or questionnaires only used for that condition)

Usage:
    python3 serialization/check_text_leaks.py <file.jsonl> <train-dir>
        [--terms configs/leak_terms/<condition>.txt] [--report <path.md>]

Standard library only.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

STOPWORDS = {"of", "the", "and", "to", "due", "with", "in", "on", "by", "for",
             "a", "an", "or", "without", "disorder", "finding"}
SIM_SCORES = ("DALY", "QALY", "QOLS")
TAG = re.compile(r"\s*\([^()]*\)\s*$")
EXAMPLES_PER_CHECK = 5


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def has_phrase(text, phrase):
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


def read_terms(path):
    """One term per line; '#' starts a comment."""
    if not path:
        return []
    terms = []
    for line in path.read_text().splitlines():
        term = line.split("#", 1)[0].strip().lower()
        if term:
            terms.append(term)
    return terms


def check_record(rec, names, terms):
    """List of (level, check, snippet) for one serialized patient."""
    text = rec["text"]
    lower = text.lower()
    desc = TAG.sub("", rec["label"]["description"]).lower()
    key_words = [w for w in words(desc) if w not in STOPWORDS]
    hits = []

    def snippet(line):
        return line.strip()[:160]

    if has_phrase(lower, rec["label"]["code"]):
        hits.append(("FAIL", "target code", rec["label"]["code"]))
    if has_phrase(lower, desc):
        hits.append(("FAIL", "target description", desc))
    for line in text.splitlines():
        lw = set(words(line))
        if len(key_words) > 1 and all(w in lw for w in key_words) and not has_phrase(line.lower(), desc):
            hits.append(("FAIL", "description words on one line", snippet(line)))
    for score in SIM_SCORES:
        if has_phrase(text, score):
            hits.append(("FAIL", "simulation score", score))
    for name in names:
        if len(name) > 2 and has_phrase(lower, name.lower()):
            hits.append(("FAIL", "patient name", name))

    seen = set()
    for a, b in zip(key_words, key_words[1:]):
        pair = f"{a} {b}"
        if pair in seen:
            continue
        seen.add(pair)
        for line in text.splitlines():
            if has_phrase(line.lower(), pair):
                hits.append(("REVIEW", f"description piece '{pair}'", snippet(line)))
    for term in terms:
        for line in text.splitlines():
            if has_phrase(line.lower(), term):
                hits.append(("REVIEW", f"term '{term}'", snippet(line)))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jsonl", type=Path, help="output of serialize.py")
    ap.add_argument("train_dir", type=Path, help="training dir the JSONL was built from (for names)")
    ap.add_argument("--terms", type=Path, help="extra condition-specific terms to flag for review")
    ap.add_argument("--report", type=Path, help="Markdown report path (default: <jsonl>.leaks.md)")
    args = ap.parse_args()

    records = [json.loads(line) for line in open(args.jsonl, encoding="utf-8")]
    names = defaultdict(list)
    with open(args.train_dir / "csv" / "patients.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            names[r["Id"]] = [r[k] for k in ("FIRST", "MIDDLE", "LAST", "MAIDEN") if r[k]]
    terms = read_terms(args.terms)

    patients_hit = defaultdict(set)   # (level, check) -> patient ids
    counts = Counter()                # (level, check) -> hits
    examples = defaultdict(list)      # (level, check) -> snippets
    for rec in records:
        for level, check, snip in check_record(rec, names[rec["patient_id"]], terms):
            key = (level, check)
            counts[key] += 1
            patients_hit[key].add(rec["patient_id"])
            if len(examples[key]) < EXAMPLES_PER_CHECK and snip not in examples[key]:
                examples[key].append(snip)

    fails = sorted(k for k in counts if k[0] == "FAIL")
    reviews = sorted(k for k in counts if k[0] == "REVIEW")
    labels = sorted({r["label"]["description"] for r in records})
    lines = [
        f"# Text Leak Report: {args.jsonl.name}",
        "",
        f"Patients: {len(records)}  ",
        f"Target: {', '.join(labels)}  ",
        f"Extra terms: {args.terms or 'none'}",
        "",
        f"**Result: {'FAIL' if fails else 'PASS'}** — "
        f"{len(fails)} failing check(s), {len(reviews)} check(s) to review",
        "",
    ]
    for title, keys in (("Failing checks", fails), ("To review (possible hints)", reviews)):
        lines += [f"## {title}", ""]
        if not keys:
            lines += ["None.", ""]
            continue
        lines += ["| Level | Check | Patients | Hits | Examples |", "|---|---|---|---|---|"]
        for key in keys:
            ex = "<br>".join(e.replace("|", "\\|") for e in examples[key])
            lines.append(f"| {key[0]} | {key[1]} | {len(patients_hit[key])} | {counts[key]} | {ex} |")
        lines.append("")

    report = args.report or args.jsonl.with_suffix(".leaks.md")
    report.write_text("\n".join(lines))
    print(f"{'FAIL' if fails else 'PASS'}: {len(records)} patients, "
          f"{len(fails)} failing check(s), {len(reviews)} to review")
    for key in fails + reviews:
        print(f"  {key[0]:6} {key[1]}: {len(patients_hit[key])} patient(s), {counts[key]} hit(s)")
    print(f"Report: {report}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

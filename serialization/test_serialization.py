#!/usr/bin/env python3
"""
test_serialization.py — Check a serialize.py output against its inputs.

Tests:
  1. Same patients     every labelled patient is in the JSONL exactly once,
                       and each record's label matches the labels file
  2. No dates          no absolute date (YYYY-MM-DD) or year in the text, and
                       every relative time is "N unit(s) ago" with N >= 0
  3. Faithful          every condition and medication in the training CSVs
                       (minus --drop rows) appears in that patient's text
  4. Latest values     for each vital/lab shown, the first value is the most
                       recent reading in the CSVs (by full timestamp)
  5. Drop list         no --drop term appears in any text
  6. Deterministic     re-running serialize.py with the same settings gives a
                       byte-identical file
  7. Compression       with --no-compress, no patient's text has fewer items

Usage:
    python3 serialization/test_serialization.py <file.jsonl> <train-dir> <labels.json>

Settings (including the drop file) are read from <file>.stats.json.
Standard library only. Exit code 1 if any test fails.
"""

import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
# A 4-digit number that looks like a year, unless a unit follows it
# (e.g. income "1967 /a", "2000 MG").
YEAR = re.compile(r"\b(19|20)\d{2}\b(?!\.\d|\s*(/|[A-Za-z%{\[]))")
AGO = re.compile(r"(-?\d+) (day|month|year)s? ago")


def read_by_patient(path, patients):
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["PATIENT"] in patients:
                rows[r["PATIENT"]].append(r)
    return rows


def dropped(row, drop):
    text = f"{row.get('DESCRIPTION', '')} {row.get('REASONDESCRIPTION', '')}".lower()
    return any(t in text for t in drop)


def serialize(train_dir, labels, out, settings):
    cmd = [sys.executable, str(HERE / "serialize.py"), str(train_dir), str(labels), "--out", str(out)]
    if settings.get("no_compress"):
        cmd.append("--no-compress")
    else:
        cmd += ["--window-years", str(settings["window_years"]),
                "--window-from", settings["window_from"],
                "--max-values", str(settings["max_values"]),
                "--max-encounters", str(settings["max_encounters"])]
    cmd += ["--min-encounters", str(settings["min_encounters"])]
    if settings.get("drop"):
        cmd += ["--drop", settings["drop"]]
    if settings.get("presenting"):
        cmd += ["--presenting", settings["presenting"]]
    if settings.get("no_history"):
        cmd.append("--no-history")
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__.split("Usage:")[1].split("Settings")[0].strip())
    jsonl, train_dir, labels_path = map(Path, sys.argv[1:])
    stats = json.loads(jsonl.with_suffix(".stats.json").read_text())
    settings = stats["settings"]
    drop = settings.get("drop_terms") or []
    labels = json.loads(labels_path.read_text())
    records = [json.loads(line) for line in open(jsonl, encoding="utf-8")]
    by_id = {r["patient_id"]: r for r in records}
    patients = set(labels)
    results = []

    def report(name, problems, detail=""):
        results.append(not problems)
        print(f"{'PASS' if not problems else 'FAIL'}  {name}{detail}")
        for p in problems[:5]:
            print(f"        {p}")

    # 1. Same patients, labels match
    problems = []
    if len(records) != len(by_id):
        problems.append(f"{len(records) - len(by_id)} duplicate patient(s)")
    skipped = stats.get("patients_skipped_min_encounters", 0)
    missing = patients - set(by_id)
    if len(missing) != skipped:
        problems.append(f"{len(missing)} labelled patient(s) missing, {skipped} expected (min-encounters)")
    if set(by_id) - patients:
        problems.append(f"{len(set(by_id) - patients)} patient(s) not in labels")
    for pid, r in by_id.items():
        want = labels.get(pid, {})
        if r["cutoff"] != want.get("cutoff") or r["label"]["code"] != want.get("code"):
            problems.append(f"{pid}: label/cutoff differs from labels file")
    report("1. same patients, labels match", problems, f" ({len(records)} records)")

    # 2. No absolute dates, no negative times
    problems = []
    for pid, r in by_id.items():
        if DATE.search(r["text"]):
            problems.append(f"{pid}: date '{DATE.search(r['text']).group()}'")
        for line in r["text"].splitlines():
            m = YEAR.search(line)
            if m:
                problems.append(f"{pid}: possible year '{m.group()}' in: {line.strip()[:80]}")
        if any(int(n) < 0 for n, _ in AGO.findall(r["text"])):
            problems.append(f"{pid}: negative time")
    report("2. no absolute dates, no negative times", problems)

    # 3 and 4 compare the text with the history; a --no-history file has none.
    csv_dir = train_dir / "csv"
    if settings.get("no_history"):
        print("SKIP  3. every condition and medication appears (--no-history)")
        print("SKIP  4. latest vital/lab value shown first (--no-history)")
    else:
        # 3. Every condition and medication appears
        problems, checked = [], 0
        for name in ("conditions", "medications"):
            rows = read_by_patient(csv_dir / f"{name}.csv", set(by_id))
            for pid, rs in rows.items():
                for row in rs:
                    if dropped(row, drop):
                        continue
                    checked += 1
                    if row["DESCRIPTION"] not in by_id[pid]["text"]:
                        problems.append(f"{pid}: {name} '{row['DESCRIPTION'][:60]}' missing")
        report("3. every condition and medication appears", problems, f" ({checked} rows checked)")

        # 4. Latest vital/lab value comes first
        problems, checked = [], 0
        obs = read_by_patient(csv_dir / "observations.csv", set(by_id))
        for pid, rs in obs.items():
            latest = defaultdict(list)   # description -> rows at the latest timestamp
            for row in rs:
                if dropped(row, drop):
                    continue
                cur = latest[row["DESCRIPTION"]]
                if not cur or row["DATE"] > cur[0]["DATE"]:
                    latest[row["DESCRIPTION"]] = [row]
                elif row["DATE"] == cur[0]["DATE"]:
                    cur.append(row)   # same exact time: any of these may come first
            text = by_id[pid]["text"]
            in_vitals = text.split("VITALS AND LABS", 1)[1].split("\n\n", 1)[0] if "VITALS AND LABS" in text else ""
            for line in in_vitals.splitlines()[1:]:
                desc, _, rest = line[2:].partition(": ")
                if desc not in latest:
                    continue
                checked += 1
                wants = []
                for row in latest[desc]:
                    try:
                        wants.append(f"{float(row['VALUE']):.1f}".rstrip("0").rstrip("."))
                    except ValueError:
                        wants.append(row["VALUE"])
                if not any(rest.startswith(w) for w in wants):
                    problems.append(f"{pid}: {desc[:40]} shows '{rest[:20]}', latest is '{wants[0]}'")
        report("4. latest vital/lab value shown first", problems, f" ({checked} tests checked)")

    # 5. Drop list honoured
    problems = [f"{pid}: '{t}'" for pid, r in by_id.items() for t in drop if t in r["text"].lower()]
    report("5. drop list honoured", problems, f" ({len(drop)} terms)" if drop else " (no drop list)")

    with tempfile.TemporaryDirectory() as tmp:
        # 6. Deterministic
        again = Path(tmp) / "again.jsonl"
        serialize(train_dir, labels_path, again, settings)
        same = again.read_bytes() == jsonl.read_bytes()
        report("6. deterministic (re-run is byte-identical)", [] if same else ["outputs differ"])

        # 7. Compression never adds content. Items are compared, not characters:
        # the compressed headings ("5 years up to the last visit") are longer
        # than "full history".
        def items(text):
            return sum(1 for line in text.splitlines() if line.startswith("- "))
        full = Path(tmp) / "full.jsonl"
        serialize(train_dir, labels_path, full, {**settings, "no_compress": True})
        full_recs = {json.loads(l)["patient_id"]: json.loads(l) for l in open(full)}
        problems = [f"{pid}: {items(r['text'])} > {items(full_recs[pid]['text'])} items"
                    for pid, r in by_id.items() if items(r["text"]) > items(full_recs[pid]["text"])]
        total = sum(r["chars"] for r in records)
        total_full = sum(full_recs[p]["chars"] for p in by_id)
        report("7. compression never adds items", problems,
               f" (total {total:,} vs {total_full:,} chars, {100 * (1 - total / total_full):.0f}% smaller)")

    print(f"\n{sum(results)}/{len(results)} tests passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

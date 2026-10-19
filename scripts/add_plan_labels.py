#!/usr/bin/env python3
"""
add_plan_labels.py — Attach plan labels to release-v2 test/dev labels.

For each example in v2's test_labels.jsonl and dev_labels.jsonl, computes a
"plan" field using the same REASONCODE logic as the corpus TREATMENT section:
medications, procedures, and care plans whose REASONCODE is the target
condition's SNOMED code, started on or before the condition's first START date
in conditions.csv.  Records with no matching items get plan=["none recorded"].

Record files (test.jsonl, dev.jsonl, corpus.jsonl) are copied byte-for-byte
from release-v2.  Only the two label files differ.

Output: data/release-v2.1/

Validation: before writing anything, checks that every test/dev example ID
exists in the source run with the same cutoff date.  Stops if more than a
handful of mismatches are found.

Usage:
    python3 scripts/add_plan_labels.py
"""

import csv
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR   = REPO_ROOT / "data" / "pop10000-seed20260916"
V2_DIR    = REPO_ROOT / "data" / "release-v2"
OUT_DIR   = REPO_ROOT / "data" / "release-v2.1"

CONDITIONS = [
    {"name": "copd",                "sha": "465e92", "codes": {"87433001", "185086009"}},
    {"name": "hypertension",        "sha": "249aea", "codes": {"59621000"}},
    {"name": "strep_throat",        "sha": "1059ea", "codes": {"43878008"}},
    {"name": "viral_pharyngitis",   "sha": "5ace2f", "codes": {"195662009"}},
    {"name": "bacterial_sinusitis", "sha": "c4404e", "codes": {"75498004"}},
    {"name": "cystitis",            "sha": "c91f0b", "codes": {"307426000"}},
    {"name": "heart_failure",       "sha": "9e8474", "codes": {"88805009"}},
]
COND_CODES = {c["name"]: c["codes"] for c in CONDITIONS}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def parse_id(ex_id):
    """(condition_name, patient_id) from '{condition}_{pid}'."""
    for cond in sorted(COND_CODES, key=len, reverse=True):
        if ex_id.startswith(cond + "_"):
            return cond, ex_id[len(cond) + 1:]
    raise ValueError(f"Cannot parse condition from id: {ex_id!r}")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


# ── Step 1: Validate ──────────────────────────────────────────────────────────

def validate(test_labels, dev_labels):
    """Return list of mismatch dicts: id, reason."""
    my_labels = {
        c["name"]: json.loads((REPO_ROOT / "data" / f"labels-{c['sha']}.json").read_text())
        for c in CONDITIONS
    }
    v2_cutoff = {}
    with open(V2_DIR / "index.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v2_cutoff[row["example_id"]] = row["cutoff_date"]

    mismatches = []
    for label in test_labels + dev_labels:
        ex_id = label["id"]
        cond, pid = parse_id(ex_id)
        if pid not in my_labels[cond]:
            mismatches.append({"id": ex_id, "reason": "pid absent from my source run"})
            continue
        mine = my_labels[cond][pid]["cutoff"][:10]
        v2   = v2_cutoff.get(ex_id, "")
        if mine != v2:
            mismatches.append({"id": ex_id, "reason": f"cutoff mine={mine} v2={v2}"})
    return mismatches


# ── Step 2: Compute plans ────────────────────────────────────────────────────

def compute_plans(examples):
    """
    examples: list of (pid, cond_name)
    Returns {(pid, cond_name): [description, ...]} — deduplicated, empty → ["none recorded"].
    """
    all_pids = {pid for pid, _ in examples}
    pid_conds = defaultdict(set)
    for pid, cond in examples:
        pid_conds[pid].add(cond)

    # Earliest condition START date per (pid, cond_name)
    cond_start = {}
    with open(SRC_DIR / "csv" / "conditions.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["PATIENT"]
            if pid not in all_pids:
                continue
            start = (row.get("START") or "")[:10]
            if not start:
                continue
            for cond_name in pid_conds[pid]:
                if row["CODE"] in COND_CODES[cond_name]:
                    key = (pid, cond_name)
                    if key not in cond_start or start < cond_start[key]:
                        cond_start[key] = start

    # Items whose REASONCODE is a target code, started on or before cond_start
    items_by_key = defaultdict(list)
    for stem in ("medications", "procedures", "careplans"):
        with open(SRC_DIR / "csv" / f"{stem}.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = row.get("PATIENT", "")
                if pid not in all_pids:
                    continue
                rc = row.get("REASONCODE", "")
                if not rc:
                    continue
                desc = row.get("DESCRIPTION", "").strip()
                if not desc:
                    continue
                item_start = (row.get("START") or "")[:10]
                for cond_name in pid_conds[pid]:
                    if rc not in COND_CODES[cond_name]:
                        continue
                    cutoff = cond_start.get((pid, cond_name), "")
                    if item_start and cutoff and item_start > cutoff:
                        continue
                    items_by_key[(pid, cond_name)].append(desc)

    result = {}
    for pid, cond_name in examples:
        seen, items = set(), []
        for desc in items_by_key.get((pid, cond_name), []):
            if desc not in seen:
                items.append(desc)
                seen.add(desc)
        result[(pid, cond_name)] = items or ["none recorded"]
    return result


# ── Step 3: Print per-condition stats ────────────────────────────────────────

def print_stats(all_labels):
    cond_total = defaultdict(int)
    cond_none  = defaultdict(int)
    cond_items = defaultdict(Counter)
    for row in all_labels:
        cond  = row["condition"]
        plan  = row["plan"]
        cond_total[cond] += 1
        if plan == ["none recorded"]:
            cond_none[cond] += 1
        else:
            for item in plan:
                cond_items[cond][item] += 1
    print(f"\n{'Condition':<26} {'N':>5} {'None':>5} {'None%':>6}  Top plan items")
    print("-" * 78)
    for c in CONDITIONS:
        cond = c["name"]
        n    = cond_total[cond]
        nn   = cond_none[cond]
        pct  = nn / n * 100 if n else 0
        top  = ", ".join(f"{desc[:35]} ({cnt})" for desc, cnt in cond_items[cond].most_common(3))
        print(f"{cond:<26} {n:>5} {nn:>5} {pct:>5.0f}%  {top or '—'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Source run : {SRC_DIR.name}")
    print(f"V2 dir     : {V2_DIR}")
    print(f"Output dir : {OUT_DIR}")

    test_labels = load_jsonl(V2_DIR / "test_labels.jsonl")
    dev_labels  = load_jsonl(V2_DIR / "dev_labels.jsonl")
    print(f"\nLoaded {len(test_labels)} test + {len(dev_labels)} dev labels from v2.")

    # ── Validate ──────────────────────────────────────────────────────────────
    print("\n[1] Validating IDs against source run ...")
    mismatches = validate(test_labels, dev_labels)
    if mismatches:
        print(f"  {len(mismatches)} mismatch(es):")
        for m in mismatches[:15]:
            print(f"    {m['id']}  →  {m['reason']}")
        if len(mismatches) > 15:
            print(f"    ... and {len(mismatches) - 15} more")
        if len(mismatches) > 20:
            print("\nERROR: Too many mismatches. Stopping.")
            sys.exit(1)
        print("  Within expected ~0.2% tolerance — continuing.")
    else:
        print("  All IDs match exactly.")

    # ── Compute plans ─────────────────────────────────────────────────────────
    print("\n[2] Computing plan labels ...")
    examples = [parse_id(r["id"])[::-1] for r in test_labels + dev_labels]  # (pid, cond)
    plan_map = compute_plans(examples)
    print(f"  Done ({len(plan_map)} examples).")

    # ── Attach plans ──────────────────────────────────────────────────────────
    def attach(labels):
        out = []
        for row in labels:
            cond, pid = parse_id(row["id"])
            out.append({**row, "plan": plan_map[(pid, cond)]})
        return out

    test_out = attach(test_labels)
    dev_out  = attach(dev_labels)

    # Plans that cannot be trusted due to cutoff mismatch between source run and v2
    # are set to null so downstream code doesn't silently use a plan from the wrong
    # episode.
    PLAN_NULLED = [
        {
            "id": "viral_pharyngitis_bd529361-c87c-b80f-2847-0050f73f3da4",
            "reason": "cutoff mismatch: source run mine=2018-02-01, v2=2016-09-26",
        },
    ]
    plan_nulled_ids = {r["id"] for r in PLAN_NULLED}
    nulled_count = 0
    for row in test_out + dev_out:
        if row["id"] in plan_nulled_ids:
            row["plan"] = None
            nulled_count += 1
    if nulled_count:
        print(f"  Set plan=null for {nulled_count} record(s) with cutoff mismatch.")

    # ── Write output ──────────────────────────────────────────────────────────
    print(f"\n[3] Writing {OUT_DIR} ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fname in ("test.jsonl", "dev.jsonl", "corpus.jsonl"):
        shutil.copy2(V2_DIR / fname, OUT_DIR / fname)

    print("  Verifying record files are byte-identical to v2 ...")
    ok = True
    for fname in ("test.jsonl", "dev.jsonl", "corpus.jsonl"):
        match = md5_file(V2_DIR / fname) == md5_file(OUT_DIR / fname)
        print(f"    {fname}: {'OK' if match else 'MISMATCH'}")
        if not match:
            ok = False
    if not ok:
        print("ERROR: byte-identity check failed.")
        sys.exit(1)

    def write_jsonl(rows, path):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Wrote {path.name} ({len(rows)} records)")

    write_jsonl(test_out, OUT_DIR / "test_labels.jsonl")
    write_jsonl(dev_out,  OUT_DIR / "dev_labels.jsonl")

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "release": "v2.1",
        "base_release": "v2",
        "plan_source_run": SRC_DIR.name,
        "plan_source_commit": git_commit(),
        "note": (
            "Record files (test.jsonl, dev.jsonl, corpus.jsonl) are byte-identical "
            "to release-v2. Plan labels were computed from the pop10000-seed20260916 "
            "source run using REASONCODE matching and attached to Lauren's v2 "
            "label files. ~0.2% of v2 IDs differ from the source run; those records "
            "received plans derived from the source run patient with the same ID."
        ),
        "mismatches_v2_vs_source_run": len(mismatches),
        "plan_nulled": PLAN_NULLED,
        "built": datetime.date.today().isoformat(),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("  Wrote manifest.json")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print_stats(test_out + dev_out)
    print("\nDone.")


if __name__ == "__main__":
    main()

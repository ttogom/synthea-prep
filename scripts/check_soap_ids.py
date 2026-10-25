#!/usr/bin/env python3
"""Check that every id in <split>.jsonl appears exactly once in soap/<split>-soap.jsonl.

Usage:
  python3 check_soap_ids.py [--split corpus|dev|test|corpus_nota|dev_nota|test_nota [...]] [--data DIR]

  --split   splits to check, e.g. --split dev test dev_nota
            (default: corpus dev test; *_nota splits only when named)
  --data    release folder (default: data/release-v2)

Reports ids that are missing, duplicated, or not in <split>.jsonl (extra).
A split whose files don't exist yet is skipped with a message.
Exits 0 if every checked split passes, else 1.
"""
import argparse, json, os, sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO_ROOT, "data", "release-v2")
SPLITS = ["dev", "test", "corpus"]  # same order as soap.py
NOTA_SPLITS = [f"{s}_nota" for s in SPLITS]  # opt-in via --split, as in soap.py
ALL_SPLITS = SPLITS + NOTA_SPLITS


def read_ids(path):
    with open(path) as f:
        return [json.loads(l)["id"] for l in f if l.strip()]


def show(label, ids, limit=20):
    print(f"{label}: {len(ids)}")
    for i in ids[:limit]:
        print(f"  {i}")
    if len(ids) > limit:
        print(f"  ... and {len(ids) - limit} more")


def check(data, split):
    """Return True if passed, False if failed, None if skipped."""
    src_path = os.path.join(data, f"{split}.jsonl")
    soap_path = os.path.join(data, "soap", f"{split}-soap.jsonl")
    print(f"== {split} ==")
    absent = [p for p in (src_path, soap_path) if not os.path.exists(p)]
    if absent:
        for p in absent:
            print(f"SKIP: {os.path.relpath(p)} does not exist")
        return None

    src_ids = read_ids(src_path)
    soap_counts = Counter(read_ids(soap_path))
    src_set = set(src_ids)

    missing = [i for i in src_ids if soap_counts[i] == 0]
    dups = [f"{i} (x{soap_counts[i]})" for i in src_ids if soap_counts[i] > 1]
    extra = sorted(i for i in soap_counts if i not in src_set)

    print(f"{split}.jsonl ids: {len(src_ids)} ({len(src_set)} unique)")
    print(f"{split}-soap.jsonl rows: {sum(soap_counts.values())} ({len(soap_counts)} unique ids)")
    show("missing", missing)
    show("duplicated", dups)
    show(f"extra (not in {split}.jsonl)", extra)

    ok = not missing and not dups and not extra
    print(f"OK: every {split} id appears exactly once" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", nargs="+", choices=ALL_SPLITS, default=SPLITS,
                    help="splits to check (default: corpus dev test; *_nota splits only when named)")
    ap.add_argument("--data", default=ROOT, help="release folder (default: data/release-v2)")
    args = ap.parse_args()

    results = {}
    for split in dict.fromkeys(args.split):  # dedupe, keep order
        results[split] = check(args.data, split)
        print()

    print("Summary: " + ", ".join(
        f"{s} {'OK' if r else 'SKIPPED' if r is None else 'FAIL'}" for s, r in results.items()))
    sys.exit(1 if any(r is False for r in results.values()) else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build med_copilot/data/patient_corpus/reference_patients.json from SOAP corpus files.

Usage (from repo root):
    python3 scripts/build_reference_corpus.py

Reads:
    data/release-v2/soap/corpus-soap.jsonl          (required)
    data/release-v2/soap/corpus_nota-soap.jsonl     (included if present)
    configs/condition_names.json                    (code -> full name mapping)

Writes:
    med_copilot/data/patient_corpus/reference_patients.json
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    REPO_ROOT / "data/release-v2/soap/corpus-soap.jsonl",
    REPO_ROOT / "data/release-v2/soap/corpus_nota-soap.jsonl",
]
CONDITION_NAMES = REPO_ROOT / "configs/condition_names.json"
OUTPUT = REPO_ROOT / "med_copilot/data/patient_corpus/reference_patients.json"

# UUID suffix: _<8>-<4>-<4>-<4>-<12> hex chars
_UUID_SUFFIX = re.compile(r"_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def condition_code(record_id):
    """Strip trailing _<uuid>, then strip leading nota_ prefix if present."""
    code = _UUID_SUFFIX.sub("", record_id)
    if code.startswith("nota_"):
        code = code[5:]
    return code


def build_record(raw, name_map):
    """Convert one SOAP record to the new schema; return None to skip."""
    subjective = raw.get("subjective", "")
    objective = raw.get("objective", "")
    if not isinstance(subjective, str) or not subjective.strip():
        return None
    if not isinstance(objective, str):
        objective = ""
    plan = raw.get("plan", [])
    if not isinstance(plan, list):
        plan = []
    plan = [str(item) for item in plan if str(item).strip()]
    if not plan:
        plan = ["none recorded"]
    code = condition_code(raw["id"])
    if code not in name_map:
        raise ValueError(f"Unknown condition code {code!r} from id {raw['id']!r}. "
                         f"Add it to configs/condition_names.json.")
    return {
        "id": raw["id"],
        "subjective": subjective,
        "objective": objective,
        "condition": name_map[code],
        "plan": plan,
    }


def main():
    name_map = json.loads(CONDITION_NAMES.read_text(encoding="utf-8"))

    records = []
    seen_ids = {}
    for source in SOURCES:
        if not source.exists():
            print(f"  skipping {source.relative_to(REPO_ROOT)} (not found)")
            continue
        file_count = 0
        with source.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                record_id = raw.get("id", "")
                built = build_record(raw, name_map)
                if built is None:
                    print(f"  SKIP {source.name}:{line_number} — blank subjective (id={record_id!r})")
                    continue
                if record_id in seen_ids:
                    print(f"  SKIP {source.name}:{line_number} — duplicate id {record_id!r} "
                          f"(first at {seen_ids[record_id]})")
                    continue
                seen_ids[record_id] = f"{source.name}:{line_number}"
                records.append(built)
                file_count += 1
        print(f"  read {file_count} records from {source.relative_to(REPO_ROOT)}")

    if not records:
        print("ERROR: no records produced. Aborting.", file=sys.stderr)
        sys.exit(1)

    counts = Counter(r["condition"] for r in records)
    print(f"\nTotal records: {len(records)}")
    print("Per condition:")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {count:5d}  {name}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {len(records)} records to {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
extract_labels.py — Extract per-patient labels and build a clean training directory.

Given a scrub output directory, produces:

  1. labels.json (sibling of the scrub dir by default): a JSON object keyed by
     patient ID with the target code, description, and cutoff date.

         { "pid": {"code": "88805009",
                   "description": "Chronic congestive heart failure (disorder)",
                   "cutoff": "2018-03-14"}, ... }

     Format rationale: JSON is naturally keyed by patient ID, handles mixed
     types and commas in description strings without quoting ambiguity, and
     lets callers look up a single patient without reading all rows.

     Labels come from the SOURCE run's conditions.csv. The diagnosis row is
     absent from the scrubbed output, so the scrub dir can't be used.
     Cutoffs are recomputed independently here (see SCRUBBING.md §Verification).

  2. A clean training directory (sibling of the scrub dir by default) containing
     only patient data (csv/, symptoms/, notes/). manifest.json and
     SCRUB_SUMMARY.md are excluded; both name the target condition and its code.
     The directory name uses the code-set sha6 but not the condition name.

  After building the training directory the script scans every file in it for
  the target SNOMED code(s) and description(s) and reports any hits.

Usage:
    python3 scripts/extract_labels.py <scrub-dir>
        [--labels <path>]      default: <scrub-dir.parent>/labels-<sha6>.json
        [--train-dir <path>]   default: <scrub-dir.parent>/<source-run>__train-<sha6>
        [--force]              replace existing output files

Python 3.9+, standard library only.
"""

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Files excluded from the clean training directory.
# manifest.json and SCRUB_SUMMARY.md both record the target code and description.
_EXCLUDE = {"manifest.json", "SCRUB_SUMMARY.md"}

# Dated CSVs that carry a REASONCODE column, for the cutoff-pull step.
# Mirrors FILTER_COL in scrub.py (excluding files without REASONCODE).
_REASONCODE_STEMS = {
    "conditions": "START",
    "encounters": "START",
    "medications": "START",
    "procedures": "START",
    "careplans": "START",
    "payer_transitions": "START_DATE",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csv(path: Path):
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _local_date(s: str):
    """Synthea date or UTC datetime string → UTC date. Returns None if empty or
    unparseable. Date-only strings are returned as-is (already local dates in
    UTC-pinned runs)."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        if len(s) == 10:
            return date.fromisoformat(s)
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo("UTC")).date()
    except ValueError:
        return None


# ── Cutoff recomputation ─────────────────────────────────────────────────────

def _enc_starts(csv_dir: Path) -> dict:
    """{encounter_id: START_string}."""
    header, rows = load_csv(csv_dir / "encounters.csv")
    idx = {c: i for i, c in enumerate(header)}
    return {r[idx["Id"]]: r[idx["START"]] for r in rows}


def _compute_cutoffs(csv_dir: Path, target_codes: set, enc_starts: dict) -> dict:
    """Recompute per-patient cutoffs from the source run.

    Returns {pid: {"code": str, "description": str, "cutoff": date}}.

    The cutoff for each patient is the earliest of:
      - for each target conditions.csv row: min(START, the local date of the
        linked encounter's START), and
      - for each row in dated CSVs whose REASONCODE is a target code (and the
        patient already has a conditions.csv match): the same min(date, enc-date).

    This is the same rule as scrub.py (reimplemented independently per
    SCRUBBING.md §Verification).
    """
    header, rows = load_csv(csv_dir / "conditions.csv")
    idx = {c: i for i, c in enumerate(header)}
    pid_i = idx["PATIENT"]
    code_i = idx["CODE"]
    start_i = idx["START"]
    desc_i = idx["DESCRIPTION"]
    enc_i = idx["ENCOUNTER"]

    result: dict = {}  # pid -> {"code", "description", "cutoff"}
    for row in rows:
        if row[code_i] not in target_codes:
            continue
        pid = row[pid_i]
        d = _local_date(row[start_i])
        if d is None:
            continue
        enc_d = _local_date(enc_starts.get(row[enc_i], ""))
        if enc_d is not None and enc_d < d:
            d = enc_d
        existing = result.get(pid)
        if existing is None or d < existing["cutoff"]:
            result[pid] = {"code": row[code_i], "description": row[desc_i],
                           "cutoff": d}

    # REASONCODE rule: move cutoffs earlier where a visit or prescription was
    # done *for* the target condition before the conditions.csv row was recorded.
    for stem, col in _REASONCODE_STEMS.items():
        path = csv_dir / f"{stem}.csv"
        if not path.is_file():
            continue
        header, rows = load_csv(path)
        idx2 = {c: i for i, c in enumerate(header)}
        if "REASONCODE" not in idx2 or "PATIENT" not in idx2:
            continue
        pid_i2 = idx2["PATIENT"]
        col_i = idx2[col]
        reason_i = idx2["REASONCODE"]
        enc_i2 = idx2.get("ENCOUNTER")
        for row in rows:
            pid = row[pid_i2]
            if row[reason_i] not in target_codes or pid not in result:
                continue
            d = _local_date(row[col_i])
            if d is None:
                continue
            enc_d = _local_date(enc_starts.get(row[enc_i2], "")) if enc_i2 is not None else None
            if enc_d is not None and enc_d < d:
                d = enc_d
            if d < result[pid]["cutoff"]:
                result[pid]["cutoff"] = d

    return result


# ── Leak scan ────────────────────────────────────────────────────────────────

def _scan(train_dir: Path, codes: list, descriptions: list) -> list:
    """Scan every file in train_dir for target codes or description text.

    For each description, the SNOMED-style tag (e.g. "(disorder)") is stripped
    and the remainder is matched case-insensitively. Returns a list of
    (rel_path, line_no, line) tuples for every matching line.
    """
    code_pats = [re.compile(re.escape(c)) for c in codes]
    desc_pats = []
    for d in descriptions:
        base = re.sub(r"\s*\([^)]+\)\s*$", "", d).strip()
        desc_pats.append(re.compile(re.escape(base), re.IGNORECASE))
    all_pats = code_pats + desc_pats

    hits = []
    for path in sorted(train_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(train_dir)
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pat in all_pats:
                if pat.search(line):
                    hits.append((str(rel), lineno, line.rstrip()))
                    break
    return hits


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract labels and build a clean training directory from a scrub output."
    )
    ap.add_argument("scrub_dir", help="Path to a scrub output directory")
    ap.add_argument("--labels",
                    help="Path for labels.json (default: <scrub-dir.parent>/labels-<sha6>.json)")
    ap.add_argument("--train-dir", dest="train_dir",
                    help="Path for training dir (default: <source-run>__train-<sha6>)")
    ap.add_argument("--force", action="store_true",
                    help="Replace existing labels file and training directory")
    args = ap.parse_args()

    scrub_dir = Path(args.scrub_dir).resolve()
    mf_path = scrub_dir / "manifest.json"
    if not mf_path.exists():
        sys.exit(f"ERROR: {mf_path} not found — is that a scrub output directory?")
    scrub_manifest = json.loads(mf_path.read_text())
    if "scrub_script_version" not in scrub_manifest:
        sys.exit(f"ERROR: {mf_path} is not a scrub manifest")

    codes = scrub_manifest["codes"]
    sha6 = scrub_manifest["codes_sha6"]
    source_run_name = scrub_manifest["source_run"]
    target_codes = set(codes)

    src_run = scrub_dir.parent / source_run_name
    if not (src_run / "csv").is_dir():
        sys.exit(f"ERROR: source run not found at {src_run}")

    labels_path = Path(args.labels).resolve() if args.labels else \
        scrub_dir.parent / f"labels-{sha6}.json"
    train_dir = Path(args.train_dir).resolve() if args.train_dir else \
        scrub_dir.parent / f"{source_run_name}__train-{sha6}"

    for p in (labels_path, train_dir):
        if p.exists() and not args.force:
            sys.exit(f"ERROR: {p} exists. Use --force to replace.")

    # ── Labels ───────────────────────────────────────────────────────────────
    print(f"Source run   : {src_run.name}")
    print(f"Codes        : {codes}")
    enc_st = _enc_starts(src_run / "csv")
    cutoffs = _compute_cutoffs(src_run / "csv", target_codes, enc_st)

    ph, pr = load_csv(scrub_dir / "csv" / "patients.csv")
    emitted = {r[{c: i for i, c in enumerate(ph)}["Id"]] for r in pr}

    missing = emitted - set(cutoffs)
    if missing:
        print(f"WARNING: {len(missing)} emitted patient(s) have no recomputed cutoff "
              f"— omitted from labels.", file=sys.stderr)

    labels = {}
    for pid in sorted(emitted & set(cutoffs)):
        info = cutoffs[pid]
        labels[pid] = {
            "code": info["code"],
            "description": info["description"],
            "cutoff": info["cutoff"].isoformat(),
        }

    if labels_path.exists():
        labels_path.unlink()
    labels_path.write_text(json.dumps(labels, indent=2) + "\n")
    print(f"Labels       : {labels_path}  ({len(labels)} patients)")

    # ── Clean training directory ──────────────────────────────────────────────
    if train_dir.exists():
        shutil.rmtree(train_dir)
    n_copied = 0
    for src_path in sorted(scrub_dir.rglob("*")):
        if not src_path.is_file():
            continue
        if src_path.name in _EXCLUDE:
            continue
        dst = train_dir / src_path.relative_to(scrub_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst)
        n_copied += 1
    print(f"Training dir : {train_dir}  ({n_copied} files copied, "
          f"{len(_EXCLUDE)} excluded: {', '.join(sorted(_EXCLUDE))})")

    # ── Leak scan ────────────────────────────────────────────────────────────
    descriptions = sorted({info["description"] for info in cutoffs.values()
                            if info["code"] in target_codes})
    hits = _scan(train_dir, codes, descriptions)
    print(f"\nLeak scan    : checked for {codes} and {descriptions}")
    if not hits:
        print("             : 0 hits — target code and description not found")
    else:
        print(f"             : {len(hits)} hit(s)")
        for rel, lineno, line in hits[:30]:
            print(f"  {rel}:{lineno}: {line[:120]}")
        if len(hits) > 30:
            print(f"  ... and {len(hits) - 30} more")

    print("\nDone.")


if __name__ == "__main__":
    main()

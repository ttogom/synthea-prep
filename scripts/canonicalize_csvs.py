#!/usr/bin/env python3
"""
canonicalize_csvs.py — Normalize every CSV in a run directory so that
byte-identical output is achieved regardless of multi-threaded write order.

For each CSV:
  1. Read as all-string (dtype=str, keep_default_na=False) to preserve values exactly.
  2. For encounters.csv only: set STOP to empty for any row whose STOP timestamp
     is later than the pinned reference date. Synthea fills those from the wall
     clock, so they vary across runs. Logging how many rows are affected.
  3. Sort all data rows by every column left-to-right.
  4. Write back with the same quoting Synthea uses (QUOTE_MINIMAL).

The header row is always kept first. Empty files (header only) are left alone.

Usage:
    python3 scripts/canonicalize_csvs.py <run_dir> [--reference-date YYYYMMDD]
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


def _parse_ref(reference_date: str) -> pd.Timestamp:
    s = reference_date.strip()
    return pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]}", tz="UTC")


def fix_open_encounters(df: pd.DataFrame, ref_dt: pd.Timestamp) -> int:
    """
    Zero out STOP for encounters still open at simulation end.
    Returns the number of rows changed.
    """
    if "STOP" not in df.columns:
        return 0
    stop_dt = pd.to_datetime(df["STOP"], errors="coerce", utc=True)
    mask = stop_dt > ref_dt
    count = int(mask.sum())
    if count:
        df.loc[mask, "STOP"] = ""
    return count


def canonicalize(path: Path, ref_dt: pd.Timestamp | None) -> None:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    if len(df) == 0:
        return

    if ref_dt is not None and path.stem == "encounters":
        changed = fix_open_encounters(df, ref_dt)
        if changed:
            print(f"    zeroed  {changed} open-encounter STOP value(s) in {path.name}")

    df.sort_values(by=list(df.columns), kind="stable", inplace=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonicalize (sort) all CSVs in a Synthea run directory."
    )
    parser.add_argument("run_dir", help="Path to data/<run-name>")
    parser.add_argument(
        "--reference-date",
        default="20260921",
        help="Pinned simulation end date (YYYYMMDD). "
             "Encounter STOP values after this date are set to empty. "
             "Default: 20260921",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    ref_dt = _parse_ref(args.reference_date)

    csvs = sorted(run_dir.glob("csv/*.csv")) + sorted(
        run_dir.glob("symptoms/csv/*.csv")
    )
    if not csvs:
        print(f"No CSVs found under {run_dir}", file=sys.stderr)
        sys.exit(1)

    for path in csvs:
        canonicalize(path, ref_dt)
        print(f"  sorted  {path.relative_to(run_dir)}")

    print(f"Canonicalized {len(csvs)} file(s).")


if __name__ == "__main__":
    main()

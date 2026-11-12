#!/usr/bin/env python3
"""Summarize a Synthea run directory and write data/<run-name>/SUMMARY.md."""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def fmt_size(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / 1024 / 1024:.1f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a Synthea run directory and write SUMMARY.md."
    )
    parser.add_argument("run_dir", help="Path to data/<run-name> directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    csv_dir = run_dir / "csv"
    symptoms_csv_path = run_dir / "symptoms" / "csv" / "symptoms.csv"
    notes_dir = run_dir / "notes"

    lines: list[str] = []

    lines.append(f"# Dataset Summary: {run_dir.name}")
    lines.append(f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")

    # ── Patient counts ──────────────────────────────────────────────────────────
    patients = pd.read_csv(csv_dir / "patients.csv", low_memory=False)
    total = len(patients)
    deceased = int(patients["DEATHDATE"].notna().sum())
    alive = total - deceased

    lines.append("## Patient Counts\n")
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Total | {total:,} |")
    lines.append(f"| Alive | {alive:,} |")
    lines.append(f"| Deceased | {deceased:,} |")

    # ── Age distribution ────────────────────────────────────────────────────────
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"])
    patients["_deathdt"] = pd.to_datetime(patients["DEATHDATE"], errors="coerce")
    ref = pd.Timestamp.now()
    patients["_age"] = patients.apply(
        lambda r: int((r["_deathdt"] - r["BIRTHDATE"]).days / 365.25)
        if pd.notna(r["_deathdt"])
        else int((ref - r["BIRTHDATE"]).days / 365.25),
        axis=1,
    )

    lines.append("\n## Age Distribution\n")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|-------|")
    stats = patients["_age"].describe()
    for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        lines.append(f"| {stat} | {stats[stat]:.1f} |")

    # ── Sex distribution ────────────────────────────────────────────────────────
    lines.append("\n## Sex Distribution\n")
    lines.append("| Sex | Count | % |")
    lines.append("|-----|-------|---|")
    for sex, count in patients["GENDER"].value_counts().items():
        lines.append(f"| {sex} | {count:,} | {100 * count / total:.1f}% |")

    # ── CSV file inventory ──────────────────────────────────────────────────────
    lines.append("\n## CSV File Inventory\n")
    lines.append("| File | Rows | Size |")
    lines.append("|------|------|------|")

    csv_frames: dict[str, pd.DataFrame] = {}
    for csv_file in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(csv_file, low_memory=False)
        csv_frames[csv_file.stem] = df
        lines.append(
            f"| {csv_file.name} | {len(df):,} | {fmt_size(csv_file.stat().st_size)} |"
        )

    sym_df: pd.DataFrame | None = None
    if symptoms_csv_path.exists():
        sym_df = pd.read_csv(symptoms_csv_path, low_memory=False)
        lines.append(
            f"| symptoms/csv/symptoms.csv | {len(sym_df):,} |"
            f" {fmt_size(symptoms_csv_path.stat().st_size)} |"
        )

    # ── Top 40 conditions by patient count ─────────────────────────────────────
    if "conditions" in csv_frames:
        top40 = (
            csv_frames["conditions"]
            .groupby("DESCRIPTION")["PATIENT"]
            .nunique()
            .sort_values(ascending=False)
            .head(40)
        )
        lines.append("\n## Top 40 Conditions by Patient Count\n")
        lines.append("| # | Condition | Patients |")
        lines.append("|---|-----------|----------|")
        for i, (cond, count) in enumerate(top40.items(), 1):
            lines.append(f"| {i} | {cond} | {count:,} |")

    # ── Symptom-bearing pathologies ─────────────────────────────────────────────
    if sym_df is not None and "NUM_SYMPTOMS" in sym_df.columns:
        bearing = sym_df[sym_df["NUM_SYMPTOMS"] > 0]
        patho_stats = (
            bearing.groupby("PATHOLOGY")
            .agg(
                patient_count=("PATIENT", "nunique"),
                mean_symptoms=("NUM_SYMPTOMS", "mean"),
            )
            .sort_values("patient_count", ascending=False)
        )
        lines.append("\n## Symptom-Bearing Pathologies (NUM_SYMPTOMS > 0)\n")
        lines.append("| Pathology | Patients | Mean Symptoms |")
        lines.append("|-----------|----------|---------------|")
        for patho, row in patho_stats.iterrows():
            lines.append(
                f"| {patho} | {int(row.patient_count):,} | {row.mean_symptoms:.2f} |"
            )

    # ── Clinical notes ──────────────────────────────────────────────────────────
    lines.append("\n## Clinical Notes\n")
    if notes_dir.exists():
        note_files = sorted(f for f in notes_dir.rglob("*") if f.is_file())
        lines.append(f"Files: {len(note_files):,}")
        if note_files:
            lines.append(f"\nExample: `{note_files[0].relative_to(run_dir)}`")
    else:
        lines.append("Notes directory not found.")

    # ── Median records per patient ──────────────────────────────────────────────
    lines.append("\n## Median Records per Patient\n")
    lines.append("| File | Median Records |")
    lines.append("|------|----------------|")
    for stem, df in csv_frames.items():
        patient_col = next((c for c in df.columns if c.upper() == "PATIENT"), None)
        if patient_col:
            median_recs = df.groupby(patient_col).size().median()
            lines.append(f"| {stem}.csv | {median_recs:.1f} |")

    # ── Write SUMMARY.md ────────────────────────────────────────────────────────
    summary_path = run_dir / "SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Written: {summary_path}")


if __name__ == "__main__":
    main()

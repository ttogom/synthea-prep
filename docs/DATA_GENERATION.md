# Data Generation

## Purpose

This pipeline generates synthetic EHR data using Synthea for a clinical-reasoning fine-tuning project. It is configured for reproducibility: the same command on any machine produces byte-identical CSV output. Synthea's disease models are unmodified, and no diagnoses are scrubbed, no diseases selected, and no training examples built at this stage.

## Why These Settings

| Setting | Reason |
|---------|--------|
| Symptoms and notes export enabled | Off by default; without them records have diagnoses but no presenting symptoms or narrative text |
| `claims.csv`, `claims_transactions.csv`, `patient_expenses.csv`, `payers.csv` excluded | Unused in this project and very large |
| FHIR export disabled | Unused; substantial overhead |
| Pinned commit, seeds (`-s`, `-cs`), reference date (`-r`) | Required for identical output across runs; `-r` prevents Synthea from using the wall clock as the simulation start time |
| Multithreaded generation + canonicalization | Threads write rows in nondeterministic finish order; `canonicalize_csvs.py` sorts every CSV after generation; `--single-thread` is substantially slower with no reproducibility benefit |
| JVM timezone pinned to UTC | Synthea writes date-only fields (`conditions.START`, `DEATHDATE`, note headers, …) in the JVM's default timezone and datetimes in UTC; unpinned, the same seed gives different dates on machines in different zones. Pinning makes date-only fields the UTC date of the same instant |
| Open encounters get empty STOP | Synthea fills the STOP of in-progress encounters from the wall clock; `canonicalize_csvs.py` clears any STOP value later than the reference date |
| `manifest.json` written per run | Records commit, seeds, reference date, timezone, and full command so any dataset can be traced back to exactly how it was generated |

## How to Run

**Prerequisites:** Java 17+, Python 3 with pandas (`pip install pandas`), ~4 GB disk for 10,000 patients (measured: 3.8 GB — csv/ 2.7 GB, notes/ 941 MB, symptoms/ 166 MB).

```bash
# Full cohort: 10,000 patients, default seed (~17 min generation + ~5 min canonicalization)
./scripts/generate_synthea.sh 10000 20260916

# Summarize the run
python3 scripts/summarize_dataset.py data/pop10000-seed20260916/

# Age-filtered subset (e.g. 40–80 only)
./scripts/generate_synthea.sh 10000 20260916 40-80
```

`--single-thread` forces single-threaded generation; use it for debugging only. The first run downloads Gradle and Synthea dependencies (~400 MB); subsequent runs use the cache.

## Output

All output goes to `data/<run-name>/`.

| Path | Key columns |
|------|-------------|
| `manifest.json` | Commit, seeds, reference date, timezone, full command, timestamp |
| `SUMMARY.md` | Auto-generated descriptive statistics |
| `csv/patients.csv` | Id, BIRTHDATE, DEATHDATE (empty for living patients), GENDER, RACE, ETHNICITY, MARITAL, INCOME |
| `csv/conditions.csv` | START, STOP, PATIENT, CODE, DESCRIPTION |
| `csv/encounters.csv` | Id, START, STOP†, PATIENT, ENCOUNTERCLASS, CODE, REASONCODE |
| `csv/medications.csv` | START, STOP, PATIENT, CODE, DESCRIPTION, REASONCODE |
| `csv/observations.csv` | DATE, PATIENT, CATEGORY, CODE, DESCRIPTION, VALUE, UNITS, TYPE |
| `csv/procedures.csv` | START, PATIENT, CODE, DESCRIPTION, REASONCODE |
| `csv/allergies.csv` | START, STOP, PATIENT, CODE, DESCRIPTION, TYPE, CATEGORY, REACTION1, SEVERITY1 |
| `csv/careplans.csv` | START, STOP, PATIENT, CODE, DESCRIPTION, REASONCODE |
| `csv/devices.csv` | START, STOP, PATIENT, CODE, DESCRIPTION, UDI |
| `csv/imaging_studies.csv` | DATE, PATIENT, BODYSITE_DESCRIPTION, MODALITY_DESCRIPTION, PROCEDURE_CODE |
| `csv/immunizations.csv` | DATE, PATIENT, CODE, DESCRIPTION |
| `csv/supplies.csv` | DATE, PATIENT, CODE, DESCRIPTION, QUANTITY |
| `csv/organizations.csv`, `providers.csv`, `payer_transitions.csv` | Supporting reference tables — facilities, providers, and coverage periods |
| `symptoms/csv/symptoms.csv` | PATIENT, PATHOLOGY, NUM_SYMPTOMS, SYMPTOMS |
| `notes/` | One file per patient; contains all encounters for that patient as a narrative |

† STOP is empty for encounters still in progress on the reference date.

Dead patients are included in all output files; filter on `DEATHDATE IS NOT NULL` in `patients.csv` if needed. `symptoms/csv/` is a separate subdirectory, not under `csv/`.

## Reproducibility

Runs are byte-identical after canonicalization given the same commit, seeds, reference date, and Java major version, on any machine (the JVM timezone is pinned to UTC). `check_reproducibility.sh` verifies this by generating 200 patients twice and diffing every CSV and note file; because both runs use the same machine, it can't detect timezone dependence on its own. Runs generated before the timezone was pinned have no `timezone` in `manifest.json` and carry the generating machine's local dates.

## Scripts

| Script | What it does |
|--------|--------------|
| `generate_synthea.sh` | Clones and pins Synthea, runs generation, canonicalizes output, writes `manifest.json` |
| `canonicalize_csvs.py` | Sorts every CSV by all columns; clears wall-clock STOP values in `encounters.csv` |
| `summarize_dataset.py` | Reads a run directory, writes `SUMMARY.md` with patient counts, top conditions, symptom pathologies, and file inventory |
| `check_reproducibility.sh` | Generates pop=200 twice and diffs all CSVs and notes |

# Scrubbing: Pre-Diagnosis Datasets

`scripts/scrub.py` turns a Synthea run (see [DATA_GENERATION.md](DATA_GENERATION.md))
into a **pre-diagnosis dataset**. Every patient who develops a target condition
has their record cut at the diagnosis. Everything before it is kept, and
everything from the diagnosis on is removed.

The data is for fine-tuning a model to reason *toward* a diagnosis. Any trace of
the diagnosis, or of anything after it, hands the model the answer. Losing real
pre-diagnosis history weakens the data, so the rules below aim to remove exactly
what happened on or after the diagnosis and nothing earlier.

This document describes current behavior (script v1.3). Design history and the
reasoning behind revisions are in [SCRUBBING_PLAN.md](SCRUBBING_PLAN.md).

## Quick start

```bash
python3 scripts/scrub.py data/pop10-seed20260916 \
    --codes configs/metabolic.txt
# → data/pop10-seed20260916__scrub-metabolic-81a87b/
```

Python 3.9+, standard library only.

## Inputs

| Argument | Required | Meaning |
|----------|----------|---------|
| `<source-run-dir>` | yes | A Synthea run dir from `generate_synthea.sh`, e.g. `data/pop10-seed20260916`. Must contain `csv/`; `notes/` and `symptoms/csv/` are used when present. Never modified. |
| `--codes <file>` | yes | Target conditions (see below). |
| `--out <dir>` | no | Output dir; default is the auto-generated name below. |
| `--force` | no | Replace an existing output dir. Only allowed if that dir is a previous scrub output. |

### Codes file

Plain text, one SNOMED-CT code per line, matched against `conditions.csv` `CODE`.
`#` starts a comment (full-line or trailing). Blank lines, surrounding
whitespace and duplicates are ignored. A non-numeric code is an error, which
catches typos such as a letter `O` for zero.

```
# configs/metabolic.txt
237602007   # Metabolic syndrome X (disorder)
```

List several codes to scrub at the earliest of any of them. Codes that match
no one produce a warning; if **no** code matches anyone, the run fails.

## Output

```
data/<source-run>__scrub-<codes-stem>-<sha6>/
  csv/                    # all source CSVs, filtered (reference tables copied)
  symptoms/csv/symptoms.csv
  notes/                  # emitted patients' notes, cut by entry date
  manifest.json           # machine-readable provenance and counts
  SCRUB_SUMMARY.md        # human summary: parameters, counts, warnings, inventory
```

- `<codes-stem>` is the codes filename without its extension. `<sha6>` is the first 6 hex chars of
  `sha1("\n".join(sorted(unique_codes)))`, so the name identifies the exact
  code set.
- Not carried over: `symptoms/text/` (unused; would re-introduce the
  diagnosis), `fhir/` and `metadata/` (hospital/practitioner reference files and Synthea run metadata, no patient records), and the source
  run's `manifest.json` / `SUMMARY.md` (they describe pre-scrub data; the source
  manifest is embedded in the new `manifest.json` instead).
- `manifest.json` records the version, codes, timezone and the result of
  its check, the cut rules, and per-file counts (dropped rows, blanked values).
  It also lists excluded patients with the reason and every warning.
- Data files are deterministic: re-running gives byte-identical CSVs and notes.
  Only the timestamps in `manifest.json` and `SCRUB_SUMMARY.md` differ.

> **Keep `manifest.json`, `SCRUB_SUMMARY.md` and the output dir name out of
> training data.** They name the target codes, so they reveal the diagnosis.

## The rules

### Cutoff

For each patient with a target condition, look at their `conditions.csv` rows
whose `CODE` is in the codes file. For each such row, the diagnosis date is the
**earlier** of:

- its `START` date, and
- the local date of the encounter it's linked to (`ENCOUNTER` → `encounters.START`).

The patient's **cutoff** is the earliest of these dates. It's a UTC date (see
[Timezone](#timezone)).

The encounter date is needed because Synthea sometimes dates `START` later than
the visit that recorded the condition. Usually it's the next day: the onset is
timestamped a little after a visit that starts just before midnight. A few are
weeks or months late. Cutting at `START` alone would keep
the diagnosing visit, including a note saying *"Patient is presenting with …"*.

**Rows done for the target.** For these patients, rows in `encounters`,
`medications`, `procedures` and `careplans` whose `REASONCODE` is a target code
also count, dated the same way (the earlier of the row's own date and its
linked encounter's). `REASONCODE` is the condition a visit, prescription,
procedure or care plan was *for*, in the same codes as `conditions.csv` `CODE`.
An earlier episode can have no `conditions.csv` row while those rows still
name it: on pop1000, 33 ER visits for Overdose predate the patient's first
Overdose condition row. Without this rule they are kept.

`REASONCODE` only moves a cutoff earlier. It doesn't add patients: a patient
with a target `REASONCODE` but no target `conditions.csv` row is not emitted.
(On pop1000 that's 673 patient/code pairs over 43 codes, e.g. 107 patients with
Sinusitis visits but no Sinusitis row. Why those rows are missing hasn't been
investigated.)

### What is removed (scrubbed patients)

Anything dated on or after the cutoff is removed. Same-day events are removed
too, since times are compared as local dates.

| File | Row dropped when | Also |
|------|------------------|------|
| `conditions.csv` | `START` ≥ cutoff (includes the diagnosis row) | `STOP` ≥ cutoff blanked |
| `encounters.csv` | `START` ≥ cutoff | `STOP` ≥ cutoff blanked |
| `medications.csv` | `START` ≥ cutoff | `STOP` ≥ cutoff blanked; see [Medications](#medications) |
| `observations.csv` | `DATE` ≥ cutoff | |
| `procedures.csv` | `START` ≥ cutoff | `STOP` ≥ cutoff blanked |
| `allergies.csv` | `START` ≥ cutoff | `STOP` ≥ cutoff blanked |
| `careplans.csv` | `START` ≥ cutoff | `STOP` ≥ cutoff blanked |
| `devices.csv` | `START` ≥ cutoff | `STOP` ≥ cutoff blanked |
| `imaging_studies.csv` | `DATE` ≥ cutoff | |
| `immunizations.csv` | `DATE` ≥ cutoff | |
| `supplies.csv` | `DATE` ≥ cutoff | |
| `payer_transitions.csv` | `START_DATE` ≥ cutoff | `END_DATE` ≥ cutoff blanked |

In addition, any row whose `ENCOUNTER` is a dropped encounter is dropped, even
if its own date is earlier. This is a backstop for inconsistent dates.

A blanked end date means the item was **still ongoing at the cutoff**. Keeping
the real end date would reveal the future; for example, a medication stopped
the day after the diagnosis is a strong hint.

### Notes

Each `notes/<Names>_<patientId>.txt` holds one entry per encounter, newest
first. Every entry starts with a bare `YYYY-MM-DD` line (local date). For
scrubbed patients, only entries dated before the cutoff are kept, byte-for-byte.
If none remain, no file is written. Before scrubbing, the number of entries
must equal the patient's encounter count. If it doesn't, an entry header wasn't
recognized, and the patient is excluded.

### Symptoms

`symptoms/csv/symptoms.csv` has no date column. `AGE_BEGIN`/`AGE_END` (whole
years) are its only time signal, and `PATHOLOGY` is a condition description,
which can name the diagnosis directly. For scrubbed patients:

- Drop rows whose `PATHOLOGY` is a target condition's description.
- Drop rows with `AGE_BEGIN` ≥ the patient's age on the cutoff date.
- On kept rows, blank `AGE_END` ≥ that age.

`≥` rather than `>` is used because a row at the cutoff age may come after the
diagnosis within that year. The cost is up to one year of pre-diagnosis symptom
rows; those conditions are still in `conditions.csv` with exact dates. Later
rows unrelated to the target are dropped too. The rule is time-based, not
relatedness-based, because later conditions are often complications of the
target.

### Medications

`DISPENSES` and `TOTALCOST` accumulate until `STOP`, or until the end of the
simulation if `STOP` is empty. For example, a 1979 prescription showed 569
dispenses, running decades past a 2010 cutoff. They are **blanked on every
emitted row whose `STOP` is empty after scrubbing**. `BASE_COST` and
`PAYER_COVERAGE` are per-dispense amounts and are kept.

### Patients

- `DEATHDATE` ≥ cutoff is blanked (scrubbed patients).
- `HEALTHCARE_EXPENSES` and `HEALTHCARE_COVERAGE` are lifetime totals. They
  can't be recomputed as of the cutoff because claims aren't exported, so they
  are blanked for **every emitted patient**.
- End-of-simulation attributes (`MARITAL`, `ADDRESS`, `INCOME`, …) are kept
  as they are. They aren't post-cutoff events and can't be reconstructed.

### Reference tables

`organizations.csv` and `providers.csv` have no patient column and are copied
unchanged. Their aggregate counts (`UTILIZATION`, `ENCOUNTERS`, …) include
post-cutoff activity across all patients. The risk is low, but they are not
scrubbed.

## Who is emitted

Only patients with a target condition, each scrubbed. Every emitted patient is
a positive case. (v1.2 also had an `all` mode that emitted everyone else
unscrubbed; it was removed in v1.3, see D9.)

## Timezone

Synthea writes datetimes in UTC but writes date-only fields (`conditions.START`,
note headers, `allergies.START`, `careplans.START`, `supplies.DATE`,
`DEATHDATE`) in the Java default timezone. `generate_synthea.sh` pins that
zone to UTC, and `scrub.py` only accepts UTC runs; everything is compared as
UTC dates.

- A run whose `manifest.json` records a different `timezone` is rejected.
- A run with no recorded zone (no manifest, or one from before the pin) is
  accepted only if its data fits UTC.

Either way, the data is checked: at least 99.9% of note entry dates must equal
the UTC date of one of the patient's encounters (or 90% of `conditions.START`
if the run has no notes). Otherwise the run exits without writing. Runs
generated in another zone must be regenerated with `generate_synthea.sh`.
On pop10 and pop1000, UTC scores 100%. On an older run generated in Los
Angeles, UTC scored 65.7%.

## Safety and error handling

- **Unparseable dates mean exclusion, not keeping.** A diagnosed patient with any
  unparseable date the cut depends on is **left out entirely**, with a warning
  giving the reason. These are: any filter or end-date column,
  `BIRTHDATE`, `DEATHDATE`, `AGE_BEGIN`/`AGE_END`, a note entry date or entry
  count, and a target `START`. Keeping such data could leak the diagnosis, and
  such values shouldn't occur in Synthea output.
- **The source is never touched.** An output dir that is, contains, or sits
  inside the source run is refused, even with `--force`.
- **`--force` only replaces scrub outputs.** It refuses to delete a dir without a
  scrub `manifest.json`. All validation runs before anything is deleted, so a
  failed run leaves an existing output in place.
- **Fails instead of writing junk:** when no code matches anyone, when no one
  would be emitted, on a run from a timezone other than UTC, or on a
  malformed codes file.
- **Unknown CSVs** in the source are copied unchanged, with a warning.

## Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Match targets by `CODE` only | `CODE`↔`DESCRIPTION` is 1-to-1 in `conditions.csv` (verified on pop10; re-check on 10k). Symptoms matching relies on it. |
| D2 | Cut at the earliest occurrence, taking the earlier of `START` and the linked encounter's date; rows with the target as `REASONCODE` also count *(v1.3)* | The first diagnosis is the answer; `START` alone is sometimes late, and earlier episodes can appear only as a visit's or prescription's reason. |
| D3 | Keep rows that started before the cutoff; blank end dates ≥ cutoff | Prior meds/conditions are wanted history, but their future end dates leak. *(v1.2 revision; previously `STOP` was left as-is)* |
| D4 | Notes cut by entry date | The task asks for notes on/after the diagnosis to be dropped; they state the diagnosis. |
| D5 | `symptoms.csv` filtered by `PATHOLOGY` and age; `symptoms/text/` not copied | Only time signal is age; text files are unused. |
| D6 | New standalone output dir; source untouched | Reproducibility and safety. |
| D7 | Blank post-cutoff `DEATHDATE`; blank lifetime totals for everyone | Remove post-diagnosis data without marking who was scrubbed. |
| D8 | Cut `payer_transitions.csv` too | Non-clinical, but its dates line up with visits and can leak. |
| D9 | Emit diagnosed patients only *(v1.3; `all` mode removed)* | In `all` mode, unscrubbed negatives ran to the end of the simulation while positives stopped at the diagnosis. On pop1000 for type 2 diabetes, "alive, but no visit in the last year" flagged 74/74 positives and 76/1,056 negatives, so the negatives taught a shortcut rather than contrast. Negatives, if needed, should be cut at matched dates. |
| D10 | Generator pins the JVM timezone to UTC; `scrub.py` accepts only UTC runs and checks the data *(v1.3; `--tz` removed)* | Date-only fields were machine-local; a wrong zone silently shifts cutoffs and breaks cross-machine reproducibility. |
| D11 | Exclude diagnosed patients with any unparseable date | Keeping them risked emitting the diagnosis row itself. |
| D12 | Blank open-ended medication running totals for everyone | Lifetime counts leak the future; blanking all avoids a marker. |

## Known limitations and open questions

- **All-positive** by construction (see D9).
- **Patients with only a `REASONCODE` for the target aren't emitted** (see
  [Cutoff](#cutoff)). Whether they should count as positives is open.
- **Related diagnoses aren't cut.** Only listed codes are matched, so a
  condition that names the target under another code survives before the
  cutoff. For example, "Microalbuminuria due to type 2 diabetes mellitus"
  6 years before a type 2 diabetes row, or "Childhood asthma" for Asthma.
  List such codes in the codes file.
- **Symptoms `AGE_BEGIN` isn't the onset age.** It can be 1–7 years below the
  age at the condition's `START`, so the age rule kept 211 rows for conditions
  starting after the cutoff across 77 of 273 codes on pop1000.
- **Symptoms coverage.** `symptoms.csv` appears to cover only about the last 10
  years of each patient's life. Patients diagnosed earlier have no symptoms
  rows left.
- **End-of-simulation snapshots** in `patients.csv` (marital status, address,
  income) describe the patient at the end, not at the cutoff.
- **Reference-table aggregates** include post-cutoff activity (see above).
- **Scale.** Source CSVs are read into memory, twice for the scrubbed
  patients' date scan. Fine for pop10; check memory and runtime on the 10k run.

## Verification

Done on `pop10-seed20260916` (v1.2), both the older Los Angeles–generated
run (`--tz America/Los_Angeles`) and the current UTC-pinned regeneration:

- **Every condition code** in the run (75 codes) was scrubbed individually in both
  modes. For every scrubbed patient, every date value in every output CSV (start, end,
  death) is before the cutoff. No kept note entry is on or after it. No
  symptoms row is at or above the cutoff age, and no note mentions the target
  description. Result: 0 failures on both runs. The same checks run on
  unscrubbed data report 12,236 failures, so they do detect leaks.
- This includes the codes where `START` is later than the diagnosing visit
  (on the LA run: gingivitis, gingival disease, dental caries, tubal ligation,
  sleep apnea, cystitis, homeless), which leaked through the diagnosing visit's
  note in v1.1. The UTC run has 13 such rows, so the encounter-date rule
  is still needed with a pinned zone.
- **Timezone (v1.2, `--tz` since removed):** UTC-pinned run → zone read from
  the manifest (100% agreement); a contradicting `--tz` rejected; older run
  without `--tz` rejected, with `--tz America/Los_Angeles` accepted.
- **Guards:** `--out` at the source, its parent or a subdir → refused; `--force`
  on a non-scrub dir → refused, contents intact; no-match with `--force` on an
  existing output → error, output intact.
- **Input errors:** inline `#` comments parsed; non-numeric code rejected;
  on the LA run, Denver and UTC rejected by the check.
- **Unparseable data:** injected bad medication `START`, bad condition `STOP`,
  and a malformed note header → the patient is excluded, with a warning naming
  the file and value.

v1.3 on `pop1000-seed20260916` (no manifest, UTC data):

- Type 2 diabetes output is byte-identical to v1.2 (cohort) apart from the
  manifest and summary.
- `--mode` and `--tz` are rejected as unknown arguments. A manifest recording
  `America/Los_Angeles` is rejected; a manifest recording `UTC` is accepted. A
  run without a manifest whose note dates were shifted by one day is rejected
  by the data check (3.7% agreement), and nothing is written.
- The v1.2 sweep of all 273 codes (independent checker) found no dates,
  rows or note entries on or after the cutoff for any code. It did find the
  target code in `REASONCODE` before the cutoff for 20 codes, which led to the
  `REASONCODE` cutoff rule; the other leaks it found are listed under Known
  limitations.
- With the `REASONCODE` rule, the sweep was re-run on 73 of the 273 codes
  (stopped early). None had the target code, a date, a row or a note entry on
  or after the cutoff, and no pre-cutoff rows were lost. Those 73 include 6 of
  the 20 codes that leaked in v1.2 (among them Overdose and Complete
  miscarriage). **The remaining 200 codes, including the other 14 that leaked,
  are not yet re-verified.**

To re-verify after changes, re-run the sweep above (reimplement the cutoff
independently; don't reuse `scrub.py`'s code), and use a copy of the source
for any fault injection.

# Implementation Plan: Diagnosis-Based Filtering & Scrubbing

> Status: **implemented** in `scripts/scrub.py` (v1.2). This is the **design
> record**. Sections 1–10 are the original spec as revised for v1.1. The v1.2
> changes in §11 supersede them where they conflict. The authoritative
> description of current behavior is [SCRUBBING.md](SCRUBBING.md).

## 1. Background

This repo generates a reproducible synthetic EHR dataset with Synthea (see
[DATA_GENERATION.md](DATA_GENERATION.md)). That is the *generation* stage. This
plan covers the next stage: **producing a "pre-diagnosis" dataset** for a
clinical-reasoning fine-tuning project.

The model is meant to reason *toward* a diagnosis from the history that preceded
it. So for each patient who develops a target condition, we cut their record at
the moment of diagnosis and keep only what came before. Leaving the diagnosis
(or any post-diagnosis treatment) in would hand the model the answer.

## 2. The core rule

For each patient **who has a target condition**:

1. **Cutoff** = the **earliest** `START` date of any target condition in that
   patient's `conditions.csv` rows. (Per-patient; different patients differ.)
2. **Keep** every record dated **strictly before** the cutoff — across all
   patient-scoped files. Prior meds/labs/etc. stay: that is the history we want.
3. **Drop** every record dated **on or after** the cutoff (`>=`), including the
   diagnosis row itself, plus any meds, labs, procedures, encounters, etc. dated
   on or after it.

The cutoff comes from `conditions.START` specifically — a plain date. All
comparisons are made on **local dates** (§2.1): an event is dropped when its
local date is `>=` the cutoff date, so every same-day event is dropped. This is
intended.

### 2.1 Timezones

Synthea writes datetimes in UTC (`2020-05-05T08:56:25Z`) but writes date-only
fields (`conditions.START`, `allergies.START`, `careplans.START`,
`supplies.DATE`, `patients.DEATHDATE`, note entry headers) as the date **in the
generating JVM's default timezone**, i.e. the generating machine's zone.
`generate_synthea.sh` does not pin a zone. Verified on pop10 (generated on an
`America/Los_Angeles` machine): 405/405 note dates and 269/276 `conditions.START`
dates equal the LA-local date of their encounter; under UTC only 161/276 match.
The remaining 7 are conditions whose onset time differs from the encounter
start, not zone errors.

So a date-only cutoff cannot be compared with a UTC instant as
`00:00Z`: evening events of the previous local day would land after midnight UTC
and be wrongly dropped. Instead:

- `--tz <IANA zone>` names the generating zone (default: this machine's zone,
  from `$TZ` or `/etc/localtime`). Recorded in the manifest.
- Datetime values are converted to a local date in `--tz`; date-only values are
  used as-is. Comparison is `local_date >= cutoff_date`.
- Sanity check: the fraction of `conditions.START` equal to its encounter's
  local `START` date is reported; below 90% → warning that `--tz` is probably
  wrong (pop10: 97% under LA, 58% under UTC).

## 3. Locked design decisions

| # | Decision | Choice |
|---|----------|--------|
| D1 | Match target conditions by CODE or DESCRIPTION? | **CODE only.** Verified `CODE`↔`DESCRIPTION` is 1-to-1 in `conditions.csv` (Synthea assigns one canonical display per code). Re-verify on the full 10k run. |
| D2 | Multiple occurrences of a target condition for one patient | Cut at the **earliest** `START`. |
| D3 | Meds (and other rows) whose `START` is before cutoff but `STOP` is after it | **Keep the row, leave `STOP` as-is.** We never filter on `STOP`; the med may be unrelated to the target condition. |
| D4 | `notes/` files | **Cut by entry date** *(revised)*. A note file is one entry per encounter, each headed by a bare local `YYYY-MM-DD` line. For scrubbed patients keep only entries dated before the cutoff; write no file if none remain. Unaffected emitted patients' files are copied unchanged; non-emitted patients' files are not copied. See §4.1. |
| D5 | `symptoms/` | **Filter `symptoms/csv/symptoms.csv`; do not carry over `symptoms/text/`** *(revised)*. See §4.2. |
| D6 | Output | A **new** standalone dataset dir; source run untouched. The output dir may not be, contain, or sit inside the source run (checked before any `--force` deletion). See §6. |
| D7 | `patients.csv` post-cutoff data | **Remove it** *(revised)*. For scrubbed patients: blank `DEATHDATE` if `>=` cutoff, and blank `HEALTHCARE_EXPENSES` / `HEALTHCARE_COVERAGE` (lifetime totals that include post-cutoff spending; they can't be recomputed as of the cutoff because claims aren't exported). End-of-simulation snapshots (`MARITAL`, `ADDRESS`, `INCOME`, …) are left as-is — they are attributes, not post-cutoff events, and cannot be reconstructed. |
| D8 | `payer_transitions.csv` | **Cut it** by `START_DATE` (non-clinical, but treat as potentially leaky). |
| D9 | Cohort selection | **Two modes, user-selected** (`--mode`). See §5. |

## 4. Per-file handling (verified against a real run)

Every patient-scoped file has exactly one column that decides the temporal cut.
"Cut" = for a scrubbed patient, drop rows where that column's local date (§2.1)
is `>=` the patient's cutoff. We never filter on `STOP`.

| File | Filter column | Format | Action |
|------|---------------|--------|--------|
| `conditions.csv` | `START` | date | **Cutoff source** + cut (drops the diagnosis row itself) |
| `encounters.csv` | `START` | datetime | Cut |
| `medications.csv` | `START` | datetime | Cut (STOP left as-is) |
| `observations.csv` | `DATE` | datetime | Cut |
| `procedures.csv` | `START` | datetime | Cut |
| `allergies.csv` | `START` | date | Cut |
| `careplans.csv` | `START` | date | Cut |
| `devices.csv` | `START` | datetime | Cut |
| `imaging_studies.csv` | `DATE` | datetime | Cut |
| `immunizations.csv` | `DATE` | datetime | Cut |
| `supplies.csv` | `DATE` | date | Cut |
| `payer_transitions.csv` | `START_DATE` | datetime | Cut (D8) |
| `patients.csv` | — | — | Membership depends on mode; post-cutoff fields blanked for scrubbed patients (D7) |
| `organizations.csv` | — (no `PATIENT` col) | — | Global reference → copy unchanged |
| `providers.csv` | — (no `PATIENT` col) | — | Global reference → copy unchanged |

Non-CSV trees:
- `symptoms/csv/symptoms.csv` — filtered per §4.2.
- `symptoms/text/` — **not** carried over (unused downstream; copying it
  unchanged would re-introduce the diagnosis).
- `notes/` — cut per §4.1 (emitted patients only).
- `fhir/`, `metadata/` — **not** part of the reproducible pipeline
  (`generate_synthea.sh` disables FHIR; these appear only because the sample run
  predates that). Do **not** copy them into scrubbed output.

### 4.1 Notes

Each `notes/<Names>_<patientId>.txt` holds one entry per encounter, newest
first. An entry starts at a line that is exactly `YYYY-MM-DD` (the encounter's
local date, §2.1) preceded by a blank line, and runs to the next such line.
Copying files unchanged (the original D4) leaked the diagnosis: on pop10 the
target patient's file had 38 of 42 entries on/after the cutoff, including
*"Patient is presenting with metabolic syndrome x (disorder)."*

For scrubbed patients, keep entries whose date is `<` the cutoff, preserving
their text byte-for-byte and in their original order; text before the first
entry is kept. If no entry remains, no file is written.

### 4.2 Symptoms

`symptoms.csv` columns: `PATIENT, GENDER, RACE, ETHNICITY, AGE_BEGIN, AGE_END,
PATHOLOGY, NUM_SYMPTOMS, SYMPTOMS`. There is no date column, but `AGE_BEGIN`
(patient age in whole years) is a coarse time signal. Rows look like yearly
snapshots of active conditions, e.g. the same chronic `PATHOLOGY` repeated at
every age, and the file appears to cover only roughly the last 10 years of
each life. `PATHOLOGY` is a condition description, so it can name the target
condition directly.

For scrubbed patients, drop a row if **either**:
- `PATHOLOGY` is a target condition's `DESCRIPTION` (from the matched
  `conditions.csv` rows; see D1), or
- `AGE_BEGIN >= ` the patient's age on the cutoff date (from `BIRTHDATE`).

`>=` instead of `>` because a row at the cutoff age may fall after the
diagnosis within that year. The cost is up to one year of pre-diagnosis rows,
which is small: those conditions are still in `conditions.csv` with exact
dates. Unrelated post-cutoff rows are dropped too; this matches the core rule
(§2), which is temporal, not relatedness-based. Later conditions are often
complications of the target, so "unrelated" can't be judged safely. On pop10
the target patient was 50 at cutoff and all of their rows were at ages 56–65,
so all were dropped. In `cohort` mode, non-emitted patients' rows are removed.

## 5. Cohort modes (`--mode`)

- **`cohort`** (default): emit **only** patients who have >=1 target condition,
  each scrubbed at their cutoff.
- **`all`**: emit **every** patient. Target-condition patients are scrubbed;
  all other patients are copied **in full, unscrubbed**.

In both modes, a patient who has a target condition but whose target `START`
dates are all unparseable has no usable cutoff; they are **excluded** (with a
warning) instead of being emitted unscrubbed.

## 6. CLI, config, and output naming

### Input: codes file
Plain `.txt`, one SNOMED code per line. Blank lines and surrounding whitespace
are ignored; `#` begins a comment line. No label, no descriptions.

```
# configs/metabolic.txt
237602007
```

### Usage
```
python3 scripts/scrub.py <source-run-dir> --codes <codes.txt> --mode {cohort|all} \
    [--tz <IANA zone>] [--out <dir>] [--force]
```
- `<source-run-dir>` — positional, e.g. `data/pop10-seed20260916`.
- `--codes` — path to the codes `.txt` (required).
- `--mode` — `cohort` (default) or `all`.
- `--tz` — zone the run was generated in (§2.1); default: this machine's zone.
  Pass it explicitly when scrubbing a run generated on another machine.
- `--out` — optional explicit output dir; overrides auto-naming below.
- `--force` — replace a non-empty output dir. Refused (with or without
  `--force`) if the output dir is, contains, or is inside the source run.

Example:
```
python3 scripts/scrub.py data/pop10-seed20260916 --codes configs/metabolic.txt --mode cohort
```

### Output dir naming
```
data/<source-run>__scrub-<codes-stem>-<sha6>-<mode>/
```
- `<source-run>` = basename of the source run dir.
- `<codes-stem>` = codes filename without extension (human-meaningful label).
- `<sha6>` = first 6 hex chars of `sha1("\n".join(sorted(unique_codes)))`
  — makes the name collision-proof and traceable regardless of filename.
- `<mode>` = `cohort` | `all`.

Example: `data/pop10-seed20260916__scrub-metabolic-a1b2c3-cohort/`

### Output contents
```
<out>/
  csv/…              # filtered CSVs (+ unchanged reference tables)
  symptoms/csv/…     # filtered per §4.2 (no symptoms/text/)
  notes/…            # cut per §4.1 (emitted patients only)
  manifest.json      # scrub provenance (machine-readable)
  SCRUB_SUMMARY.md   # self-contained human summary
```
Do **not** copy the source run's `manifest.json` / `SUMMARY.md` — they describe
the pre-scrub data and would mislead. Both artifacts below are freshly written
by `scrub.py`.

**`manifest.json`** (scrub provenance):
- `source_run` (name) and a nested copy of the source `manifest.json`
- `codes` (sorted unique list) and `codes_file`
- `codes_sha6`
- `mode`
- `timezone` and `timezone_agreement` (§2.1)
- cut rule descriptor (`comparison: ">="`, cutoff, symptoms and patients rules)
- counts: patients in source / matched (affected) / excluded (unparseable) /
  emitted, `DEATHDATE`s blanked, note files written and note entries kept, and
  rows source/written/dropped/unparseable-kept per file (incl. `symptoms.csv`)
- `warnings` (every warning printed during the run)
- `timestamp` (UTC), `scrub_script_version`

**`SCRUB_SUMMARY.md`** (self-contained — do not reuse `summarize_dataset.py`):
- Scrub parameters: source run, codes, mode, cut rule
- Scrub stats: patients matched/emitted, per-file rows kept/dropped, count of
  patients left with an empty pre-diagnosis history
- Descriptive stats of the **resulting** data: patient counts, top conditions,
  file inventory (may share internal helpers with `summarize_dataset.py`, but
  output is one standalone file)

## 7. Open questions / assumptions to confirm

- **1-to-1 CODE↔DESCRIPTION** confirmed only on pop10 (75 conditions). Re-run
  the check on the full 10k dataset. §4.2 relies on it (symptoms are matched
  by `PATHOLOGY` = target `DESCRIPTION`).
- **Zero matches.** If no patient has any target code: warn loudly. In `cohort`
  mode this yields an empty dataset — exit non-zero rather than silently writing
  junk.
- **Unparseable dates**: should be impossible in Synthea output. Every case
  emits a warning (stderr, `manifest.json` `warnings`, `SCRUB_SUMMARY.md`):
  - event-CSV filter column, `symptoms.csv` `AGE_BEGIN`, note entry date,
    `DEATHDATE` → the value/row is **kept** (not silently dropped);
  - target-row `conditions.START` → ignored when computing the cutoff; if the
    patient is left with no cutoff they are excluded (§5).
- **`STOP` after cutoff (D3).** Kept `STOP` values can fall after the cutoff
  (e.g. a med stopped on the diagnosis date). Left as-is per D3; worth
  revisiting now that D7 removes other post-cutoff data.
- **Generator timezone.** `generate_synthea.sh` doesn't pin the JVM timezone,
  so date-only fields depend on the generating machine (§2.1). Pinning it (e.g.
  `-Duser.timezone=UTC`) and recording it in the source manifest would let
  `scrub.py` read it instead of relying on `--tz`.

## 8. Algorithm

1. Parse args; read + clean codes file into a sorted unique code list; compute `sha6`.
2. Resolve `--tz` (§2.1). Resolve output dir (`--out` or auto-name); refuse if
   it overlaps the source run; refuse to overwrite a non-empty existing dir
   unless `--force`.
3. Timezone sanity check (§2.1). Read `csv/conditions.csv`. Find rows whose
   `CODE` is in the target set → group by `PATIENT` → per-patient cutoff =
   min(`START`) as a local date. Build `affected = {patient: cutoff}`; collect
   target `DESCRIPTION`s; patients with no parseable target `START` are
   excluded (§5).
4. Determine emitted patient set: `cohort` → keys of `affected`; `all` → every
   patient in `patients.csv` minus excluded patients.
5. For each patient-scoped CSV in §4:
   - Restrict to emitted patients.
   - For affected patients, drop rows whose `filter_col` local date `>= cutoff`.
   - For unaffected patients (only reachable in `all` mode), keep all rows.
   - Track dropped-row counts.
6. `patients.csv`: restrict to emitted patients; apply D7 to affected ones.
7. Copy `organizations.csv`, `providers.csv` unchanged.
8. Filter `symptoms/csv/symptoms.csv` per §4.2; cut `notes/` per §4.1.
9. Write `manifest.json` and `SCRUB_SUMMARY.md`.

### Determinism
Source CSVs are already canonicalized/sorted (`canonicalize_csvs.py`). Scrubbing
only removes rows and preserves order, so output stays sorted and re-running
produces byte-identical data files. Values are read and written as plain strings
with the stdlib `csv` module (`QUOTE_MINIMAL`, `\n` line endings, same as
`canonicalize_csvs.py`), so nothing is reformatted. Only the `timestamp` in
`manifest.json` and the "Generated" line in `SCRUB_SUMMARY.md` differ between
runs.

## 9. Deliverables

- `scripts/scrub.py` — the scrubber (this spec).
- `configs/metabolic.txt` — example codes file (`237602007`).
- Docs: a short "Scrubbing" section in `README.md`, and either fold this file
  into a user-facing `docs/SCRUBBING.md` or keep it as the design record.

## 10. Verification (use the existing `pop10-seed20260916` run)

Target code `237602007` ("Metabolic syndrome X (disorder)") matches exactly 1
patient in that run — a convenient smoke test.

- **cohort mode:** output `patients.csv` has 1 row; that patient's
  `conditions.csv` has no row with `START >= cutoff` and the target-code row is
  gone; at least the rows dated before the cutoff survive; `notes/` contains only
  that patient's file, holding only entries dated before the cutoff (pop10: 4 of
  42, matching the 4 kept encounters); `symptoms.csv` has none of their rows at
  or above the cutoff age; `DEATHDATE`/expense columns blanked per D7.
- **No leak:** `grep -ri "<target description>"` over the output finds nothing.
- **all mode:** output has 10 patients; the 9 unaffected patients' rows
  (CSVs, `symptoms.csv`) and note files are byte-identical to source; the 1
  affected patient is scrubbed.
- **Cross-check:** for every scrubbed patient, assert no row in any cut file has
  its filter column's local date `>= cutoff` (independent reimplementation).
- **Timezone:** default zone reports ~97% agreement; `--tz UTC` warns.
- **Guard:** `--out` = the source run, its parent, or a subdir of it → refused,
  even with `--force`; source untouched.
- **Unparseable dates:** on a copy with injected bad dates, each case warns
  and is recorded in `manifest.json`; a patient whose only target `START` is bad
  is excluded in `all` mode, and `cohort` mode exits non-zero.
- **Determinism:** run twice → identical output (apart from timestamps).
- **Artifacts:** `manifest.json` counts reconcile with actual dropped rows;
  `SCRUB_SUMMARY.md` renders and is self-contained.

## 11. Revision log

v1.1 — after review of the v1.0 implementation against the original task:

| Change | Reason |
|--------|--------|
| D4: notes cut by entry date (§4.1) | Task says drop notes on/after the diagnosis; v1.0 copied whole files and leaked it. |
| D5: `symptoms.csv` filtered by `PATHOLOGY` + `AGE_BEGIN` (§4.2); `symptoms/text/` not copied | `PATHOLOGY` named the target condition; all of the target patient's rows were post-cutoff. |
| D7 reversed: blank post-cutoff `DEATHDATE` and lifetime cost totals | Remove data from after the condition's START. |
| Local-date comparison with `--tz` (§2.1) | Date-only fields are in the generating machine's zone; comparing them to UTC midnight dropped the previous local evening. |
| Output dir may not overlap the source run | `--force` could `rmtree` the source run or `data/`. |
| Warnings for every unparseable date; exclude patients with no usable cutoff | v1.0 skipped bad target dates silently, which could emit a diagnosed patient unscrubbed in `all` mode. |

v1.2 — after an independent validation of v1.1 (a separate session
reimplemented the cut and scanned for leaks across many target codes):

| Change | Reason |
|--------|--------|
| Cutoff per target row = earlier of `START` and its linked encounter's local date; also drop rows linked to dropped encounters | 7 of 180 patient/code pairs on pop10 have `START` later than the diagnosing visit (4 by one day — the UTC date of an evening visit — 2 by weeks/months). No `--tz` fixes this. v1.1 kept that visit and its note *"Patient is presenting with …"* for 7 codes, with no warning. This replaces the claim in §2.1 that the non-matching rows were harmless. |
| `all` mode: patient lifetime totals blanked for **every** emitted patient | v1.1 blanked them only for scrubbed patients, so a blank marked exactly the diagnosed patients (10/10). |
| D3 revised: end dates (`STOP`, `END_DATE`, symptoms `AGE_END`) ≥ cutoff blanked on kept rows | They revealed the future, e.g. lisinopril `STOP` one day after the cutoff; payer `END_DATE` equal to the diagnosing visit's timestamp. |
| Medication `DISPENSES`/`TOTALCOST` blanked on every emitted row with an empty `STOP` | Lifetime running totals (569 dispenses for a 1979 prescription with a 2010 cutoff). Applied to all patients to avoid a marker. `PAYER_COVERAGE` is per-dispense (never > `BASE_COST`), so it's kept. |
| Unparseable dates now **exclude** the diagnosed patient (previously "keep + warn") | With keep + warn, a bad earliest target `START` wrote the diagnosis row itself to the output. |
| Notes: entry count must equal the patient's encounter count, or the patient is excluded | A malformed header merged an entry into its neighbour and silently dropped pre-cutoff history. |
| `--tz` required (no machine default); check uses note dates (≥ 99.9%) and fails instead of warning | Wrong zones still wrote a different dataset. Denver/Chicago/Anchorage passed the 90% `conditions.START` check; note dates separate them (LA 100%, Denver 99.3%). |
| All validation before any deletion; `--force` only replaces a dir with a scrub `manifest.json` | A no-match run with `--force` deleted an existing dir before failing; `--force` could delete any non-overlapping dir. |
| Generator pins the JVM timezone to UTC and records `timezone` in the run manifest; `scrub.py` reads it, `--tz` only for older runs, a contradicting `--tz` is an error | Synthea's `dateFromTimestamp` uses the JVM default zone (`ExportHelper`), so date-only fields depended on the generating machine. A UTC-pinned pop10 regeneration has identical datetimes and 405/405 note dates matching the encounters' UTC dates. |
| Codes file: trailing `#` comments stripped, non-numeric codes rejected; zero matches fails in both modes | `237602007  # metabolic` was read as one code, matched no one, and `all` mode then emitted everyone unscrubbed. |

v1.3 — after an independent validation of v1.2 on `pop1000-seed20260916`
(all 273 condition codes swept with a separately written checker):

| Change | Reason |
|--------|--------|
| `--mode` removed; only diagnosed patients are emitted (§5 superseded) | `all`-mode negatives ran to the end of the simulation, positives stopped at the diagnosis. For type 2 diabetes, "alive, but no visit in the last year" flagged 74/74 positives and 76/1,056 negatives. Output dir names drop the `-<mode>` suffix. |
| `--tz` removed; only UTC runs accepted (§2.1 superseded). A manifest recording another zone is an error; runs without a recorded zone must pass the UTC data check | The generator pins UTC. pop1000 has no manifest, so v1.2 refused it without `--tz`; its data fits UTC 100%. |
| Cutoff also taken from `encounters`/`medications`/`procedures`/`careplans` rows whose `REASONCODE` is a target code, for patients who already have a target `conditions.csv` row | v1.2 kept the target code in `REASONCODE` before the first condition row for 20 of 273 codes (109 patient/code pairs), e.g. 33 ER visits for Overdose. Adding REASONCODE-only patients (673 pairs over 43 codes) was considered and deferred: weaker evidence, unexplained missing rows, and 2–3× cohort changes. |

Open after v1.3 (see SCRUBBING.md, Known limitations): related diagnoses under
other codes, `symptoms.csv` `AGE_BEGIN` understating onset, and whether
REASONCODE-only patients should count as positives.

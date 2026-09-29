# Serialization: Training Data to Free Text

`serialization/serialize.py` turns a clean training directory (from
[`extract_labels.py`](LABELS.md)) into one free-text summary per patient for an
LLM. `serialization/check_text_leaks.py` then checks that text for anything
that names or hints at the diagnosis.

```
Synthea run ──scrub.py──▶ __scrub-… ──extract_labels.py──▶ __train-<sha6> + labels
            ──serialize.py──▶ <name>.jsonl ──check_text_leaks.py──▶ <name>.leaks.md
```

Python 3.9+, standard library only.

## Quick start: any condition, one command

```bash
serialization/run_condition.sh data/pop10000-seed20260916 configs/heart_failure.txt
# → data/serialized/pop10000-seed20260916-heart_failure.jsonl  (+ .stats.json, .leaks.md)
```

`run_condition.sh` runs four steps: scrub, labels, serialize and the text leak
check. Any extra options go to `serialize.py`. For a new condition, write a
codes file (one SNOMED code per line, as for `scrub.py`) and run the same
command.

Nothing in `serialize.py` or `check_text_leaks.py` is specific to one
condition, because the target comes from each patient's label. Two optional
files per condition hold what is:

| File | Used by | Purpose |
|---|---|---|
| `configs/drop/<stem>.txt` | `serialize.py --drop` | Rows to leave out of the text because their description or reason names or grades the target |
| `configs/leak_terms/<stem>.txt` | `check_text_leaks.py --terms` | Terms to flag for human review |

`<stem>` is the codes file name without `.txt`. Both files hold one term per
line, matched case-insensitively. `#` starts a comment.

For a multi-condition dataset, run each condition and concatenate the JSONL
files. A patient can then appear once per condition, each time with its own
cutoff.

> `extract_labels.py` names the labels file `labels-<sha6>.json` by code set
> only, so a 1k and a 10k run of the same condition overwrite each other.
> `run_condition.sh` passes `--labels data/<run>__labels-<sha6>.json` to keep
> them apart.

## Output

`<name>.jsonl`, one line per patient:

```json
{"patient_id": "…", "cutoff": "2019-12-28",
 "label": {"code": "88805009", "description": "Chronic congestive heart failure (disorder)"},
 "text": "PATIENT\n- 81-year-old male, …", "chars": 17844, "est_tokens": 4461}
```

The label is only in `label`, never in `text`. `<name>.stats.json` records the
settings, the drop terms and token statistics.

Shortened example `text` (10k heart failure patient):

```
PATIENT
- 81-year-old male, white, non-Hispanic, divorced
- Tobacco smoking status: Ex-smoker (finding) (3 months ago)

ACTIVE PROBLEMS
- Essential hypertension (disorder), since 31 years ago
- Ischemic heart disease (disorder), since 21 years ago
- Aortic valve stenosis (disorder), since 3 years ago

CURRENT MEDICATIONS
- Hydrochlorothiazide 25 MG Oral Tablet (started 5 years ago; for Essential hypertension (disorder))
- lisinopril 10 MG Oral Tablet (started 5 years ago; for Essential hypertension (disorder))

VITALS AND LABS (5 years up to the last visit, latest first)
- Systolic Blood Pressure: 112 mm[Hg] (3 months ago); earlier 104 mm[Hg] (1 year ago), 106 mm[Hg] (2 years ago)

RECENT VISITS (5 years up to the last visit, newest first)
- 3 years ago: inpatient visit, Hospital admission (procedure); reason: Aortic valve stenosis (disorder)
```

## What the text contains

Sections, in order (empty sections are omitted):

| Section | Coverage | Compression |
|---|---|---|
| PATIENT | Age at cutoff, sex, race, ethnicity, marital status, latest social items (smoking, employment, education, housing, stress, income) | Latest answer only |
| CHIEF COMPLAINTS AT VISITS | From notes, within the window | One line per complaint with count |
| ACTIVE / PAST PROBLEMS | `(disorder)` conditions, whole history | One line per condition with count and dates |
| OTHER FINDINGS | Other conditions (`(finding)`, `(situation)`), whole history | One line each |
| ALLERGIES, ACTIVE CARE PLANS, DEVICES IN USE | Active at cutoff | — |
| CURRENT / PAST MEDICATIONS | Whole history, with reason | One line per drug with count |
| VITALS AND LABS | Within the window | Latest `--max-values` readings per test |
| PROCEDURES | Within the window | One line per procedure with count |
| IMMUNIZATIONS | Whole history | One line per vaccine with count |
| RECENT VISITS | Within the window | Identical visits collapsed; newest `--max-encounters` |

Always left out:
- **Identifying data:** names, address, SSN and IDs. Note text is used only for chief complaints, because note bodies repeat the patient's name.
- **Synthea's burden scores** (DALY, QALY, QOLS). They're simulation output, not clinical data, and they were among the leak probe's top features in [LEAK_REPORT_10k.md](LEAK_REPORT_10k.md).
- **Yes/no screening questions** (PRAPARE survey items).
- **`symptoms.csv`.** Its age-based rows can pass the cutoff (SCRUBBING.md §Known limitations). Chief complaints from notes carry the same information with dates.

Dates are written relative to the cutoff ("3 years ago"), so the text never
states the diagnosis date.

## Compression options

| Option | Default | Meaning |
|---|---|---|
| `--window-years` | 5 | Window for visits, procedures, vitals/labs, complaints |
| `--window-from` | `last-visit` | Window ends at the patient's last visit, or at `cutoff` |
| `--max-values` | 3 | Readings kept per vital/lab test |
| `--max-encounters` | 15 | Visit lines kept |
| `--min-encounters` | 0 | Skip patients with fewer pre-cutoff visits |
| `--drop` | none | Condition-specific rows to leave out (see above) |
| `--no-compress` | off | Full history, no caps (for comparison) |

The window ends at the last visit because some patients have no visits shortly
before diagnosis: 270 of 848 type 2 diabetes patients (10k) have none in the 5
years before, so a cutoff-anchored window would be empty.

Token counts assume 4 characters per token until a tokenizer is chosen.

## Leak check on the text

`check_text_leaks.py` checks each patient's text against that patient's label.
It exits with code 1 if any check fails.

| Level | Check |
|---|---|
| FAIL | Target code; full description; all distinctive description words on one line; DALY/QALY/QOLS; the patient's name |
| REVIEW | A two-word piece of the description; any term in `--terms` |

REVIEW hits are not failures. Each one needs a decision, recorded below.

## Tests on the output

`test_serialization.py` checks a JSONL file against the data it was built
from. Settings are read from `<file>.stats.json`.

```bash
python3 serialization/test_serialization.py \
    data/serialized/pop10000-seed20260916-heart_failure.jsonl \
    data/pop10000-seed20260916__train-9e8474 \
    data/pop10000-seed20260916__labels-9e8474.json
```

| # | Test | What it catches |
|---|---|---|
| 1 | Same patients, labels match | Lost or duplicated patients; a label or cutoff attached to the wrong patient |
| 2 | No absolute dates, no negative times | A date that would reveal the cutoff; an event after it |
| 3 | Every condition and medication appears | History lost by compression |
| 4 | Latest vital/lab value shown first | Wrong ordering of readings |
| 5 | Drop list honoured | Dropped terms still in the text |
| 6 | Deterministic | Different output on a re-run |
| 7 | Compression never adds items | Compression that makes a record longer |

All 7 tests pass on all 14 datasets run (2026-09-29). Test 4 found a bug in the
first version: readings of one test on the same day (e.g. during a hospital
stay) were ordered by value, not by time, so an earlier reading could be
shown as the latest. Readings are now ordered by full timestamp.

Compression reduces total text by 52% (heart failure, 10k), 24% (type 2
diabetes, 10k) and 15–56% on the pop1000 sets.

## Data check (2026-09-29)

Both runs were generated with `generate_synthea.sh` after the UTC fix
(`manifest.json`: `"timezone": "UTC"`, 2026-09-29, Synthea `d9d07a6`,
reference date 20260921). The generator and `scrub.py` are identical on
`scrub-by-diagnosis` and `labels-and-leak-checks`.

| Run | Patients | Heart failure | Matches LABELS.md / CHF_FINDINGS.md |
|---|---|---|---|
| pop10000 | 11,504 | 313 | Yes (11,504 / 313) |
| pop1000 | 1,131 | 34 | 34 matches; LABELS.md says 1,130 source patients — to confirm |

## Results (2026-09-29)

| Dataset | Patients | Est. tokens median / p90 / max | Text leak check |
|---|---|---|---|
| pop1000 heart failure | 34 | 2,412 / 4,300 / 6,090 | PASS, 3 to review |
| pop10000 heart failure | 313 | 2,430 / 4,300 / 6,218 | PASS, 4 to review |
| pop10000 heart failure, `--no-compress` | 313 | 3,221 / 9,740 / 60,148 | — |
| pop1000 type 2 diabetes | 74 | 304 / 2,424 / 5,474 | PASS, 2 to review |
| pop10000 type 2 diabetes | 848 | 352 / 2,413 / 5,567 | PASS, 2 to review |
| pop1000 essential hypertension | 250 | 264 / 2,003 / 3,826 | PASS |
| pop1000 asthma | 46 | 284 / 2,361 / 3,870 | FAIL: 43 patients keep "Childhood asthma" (see below) |

Before its drop list, pop10000 type 2 diabetes failed: 12 patients kept
"Microalbuminuria/Proteinuria due to type 2 diabetes mellitus".

To check that the pipeline works for conditions it wasn't built around,
`run_condition.sh` was also run with no drop or review lists on the 8 most
common disorders in pop1000: gingivitis, viral sinusitis, gingival disease,
acute viral pharyngitis, primary dental caries, acute bronchitis, anemia and
chronic sinusitis (233–884 patients each).
- **Leak check:** all 8 pass, with 0 items to review.
- **Output tests:** all 8 pass all 7 tests.

**Asthma still fails.** 43 of 46 patients have childhood asthma before their
first "Asthma" row:
- "Childhood asthma" as a condition
- asthma care plans and asthma follow-up visits
- asthma emergency admissions
- inhaler prescriptions whose reason is childhood asthma

A drop list can't fix this. Dropping everything that names asthma would
remove most of these patients' history, and the inhalers would still point
to asthma. Treating childhood asthma as the same condition is a scrub-level
decision (list both codes, which cuts at the childhood diagnosis) for the
team. Until then, asthma should not be used.

## Decisions on flagged items

### Heart failure (pop10000, 313 patients)

The earlier version of this document said the heart-failure-related tests
"are not a leak". That was too broad. Each test was checked on its own:

| Item | Patients before diagnosis | Check | Decision |
|---|---|---|---|
| KCCQ-12 (Kansas City Cardiomyopathy Questionnaire) | 2 | Also in 37 patients who never get heart failure (valve/bypass work-up), so in this data it is not specific. But the model reads the name with medical knowledge: KCCQ exists to grade heart failure. | **Dropped** (`configs/drop/heart_failure.txt`) |
| NYHA functional class, and "Assessment using New York Heart Association Classification" | 2 (Class II, Class IV) | Also in 37 patients who never get heart failure (mostly Class III). As with KCCQ, the name is a heart failure grading scale. | **Dropped** |
| Left ventricular ejection fraction | 11 in text (14 in data; 3 fall outside the window) | Low values are not specific to heart failure: patients who never get it have lower EF (median 39.5%, 54% below 40%) than heart failure patients before diagnosis (median 43.6%, 29% below 40%). All pre-diagnosis values in both groups are 30–50%. | **Kept** |
| NT-proBNP | 12 in text (15 in data) | Before diagnosis, values are 1.1–1.9 pg/mL, the same as patients who never get heart failure (1.0–2.0). All 313 are 238–2,000 after diagnosis, and the scrub removes those. | **Kept** |
| Dyspnea as an allergy reaction (fish, latex, lisinopril) | 5 | Reaction to an allergen, unrelated to heart failure | **Kept** |
| Dyspnea (finding) | 2 | Last noted 4–5 years before diagnosis. Heart failure's own symptoms are recorded at the diagnosis and scrubbed (CHF_FINDINGS.md), so these are earlier, unrelated episodes | **Kept** |
| Macular edema and retinopathy due to type 2 diabetes | 1 | Diabetic eye disease, not heart failure edema | **Kept** |

Where these tests come from: 905 patients in pop10000 have EF, NT-proBNP,
NYHA or KCCQ at some point, and 592 of them never get heart failure. In
Synthea they come from the work-up for aortic valve replacement and bypass
surgery.

Why EF and NT-proBNP are kept but NYHA and KCCQ are dropped: the model sees
EF and NT-proBNP as numbers, and before diagnosis those numbers look the same
in both groups. NYHA and KCCQ carry the answer in their names, whatever the
value. The drop costs little: 2 of 313 patients.

**If matched negatives are added later,** apply the same drop list to them.
Otherwise NYHA/KCCQ would appear only in negatives, and their presence would
become a shortcut to "not heart failure".

### Type 2 diabetes (pop10000, 848 patients)

| Item | Patients | Decision |
|---|---|---|
| Microalbuminuria / Proteinuria due to type 2 diabetes mellitus | 12 | **Dropped** (`configs/drop/diabetes_t2.txt`). Names the target. |
| Disorder of kidney due to diabetes mellitus | 14 | **Dropped.** Implies established diabetes. |
| Diabetes self management plan (for Prediabetes) | 612 | **Kept.** Prediabetes is a real, earlier risk factor, not the diagnosis. |
| Urine screening test for diabetes | 27 | **Kept.** Routine screening; it names what is looked for, not a result (like colon cancer screening in SCRUBBING.md). |
| Diabetes mellitus due to cystic fibrosis | 3 | **Kept.** A different disease (CF-related diabetes). |

The dropped complications are a symptom of the scrub limitation in
SCRUBBING.md: the diabetes complications are coded before the type 2 diabetes
row. Dropping them from the text removes the naming. The underlying labs
(e.g. urine albumin) stay. Whether the cutoff should move to the first
complication is a scrub-level decision for the team.

## Other findings

1. **Type 2 diabetes histories are thin.** The median is 8 pre-diagnosis visits (heart failure: 36), and 198 of 848 patients have fewer than 5. `--min-encounters 5` would drop them.
2. **The presenting symptoms are missing.** As CHF_FINDINGS.md notes, CHF symptoms are scrubbed with the diagnosis. Chief complaints in the text come from earlier, unrelated visits.

## Not done yet

- **No negatives.** Every patient is a positive, as in the scrub. Matched negatives need a cutoff rule first (SCRUBBING.md D9).
- **Tokenizer.** Token counts are estimates until the model is chosen.
- **Probe on text.** `check_leaks.py`'s classifier probe runs on the CSVs. The same probe on serialized text would also cover notes.
- **Other conditions** should get a review list, and a drop list if the leak check flags names; asthma needs a scrub-level decision first.

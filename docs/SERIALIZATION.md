# Serialization: Patient Records to LLM Text

This step turns each patient's scrubbed record into one readable text summary
that an LLM can take as input. The diagnosis is kept in a separate field, never
in the text.

```
scrub.py ──▶ extract_labels.py ──▶ serialize.py ──▶ check_text_leaks.py ──▶ test_serialization.py
(cut at dx)   (labels + clean dir)   (text summary)   (leak check on text)   (output correct?)
```

Python 3.9+, standard library only.

## 1. How to run

One command runs the whole pipeline for one condition:

```bash
serialization/run_condition.sh data/pop10000-seed20260916 configs/heart_failure.txt
```

Output goes to `data/serialized/<run>-<condition>.jsonl`, with a
`.leaks.md` leak report and a `.stats.json` of settings and lengths.

**New condition:** write a codes file with one SNOMED code per line, e.g.
`configs/asthma.txt` containing `195967001   # Asthma (disorder)`. Then run the
same command. The code is not specific to any condition.

**Which conditions:** The seven target conditions are fixed. See
[`docs/CONDITION_SELECTION.md`](CONDITION_SELECTION.md) for the selection
rationale. The codes files are in `configs/bins/`.

**Optional per-condition files.** Most conditions don't need them.

| File | What it does | When to add one |
|---|---|---|
| `configs/drop/<condition>.txt` | Leaves out rows whose name gives the diagnosis away | The leak check reports FAIL, or a REVIEW item that names the target |
| `configs/leak_terms/<condition>.txt` | Extra words the leak check flags for review | You know hints the check wouldn't find by itself (tests, drugs, symptoms) |
| `data/presenting/<condition>.json` | Presenting symptoms from `extract_presenting.py` | Use with `--presenting` to add the PRESENTING COMPLAINT section |

**Presenting complaint flag.** Pass `--presenting data/presenting/<condition>.json`
(output of `scripts/extract_presenting.py`) to add a `PRESENTING COMPLAINT` section.
The section contains symptom names and severities from `symptoms.csv`, followed by a
`Vital signs at this visit:` line listing whitelisted measurements (body temperature,
BP, heart rate, respiratory rate, SpO2) from the diagnosing encounter. Records with
neither symptoms nor whitelisted vitals are written without the section. The flag is
intentionally separate from the core pipeline so that history and presentation can be
ablated independently.

**No-history flag (optional).** Pass `--no-history` together with `--presenting` to
write only the demographics line and the `PRESENTING COMPLAINT` section, for the
"remove history" ablation in [CONDITION_SELECTION.md](CONDITION_SELECTION.md). It is
off by default.

| Variant | Options | Contains |
|---|---|---|
| History only | (none) | Pre-diagnosis record |
| History + presenting | `--presenting <file>` | Pre-diagnosis record + presenting complaint |
| Presenting only | `--presenting <file> --no-history` | Demographics + presenting complaint |

`.stats.json` records `presenting` and `no_history`, and `test_serialization.py`
re-runs with the same options. Tests 3 and 4 compare the text with the history, so
they are skipped for `--no-history` files.

When running `scripts/extract_presenting.py` on a test run, pass `--report` with a
path outside `docs/`. Its default report path overwrites the committed
`docs/PRESENTING_<condition>.md`.

To check the output:

```bash
python3 serialization/test_serialization.py data/serialized/<run>-<condition>.jsonl \
    data/<run>__train-<sha6> data/<run>__labels-<sha6>.json
```

> `extract_labels.py` names its labels file by condition only, so 1k and 10k
> runs overwrite each other. `run_condition.sh` works around this by writing
> `data/<run>__labels-<sha6>.json`.

## 2. What the output looks like

Each run writes three files to `data/serialized/`:

| File | Contents |
|---|---|
| `<run>-<condition>.jsonl` | One line per patient: the text for the model, and the answer |
| `<run>-<condition>.leaks.md` | Leak check report |
| `<run>-<condition>.stats.json` | Settings used and text lengths |

One line of the `.jsonl` file (the text is shortened here and shown in full
below):

```json
{"patient_id": "dfeafd76-a0d6-326d-3640-5e71d9b8df65",
 "cutoff": "1999-07-27",
 "label": {"code": "88805009", "description": "Chronic congestive heart failure (disorder)"},
 "text": "PATIENT\n- 55-year-old male, black, Hispanic, single\n…",
 "chars": 6221,
 "est_tokens": 1555}
```

| Field | Meaning |
|---|---|
| `patient_id` | Synthea patient ID |
| `cutoff` | Diagnosis date. Everything in `text` is before it. Not given to the model. |
| `label` | The answer: the target condition. Not given to the model. |
| `text` | What the model reads |
| `chars`, `est_tokens` | Length of `text` (tokens estimated at 4 characters each) |

The full `text` of this record, unedited (10k heart failure run, 1,555
tokens; the median record is 2,430). The patient is diagnosed with heart failure
at the cutoff. The text doesn't say so.

```
PATIENT
- 55-year-old male, black, Hispanic, single
- Employment status - current: Part-time or temporary work (7 months ago)
- Highest level of education: High school diploma or GED (7 months ago)
- Housing status: I have housing (7 months ago)
- How many people are living or staying at this address [#]: 5 (7 months ago)
- Stress level: Somewhat (7 months ago)
- Tobacco smoking status: Never smoked tobacco (finding) (7 months ago)
- What was your best estimate of the total income of all family members from all sources  before taxes  in last year [PhenX]: 57649 /a (7 months ago)

ACTIVE PROBLEMS
- Essential hypertension (disorder), since 9 years ago
- Sleep disorder (disorder), since 8 years ago
- Obstructive sleep apnea syndrome (disorder), since 8 years ago
- Ischemic heart disease (disorder), since 5 years ago
- Sepsis (disorder), since 1 year ago

PAST PROBLEMS
- Gingivitis (disorder) (7 months ago)

OTHER FINDINGS
- Part-time employment (finding) (last noted 7 months ago)
- Medication review due (situation) (last noted 7 months ago)
- Full-time employment (finding) (last noted 2 years ago)
- Stress (finding) (last noted 3 years ago)
- Not in labor force (finding) (last noted 4 years ago)
- Limited social contact (finding) (last noted 4 years ago)
- History of coronary artery bypass grafting (situation) (last noted 5 years ago)
- Abnormal findings diagnostic imaging heart+coronary circulat (finding) (last noted 5 years ago)
- Refugee (person) (last noted 18 years ago)
- Educated to high school level (finding) (last noted 37 years ago)

CURRENT MEDICATIONS
- 24 HR metoprolol succinate 100 MG Extended Release Oral Tablet (started 5 years ago)
- Nitroglycerin 0.4 MG/ACTUAT Mucosal Spray (started 5 years ago)
- Simvastatin 20 MG Oral Tablet (started 5 years ago)
- Hydrochlorothiazide 25 MG Oral Tablet (started 4 years ago; for Essential hypertension (disorder))
- lisinopril 10 MG Oral Tablet (started 4 years ago; for Essential hypertension (disorder))

PAST MEDICATIONS
- piperacillin 2000 MG / tazobactam 250 MG Injection (1 year ago; for Sepsis (disorder))
- 150 ML vancomycin 5 MG/ML Injection (1 year ago; for Sepsis (disorder))

ACTIVE CARE PLANS
- Lifestyle education regarding hypertension (procedure) (for Essential hypertension (disorder))
- Care plan (record artifact) (for Obstructive sleep apnea syndrome (disorder))

DEVICES IN USE
- Home continuous positive airway pressure unit (physical object)
- Respiratory humidifier (physical object)

VITALS AND LABS (5 years up to the last visit, latest first)
- Body Height: 191.4 cm (7 months ago); earlier 191.4 cm (1 year ago), 191.4 cm (2 years ago)
- Body Weight: 101.6 kg (7 months ago); earlier 101.6 kg (1 year ago), 101.6 kg (2 years ago)
- Body mass index (BMI) [Ratio]: 27.7 kg/m2 (7 months ago); earlier 27.7 kg/m2 (1 year ago), 27.7 kg/m2 (2 years ago)
- Capillary refill [Time] of Nail bed: Increased capillary filling time (finding) (1 year ago)
- Cholesterol [Mass/volume] in Serum or Plasma: 222.7 mg/dL (7 months ago); earlier 191.3 mg/dL (3 years ago)
- Cholesterol in HDL [Mass/volume] in Serum or Plasma: 31.2 mg/dL (7 months ago); earlier 33.3 mg/dL (3 years ago)
- Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay: 164 mg/dL (7 months ago); earlier 133.8 mg/dL (3 years ago)
- Diastolic Blood Pressure: 69 mm[Hg] (7 months ago); earlier 69 mm[Hg] (1 year ago), 64 mm[Hg] (2 years ago)
- Generalized anxiety disorder 7 item (GAD-7) total score [Reported.PHQ]: 1 (1 year ago); earlier 2 (3 years ago)
- Gram positive blood culture panel by Probe in Positive blood culture: Positive (qualifier value) (1 year ago)
- Heart rate: 78 /min (7 months ago); earlier 83 /min (1 year ago), 71 /min (2 years ago)
- Lactate [Moles/volume] in Blood: 2.7 mmol/L (1 year ago)
- Mean blood pressure: 96.5 mm[Hg] (1 year ago)
- Oxygen saturation in Arterial blood: 90 % (1 year ago)
- Pain severity - 0-10 verbal numeric rating [Score] - Reported: 3 (7 months ago); earlier 2 (1 year ago), 3 (2 years ago)
- Patient Health Questionnaire 2 item (PHQ-2) total score [Reported]: 0 (7 months ago); earlier 2 (1 year ago), 1 (2 years ago)
- Respiratory rate: 12 /min (7 months ago); earlier 13 /min (1 year ago), 13 /min (2 years ago)
- Systolic Blood Pressure: 118 mm[Hg] (7 months ago); earlier 113 mm[Hg] (1 year ago), 121 mm[Hg] (2 years ago)
- Total score [AUDIT-C]: 2 (7 months ago); earlier 1 (3 years ago)
- Total score [DAST-10]: 2 (1 year ago)
- Triglyceride [Mass/volume] in Serum or Plasma: 137.8 mg/dL (7 months ago); earlier 121 mg/dL (3 years ago)

PROCEDURES (5 years up to the last visit)
- Removal of supragingival plaque and calculus from all teeth using dental instrument (procedure) (6 months ago)
- Removal of subgingival plaque and calculus from all teeth using dental instrument (procedure) (6 months ago)
- Oral health education (procedure) (6 months ago)
- Examination of gingivae (procedure) (6 months ago)
- Dental care (regime/therapy) (6 months ago)
- Dental consultation and report (procedure) (6 months ago)
- Patient referral for dental care (procedure) (7 months ago)
- Depression screening (procedure) (8 times, last 7 months ago)
- Assessment using Alcohol Use Disorders Identification Test - Consumption (procedure) (2 times, last 7 months ago)
- Assessment of substance use (procedure) (3 times, last 7 months ago)
- Sleep apnea assessment (procedure) (4 times, last 7 months ago)
- Assessment of health and social care needs (procedure) (4 times, last 7 months ago)
- Transfer to stepdown unit (procedure) (1 year ago)
- Resuscitation using intravenous fluid (procedure) (1 year ago)
- Screening for drug abuse (procedure) (1 year ago)
- Assessment of anxiety (procedure) (2 times, last 1 year ago)
- Medication reconciliation (procedure) (1 year ago)

IMMUNIZATIONS
- Influenza  split virus  trivalent  PF (4 times, last 7 months ago)

RECENT VISITS (5 years up to the last visit, newest first)
- 6 months ago: ambulatory visit, Encounter for check up (procedure); reason: Gingivitis (disorder)
- 7 months ago: wellness visit, General examination of patient (procedure) (5 visits)
- 1 year ago: emergency visit, Encounter for problem (procedure); reason: Sepsis (disorder)
```

The `.stats.json` for the same run:

```json
{
  "train_dir": "pop10000-seed20260916__train-9e8474",
  "patients_written": 313,
  "settings": {
    "window_years": 5,
    "window_from": "last-visit",
    "max_values": 3,
    "max_encounters": 15,
    "min_encounters": 0,
    "no_compress": false,
    "drop": "configs/drop/heart_failure.txt",
    "drop_terms": [
      "nyha",
      "new york heart association",
      "kccq",
      "kansas city"
    ]
  },
  "est_tokens": {
    "min": 221,
    "median": 2430,
    "p90": 4300,
    "max": 6218
  }
}
```

## 3. How the text is built

**Sections:** patient and social background, chief complaints from notes,
active and past problems, other findings, allergies, medications, care plans,
devices, vitals and labs, procedures, immunizations and recent visits.

**Compression** keeps the text short. Every setting is an option of
`serialize.py`:

| Rule | Default |
|---|---|
| Repeats collapsed into one line with a count | always |
| Visits, procedures, vitals/labs and complaints limited to a window ending at the last visit | `--window-years 5` |
| Readings kept per vital/lab test | `--max-values 3` |
| Visit lines kept | `--max-encounters 15` |

The window ends at the last visit, not the diagnosis, because some patients
have long gaps before diagnosis (270 of 848 type 2 diabetes patients had no
visit in the 5 years before). `--no-compress` turns compression off for
comparison.

**Always left out:**
- **Names, addresses and IDs.**
- **Synthea's burden scores (DALY, QALY, QOLS).** They're simulation output, and the leak probe's top features.
- **Yes/no survey questions.**
- **`symptoms.csv`.** Its age-based rows can pass the cutoff.

**Dates** are written as "3 years ago", so the diagnosis date never appears.

## 4. How it is checked

**Leak check** (`check_text_leaks.py`) reads the final text, notes included:

| Result | Triggered by |
|---|---|
| FAIL | The target's code or name; all its words on one line; DALY/QALY/QOLS; the patient's name |
| REVIEW | Part of the target's name; a word from `leak_terms` |

**Output tests** (`test_serialization.py`):

| # | Test |
|---|---|
| 1 | Every patient appears once, with the right label |
| 2 | No real dates, no negative times |
| 3 | Every condition and medication from the data appears |
| 4 | The latest lab value is shown first |
| 5 | Dropped terms are gone |
| 6 | Running twice gives identical output |
| 7 | Compression never adds content |

Test 4 found a bug, now fixed: same-day readings were ordered by value
instead of time.

## 5. Results (2026-09-29)

Data: pop1000 and pop10000, regenerated with the UTC fix. The 10k run matches
CHF_FINDINGS.md (11,504 patients, 313 heart failure).

| Condition | Patients | Typical length (tokens) | Leak check | Tests |
|---|---|---|---|---|
| Heart failure, 10k | 313 | 2,430 (max 6,218) | ✅ PASS | ✅ 7/7 |
| Type 2 diabetes, 10k | 848 | 352 (max 5,567) | ✅ PASS | ✅ 7/7 |
| Heart failure, 1k | 34 | 2,412 | ✅ PASS | ✅ 7/7 |
| Type 2 diabetes, 1k | 74 | 304 | ✅ PASS | ✅ 7/7 |
| Hypertension, 1k | 250 | 264 | ✅ PASS | ✅ 7/7 |
| 8 most common disorders, 1k, no config (gingivitis, sinusitis, anemia, …) | 233–884 each | — | ✅ PASS | ✅ 7/7 |
| Asthma, 1k | 46 | 284 | ❌ FAIL | ✅ 7/7 |

Compression shortens heart failure text by 52% overall. Without it, the
longest record is 60,148 tokens instead of 6,218.

## 6. Decisions per condition

**Heart failure** (`configs/drop/heart_failure.txt`)

| Item | Patients before diagnosis | Decision | Why |
|---|---|---|---|
| NYHA class, KCCQ questionnaire | 2 | Dropped | Their names are heart failure grading scales, so the model would read the answer from the name. 37 patients who never get heart failure also have them. |
| Ejection fraction | 11 | Kept | Low values are not specific: patients who never get heart failure have lower values (median 39.5% vs 43.6%). |
| NT-proBNP | 12 | Kept | Before diagnosis, values are normal in both groups (1–2 pg/mL). They are high only after diagnosis, and the scrub removes those. |
| Dyspnea as an allergy reaction, earlier dyspnea episodes, diabetic macular edema | 8 | Kept | Unrelated to heart failure |

These tests come from valve and bypass surgery work-ups, which 592 patients
who never get heart failure also have.

**Type 2 diabetes** (`configs/drop/diabetes_t2.txt`)

| Item | Patients | Decision | Why |
|---|---|---|---|
| "Microalbuminuria/Proteinuria due to type 2 diabetes", "Disorder of kidney due to diabetes" | 26 | Dropped | Names the diagnosis |
| Prediabetes care plan | 612 | Kept | A real earlier risk factor |
| Urine screening for diabetes | 27 | Kept | Routine screening |
| Diabetes due to cystic fibrosis | 3 | Kept | A different disease |

**Asthma: not usable yet.** 43 of 46 patients had childhood asthma first:
the condition itself, asthma care plans, inhalers and follow-up visits. A drop
list can't fix this without deleting most of their history.

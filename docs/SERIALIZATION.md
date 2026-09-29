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

**Optional per-condition files.** Most conditions don't need them.

| File | What it does | When to add one |
|---|---|---|
| `configs/drop/<condition>.txt` | Leaves out rows whose name gives the diagnosis away | The leak check reports FAIL, or a REVIEW item that names the target |
| `configs/leak_terms/<condition>.txt` | Extra words the leak check flags for review | You know hints the check wouldn't find by itself (tests, drugs, symptoms) |

To check the output:

```bash
python3 serialization/test_serialization.py data/serialized/<run>-<condition>.jsonl \
    data/<run>__train-<sha6> data/<run>__labels-<sha6>.json
```

## 2. What the output looks like

One JSON line per patient:

```json
{"patient_id": "…", "cutoff": "2019-12-28",
 "label": {"code": "88805009", "description": "Chronic congestive heart failure (disorder)"},
 "text": "PATIENT\n- 81-year-old male, …", "est_tokens": 4461}
```

The model reads `text`. `label` is the answer. Shortened example `text`:

```
PATIENT
- 81-year-old male, white, non-Hispanic, divorced

ACTIVE PROBLEMS
- Essential hypertension (disorder), since 31 years ago
- Aortic valve stenosis (disorder), since 3 years ago

CURRENT MEDICATIONS
- lisinopril 10 MG Oral Tablet (started 5 years ago; for Essential hypertension (disorder))

VITALS AND LABS (5 years up to the last visit, latest first)
- Systolic Blood Pressure: 112 mm[Hg] (3 months ago); earlier 104 mm[Hg] (1 year ago)
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

## 7. Open questions for the team

1. **Asthma:** treat childhood asthma as the same condition (cut earlier in `scrub.py`)?
2. **Diabetes complications:** should the cutoff move to the first diabetes complication instead of dropping the rows?
3. **Negatives:** every patient is a positive. If matched negatives are added, apply the same drop lists to them. Otherwise, for example, NYHA would appear only in negatives and become a shortcut.
4. **Short histories:** 198 of 848 diabetes patients have fewer than 5 visits. Filter them with `--min-encounters 5`?
5. **Tokenizer:** token counts assume 4 characters per token until a model is chosen.
6. **pop1000 count:** local run has 1,131 patients; LABELS.md says 1,130. The heart failure count (34) matches.

> `extract_labels.py` names its labels file by condition only, so 1k and 10k
> runs overwrite each other. `run_condition.sh` works around this by writing
> `data/<run>__labels-<sha6>.json`.

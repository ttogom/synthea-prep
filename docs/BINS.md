# Condition Bins and Combined-Dataset Checks

Bin definitions are in `configs/bins/` (see [`configs/bins/README.md`](../configs/bins/README.md)). The antecedent screen and symptom availability results that motivate the grouping are in [`docs/CONDITION_SCREEN.md`](CONDITION_SCREEN.md).

## Presenting evidence coverage

Coverage after adding whitelisted vitals (body temperature, systolic/diastolic BP, heart rate, respiratory rate, SpO2) from the diagnosing encounter:

| Condition | Positives | With symptoms | With vitals | With any evidence |
|-----------|----------:|-------------:|------------:|------------------:|
| COPD | 384 | 141 | 139 | 144 (38%) |
| Hypertension | 2650 | 0 | 757 | 757 (29%) |
| Strep throat | 1499 | 1486 | 1493 | 1493 (100%) |
| Viral pharyngitis | 5108 | 5076 | 5088 | 5089 (100%) |
| Bacterial sinusitis | 656 | 600 | 0 | 600 (91%) |
| Cystitis | 1160 | 856 | 277 | 912 (79%) |
| CHF | 313 | 311 | 0 | 311 (99%) |

> Hypertension now has presenting evidence (elevated BP) for 29% of positives. The remaining 71% had their diagnosis recorded at an encounter with no standard vital signs in Synthea (e.g. specialist referral encounters). Bacterial sinusitis and CHF have no whitelisted vitals at their diagnosing encounters in this run.

**Hypertension BP at diagnosing encounter** (n=757): systolic median 149 mmHg (81% ≥140); diastolic median 104 mmHg (89% ≥90). Readings are genuinely elevated — this is a real diagnostic signal, not noise.

**Observation whitelist decisions** — every observation class at each diagnosing encounter was reviewed. Excluded even when recorded in the `vital-signs` FHIR category:

| Code | Description | Decision | Reason |
|------|-------------|----------|--------|
| 19926-5 | FEV1/FVC (spirometry ratio) | **EXCLUDE** | < 0.70 is the diagnostic criterion for COPD; including it names the answer |
| 88020-3 | NYHA Functional Capacity | **EXCLUDE** | Directly classifies CHF severity; present only at CHF encounters |
| 88021-1 | NYHA Objective Assessment | **EXCLUDE** | Same |
| 33762-6 | NT-proBNP | **EXCLUDE** | Diagnostic marker for CHF; elevated values would name the condition |
| 89579-7 | Troponin I (high sensitivity) | **EXCLUDE** | Cardiac injury marker at CHF encounters; present only for CHF |
| 29463-7 | Body Weight | **EXCLUDE** | Not on six-item whitelist |
| 39156-5 | BMI | **EXCLUDE** | Not on six-item whitelist; derived from weight and height |
| 72514-3 | Pain severity (0–10) | **EXCLUDE** | A numeric rating scale, not a physiological measurement |
| 8302-2 | Body Height | **EXCLUDE** | Not on six-item whitelist |

## Bin summary

| Bin | Conditions | Primary signal | Notes |
|-----|-----------|----------------|-------|
| history | COPD, Hypertension | Pre-diagnosis antecedents | COPD: smoking_ever z=+7.8; HTN: sleep disorder z=+14.9, diabetes z=+11.5 |
| acute | Strep, Viral pharyngitis, Bacterial sinusitis, Cystitis | Presenting symptoms (91–99% coverage) | No non-artifact history signal |
| control | CHF | Demographic baseline | 99% symptom coverage; no clean history signal; prior case-study condition |

## d. Class balance

**Total positives across 7 conditions: 11770**

| Bin | Condition | n | % of dataset |
|-----|-----------|--:|-------------:|
| history | Hypertension | 2650 | 22.5% |
| history | COPD | 384 | 3.3% |
| acute | Viral pharyngitis | 5108 | 43.4% |
| acute | Strep throat | 1499 | 12.7% |
| acute | Cystitis | 1160 | 9.9% |
| acute | Bacterial sinusitis | 656 | 5.6% |
| control | CHF | 313 | 2.7% |

> Viral pharyngitis alone accounts for 43% of the dataset. Hypertension accounts for 23%. Class weights are needed for any multi-class model.

## a. Symptoms-plus-vitals multi-class classifier

**Features:** bag of symptom names + whitelisted vital sign values (z-scored over patients who have each vital). **Model:** multinomial logistic regression (one-vs-rest), class_weight=balanced, 5-fold stratified CV.

**Overall accuracy:** 83.9%  **Balanced accuracy:** 79.5%

**Confusion matrix** (rows = true, cols = predicted):  

| True \ Pred | `Bacterial si` | `CHF` | `COPD` | `Cystitis` | `Hypertension` | `Strep throat` | `Viral pharyn` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `Bacterial si` | 600 | 0 | 0 | 0 | 56 | 0 | 0 |
| `CHF` | 0 | 311 | 0 | 0 | 2 | 0 | 0 |
| `COPD` | 0 | 0 | 141 | 2 | 241 | 0 | 0 |
| `Cystitis` | 0 | 0 | 10 | 892 | 257 | 1 | 0 |
| `Hypertension` | 0 | 0 | 82 | 9 | 2559 | 0 | 0 |
| `Strep throat` | 0 | 0 | 0 | 0 | 12 | 1061 | 426 |
| `Viral pharyn` | 0 | 0 | 0 | 0 | 32 | 767 | 4309 |

**Strep vs viral pharyngitis:**
- Strep correctly predicted: 1061/1499 (70.8%);  predicted as viral: 426 (28.4%)
- Viral correctly predicted: 4309/5108 (84.4%);  predicted as strep: 767 (15.0%)
- **Features more predictive of strep than viral:** `Body temperature`, `Systolic Blood Pressure`, `Diastolic Blood Pressure`, `Heart rate`, `Headache`
- **Features more predictive of viral than strep:** `Respiratory rate`, `Body Aches`, `Difficulty Swallowing`, `Decreased Appetite`, `Runny/Stuffy Nose`

**Top discriminating features per condition:**

- **Bacterial sinusitis:** `Pain with Bright Lights`, `Headache`, `Sinus Pain`, `Nasal Congestion`, `Nasal Discharge`
- **CHF:** `Rales (finding)`, `Edema (finding)`, `Paroxysmal dyspnea (finding)`, `Dyspnea (finding)`, `Orthopnea (finding)`
- **COPD:** `Shortness of Breath`, `Cough`, `Respiratory rate`, `Oxygen saturation`, `Heart rate`
- **Cystitis:** `Urgent desire to urinate`, `Dysuria`, `Suprapubic pain`, `Increased frequency of urination`, `Heart rate`
- **Hypertension:** `Systolic Blood Pressure`, `Diastolic Blood Pressure`, `Respiratory rate`, `Oxygen saturation`, `Heart rate`
- **Strep throat:** `Body temperature`, `Decreased Appetite`, `Runny/Stuffy Nose`, `Difficulty Swallowing`, `Fatigue`
- **Viral pharyngitis:** `Runny/Stuffy Nose`, `Decreased Appetite`, `Body Aches`, `Difficulty Swallowing`, `Fatigue`

## b. Shortcut probe

**Features:** age at diagnosis, sex (binary), number of pre-cutoff encounters, years of recorded history (earliest encounter to cutoff). **Intent:** record shape alone — not clinical content.

**Overall accuracy:** 30.0%  **Balanced accuracy:** 33.6%  **Chance level:** 14.3% (1/7 classes)

**Feature importance** (mean |coef| across all condition classifiers):

- `n_enc`: 1.272
- `age`: 0.825
- `years_hist`: 0.377
- `sex_male`: 0.074

> **Record shape is 2.4× chance (33.6% vs 14.3% baseline).** Conditions differ in age profile, sex distribution, and encounter density. Any symptom/vital-based model must beat this floor, not random chance.

## c. Presence probe

**Feature:** `has_evidence` — binary flag for whether the patient has any presenting evidence (symptoms from `symptoms.csv` or whitelisted vitals from the diagnosing encounter).

**Overall accuracy:** 39.0%  **Balanced accuracy:** 24.4%  **Chance level:** 14.3%

**Per-condition absence rate (no evidence at all):**

| Condition | n | No evidence | % absent | Interpretation |
|-----------|--:|------------:|--------:|----------------|
| Hypertension | 2650 | 1893 | 71.4% | **Strong absence signal** |
| COPD | 384 | 240 | 62.5% | Partial absence |
| Cystitis | 1160 | 248 | 21.4% | Partial absence |
| Bacterial sinusitis | 656 | 56 | 8.5% | Mostly present |
| CHF | 313 | 2 | 0.6% | Mostly present |
| Strep throat | 1499 | 6 | 0.4% | Mostly present |
| Viral pharyngitis | 5108 | 19 | 0.4% | Mostly present |

> **Hypertension absence rate is now 71.4%** (down from 100% when only symptoms were used). The 71% without evidence were likely diagnosed at specialist encounters that Synthea does not record a vital signs panel for. Presence alone no longer perfectly identifies hypertension, but remains a partial signal.

## Open design questions

These are unresolved decisions the team needs to make before training:

1. ~~**Strep vs viral pharyngitis separability.**~~ **Resolved.** Strep recall is 71% once body temperature is included. Body temperature cleanly separates the two: strep patients present with fever (median 38.3°C, 67% ≥38°C); viral pharyngitis does not (median 37.5°C, 0% ≥38.5°C). Keep them as separate classes.

2. **Hypertension partial absence.** 71.4% of hypertension patients still have no presenting evidence. A multi-task model that observes whether a presenting section exists will still find partial signal for hypertension. Decide whether to ablate the presence flag or accept this as a known confound.

3. **Demographic confounding.** The shortcut probe shows record shape alone is discriminative (2.3× chance). Age, sex, and encounter density differ systematically across conditions. Any evaluation of symptom/vital-based models should be compared against this floor, not against random chance.

4. **Class imbalance.** Viral pharyngitis (43%) and hypertension (23%) dominate the dataset. Confirm class-weighting strategy before training.

5. **COPD episode coverage (37%).** Most COPD patients lack presenting evidence — Synthea's symptom coverage is sparse for chronic conditions, and the diagnosing encounter only records BP/HR (not pulmonary-specific vitals beyond the excluded FEV1/FVC). Decide whether to train COPD with history only, or exclude it from symptom-based evaluations.


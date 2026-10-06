# Condition Bins and Combined-Dataset Checks

Bin definitions are in `configs/bins/` (see [`configs/bins/README.md`](../configs/bins/README.md)). The antecedent screen and symptom availability results that motivate the grouping are in [`docs/CONDITION_SCREEN.md`](CONDITION_SCREEN.md).

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

> Viral pharyngitis alone accounts for 45% of the dataset. Hypertension accounts for 23%. Class weights are needed for any multi-class model.

## a. Symptoms-only multi-class classifier

**Features:** bag of symptom names from `data/presenting/`. **Model:** multinomial logistic regression (one-vs-rest), class_weight=balanced, 5-fold stratified CV.

**Overall accuracy:** 81.8%  **Balanced accuracy:** 71.5%

**Confusion matrix** (rows = true, cols = predicted):  

| True \ Pred | `Bacterial si` | `CHF` | `COPD` | `Cystitis` | `Hypertension` | `Strep throat` | `Viral pharyn` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `Bacterial si` | 600 | 0 | 0 | 0 | 56 | 0 | 0 |
| `CHF` | 0 | 311 | 0 | 0 | 2 | 0 | 0 |
| `COPD` | 0 | 0 | 141 | 0 | 243 | 0 | 0 |
| `Cystitis` | 0 | 0 | 0 | 856 | 304 | 0 | 0 |
| `Hypertension` | 0 | 0 | 0 | 0 | 2650 | 0 | 0 |
| `Strep throat` | 0 | 0 | 0 | 0 | 13 | 0 | 1486 |
| `Viral pharyn` | 0 | 0 | 0 | 0 | 32 | 0 | 5076 |

**Strep vs viral pharyngitis:**
- Strep correctly predicted: 0/1499 (0.0%);  predicted as viral: 1486 (99.1%)
- Viral correctly predicted: 5076/5108 (99.4%);  predicted as strep: 0 (0.0%)

**Top discriminating symptoms per condition:**

- **Bacterial sinusitis:** `Sinus Pain`, `Pain with Bright Lights`, `Nasal Discharge`, `Headache`, `Facial Swelling`
- **CHF:** `Paroxysmal dyspnea (finding)`, `Rales (finding)`, `Orthopnea (finding)`, `Edema (finding)`, `Dyspnea on exertion (finding)`
- **COPD:** `Shortness of Breath`, `Cough`, `Sinus Pain`, `Nasal Congestion`, `Facial Swelling`
- **Cystitis:** `Urgent desire to urinate`, `Suprapubic pain`, `Increased frequency of urination`, `Dysuria`, `Nasal Congestion`
- **Hypertension:** `Sinus Pain`, `Pain with Bright Lights`, `Nasal Discharge`, `Headache`, `Facial Swelling`
- **Strep throat:** `Swollen Tonsils`, `Swollen Lymph Nodes`, `Runny/Stuffy Nose`, `Decreased Appetite`, `Body Aches`
- **Viral pharyngitis:** `Swollen Tonsils`, `Swollen Lymph Nodes`, `Runny/Stuffy Nose`, `Decreased Appetite`, `Body Aches`

> **Strep throat and viral pharyngitis share an identical top-5 feature list.** This directly explains the 0% strep recall: the two symptom distributions are indistinguishable to the classifier. Note also that hypertension "top features" are sinusitis symptoms — an artifact of the one-vs-rest classifier learning symptom absence as a discriminant when a class has no symptom data at all.

## b. Shortcut probe

**Features:** age at diagnosis, sex (binary), number of pre-cutoff encounters, years of recorded history (earliest encounter to cutoff). **Intent:** record shape alone — not clinical content.

**Overall accuracy:** 30.0%  **Balanced accuracy:** 33.6%  **Chance level:** 14.3% (1/7 classes)

**Feature importance** (mean |coef| across all condition classifiers):

- `n_enc`: 1.272
- `age`: 0.825
- `years_hist`: 0.377
- `sex_male`: 0.074

> **Record shape is 2.3× chance (33.6% vs 14.3% baseline).** Conditions differ in age profile (CHF skews older; paediatric-heavy viral pharyngitis skews younger), sex distribution (cystitis predominantly female), and encounter density (acute conditions have shorter histories). This is a measurable confound. Any symptom-based model should be benchmarked against the 33.6% shortcut balanced-accuracy floor, not against random chance.

## c. Presence probe

**Feature:** `has_episode` — binary flag for whether a presenting symptom episode exists in `symptoms.csv` for this patient.

**Overall accuracy:** 51.4%  **Balanced accuracy:** 28.4%  **Chance level:** 14.3%

**Per-condition episode absence rate:**

| Condition | n | No episode | % absent | Interpretation |
|-----------|--:|----------:|--------:|----------------|
| Hypertension | 2650 | 2650 | 100.0% | **Strong absence signal** |
| COPD | 384 | 243 | 63.3% | Partial absence |
| Cystitis | 1160 | 304 | 26.2% | Partial absence |
| Bacterial sinusitis | 656 | 56 | 8.5% | Mostly present |
| Strep throat | 1499 | 13 | 0.9% | Mostly present |
| CHF | 313 | 2 | 0.6% | Mostly present |
| Viral pharyngitis | 5108 | 32 | 0.6% | Mostly present |

> **Hypertension is 100% identifiable by absence.** Every hypertension patient has no presenting episode (Synthea generates no symptoms for it). Any model that can observe whether a presenting section exists will perfectly identify hypertension without reading any clinical content. The `--presenting` flag in `serialize.py` is the correct interface for controlling this.

## Open design questions

These are unresolved decisions the team needs to make before training:

1. **Hypertension without symptoms.** Including hypertension in the acute bin is invalid (it has no symptoms). Including it in history-only evaluation is legitimate, but any multi-task model that also sees a `has_presenting` flag will trivially identify it. Decide: ablate presence entirely, or treat hypertension as a history-only condition and the others as symptom-available?

2. **Strep vs viral pharyngitis separability.** These two conditions share overlapping symptoms in Synthea. If the classifier cannot separate them reliably from symptoms alone, should they be merged into a single respiratory-infection class, or kept separate with the understanding that a final model must use additional context (e.g. test results)?

3. **Demographic confounding.** The shortcut probe shows record shape alone is discriminative. Age, sex, and encounter density differ systematically across conditions. Any evaluation of symptom-based models should be compared against the shortcut-probe baseline, not against random chance.

4. **Class imbalance.** Viral pharyngitis (45%) and hypertension (23%) dominate the dataset. A model trained with standard cross-entropy will collapse to predicting these two conditions. Confirm class-weighting strategy before training.

5. **COPD episode coverage (37%).** Most COPD patients lack a presenting episode — Synthea's symptom coverage is sparse for long-term chronic conditions. Decide whether to train COPD with history only, or exclude COPD from symptom-based evaluations.


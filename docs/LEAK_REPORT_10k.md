# Leak Check Report: pop10000-seed20260916__scrub-heart_failure-9e8474

Generated: 2026-10-06 08:26 UTC

Target codes: ['88805009']  
Descriptions: ['Chronic congestive heart failure (disorder)']

## Summary

| Check | Result |
|-------|--------|
| 1. Date assertion | PASS — 0 violations |
| 2. Text/code scan | PASS (0 code/full-desc hits, 14190 partial match(es)) |
| 3. Classifier probe | acc 70.7% ± 2.2%, baseline 75.0%, lift -4.3%, bal-acc 76.8%, AUC 0.862 |
| 4. Probe validation | shifted acc 100.0%, improvement +29.3% — ✓ probe valid |

## Check 1: Date assertion

Every filter date, end date, and note entry date in the scrubbed output must be before the patient's cutoff. ENCOUNTER references must appear in the scrubbed encounters.csv (which contains only pre-cutoff encounters).

**PASS.** No violations found.

## Check 2: Text and code scan

Checks: exact SNOMED code; full description (tag stripped, case-insensitive); word-order-independent (all description words present); and partial matches (distinctive sub-phrases and long individual words).

**14190 hit(s)** — 0 exact code, 0 full/word-order, 14190 partial.

Partial matches are expected before the cutoff (see SCRUBBING.md §Known limitations). Only code and full-description hits are hard failures.

| File | Row | Col | Match type | Fragment |
|------|-----|-----|------------|---------|
| csv/careplans.csv | 3 | DESCRIPTION | `word:chronic` | Chronic obstructive pulmonary disease clinical management plan (record artifact) |
| csv/careplans.csv | 13 | DESCRIPTION | `word:chronic` | Chronic obstructive pulmonary disease clinical management plan (record artifact) |
| csv/careplans.csv | 421 | DESCRIPTION | `word:chronic` | Chronic obstructive pulmonary disease clinical management plan (record artifact) |
| csv/careplans.csv | 465 | DESCRIPTION | `word:chronic` | Chronic obstructive pulmonary disease clinical management plan (record artifact) |
| csv/careplans.csv | 472 | DESCRIPTION | `word:chronic` | Chronic obstructive pulmonary disease clinical management plan (record artifact) |
| csv/careplans.csv | 697 | DESCRIPTION | `word:chronic` | Chronic obstructive pulmonary disease clinical management plan (record artifact) |
| csv/careplans.csv | 777 | DESCRIPTION | `word:chronic` | Chronic obstructive pulmonary disease clinical management plan (record artifact) |
| csv/conditions.csv | 2 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 3 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 50 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 82 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 84 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 85 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 104 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 124 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 127 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 230 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 280 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 297 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 314 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 1 (disorder) |
| csv/conditions.csv | 317 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 320 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 348 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 376 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 418 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 461 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 468 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 486 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 541 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 543 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 546 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 562 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 593 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 635 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 648 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 660 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 690 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 2 (disorder) |
| csv/conditions.csv | 710 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 726 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 776 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 793 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 809 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 856 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 915 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 927 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 950 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 1000 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 1 (disorder) |
| csv/conditions.csv | 1031 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 1 (disorder) |
| csv/conditions.csv | 1076 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 2 (disorder) |
| csv/conditions.csv | 1109 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 2 (disorder) |
| csv/conditions.csv | 1111 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 3 (disorder) |
| csv/conditions.csv | 1131 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 1154 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 3 (disorder) |
| csv/conditions.csv | 1158 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 1 (disorder) |
| csv/conditions.csv | 1288 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 1300 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 1409 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 1414 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 1428 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 1494 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |

*(and 14130 more — see full output)*

## Check 3: Classifier probe

TF-IDF (1–2 grams) + logistic regression, 5-fold stratified CV. Positives: scrubbed Chronic congestive heart failure patients. Negatives: source-run patients without Chronic congestive heart failure, age-matched (birth year ±10) and truncated at the same age to prevent the D9 record-length shortcut.

> **Interpretation note:** The target condition may have legitimate clinical antecedents that distinguish it from controls. What matters is **what the classifier keys on**. Features labelled `POSSIBLE_LEAK` (see table) name the target directly or are specific to Chronic congestive heart failure treatment; they warrant manual review. The final call is yours.

**Accuracy: 70.7% ± 2.2%** (majority-class baseline: 75.0%, lift: -4.3%)
**Balanced accuracy: 76.8% ± 2.5%** (chance = 50%)
**AUC: 0.862 ± 0.018** (chance = 0.500; note: above 0.5 means better than random even when accuracy is below the majority-class baseline)

n = 313 positive, 939 negative. CV scores: 0.69, 0.70, 0.72, 0.74, 0.68


### Top 30 features — Chronic congestive heart failure-positive (positive coefficient)

| Feature | Weight | Category |
|---------|--------|----------|
| `count` | +0.776 | other |
| `automated` | +0.774 | other |
| `by automated` | +0.768 | other |
| `automated count` | +0.768 | other |
| `blood by` | +0.684 | other |
| `entitic` | +0.600 | other |
| `procedure normal` | +0.596 | other |
| `normal` | +0.565 | other |
| `normal pregnancy` | +0.565 | other |
| `pregnancy finding` | +0.565 | other |
| `by` | +0.534 | other |
| `viral sinusitis` | +0.454 | other |
| `cells` | +0.439 | other |
| `viral` | +0.435 | other |
| `blood cells` | +0.423 | other |
| `red` | +0.414 | other |
| `red blood` | +0.414 | other |
| `physical examination` | +0.406 | other |
| `platelet` | +0.404 | other |
| `28` | +0.400 | other |
| `mean` | +0.395 | other |
| `in red` | +0.392 | other |
| `cells by` | +0.392 | other |
| `entitic mass` | +0.392 | other |
| `entitic mean` | +0.385 | other |
| `mean volume` | +0.385 | other |
| `estradiol` | +0.383 | other |
| `test procedure` | +0.367 | other |
| `in blood` | +0.364 | other |
| `therapy consultation` | +0.362 | other |

### Top 30 features — Chronic congestive heart failure-negative (negative coefficient)

| Feature | Weight | Category |
|---------|--------|----------|
| `death` | -1.500 | other |
| `qols` | -1.211 | other |
| `qaly qols` | -1.211 | other |
| `qaly` | -1.211 | other |
| `daly` | -1.211 | other |
| `daly qaly` | -1.211 | other |
| `of death` | -1.202 | other |
| `hospice` | -1.148 | other |
| `hospice care` | -1.121 | other |
| `therapy hospice` | -1.101 | other |
| `qols tobacco` | -0.996 | other |
| `certification` | -0.816 | other |
| `disorder daly` | -0.762 | other |
| `us` | -0.724 | other |
| `certificate of` | -0.710 | other |
| `us standard` | -0.710 | other |
| `standard certificate` | -0.710 | other |
| `death certification` | -0.710 | other |
| `cause` | -0.710 | other |
| `certificate` | -0.710 | other |
| `death us` | -0.710 | other |
| `cause of` | -0.710 | other |
| `qols daly` | -0.621 | other |
| `qols cholesterol` | -0.555 | clinical |
| `renal disease` | -0.549 | clinical |
| `stage renal` | -0.549 | clinical |
| `end stage` | -0.549 | other |
| `end` | -0.549 | other |
| `procedure end` | -0.545 | other |
| `finding chronic` | -0.512 | other |

## Check 4: Probe validation

Shifts each positive's cutoff one encounter forward so the diagnosing visit (and its note) is included. This dataset deliberately contains the leak. The shifted probe must score clearly higher (>5 pp) than the clean probe to confirm the probe can detect a leak of this kind.

**Shifted accuracy: 100.0% ± 0.0%** (clean: 70.7%, improvement: +29.3%)

313/313 positives shifted by ≥1 encounter.

**PROBE VALID.** Shifted accuracy is clearly higher; the probe can detect this class of leak.

### Top 30 features — shifted CHF-positive (positive coefficient)

| Feature | Weight | Category |
|---------|--------|----------|
| `is` | +1.189 | other |
| `patient has` | +1.147 | other |
| `allergies` | +1.133 | other |
| `medications` | +1.091 | other |
| `patient is` | +1.080 | other |
| `no` | +1.073 | other |
| `present illness` | +0.987 | other |
| `present` | +0.987 | other |
| `old` | +0.987 | other |
| `patient comes` | +0.987 | other |
| `and plan` | +0.987 | other |
| `background patient` | +0.987 | other |
| `background` | +0.987 | other |
| `patient currently` | +0.987 | other |
| `history patient` | +0.987 | other |
| `illness` | +0.987 | other |
| `socioeconomic background` | +0.987 | other |
| `chief` | +0.987 | other |
| `chief complaint` | +0.987 | other |
| `social history` | +0.987 | other |
| `socioeconomic` | +0.987 | other |
| `comes` | +0.987 | other |
| `comes from` | +0.987 | other |
| `complaint` | +0.987 | other |
| `currently has` | +0.987 | other |
| `of present` | +0.987 | other |
| `assessment and` | +0.987 | other |
| `year old` | +0.986 | other |
| `known` | +0.984 | other |
| `known allergies` | +0.984 | other |

## What this suite cannot detect

- **Related codes and text surviving before the cutoff** (SCRUBBING.md §Known
  limitations): conditions like childhood asthma under asthma, or sinusitis
  variants. CHF has no such related Synthea codes, but text like "heart failure"
  may appear in notes before diagnosis (monitoring, family history). The partial
  scan flags these; they fall under known limitations, not scrubbing failures.
- **Numerical signals**: blood pressure trends, BNP levels, and ejection fractions
  in observations.csv are not scanned by the text check. The probe can detect them
  as features (look for observation DESCRIPTION names in the feature table).
- **Implicit information**: a record that stops updating in 2005 may be from a
  patient who died then; the fact of stopping is not a text leak but may be
  clinically informative. The age-matching in check 3 mitigates this for the probe.
- **Probe sensitivity**: the probe uses only condition/medication/procedure/encounter
  descriptions, not note text for the clean dataset. It may miss leaks that only
  appear in notes.

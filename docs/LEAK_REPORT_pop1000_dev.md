> **Development run.** This report covers the pop1000 pilot. The canonical leak report is [LEAK_REPORT_10k.md](LEAK_REPORT_10k.md).

# Leak Check Report: pop1000-seed20260916__scrub-heart_failure-9e8474

Generated: 2026-09-28 22:03 UTC

Target codes: ['88805009']  
Descriptions: ['Chronic congestive heart failure (disorder)']

## Summary

| Check | Result |
|-------|--------|
| 1. Date assertion | PASS — 0 violations |
| 2. Text/code scan | PASS (0 code/full-desc hits, 2013 partial match(es)) |
| 3. Classifier probe | acc 74.3% ± 6.8%, baseline 75.0%, lift -0.7% |
| 4. Probe validation | shifted acc 100.0%, improvement +25.7% — ✓ probe valid |

## Check 1: Date assertion

Every filter date, end date, and note entry date in the scrubbed output must be before the patient's cutoff. ENCOUNTER references must appear in the scrubbed encounters.csv (which contains only pre-cutoff encounters).

**PASS.** No violations found.

## Check 2: Text and code scan

Checks: exact SNOMED code; full description (tag stripped, case-insensitive); word-order-independent (all description words present); and partial matches (distinctive sub-phrases and long individual words).

**2013 hit(s)** — 0 exact code, 0 full/word-order, 2013 partial.

Partial matches are expected before the cutoff (see SCRUBBING.md §Known limitations). Only code and full-description hits are hard failures.

| File | Row | Col | Match type | Fragment |
|------|-----|-----|------------|---------|
| csv/careplans.csv | 2 | DESCRIPTION | `word:chronic` | Chronic obstructive pulmonary disease clinical management plan (record artifact) |
| csv/conditions.csv | 59 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 60 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 99 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 153 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 176 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 1 (disorder) |
| csv/conditions.csv | 203 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 206 | DESCRIPTION | `word:chronic` | Chronic intractable migraine without aura (disorder) |
| csv/conditions.csv | 207 | DESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/conditions.csv | 220 | DESCRIPTION | `word:chronic` | Chronic intractable migraine without aura (disorder) |
| csv/conditions.csv | 237 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 1 (disorder) |
| csv/conditions.csv | 254 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 2 (disorder) |
| csv/conditions.csv | 262 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 1 (disorder) |
| csv/conditions.csv | 331 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 2 (disorder) |
| csv/conditions.csv | 367 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 2 (disorder) |
| csv/conditions.csv | 386 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 430 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 3 (disorder) |
| csv/conditions.csv | 455 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 3 (disorder) |
| csv/conditions.csv | 576 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 593 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/conditions.csv | 882 | DESCRIPTION | `word:chronic` | Chronic low back pain (finding) |
| csv/conditions.csv | 883 | DESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/conditions.csv | 884 | DESCRIPTION | `word:chronic` | Chronic neck pain (finding) |
| csv/conditions.csv | 885 | DESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/conditions.csv | 886 | DESCRIPTION | `word:chronic` | Chronic low back pain (finding) |
| csv/conditions.csv | 897 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 1 (disorder) |
| csv/conditions.csv | 1139 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/encounters.csv | 53 | REASONDESCRIPTION | `word:chronic` | Chronic intractable migraine without aura (disorder) |
| csv/encounters.csv | 89 | REASONDESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/encounters.csv | 93 | REASONDESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/encounters.csv | 100 | REASONDESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/encounters.csv | 105 | REASONDESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/encounters.csv | 106 | REASONDESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/encounters.csv | 109 | REASONDESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/encounters.csv | 111 | REASONDESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/encounters.csv | 113 | REASONDESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/encounters.csv | 121 | REASONDESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/encounters.csv | 136 | REASONDESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/encounters.csv | 142 | REASONDESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/encounters.csv | 152 | REASONDESCRIPTION | `word:chronic` | Chronic pain (finding) |
| csv/encounters.csv | 274 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 275 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 276 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 277 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 280 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 282 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 283 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 284 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 285 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 286 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 287 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 289 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 290 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 291 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 292 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 293 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 295 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 297 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 298 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |
| csv/encounters.csv | 299 | REASONDESCRIPTION | `word:chronic` | Chronic kidney disease stage 4 (disorder) |

*(and 1953 more — see full output)*

## Check 3: Classifier probe

TF-IDF (1–2 grams) + logistic regression, 5-fold stratified CV. Positives: scrubbed CHF patients. Negatives: source-run patients without CHF, age-matched (birth year ±10) and truncated at the same age to prevent the D9 record-length shortcut.

> **Interpretation note:** CHF is a progressive condition. A high accuracy is expected from legitimate clinical features (prior cardiac disease, hypertension, kidney disease). What matters is **what the classifier keys on**. Features labelled `POSSIBLE_LEAK` (see table) name the target directly or are specific to CHF treatment; they warrant manual review. The final call is yours.

**Accuracy: 74.3% ± 6.8%** (majority-class baseline: 75.0%, lift: -0.7%)

n = 34 positive, 102 negative. CV scores: 0.68, 0.67, 0.85, 0.74, 0.78


### Top 30 features — CHF-positive (positive coefficient)

| Feature | Weight | Category |
|---------|--------|----------|
| `pregnancy finding` | +0.501 | other |
| `normal pregnancy` | +0.501 | other |
| `normal` | +0.501 | other |
| `procedure normal` | +0.485 | other |
| `fetal` | +0.256 | other |
| `therapy normal` | +0.245 | other |
| `automated` | +0.245 | other |
| `count` | +0.245 | other |
| `automated count` | +0.242 | other |
| `by automated` | +0.242 | other |
| `microscopy` | +0.234 | other |
| `blood by` | +0.221 | other |
| `sediment by` | +0.220 | other |
| `urine sediment` | +0.220 | other |
| `sediment` | +0.220 | other |
| `physical examination` | +0.217 | other |
| `pregnancy` | +0.200 | other |
| `entitic` | +0.199 | other |
| `pregnancy test` | +0.195 | other |
| `standard pregnancy` | +0.195 | other |
| `power` | +0.190 | other |
| `high power` | +0.190 | other |
| `by microscopy` | +0.190 | other |
| `power field` | +0.190 | other |
| `field` | +0.190 | other |
| `microscopy high` | +0.190 | other |
| `in blood` | +0.189 | other |
| `gingivitis` | +0.181 | other |
| `gingivitis disorder` | +0.181 | other |
| `prenatal initial` | +0.180 | other |

### Top 30 features — CHF-negative (negative coefficient)

| Feature | Weight | Category |
|---------|--------|----------|
| `disorder` | -0.265 | other |
| `procedure` | -0.260 | other |
| `hypertension` | -0.254 | clinical |
| `finding` | -0.240 | other |
| `hypertension disorder` | -0.228 | clinical |
| `essential` | -0.228 | other |
| `essential hypertension` | -0.228 | clinical |
| `patient procedure` | -0.219 | other |
| `general examination` | -0.219 | other |
| `of patient` | -0.219 | other |
| `general` | -0.218 | other |
| `of` | -0.218 | other |
| `examination of` | -0.217 | other |
| `examination` | -0.210 | other |
| `chronic` | -0.199 | other |
| `disorder disorder` | -0.199 | other |
| `pain finding` | -0.192 | other |
| `sleep` | -0.187 | other |
| `patient` | -0.187 | other |
| `amlodipine mg` | -0.176 | other |
| `amlodipine` | -0.176 | other |
| `asthma` | -0.175 | other |
| `loss of` | -0.166 | other |
| `loss` | -0.166 | other |
| `visit procedure` | -0.163 | other |
| `asthma disorder` | -0.163 | other |
| `finding chronic` | -0.160 | other |
| `problem` | -0.158 | other |
| `daly qaly` | -0.155 | other |
| `qaly qols` | -0.155 | other |

## Check 4: Probe validation

Shifts each positive's cutoff one encounter forward so the diagnosing visit (and its note) is included. This dataset deliberately contains the leak. The shifted probe must score clearly higher (>5 pp) than the clean probe to confirm the probe can detect a leak of this kind.

**Shifted accuracy: 100.0% ± 0.0%** (clean: 74.3%, improvement: +25.7%)

34/34 positives shifted by ≥1 encounter.

**PROBE VALID.** Shifted accuracy is clearly higher; the probe can detect this class of leak.

### Top 30 features — shifted CHF-positive (positive coefficient)

| Feature | Weight | Category |
|---------|--------|----------|
| `is` | +0.550 | other |
| `patient has` | +0.531 | other |
| `allergies` | +0.520 | other |
| `medications` | +0.502 | other |
| `patient is` | +0.500 | other |
| `no` | +0.492 | other |
| `patient comes` | +0.453 | other |
| `patient currently` | +0.453 | other |
| `present illness` | +0.453 | other |
| `present` | +0.453 | other |
| `and plan` | +0.453 | other |
| `socioeconomic background` | +0.453 | other |
| `chief` | +0.453 | other |
| `old` | +0.453 | other |
| `chief complaint` | +0.453 | other |
| `background patient` | +0.453 | other |
| `comes from` | +0.453 | other |
| `comes` | +0.453 | other |
| `illness` | +0.453 | other |
| `currently has` | +0.453 | other |
| `background` | +0.453 | other |
| `of present` | +0.453 | other |
| `assessment and` | +0.453 | other |
| `complaint` | +0.453 | other |
| `social history` | +0.453 | other |
| `socioeconomic` | +0.453 | other |
| `history patient` | +0.453 | other |
| `year old` | +0.453 | other |
| `identifies as` | +0.452 | other |
| `as` | +0.452 | other |

## Check 5: Cross-module attribute leak

The `chf` attribute is the mechanism through which Synthea communicates a CHF
diagnosis across modules. Five external modules read it:

| Module | State | Condition | Effect if true |
|--------|-------|-----------|----------------|
| `hypertension` | `Check for Exclusions` | `chf is not nil` | Suppresses new hypertension treatment onset |
| `sleep_apnea` | `Sleep Apnea Care Plan` | `chf is not nil` | Forces CPAP (vs. 80%/20% CPAP/oral split) |
| `covid19/determine_risk` | `Determine Risk` | `chf is not nil` | Increases COVID-19 severe risk classification |
| `heart/cabg/preoperative` | `Check CHF` | `chf is not nil` | Adds NTproBNP to CABG preoperative workup |
| `home_hospice_snf` | `Hospice Check` | `chf is not nil` | Routes to CHF hospice pathway |

**The attribute is set at one state only: `CHF Condition Start`**
(`assign_to_attribute: "chf"`, type=ConditionOnset). No other state in
`congestive_heart_failure.json` sets the bare `chf` attribute. Between `CHF
onset` and `CHF Condition Start` there are zero Delay states; all intermediate
states are SetAttribute (counter resets) and Symptom (symptom severity integers,
not time). Zero simulated time passes between `CHF onset` and the attribute
being set.

Because `chf` does not exist before `CHF Condition Start`, every module that
reads it can only branch on it after the diagnosis. All post-diagnosis encounters
and records are removed by scrub.py at or after the cutoff date.

**No cross-module attribute leak exists for CHF.**

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

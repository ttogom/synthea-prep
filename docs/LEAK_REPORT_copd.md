# Leak Check Report: pop10000-seed20260916__scrub-copd-465e92

Generated: 2026-10-06 01:36 UTC

Target codes: ['185086009', '87433001']  
Descriptions: ['Chronic obstructive bronchitis (disorder)', 'Pulmonary emphysema (disorder)']

## Summary

| Check | Result |
|-------|--------|
| 1. Date assertion | PASS — 0 violations |
| 2. Text/code scan | PASS (0 code/full-desc hits, 4500 partial match(es)) |
| 3. Classifier probe | ERROR: skipped (--skip-probe) |
| 4. Probe validation | skipped (--skip-probe) |

## Check 1: Date assertion

Every filter date, end date, and note entry date in the scrubbed output must be before the patient's cutoff. ENCOUNTER references must appear in the scrubbed encounters.csv (which contains only pre-cutoff encounters).

**PASS.** No violations found.

## Check 2: Text and code scan

Checks: exact SNOMED code; full description (tag stripped, case-insensitive); word-order-independent (all description words present); and partial matches (distinctive sub-phrases and long individual words).

**4500 hit(s)** — 0 exact code, 0 full/word-order, 4500 partial.

Partial matches are expected before the cutoff (see SCRUBBING.md §Known limitations). Only code and full-description hits are hard failures.

| File | Row | Col | Match type | Fragment |
|------|-----|-----|------------|---------|
| csv/careplans.csv | 139 | REASONDESCRIPTION | `word:obstructive` | Obstructive sleep apnea syndrome (disorder) |
| csv/careplans.csv | 149 | REASONDESCRIPTION | `word:obstructive` | Obstructive sleep apnea syndrome (disorder) |
| csv/careplans.csv | 156 | REASONDESCRIPTION | `word:obstructive` | Obstructive sleep apnea syndrome (disorder) |
| csv/careplans.csv | 201 | REASONDESCRIPTION | `word:obstructive` | Obstructive sleep apnea syndrome (disorder) |
| csv/careplans.csv | 314 | REASONDESCRIPTION | `word:obstructive` | Obstructive sleep apnea syndrome (disorder) |
| csv/careplans.csv | 364 | REASONDESCRIPTION | `word:obstructive` | Obstructive sleep apnea syndrome (disorder) |
| csv/careplans.csv | 456 | REASONDESCRIPTION | `word:obstructive` | Obstructive sleep apnea syndrome (disorder) |
| csv/careplans.csv | 464 | REASONDESCRIPTION | `word:obstructive` | Obstructive sleep apnea syndrome (disorder) |
| csv/careplans.csv | 597 | REASONDESCRIPTION | `word:obstructive` | Obstructive sleep apnea syndrome (disorder) |
| csv/conditions.csv | 3 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 12 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 50 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 72 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 98 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 113 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 122 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 159 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 171 | DESCRIPTION | `word:bronchitis` | Acute bronchitis (disorder) |
| csv/conditions.csv | 185 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 207 | DESCRIPTION | `word:obstructive` | Obstructive sleep apnea syndrome (disorder) |
| csv/conditions.csv | 218 | DESCRIPTION | `word:bronchitis` | Acute bronchitis (disorder) |
| csv/conditions.csv | 223 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 224 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 231 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 235 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 1 (disorder) |
| csv/conditions.csv | 252 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 2 (disorder) |
| csv/conditions.csv | 256 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 261 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 268 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 280 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 282 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 3 (disorder) |
| csv/conditions.csv | 325 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 335 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 355 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 357 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 407 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 416 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 417 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 436 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 445 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 446 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 451 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 508 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 529 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 534 | DESCRIPTION | `word:obstructive` | Obstructive sleep apnea syndrome (disorder) |
| csv/conditions.csv | 552 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 609 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 621 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 642 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 658 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 665 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 688 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 700 | DESCRIPTION | `word:bronchitis` | Acute bronchitis (disorder) |
| csv/conditions.csv | 709 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 752 | DESCRIPTION | `word:chronic` | Chronic kidney disease stage 1 (disorder) |
| csv/conditions.csv | 769 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 781 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 782 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 800 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |
| csv/conditions.csv | 819 | DESCRIPTION | `word:chronic` | Chronic sinusitis (disorder) |

*(and 4440 more — see full output)*

## Check 3: Classifier probe

TF-IDF (1–2 grams) + logistic regression, 5-fold stratified CV. Positives: scrubbed Chronic obstructive bronchitis / Pulmonary emphysema patients. Negatives: source-run patients without Chronic obstructive bronchitis / Pulmonary emphysema, age-matched (birth year ±10) and truncated at the same age to prevent the D9 record-length shortcut.

> **Interpretation note:** The target condition may have legitimate clinical antecedents that distinguish it from controls. What matters is **what the classifier keys on**. Features labelled `POSSIBLE_LEAK` (see table) name the target directly or are specific to Chronic obstructive bronchitis / Pulmonary emphysema treatment; they warrant manual review. The final call is yours.

ERROR: skipped (--skip-probe)

## Check 4: Probe validation

Shifts each positive's cutoff one encounter forward so the diagnosing visit (and its note) is included. This dataset deliberately contains the leak. The shifted probe must score clearly higher (>5 pp) than the clean probe to confirm the probe can detect a leak of this kind.

ERROR: skipped (--skip-probe)

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

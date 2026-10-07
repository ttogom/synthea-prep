# Leak Check Report: pop10000-seed20260916__scrub-hypertension-249aea

Generated: 2026-10-06 01:49 UTC

Target codes: ['59621000']  
Descriptions: ['Essential hypertension (disorder)']

## Summary

| Check | Result |
|-------|--------|
| 1. Date assertion | PASS — 0 violations |
| 2. Text/code scan | PASS (0 code/full-desc hits, 0 partial match(es)) |
| 3. Classifier probe | ERROR: skipped (--skip-probe) |
| 4. Probe validation | skipped (--skip-probe) |

## Check 1: Date assertion

Every filter date, end date, and note entry date in the scrubbed output must be before the patient's cutoff. ENCOUNTER references must appear in the scrubbed encounters.csv (which contains only pre-cutoff encounters).

**PASS.** No violations found.

## Check 2: Text and code scan

Checks: exact SNOMED code; full description (tag stripped, case-insensitive); word-order-independent (all description words present); and partial matches (distinctive sub-phrases and long individual words).

**PASS.** No hits of any kind.

## Check 3: Classifier probe

TF-IDF (1–2 grams) + logistic regression, 5-fold stratified CV. Positives: scrubbed Essential hypertension patients. Negatives: source-run patients without Essential hypertension, age-matched (birth year ±10) and truncated at the same age to prevent the D9 record-length shortcut.

> **Interpretation note:** The target condition may have legitimate clinical antecedents that distinguish it from controls. What matters is **what the classifier keys on**. Features labelled `POSSIBLE_LEAK` (see table) name the target directly or are specific to Essential hypertension treatment; they warrant manual review. The final call is yours.

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

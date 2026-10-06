# Leak Check Report: pop10000-seed20260916__scrub-polyp_colon-33a2f4

Generated: 2026-10-06 01:42 UTC

Target codes: ['68496003']  
Descriptions: ['Polyp of colon (disorder)']

## Summary

| Check | Result |
|-------|--------|
| 1. Date assertion | PASS — 0 violations |
| 2. Text/code scan | PASS (0 code/full-desc hits, 292 partial match(es)) |
| 3. Classifier probe | ERROR: skipped (--skip-probe) |
| 4. Probe validation | skipped (--skip-probe) |

## Check 1: Date assertion

Every filter date, end date, and note entry date in the scrubbed output must be before the patient's cutoff. ENCOUNTER references must appear in the scrubbed encounters.csv (which contains only pre-cutoff encounters).

**PASS.** No violations found.

## Check 2: Text and code scan

Checks: exact SNOMED code; full description (tag stripped, case-insensitive); word-order-independent (all description words present); and partial matches (distinctive sub-phrases and long individual words).

**292 hit(s)** — 0 exact code, 0 full/word-order, 292 partial.

Partial matches are expected before the cutoff (see SCRUBBING.md §Known limitations). Only code and full-description hits are hard failures.

| File | Row | Col | Match type | Fragment |
|------|-----|-----|------------|---------|
| csv/encounters.csv | 53 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 78 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 116 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 152 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 158 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 205 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 258 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 470 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 568 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 661 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 676 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 701 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 852 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 971 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 1065 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 1086 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 1615 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 1657 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 1950 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 1977 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 2143 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 2195 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 2199 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 2242 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 2408 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 2542 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 2656 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 2766 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 2803 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 2814 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 2878 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 2927 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 3549 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 3643 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 3683 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 3924 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 4019 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 4053 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 4071 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 4329 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 4406 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 4563 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 4582 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 4851 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 4891 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 4950 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 4992 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 5035 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 5187 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 5256 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 5373 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 5481 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 5627 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 5652 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 5793 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 5840 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 5899 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 5988 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 6042 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |
| csv/encounters.csv | 6276 | REASONDESCRIPTION | `partial:of colon` | Screening for malignant neoplasm of colon (procedure) |

*(and 232 more — see full output)*

## Check 3: Classifier probe

TF-IDF (1–2 grams) + logistic regression, 5-fold stratified CV. Positives: scrubbed Polyp of colon patients. Negatives: source-run patients without Polyp of colon, age-matched (birth year ±10) and truncated at the same age to prevent the D9 record-length shortcut.

> **Interpretation note:** The target condition may have legitimate clinical antecedents that distinguish it from controls. What matters is **what the classifier keys on**. Features labelled `POSSIBLE_LEAK` (see table) name the target directly or are specific to Polyp of colon treatment; they warrant manual review. The final call is yours.

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

# Leak Check Report: pop10000-seed20260916__scrub-strep_throat-1059ea

Generated: 2026-10-06 01:55 UTC

Target codes: ['43878008']  
Descriptions: ['Streptococcal sore throat (disorder)']

## Summary

| Check | Result |
|-------|--------|
| 1. Date assertion | PASS — 0 violations |
| 2. Text/code scan | PASS (0 code/full-desc hits, 2193 partial match(es)) |
| 3. Classifier probe | ERROR: skipped (--skip-probe) |
| 4. Probe validation | skipped (--skip-probe) |

## Check 1: Date assertion

Every filter date, end date, and note entry date in the scrubbed output must be before the patient's cutoff. ENCOUNTER references must appear in the scrubbed encounters.csv (which contains only pre-cutoff encounters).

**PASS.** No violations found.

## Check 2: Text and code scan

Checks: exact SNOMED code; full description (tag stripped, case-insensitive); word-order-independent (all description words present); and partial matches (distinctive sub-phrases and long individual words).

**2193 hit(s)** — 0 exact code, 0 full/word-order, 2193 partial.

Partial matches are expected before the cutoff (see SCRUBBING.md §Known limitations). Only code and full-description hits are hard failures.

| File | Row | Col | Match type | Fragment |
|------|-----|-----|------------|---------|
| csv/conditions.csv | 20128 | DESCRIPTION | `partial:sore throat` | Sore throat (finding) |
| csv/conditions.csv | 20426 | DESCRIPTION | `partial:sore throat` | Sore throat (finding) |
| csv/conditions.csv | 20509 | DESCRIPTION | `partial:sore throat` | Sore throat (finding) |
| csv/conditions.csv | 20619 | DESCRIPTION | `partial:sore throat` | Sore throat (finding) |
| csv/conditions.csv | 20659 | DESCRIPTION | `partial:sore throat` | Sore throat (finding) |
| csv/conditions.csv | 20756 | DESCRIPTION | `partial:sore throat` | Sore throat (finding) |
| csv/conditions.csv | 20763 | DESCRIPTION | `partial:sore throat` | Sore throat (finding) |
| csv/conditions.csv | 20935 | DESCRIPTION | `partial:sore throat` | Sore throat (finding) |
| csv/conditions.csv | 21147 | DESCRIPTION | `partial:sore throat` | Sore throat (finding) |
| csv/conditions.csv | 21275 | DESCRIPTION | `partial:sore throat` | Sore throat (finding) |
| symptoms/csv/symptoms.csv | 16 | SYMPTOMS | `partial:sore throat` | Sinus Pain:53:0;Headache:12:0;Pain with Bright Lights:31:0;Facial Swelling:1:0;N |
| symptoms/csv/symptoms.csv | 43 | SYMPTOMS | `partial:sore throat` | Body Aches:17:0;Diarrhea:7:0;Fatigue:68:0;Mucus:28:0;Nasal Congestion:44:0;Short |
| symptoms/csv/symptoms.csv | 61 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:24:0;Body Aches:22:0;Runny/Stuffy Nose:32:0;Fatigue:24:0;S |
| symptoms/csv/symptoms.csv | 64 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:24:0;Body Aches:26:0;Runny/Stuffy Nose:38:0;Fatigue:7:0;Sw |
| symptoms/csv/symptoms.csv | 69 | SYMPTOMS | `partial:sore throat` | Body Aches:22:0;Diarrhea:23:0;Fatigue:61:0;Mucus:36:0;Nasal Congestion:30:0;Shor |
| symptoms/csv/symptoms.csv | 131 | SYMPTOMS | `partial:sore throat` | Headache:33:0;Sinus Pain:64:0;Pain with Bright Lights:40:0;Facial Swelling:37:0; |
| symptoms/csv/symptoms.csv | 136 | SYMPTOMS | `partial:sore throat` | Body Aches:11:0;Diarrhea:8:0;Fatigue:68:0;Mucus:41:0;Nasal Congestion:49:0;Short |
| symptoms/csv/symptoms.csv | 141 | SYMPTOMS | `partial:sore throat` | Body Aches:2:0;Diarrhea:5:0;Fatigue:73:0;Mucus:40:0;Nasal Congestion:46:0;Shortn |
| symptoms/csv/symptoms.csv | 146 | SYMPTOMS | `partial:sore throat` | Headache:22:0;Sinus Pain:77:0;Pain with Bright Lights:38:0;Facial Swelling:22:0; |
| symptoms/csv/symptoms.csv | 191 | SYMPTOMS | `partial:sore throat` | Body Aches:4:0;Diarrhea:7:0;Fatigue:60:0;Mucus:34:0;Nasal Congestion:46:0;Shortn |
| symptoms/csv/symptoms.csv | 192 | SYMPTOMS | `partial:sore throat` | Sinus Pain:66:0;Headache:19:0;Pain with Bright Lights:34:0;Facial Swelling:45:0; |
| symptoms/csv/symptoms.csv | 195 | SYMPTOMS | `partial:sore throat` | Sinus Pain:92:102:103:91:0;Headache:0;Pain with Bright Lights:0;Facial Swelling: |
| symptoms/csv/symptoms.csv | 200 | SYMPTOMS | `partial:sore throat` | Sinus Pain:72:92:102:103:91:0;Headache:47:0;Pain with Bright Lights:43:0;Facial  |
| symptoms/csv/symptoms.csv | 232 | SYMPTOMS | `partial:sore throat` | Sinus Pain:66:0;Headache:11:0;Pain with Bright Lights:36:0;Facial Swelling:33:0; |
| symptoms/csv/symptoms.csv | 234 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:25:0;Body Aches:26:0;Runny/Stuffy Nose:12:0;Fatigue:43:0;S |
| symptoms/csv/symptoms.csv | 250 | SYMPTOMS | `partial:sore throat` | Headache:5:0;Sinus Pain:67:0;Pain with Bright Lights:33:0;Facial Swelling:32:0;N |
| symptoms/csv/symptoms.csv | 282 | SYMPTOMS | `partial:sore throat` | Headache:46:0;Sinus Pain:36:0;Pain with Bright Lights:35:0;Facial Swelling:29:0; |
| symptoms/csv/symptoms.csv | 292 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:17:0;Body Aches:22:0;Runny/Stuffy Nose:20:0;Fatigue:31:0;S |
| symptoms/csv/symptoms.csv | 333 | SYMPTOMS | `partial:sore throat` | Body Aches:8:0;Diarrhea:4:0;Fatigue:27:0;Mucus:32:0;Nasal Congestion:45:0;Shortn |
| symptoms/csv/symptoms.csv | 344 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:9:0;Body Aches:8:0;Runny/Stuffy Nose:48:0;Fatigue:30:0;Swo |
| symptoms/csv/symptoms.csv | 362 | SYMPTOMS | `partial:sore throat` | Sinus Pain:55:0;Headache:11:0;Pain with Bright Lights:38:0;Facial Swelling:18:0; |
| symptoms/csv/symptoms.csv | 364 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:40:0;Body Aches:49:0;Runny/Stuffy Nose:32:0;Fatigue:21:0;S |
| symptoms/csv/symptoms.csv | 374 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:38:0;Body Aches:15:0;Runny/Stuffy Nose:8:0;Fatigue:2:0;Swo |
| symptoms/csv/symptoms.csv | 398 | SYMPTOMS | `partial:sore throat` | Body Aches:15:0;Diarrhea:23:0;Fatigue:27:0;Mucus:36:0;Nasal Congestion:34:0;Shor |
| symptoms/csv/symptoms.csv | 433 | SYMPTOMS | `partial:sore throat` | Headache:28:0;Sinus Pain:67:0;Pain with Bright Lights:42:0;Facial Swelling:13:0; |
| symptoms/csv/symptoms.csv | 480 | SYMPTOMS | `partial:sore throat` | Headache:0:0;Sinus Pain:51:0;Pain with Bright Lights:26:0;Facial Swelling:45:0;N |
| symptoms/csv/symptoms.csv | 493 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:30:0;Body Aches:14:0;Runny/Stuffy Nose:40:0;Fatigue:24:0;S |
| symptoms/csv/symptoms.csv | 504 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:18:0;Body Aches:48:0;Runny/Stuffy Nose:39:0;Fatigue:36:0;S |
| symptoms/csv/symptoms.csv | 516 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:47:0;Body Aches:34:0;Runny/Stuffy Nose:37:0;Fatigue:6:0;Sw |
| symptoms/csv/symptoms.csv | 530 | SYMPTOMS | `partial:sore throat` | Sinus Pain:67:0;Headache:32:0;Pain with Bright Lights:45:0;Facial Swelling:47:0; |
| symptoms/csv/symptoms.csv | 548 | SYMPTOMS | `partial:sore throat` | Body Aches:16:0;Diarrhea:11:0;Fatigue:52:0;Mucus:30:0;Nasal Congestion:43:0;Shor |
| symptoms/csv/symptoms.csv | 566 | SYMPTOMS | `partial:sore throat` | Headache:42:0;Sinus Pain:36:0;Pain with Bright Lights:44:0;Facial Swelling:6:0;N |
| symptoms/csv/symptoms.csv | 567 | SYMPTOMS | `partial:sore throat` | Headache:27:0;Sinus Pain:59:0;Pain with Bright Lights:49:0;Facial Swelling:31:0; |
| symptoms/csv/symptoms.csv | 594 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:47:0;Body Aches:35:0;Runny/Stuffy Nose:30:0;Fatigue:37:0;S |
| symptoms/csv/symptoms.csv | 620 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:2:0;Body Aches:34:0;Runny/Stuffy Nose:31:0;Fatigue:12:0;Sw |
| symptoms/csv/symptoms.csv | 739 | SYMPTOMS | `partial:sore throat` | Body Aches:20:0;Diarrhea:11:0;Fatigue:39:0;Mucus:45:0;Nasal Congestion:45:0;Shor |
| symptoms/csv/symptoms.csv | 741 | SYMPTOMS | `partial:sore throat` | Headache:33:0;Sinus Pain:69:0;Pain with Bright Lights:43:0;Facial Swelling:40:0; |
| symptoms/csv/symptoms.csv | 749 | SYMPTOMS | `partial:sore throat` | Headache:42:0;Sinus Pain:78:0;Pain with Bright Lights:32:0;Facial Swelling:41:0; |
| symptoms/csv/symptoms.csv | 850 | SYMPTOMS | `partial:sore throat` | Sinus Pain:72:0;Headache:0:0;Pain with Bright Lights:28:0;Facial Swelling:42:0;N |
| symptoms/csv/symptoms.csv | 864 | SYMPTOMS | `partial:sore throat` | Body Aches:3:0;Diarrhea:7:0;Fatigue:60:0;Mucus:46:0;Nasal Congestion:40:0;Shortn |
| symptoms/csv/symptoms.csv | 873 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:48:0;Body Aches:32:0;Runny/Stuffy Nose:14:0;Fatigue:14:0;S |
| symptoms/csv/symptoms.csv | 878 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:32:0;Body Aches:17:0;Runny/Stuffy Nose:46:0;Fatigue:8:0;Sw |
| symptoms/csv/symptoms.csv | 880 | SYMPTOMS | `partial:sore throat` | Sinus Pain:71:0;Headache:19:0;Pain with Bright Lights:32:0;Facial Swelling:48:0; |
| symptoms/csv/symptoms.csv | 890 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:2:0;Body Aches:25:0;Runny/Stuffy Nose:15:0;Fatigue:30:0;Sw |
| symptoms/csv/symptoms.csv | 952 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:42:0;Body Aches:30:0;Runny/Stuffy Nose:14:0;Fatigue:49:0;S |
| symptoms/csv/symptoms.csv | 965 | SYMPTOMS | `partial:sore throat` | Sinus Pain:77:0;Headache:35:0;Pain with Bright Lights:28:0;Facial Swelling:3:0;N |
| symptoms/csv/symptoms.csv | 1016 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:40:0;Body Aches:27:0;Runny/Stuffy Nose:24:0;Fatigue:7:0;Sw |
| symptoms/csv/symptoms.csv | 1019 | SYMPTOMS | `partial:sore throat` | Sinus Pain:72:0;Headache:7:0;Pain with Bright Lights:30:0;Facial Swelling:25:0;N |
| symptoms/csv/symptoms.csv | 1076 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:23:0;Body Aches:7:0;Runny/Stuffy Nose:49:0;Fatigue:24:0;Sw |
| symptoms/csv/symptoms.csv | 1077 | SYMPTOMS | `partial:sore throat` | Difficulty Swallowing:37:0;Body Aches:25:0;Runny/Stuffy Nose:37:0;Fatigue:7:0;Sw |

*(and 2133 more — see full output)*

## Check 3: Classifier probe

TF-IDF (1–2 grams) + logistic regression, 5-fold stratified CV. Positives: scrubbed Streptococcal sore throat patients. Negatives: source-run patients without Streptococcal sore throat, age-matched (birth year ±10) and truncated at the same age to prevent the D9 record-length shortcut.

> **Interpretation note:** The target condition may have legitimate clinical antecedents that distinguish it from controls. What matters is **what the classifier keys on**. Features labelled `POSSIBLE_LEAK` (see table) name the target directly or are specific to Streptococcal sore throat treatment; they warrant manual review. The final call is yours.

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

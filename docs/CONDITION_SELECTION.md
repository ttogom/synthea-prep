# Condition Selection

Seven conditions were selected from a 10,000-patient Synthea run
(`pop10000-seed20260916`, reference date 2026-09-21) and grouped into three bins
by signal type: two history-driven conditions, four acute/symptom-driven
conditions, and one demographic control. The selection used a module-level static
analysis to identify history-dependent candidates, a data screen to verify the
antecedent structure survives in actual patient records, and a presenting-evidence
extraction to confirm what is observable at the time of diagnosis.

## The funnel

**Step 1 — Module scan** (`scripts/scan_modules.py`, output `docs/MODULE_SCAN.md`).
Static analysis of 242 Synthea module JSON files found 333 unique SNOMED
ConditionOnset codes. Of these, 127 are history-dependent (at least one
clinically visible gate on the path to onset), 100 are demographic-only, and 45
are excluded by known leakage flags (symptom codes, precursor-names-target, etc.).
The remaining 61 are history-dependent via hidden attributes only (internal
simulation flags not present in the patient CSV).

**Step 2 — Candidate selection.** Eleven conditions were hand-picked from the 127
visible-history and 100 demographic-only groups, chosen to span the history/no-history
and symptom/no-symptom axes, and to include at least one condition (COPD) that serves
as a positive control with a known causal antecedent (smoking).

**Step 3 — Data screen** (`scripts/quick_screen.py`, output `docs/CONDITION_SCREEN.md`).
Each candidate was screened against exact birth-year + sex matched negatives using a
two-proportion z-test (Bonferroni corrected). Four artifacts were identified and marked
across all conditions: survivor bias (cause-of-death appears in negatives who died
before the positive's diagnosis age), QALY/QOLS/DALY simulation metrics, healthcare
utilization vitals cluster (all standard vitals co-occurring in a single prior wellness
encounter, identical z-scores), and the Synthea OSA precursor (Sleep disorder is
scripted 100% before every OSA diagnosis).

**Results:** 7 of 11 candidates kept; 4 dropped.

| Dropped condition | Reason |
|-------------------|--------|
| Polyp of colon | Survivor-bias artifact dominates (rank 1); non-artifact antecedents marginal; 10% symptom availability |
| Ischemic heart disease | Utilization artifact is inverted (older positives have *fewer* prior wellness visits); no genuine history signal; 45% symptom availability |
| Obstructive sleep apnea | Only Bonferroni-significant feature is the Synthea model precursor (Sleep disorder 100% vs 3%); no symptom availability |
| Alzheimer's disease | Survivor bias at rank 1 (43% of matched negatives died before diagnosis age); remaining features are cardiac comorbidity co-assignments; 24% symptom availability |

## Final bins

| Bin | Condition | SNOMED code(s) | Positives | Presenting evidence |
|-----|-----------|----------------|----------:|---------------------|
| history | COPD (chronic obstructive bronchitis + pulmonary emphysema) | 87433001, 185086009 | 384 | 144 (38%): mostly symptoms; sparse coverage because Synthea assigns symptoms late in disease course |
| history | Essential hypertension | 59621000 | 2650 | 757 (29%): vitals only — elevated BP at the diagnosing wellness encounter (systolic median 149 mmHg, diastolic median 104 mmHg); Synthea generates no symptoms for this condition |
| acute | Streptococcal sore throat | 43878008 | 1499 | 1493 (100%): symptoms + temperature (median 38.3°C) |
| acute | Acute viral pharyngitis | 195662009 | 5108 | 5089 (100%): symptoms + temperature (median 37.5°C) |
| acute | Acute bacterial sinusitis | 75498004 | 656 | 600 (91%): symptoms only; no whitelisted vitals at Synthea's sinusitis encounters |
| acute | Acute infective cystitis | 307426000 | 1160 | 912 (79%): symptoms + BP/HR at some encounters |
| control | Congestive heart failure | 88805009 | 313 | 311 (99%): symptoms only; no whitelisted vitals at Synthea's CHF encounters |

**Total:** 11,770 labeled positives. The presenting-evidence counts come from
`scripts/extract_presenting.py`; the symptom-availability counts in
`docs/CONDITION_SCREEN.md` differ slightly because that screen uses raw
`symptoms.csv` counts before the AGE_BEGIN window filter.

**Bin definitions** (SNOMED codes and config files): `configs/bins/`. Rationale
and excluded candidates: `docs/CONDITION_SCREEN.md` and `configs/bins/README.md`.

## The design

The bins define a 2×2 of signal availability:

|  | Has presenting evidence | No presenting evidence |
|--|------------------------|------------------------|
| **Has history antecedent** | COPD (partial), Hypertension (partial) | — |
| **No history antecedent** | Strep, Viral pharyngitis, Sinusitis, Cystitis, CHF | Hypertension (71% of cases) |

Expected ablation results:
- Remove history (do not pass pre-diagnosis record): should hurt the history bin
  (COPD, hypertension); acute and control conditions are expected to be unaffected.
- Remove presenting evidence (do not pass `--presenting`): should hurt the acute
  bin and CHF; history-bin conditions that rely on antecedents should be unaffected.

## Presenting evidence

The PRESENTING COMPLAINT section in each serialized record contains:

1. **Symptoms** from raw `symptoms.csv`, matched to the condition episode whose
   `AGE_BEGIN` is closest to the diagnosis age (within a 7-year lookback). Source:
   Synthea's symptom simulation, not the patient's own encounter notes.

2. **Whitelisted vitals** from the diagnosing encounter in `observations.csv`:
   body temperature, systolic blood pressure, diastolic blood pressure, heart rate,
   respiratory rate, and oxygen saturation. All other observations at that encounter
   were reviewed and excluded.

The full exclusion list with justifications is in `docs/BINS.md` (Observation
whitelist decisions). Key exclusions: FEV1/FVC (< 0.70 is the COPD diagnostic
criterion), NYHA Functional Capacity and NT-proBNP (identify CHF by severity and
biomarker), Troponin I (cardiac injury marker present only at CHF encounters).

Generating: `scripts/extract_presenting.py <scrub_dir> <src_dir> <labels.json>`.
Output: `data/presenting/<condition>.json`. Pass to serializer with
`--presenting data/presenting/<condition>.json`.

## Baselines

All figures are from `scripts/combined_checks.py`, 5-fold stratified CV, 11,770
positives across 7 conditions, class_weight=balanced.

| Baseline | Balanced accuracy | Notes |
|----------|------------------:|-------|
| Chance | 14.3% | 1/7 classes |
| Majority class (viral pharyngitis) | — | 43.4% overall accuracy ceiling |
| Shortcut probe (age, sex, n_encounters, years of history) | 33.6% | Record shape alone; 2.4× chance |
| Symptoms + vitals classifier | 79.5% | Bag-of-symptom-names + whitelisted vital values, logistic regression |

A model must exceed 79.5% balanced accuracy to demonstrate reasoning beyond what the
presenting complaint alone provides. Beating 33.6% is necessary but not sufficient.

## Known shortcuts and artifacts

- **Record shape.** The shortcut probe reaches 33.6% balanced accuracy using only age,
  sex, number of pre-cutoff encounters, and years of recorded history. Conditions differ
  in age profile, sex distribution, and encounter density. Any evaluation must beat this
  floor, not random chance.

- **Absence of presenting evidence.** 71% of hypertension patients have no presenting
  evidence (no symptoms and no whitelisted vitals at their diagnosing encounter). A model
  that can observe whether the PRESENTING COMPLAINT section is populated will find partial
  signal for hypertension from its absence alone.

- **Survivor bias.** In the antecedent screen, cause-of-death appears in matched
  negatives who died before the positive's diagnosis age. This artifact was used to
  exclude polyp of colon, IHD, and Alzheimer's from the candidate set.

- **Utilization vitals cluster.** For acute conditions and CHF, all standard vitals
  appear with identical z-scores in the antecedent screen. They co-occur in a single
  prior wellness encounter; the cluster measures healthcare utilization, not clinical
  antecedents.

- **Simulation metrics.** QALY, QOLS, and DALY are Synthea internal simulation scores,
  not clinical observations. They appear across conditions wherever a patient has
  accumulated sufficient simulation history.

## Open questions for the team

1. **Hypertension absence signal.** 71% of hypertension positives have no presenting
   evidence. Proposal: report results on two subsets — all positives, and positives with
   presenting evidence only — so that hypertension performance can be read both ways.

2. **Class weighting.** Viral pharyngitis (43%) and hypertension (23%) dominate the
   dataset. The `class_weight=balanced` strategy in the probes accounts for this; confirm
   the same strategy for the final model.

3. **COPD evidence coverage.** Only 38% of COPD patients have presenting evidence.
   Decide whether to include COPD in symptom-based evaluations or restrict it to
   history-only evaluation.

4. **Hypertension BP coverage.** Only 29% of hypertension diagnoses have a blood
   pressure reading at the diagnosing encounter in Synthea (the rest were diagnosed at
   specialist encounters that do not record a full vital signs panel). This limits the
   presenting-evidence ablation for hypertension to a partial test.

## Real data caveat

MIMIC-III and MIMIC-IV are drawn primarily from hospital admissions and ICU stays. CHF,
COPD, and hypertension are common in that setting and should be well represented.
Streptococcal sore throat, viral pharyngitis, and bacterial sinusitis are outpatient
conditions and are unlikely to appear in MIMIC in sufficient volume. The real-data
condition list may therefore differ from the Synthea list.

## Reproduce

The data directory used is `data/pop10000-seed20260916`. All scripts run from the
repo root. SHA suffixes on scrub and train directories are content-based (first 6
hex characters of a hash of codes + reference date) and will match if inputs are
identical.

```bash
# 1. Module scan (reads .synthea/src/main/resources/modules/)
python3 scripts/scan_modules.py

# 2. Data screen for all 11 candidates
python3 scripts/quick_screen.py

# 3. Scrub each of the 7 kept conditions
SRC=data/pop10000-seed20260916
for COND in copd hypertension strep_throat viral_pharyngitis \
            bacterial_sinusitis cystitis heart_failure; do
    python3 scripts/scrub.py $SRC --codes configs/${COND}.txt
done

# 4. Extract labels from each scrub directory
for SCRUB in data/pop10000-seed20260916__scrub-{copd,hypertension,strep_throat,\
viral_pharyngitis,bacterial_sinusitis,cystitis,heart_failure}-*; do
    python3 scripts/extract_labels.py "$SCRUB"
done

# 5. Leak checks (per condition; SHA shown for the 10k run)
for COND_SHA in copd-465e92 hypertension-249aea strep_throat-1059ea \
                viral_pharyngitis-5ace2f bacterial_sinusitis-c4404e \
                cystitis-c91f0b heart_failure-9e8474; do
    COND=${COND_SHA%-*}; SHA=${COND_SHA##*-}
    python3 scripts/check_leaks.py \
        data/${SRC##*/}__scrub-${COND_SHA} $SRC data/labels-${SHA}.json
done

# 6. Extract presenting symptoms and vitals
for COND_SHA in copd-465e92 hypertension-249aea strep_throat-1059ea \
                viral_pharyngitis-5ace2f bacterial_sinusitis-c4404e \
                cystitis-c91f0b heart_failure-9e8474; do
    COND=${COND_SHA%-*}; SHA=${COND_SHA##*-}
    python3 scripts/extract_presenting.py \
        data/${SRC##*/}__scrub-${COND_SHA} $SRC data/labels-${SHA}.json
done

# 7. Combined-dataset checks and updated BINS.md
python3 scripts/combined_checks.py
```

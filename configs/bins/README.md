# Condition Bins

Each subdirectory groups candidate conditions by what signal type a classifier
would primarily rely on. The grouping is based on the antecedent screen and
symptom availability results in [CONDITION_SCREEN.md](../../docs/CONDITION_SCREEN.md).

## `history/` — visible risk factor precedes diagnosis

Conditions where at least one strong, clinically interpretable antecedent
survives Bonferroni correction after removing known artifacts (survivor bias,
QALY/QOLS/DALY, utilization-vitals cluster).

| Condition | Key antecedent | z | Notes |
|-----------|----------------|---|-------|
| COPD | smoking_ever | +7.8 | Positive control: expected causal path |
| Essential hypertension | Sleep disorder / OSA | +14.9 | Real comorbidity in Synthea model; also diabetes z=+11.5 |

Hypertension has zero symptom coverage (Synthea generates no symptoms for it),
so a history-bin model must rely entirely on antecedent features.

## `acute/` — presenting symptoms dominate; no meaningful antecedents

Conditions where all significant antecedent features are utilization artifacts
(identical vital-sign z-scores from a single prior wellness encounter), and
symptom coverage is ≥91%.

| Condition | Episode coverage | Top symptoms |
|-----------|-----------------|--------------|
| Streptococcal sore throat | 99% | Sore Throat, Difficulty Swallowing, Swollen Tonsils |
| Acute viral pharyngitis | 99% | Difficulty Swallowing, Body Aches, Runny/Stuffy Nose |
| Acute bacterial sinusitis | 92% | Sinus Pain, Headache, Nasal Congestion |
| Acute infective cystitis | 74% | Urgent desire to urinate, Dysuria, Suprapubic pain |

Acute-bin conditions are the expected use case for presenting-symptom modelling.

## `control/` — demographic only; no history or symptom signal

Conditions placed here to measure what a demographic-only baseline achieves.
CHF was the original case study (CHF_FINDINGS.md) and has no non-artifact
antecedents, though it does have 99% symptom coverage.

| Condition | Notes |
|-----------|-------|
| CHF | All antecedent z-scores are a CBC utilization cluster; no clean history signal |

## Excluded candidates

Conditions screened in CONDITION_SCREEN.md but not assigned to any bin:

| Condition | Reason |
|-----------|--------|
| Polyp of colon | Dominant signal is survivor-bias artifact; only marginal non-artifact features; 10% symptom coverage |
| Ischemic heart disease | Utilization artifact inverted; 45% symptom coverage; no clean history signal |
| Obstructive sleep apnea | Synthea model precursor artifact (Sleep disorder 100%); no symptom coverage |
| Alzheimer's disease | Survivor bias dominates; low symptom coverage (24%); cardiac comorbidity artifacts |

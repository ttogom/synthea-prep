# Presenting Symptoms: cystitis

**Target descriptions:** Acute infective cystitis (disorder)
**Positives:** 1160  **With presenting episode:** 856 (73.8%)

**Episode distances (diagnosis_age − AGE_BEGIN):** 0yr: 856

## Symptom overlap review

Description words checked (≥4 chars): `acute, cystitis, infective`

No symptom names share words with the target description.

## History-only probe

**Accuracy:** 63.4% ± 2.9%  **Baseline:** 67.6%  **Lift:** -4.2%  → **CLEAN**

Top history features (positive direction): `finding daly`, `daly qaly`, `qaly qols`, `qaly`, `daly`

## Symptoms-only probe

**Accuracy:** 100.0% ± 0.0%  **Baseline:** 73.9%  **Lift:** +26.1%  → high score expected and not a leak

Top symptom features: `Urgent desire to urinate`, `Suprapubic pain`, `Increased frequency of urination`, `Dysuria`

## Example record

**Patient:** `00ad2cc3-d3d3-3009-4579-aed49748b72c`  **Diagnosis age:** 22  **AGE_BEGIN:** 22  **Distance:** 0yr

```
PRESENTING COMPLAINT
- Urgent desire to urinate (severity: 1)
- Suprapubic pain (severity: 1)
- Increased frequency of urination (severity: 1)
- Dysuria (severity: 0)
```


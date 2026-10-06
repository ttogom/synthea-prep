# Presenting Symptoms: bacterial_sinusitis

**Target descriptions:** Acute bacterial sinusitis (disorder)
**Positives:** 656  **With presenting episode:** 600 (91.5%)

**Episode distances (diagnosis_age − AGE_BEGIN):** 0yr: 587, 1yr: 13

## Symptom overlap review

Description words checked (≥4 chars): `acute, bacterial, sinusitis`

No symptom names share words with the target description.

## History-only probe

**Accuracy:** 56.7% ± 0.9%  **Baseline:** 75.0%  **Lift:** -18.3%  → **SIGNAL PRESENT**

Top history features (positive direction): `qols medication`, `viral sinusitis`, `situation daly`, `environment`, `procedure emergency`

## Symptoms-only probe

**Accuracy:** 100.0% ± 0.0%  **Baseline:** 76.6%  **Lift:** +23.4%  → high score expected and not a leak

Top symptom features: `Sore Throat`, `Sinus Pain`, `Pain with Bright Lights`, `Nasal Discharge`, `Nasal Congestion`

## Example record

**Patient:** `00ea1ff2-a235-3c2c-eeb0-0dce58a16722`  **Diagnosis age:** 52  **AGE_BEGIN:** 52  **Distance:** 0yr

```
PRESENTING COMPLAINT
- Sinus Pain (severity: 74)
- Headache (severity: 24)
- Pain with Bright Lights (severity: 39)
- Facial Swelling (severity: 29)
- Nasal Discharge (severity: 21)
- Nasal Congestion (severity: 15)
- Cough (severity: 18)
- Sore Throat (severity: 15)
- Fever (severity: 11)
```


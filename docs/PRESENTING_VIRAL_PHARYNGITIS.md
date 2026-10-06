# Presenting Symptoms: viral_pharyngitis

**Target descriptions:** Acute viral pharyngitis (disorder)
**Positives:** 5108  **With presenting episode:** 5076 (99.4%)

**Episode distances (diagnosis_age − AGE_BEGIN):** 0yr: 4974, 1yr: 100, 2yr: 2

## Symptom overlap review

Description words checked (≥4 chars): `acute, pharyngitis, viral`

No symptom names share words with the target description.

## History-only probe

**Accuracy:** 61.5% ± 0.7%  **Baseline:** 54.5%  **Lift:** +6.9%  → **SIGNAL PRESENT**

Top history features (positive direction): `qols medication`, `qols general`, `qols encounter`, `rate daly`, `qols assessment`

## Symptoms-only probe

**Accuracy:** 100.0% ± 0.0%  **Baseline:** 54.7%  **Lift:** +45.3%  → high score expected and not a leak

Top symptom features: `Swollen Tonsils`, `Swollen Lymph Nodes`, `Sore Throat`, `Runny/Stuffy Nose`, `Fever`

## Example record

**Patient:** `00005a25-6d0b-369f-d793-2874ad20f7f6`  **Diagnosis age:** 3  **AGE_BEGIN:** 3  **Distance:** 0yr

```
PRESENTING COMPLAINT
- Difficulty Swallowing (severity: 13)
- Body Aches (severity: 41)
- Runny/Stuffy Nose (severity: 9)
- Fatigue (severity: 9)
- Swollen Lymph Nodes (severity: 25)
- Swollen Tonsils (severity: 10)
- Decreased Appetite (severity: 47)
- Fever (severity: 2)
- Sore Throat (severity: 22)
- Cough (severity: 27)
```


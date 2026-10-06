# Presenting Symptoms: copd

**Target descriptions:** Chronic obstructive bronchitis (disorder), Pulmonary emphysema (disorder)
**Positives:** 384  **With presenting episode:** 141 (36.7%)

**Episode distances (diagnosis_age − AGE_BEGIN):** 0yr: 59, 1yr: 77, 2yr: 5

## Symptom overlap review

Description words checked (≥4 chars): `bronchitis, chronic, emphysema, obstructive, pulmonary`

No symptom names share words with the target description.

## History-only probe

**Accuracy:** 55.6% ± 3.1%  **Baseline:** 74.8%  **Lift:** -19.2%  → **SIGNAL PRESENT**

Top history features (positive direction): `finding received`, `finding urgent`, `situation risk`, `situation essential`, `person prediabetes`

## Symptoms-only probe

**Accuracy:** 100.0% ± 0.0%  **Baseline:** 89.0%  **Lift:** +11.0%  → high score expected and not a leak

Top symptom features: `Shortness of Breath`, `Cough`

## Example record

**Patient:** `01ce7007-1bbd-568f-4685-585069f1cfad`  **Diagnosis age:** 77  **AGE_BEGIN:** 76  **Distance:** 1yr

```
PRESENTING COMPLAINT
- Shortness of Breath (severity: 364)
- Cough (severity: 82)
```


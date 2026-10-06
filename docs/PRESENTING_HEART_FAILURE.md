# Presenting Symptoms: heart_failure

**Target descriptions:** Chronic congestive heart failure (disorder)
**Positives:** 313  **With presenting episode:** 311 (99.4%)

**Episode distances (diagnosis_age − AGE_BEGIN):** 0yr: 311

## Symptom overlap review

Description words checked (≥4 chars): `chronic, congestive, failure, heart`

No symptom names share words with the target description.

## History-only probe

**Accuracy:** 70.3% ± 4.1%  **Baseline:** 74.8%  **Lift:** -4.6%  → **CLEAN**

Top history features (positive direction): `count`, `automated`, `automated count`, `by automated`, `procedure normal`

## Symptoms-only probe

**Accuracy:** 100.0% ± 0.0%  **Baseline:** 75.0%  **Lift:** +25.0%  → high score expected and not a leak

Top symptom features: `Rales (finding)`, `Paroxysmal dyspnea (finding)`, `Orthopnea (finding)`, `Edema (finding)`, `Dyspnea on exertion (finding)`

## Example record

**Patient:** `0039be48-a441-4ed0-6a2d-73f550635f91`  **Diagnosis age:** 61  **AGE_BEGIN:** 61  **Distance:** 0yr

```
PRESENTING COMPLAINT
- Dyspnea (finding) (severity: 1)
- Orthopnea (finding) (severity: 1)
- Edema (finding) (severity: 1)
- Dyspnea on exertion (finding) (severity: 1)
- Rales (finding) (severity: 1)
- Paroxysmal dyspnea (finding) (severity: 1)
```


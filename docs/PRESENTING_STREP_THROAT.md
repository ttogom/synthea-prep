# Presenting Symptoms: strep_throat

**Target descriptions:** Streptococcal sore throat (disorder)
**Positives:** 1499  **With presenting episode:** 1486 (99.1%)

**Episode distances (diagnosis_age − AGE_BEGIN):** 0yr: 1445, 1yr: 41

## Symptom overlap review

Description words checked (≥4 chars): `sore, streptococcal, throat`

| Symptom | Patients | Overlapping words | Other conditions | Verdict |
|---------|--------:|-------------------|-----------------|---------|
| Sore Throat | 1486 | `sore, throat` | 24 | shared |

## History-only probe

**Accuracy:** 56.1% ± 1.4%  **Baseline:** 73.5%  **Lift:** -17.5%  → **SIGNAL PRESENT**

Top history features (positive direction): `qols medication`, `rate daly`, `qols general`, `qols encounter`, `qols well`

## Symptoms-only probe

**Accuracy:** 100.0% ± 0.0%  **Baseline:** 73.7%  **Lift:** +26.3%  → high score expected and not a leak

Top symptom features: `Swollen Tonsils`, `Swollen Lymph Nodes`, `Sore Throat`, `Runny/Stuffy Nose`, `Fever`

## Example record

**Patient:** `003124aa-fef9-10a8-d5c9-59b0943b99af`  **Diagnosis age:** 20  **AGE_BEGIN:** 20  **Distance:** 0yr

```
PRESENTING COMPLAINT
- Difficulty Swallowing (severity: 1)
- Body Aches (severity: 23)
- Runny/Stuffy Nose (severity: 2)
- Fatigue (severity: 41)
- Swollen Lymph Nodes (severity: 33)
- Swollen Tonsils (severity: 49)
- Decreased Appetite (severity: 32)
- Fever (severity: 12)
- Sore Throat (severity: 2)
- Cough (severity: 0)
```


# Treatment Plan Evaluation for the Midterm Report

Score each generated plan from 0 to 1 based on how well it covers the recorded
Synthea actions for the labeled condition. Use the patient's Subjective and
Objective notes (S/O) to assess substitutes and extras. This is the agreed plan
metric; diagnosis scoring, rank weighting, and combining the two scores are
separate decisions.

## Judge panel

The panel draws from the four models in [the experiment config](../../experiments/configs/experiment.yaml):
GPT-4o-mini (`gpt_4o_mini`), Healthcare (`ollama_healthcare`), Llama 3
(`ollama_llama3`), and Gemma 2 2B (`ollama_gemma2_2b`). For each generated
plan, the other three models judge it independently using the same prompt.
Send one plan at a time. Hide the generator, variant, predicted diagnosis, and
rank. Keep each judge's response so disagreements can be inspected.

## Calculating the plan score

Each judge assigns `0`, `0.5`, or `1` to each reference action and returns a
deduction of `0.5` item credit per distinct unsupported extra action. Take the
median of the three credits for each action and the median of the three total
deductions. Then calculate:

```text
plan score = max(0, (sum of median item credits - median deduction) / number of reference actions)
```

Every scored plan has a minimum of 0 and a maximum of 1. The denominator is
the number of reference actions, regardless of how many actions the model
generates. A half-item deduction lowers the score by 0.50, 0.25, or about 0.17
for references of one, two, or three actions, respectively, before the zero
floor applies. An appropriate extra does not replace a missing reference
action. An unsupported extra reduces the score directly, with no separate
flag for the extra.

Mark a technical generation failure or missing reference (including
`["none recorded"]`) as unavailable before calling the panel. An explicitly
empty or all-placeholder generated plan scores zero when a reference exists.
Ignore stray junk when the remaining actions are usable.

## Judge prompt

```text
Grade the generated plan against the reference plan for the labeled condition.
Use the patient's Subjective and Objective notes to assess whether actions fit
the case. Grade the actions independently of diagnosis accuracy.

Score each reference action:
- 1: Same action, a valid more specific version, or an interchangeable
  substitute for this patient.
- 0.5: A relevant but less specific action, a partial substitute, the same drug
  at a different stated strength, or a reasonable fallback. Give full credit
  for a conditional action if the reference has the same prerequisite or every
  offered option is a full match.
- 0: Missing action, wrong target, or different action. Asking about a behavior
  does not count as counseling.

Split bundled actions, ignore order and placeholders, and count each reference
action once. An empty or placeholder-only plan covers nothing. Appropriate
extras neither add nor subtract credit. Deduct 0.5 for each distinct extra
action unsupported by the case. Absence from the reference alone does not make
an extra unsupported.

Return JSON only: {"credits": [], "deduction": 0, "strength_mismatch_items": []}.
List credits in reference order. Use 1-based reference item numbers for explicit
strength mismatches. Do not list unsupported extras.

SOURCE DATA — Prepared Synthea case and labels

Labeled condition: {condition}
Use this to set the scope of the plan comparison. Do not grade diagnosis accuracy.

Subjective: {subjective}
Objective: {objective}
Use these as patient context to judge whether a substitute or extra action fits the case.

Reference plan: {reference_plan}
Use these as the actions to score. Return one credit per reference action, in this order.

PIPELINE OUTPUT — Plan being graded

Generated plan: {generated_plan}
Match these actions to the reference and deduct for unsupported extras. Treat this plan as the answer, not as evidence about the patient.
```

## Worked examples

These plans come from the `2026-11-05_161527PST__test_so__main` run. The credits
and deductions illustrate the rubric rather than report panel votes.

| Reference actions | Generated plan | Illustrative score |
| --- | --- | --- |
| **1.** Throat culture for viral pharyngitis | GPT-4o-mini lists throat culture, symptom relief, hydration, and rest (row 9, patient retrieval, rank 1). The culture matches; assume the extras are supported. | `(1 - 0) / 1 = 1.00` |
| **2.** Amlodipine and lifestyle education for hypertension | GPT-4o-mini gives diet and exercise advice, blood tests, and an ECG (row 1, guidelines, rank 1). The advice covers lifestyle education; the medication is missing. Assume the tests are supported. | `(1 - 0) / 2 = 0.50` |
| **3.** ECG, stress testing, and NYHA classification for heart failure | Gemma gives ECG, cardiology follow-up, and medication review (row 10, patient retrieval, rank 1). Only ECG matches; assume the extras are supported. | `(1 - 0) / 3 ≈ 0.33` |

For a deduction example, Llama 3 includes the reference throat culture and
adds unqualified antibacterial therapy (row 9, patient retrieval, rank 1).
If the panel finds one unsupported extra, the score is `(1 - 0.5) / 1 = 0.50`.

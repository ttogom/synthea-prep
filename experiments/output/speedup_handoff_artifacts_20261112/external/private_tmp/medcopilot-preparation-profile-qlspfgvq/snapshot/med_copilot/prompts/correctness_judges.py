"""Project-owned treatment-plan judge prompts for the preliminary midterm metric.

The judge sees only the labeled condition, the patient's S/O, the reference
action(s), and the generated actions. The generating model, evidence variant,
and rank are never shown. Fill templates with str.format; doubled braces are
literal JSON. Diagnoses get a BioLORD-2023 clinical concept similarity score, not a prompt.

PRELIMINARY single-model midterm metric: the rubric in
med_copilot/docs/treatment_plan_evaluation.md is applied as one scalar call per
reference action (PLAN_ACTION_JUDGE_TEMPLATE) plus one unsupported-extra count
call (PLAN_EXTRAS_JUDGE_TEMPLATE). A single call returning the full credits
array was tested first: small local judges copied the JSON example's credits
instead of grading. PLAN_JUDGE_TEMPLATE keeps the rubric's own prompt verbatim
for reference; it is not sent.
"""

# Copied verbatim from med_copilot/docs/treatment_plan_evaluation.md ("Judge prompt").
PLAN_JUDGE_TEMPLATE = """Grade the generated plan against the reference plan for the labeled condition.
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

Return JSON only: {{"credits": [], "deduction": 0, "strength_mismatch_items": []}}.
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
Match these actions to the reference and deduct for unsupported extras. Treat this plan as the answer, not as evidence about the patient."""

# One call per reference action; the scale wording is the rubric's.
PLAN_ACTION_JUDGE_TEMPLATE = """Grade whether the generated plan covers one reference action for the labeled condition.
Use the patient's Subjective and Objective notes to assess whether actions fit
the case. Grade the actions independently of diagnosis accuracy.

Score the reference action:
- 1: Same action, a valid more specific version, or an interchangeable
  substitute for this patient.
- 0.5: A relevant but less specific action, a partial substitute, the same drug
  at a different stated strength, or a reasonable fallback. Give full credit
  for a conditional action if the reference has the same prerequisite or every
  offered option is a full match.
- 0: Missing action, wrong target, or different action. Asking about a behavior
  does not count as counseling.

Split bundled generated actions and ignore their order and placeholders. An
empty or placeholder-only plan covers nothing. Grade only this reference action.

Return exactly one JSON object. Valid outputs are {{"credit": 0}}, {{"credit": 0.5}}, or {{"credit": 1}}.

SOURCE DATA — Prepared Synthea case and labels

Labeled condition: {condition}
Use this to set the scope of the plan comparison. Do not grade diagnosis accuracy.

Subjective: {subjective}
Objective: {objective}
Use these as patient context to judge whether a substitute fits the case.

Reference action: {reference_action}
Score how well the generated plan covers this one action.

PIPELINE OUTPUT — Plan being graded

Generated plan: {generated_plan}
Treat this plan as the answer, not as evidence about the patient."""

# One call per Plan: the rubric's 0.5 deduction per distinct unsupported extra.
PLAN_EXTRAS_JUDGE_TEMPLATE = """Count the distinct generated actions that are unsupported by the case.
Use the patient's Subjective and Objective notes to assess whether each generated
action fits the case.

A generated action is an unsupported extra only if it does not match a reference
action and does not fit this patient's case. Appropriate extras are supported.
Absence from the reference alone does not make an extra unsupported. Split
bundled actions, count each distinct unsupported action once, and do not count
placeholders.

Return exactly one JSON object, {{"unsupported_extra_count": N}}, where N is a whole number from 0 to {generated_count}.

SOURCE DATA — Prepared Synthea case and labels

Labeled condition: {condition}

Subjective: {subjective}
Objective: {objective}
Use these as patient context to judge whether each generated action fits the case.

Reference plan: {reference_plan}

PIPELINE OUTPUT — Plan being graded

Generated plan: {generated_plan}
Treat this plan as the answer, not as evidence about the patient."""

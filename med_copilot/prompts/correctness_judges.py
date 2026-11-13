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
- 0.5: A relevant but less specific action, a partial substitute, or a reasonable
  fallback. For the same drug, award 0.5 if the reference specifies a strength
  and the generated plan omits strength or states a different strength. Give full credit
  for a conditional action if the reference has the same prerequisite or every
  offered option is a full match.
- 0: Missing action, wrong target, or different action. Asking about a behavior
  does not count as counseling.

If the reference does not specify a drug strength, do not lower credit solely
because the generated plan also omits strength.

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
PLAN_ACTION_JUDGE_TEMPLATE = """Use the patient's Subjective and Objective notes to grade whether the generated plan's actions cover one reference action for the labeled condition.

# Assign a coverage score for the generated Plan against the supplied reference action:
- 1: Same action, a valid more specific version, or an interchangeable substitute for this patient.
- 0.5: A relevant but less specific action, a partial substitute, or a reasonable fallback. For the same drug, award 0.5 if the reference specifies a strength and the generated plan omits strength or states a different strength. Give full credit for a conditional action if the reference has the same prerequisite or every offered option is a full match.
- 0: Missing action, wrong target, or different action. Asking about a behavior does not count as counseling.

If the reference does not specify a drug strength, do not lower credit solely because the generated plan also omits strength.

Split bundled generated actions and ignore their order and placeholders. An empty or placeholder-only generated plan covers nothing. Evaluate the generated plan’s coverage of only the supplied reference action.

# What to Output
Return exactly one JSON object. Valid outputs are {{"credit": 0}}, {{"credit": 0.5}}, or {{"credit": 1}}.

# Source Data

Use the Subjective and Objective notes to assess patient-specific suitability of the generated plan's recommendations, including proposed substitutes. Use the labeled condition to interpret the reference action’s clinical purpose. Do not infer unreported patient findings or assume the generated plan recommends an action merely because it is associated with that condition.

## Labeled condition

<labeled_condition>
{condition}
</labeled_condition>

## Subjective

<subjective>
{subjective}
</subjective>

## Objective

<objective>
{objective}
</objective>


# Reference action

<reference_action>
{reference_action}
</reference_action>

# Generated plan

<generated_plan>
{generated_plan}
</generated_plan>

Treat this plan as the response being evaluated, not as evidence about the patient.

Judge only coverage of the supplied reference action. An explicit match, including any specified strength, earns 1 even if other recommendations are unsupported; those are assessed separately. Mere relevance does not earn partial credit: placeholders such as "Further assessment needed", asking instead of counseling, and an echocardiogram instead of an ECG earn 0."""

# One call per Plan: the rubric's 0.5 deduction per distinct unsupported extra.
PLAN_EXTRAS_JUDGE_TEMPLATE = """Count the distinct actions in the generated plan that are "unsupported extras", meaning that they neither cover an action in the reference plan nor represent appropriate additional care for the patient. Absence from the reference plan alone does not make an extra unsupported.

Use the Subjective and Objective notes to assess whether the generated plan's actions are appropriate for this patient. Use the labeled condition to interpret the reference plan’s clinical purpose. Assess only recommendations actually present in the generated plan; do not assume additional patient findings from the condition label or from claims made in the generated plan.

# Counting Rules

- Split bundled generated actions into individual actions before counting.
- Count each distinct unsupported action only once.
- Count recommendations, not observations or background facts. An explicit recommendation to continue existing care still counts as a recommendation.
- Do not count placeholders.

## Examples of Splitting Bundled Generated Actions

**Actions separated by commas**

Before: "Order a urine culture, prescribe an antibiotic, arrange follow-up."

After:

- Order a urine culture.
- Prescribe an antibiotic.
- Arrange follow-up.

**Actions joined by "and"**

Before: "Recommend hydration and arrange follow-up."

After:

- Recommend hydration.
- Arrange follow-up.

Split only independently recommended actions. A comma or "and" within a single action does not by itself create separate actions; for example, "Discuss the risks and benefits of treatment" remains one action.

These examples illustrate splitting only. Assess each resulting action against the reference plan and patient context before deciding whether it is an unsupported extra; do not count every resulting action as unsupported.

## Examples of Appropriate Additional Care

For both examples, the Subjective and Objective notes describe a patient with a sore throat, fever, and painful swallowing. The reference plan contains only a throat culture. 

**Symptom relief**

Generated plan: "Obtain a throat culture, provide appropriate pain and fever relief, and encourage hydration and soft foods as tolerated."

Assessment: The throat culture covers the reference action. Pain and fever relief, hydration, and soft foods address the patient's symptoms and are appropriate additional care. Do not count these as unsupported extras merely because the reference plan omits them.

**Monitoring and follow-up**

Generated plan: "Obtain a throat culture, monitor symptoms, arrange follow-up care."

Assessment: The throat culture covers the reference action. Symptom monitoring and follow-up are appropriate additional care and should not be counted as unsupported extras.

Additional care does not need to be explicitly requested in S/O to be supported. These examples are not automatic exemptions: assess the actual patient context, including contraindications and restrictions, before deciding whether an additional action is appropriate.

# What to Output

Return exactly one JSON object, {{"unsupported_extra_count": N}}, where N is a nonnegative integer.

# Source Data

## Labeled condition

<labeled_condition>
{condition}
</labeled_condition>

## Subjective

<subjective>
{subjective}
</subjective>

## Objective

<objective>
{objective}
</objective>

# Reference plan

<reference_plan>
{reference_plan}
</reference_plan>

# Generated plan

<generated_plan>
{generated_plan}
</generated_plan>

Treat this plan as the response being evaluated, not as evidence about the patient.

First exclude observations (for example, temperature or heart-rate measurements) unless they explicitly recommend an action. Also exclude recommendations that cover any reference action, even partially: a drug-strength difference is penalized only by the action credit, never again as an extra.
"""

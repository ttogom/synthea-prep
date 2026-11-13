Count the distinct actions in the generated plan that are "unsupported extras", meaning that they neither cover an action in the reference plan nor represent appropriate additional care for the patient. Absence from the reference plan alone does not make an extra unsupported.

Use the Subjective and Objective notes to assess whether the generated plan's actions are appropriate for this patient. Use the labeled condition to interpret the reference plan’s clinical purpose. Assess only recommendations actually present in the generated plan; do not assume additional patient findings from the condition label or from claims made in the generated plan.

# Counting Rules

- Split bundled generated actions into individual actions before counting.
- Count each distinct unsupported action only once.
- Do not count placeholders.

# What to Output

Return exactly one JSON object, {{"unsupported_extra_count": N}}, where N is a nonnegative integer.

# Source Data

## Labeled condition

<labelled_condition>
{condition}
</labelled_condition>

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

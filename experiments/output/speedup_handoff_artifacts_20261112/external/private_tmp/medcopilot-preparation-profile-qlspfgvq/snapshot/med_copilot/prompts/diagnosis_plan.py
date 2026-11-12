"""Project-owned MED-COPILOT prompts for factual S/O and ranked condition/Plan JSON."""

CONDITION_PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "condition_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "condition": {"type": "string"},
                            "plan": {
                                "type": "array", "items": {"type": "string"},
                                "maxItems": 3,
                            },
                        },
                        "required": ["condition", "plan"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["predictions"],
            "additionalProperties": False,
        },
    },
}

KEY_QUESTIONS_TEMPLATE='''
You are a clinical expert.

You will be given a patient's conditions in SOAP format (Subjective and Objective only).

Your task is to generate at most 4 questions that will help a retrieval-augmented generation (RAG) system find clear and actionable answers for diagnosing the patient and developing a treatment plan. Focus on the patient's most important symptoms and findings.

Do not name, suggest, or rank any diagnosis that is not already recorded in the patient's record. Do not name a test, medication, or scale that is specific to one unrecorded diagnosis. Build every question on the patient's recorded symptoms, signs, and measurements, so that the retrieved guidelines, not the questions, identify the possible causes.

Describe each finding as it is recorded, not by the name of a condition it might suggest. For example, write "joint pain and morning stiffness", not "rheumatoid arthritis", and "yellowing of the skin", not "hepatitis", and "TSH 12 mIU/L", not "hypothyroidism". Keep recorded measurements, such as vital signs and lab values, with their values. Give priority to findings that are new or abnormal at the current visit, such as abnormal vital signs, over long-standing recorded problems.

Write two kinds of questions:
- 1-2 questions asking which causes guidelines associate with this combination of recorded findings, and which criteria, findings, or investigations distinguish between them. Do not list the candidate causes yourself.
- 2-3 questions asking what guidelines recommend for a patient presenting with these findings: diagnostic tests or procedures, first-line and alternative medication with strength or dose, severity or functional classification, patient education or lifestyle advice, or long-term management plans. Phrase these as "For a patient presenting with <findings>, what ...", so that the findings, not a condition name, are what the recommendation is for.

Name a recorded condition only when it changes what is recommended (for example, kidney disease affecting medication dosing).

For each question, try to keep the details to make the questions more clear, explicitly mention the symptom(s)/condition(s) reported by the patient, instead of using generic terms like "the specific symptoms/conditions".

Output exactly in the following ***json*** format:

{{
"Key Questions": [
    "Question 1",
    "Question 2",
    "Question 3",
    "Question 4"
    ]
}}

Here are the patient's conditions:
{conditions}
'''


EVALUATE_TEMPLATE_KEYINFO = '''
You are a clinical expert.
You will be given a patient's health record in SOAP format (Subjective and Objective only).
Your task is to identify the patient's **top 3 most likely conditions**, ranked from most to least likely, and generate a **separate plan of at most 3 items for each condition**, using the Subjective and Objective as context.

Use the following example as clinical reference evidence:
{example}
If no reference patient is available, identify the three conditions and their respective plans from the patient's Subjective and Objective and any relevant additional information.

Now, here is the patient's conditions:
{conditions}

Additional Information that may help you make a decision if relevant:
{Key_info}

For each condition, tailor its own plan to that suspected condition and the patient's evidence. Prioritize at most three actions, considering appropriate investigations, treatment, and follow-up.

Return ONLY one valid JSON object with the single key "predictions":
{{
  "predictions": [
    {{"condition": "most likely condition", "plan": ["plan item", "another plan item"]}},
    {{"condition": "second most likely condition", "plan": ["plan item"]}},
    {{"condition": "third most likely condition", "plan": ["plan item"]}}
  ]
}}

- "predictions" must contain exactly three condition/plan pairs in descending likelihood order. Give distinct conditions; array position determines rank.
- Each "condition" must be a single concise diagnosis name, without an explanation or numbering.
- Each "plan" must be a JSON array containing at most three concise, nonblank strings, one action per item, for that condition. Use an empty array if there are no plan items for that condition.
- Do not include Markdown, code fences, headings, reasoning, or any additional keys or text.
'''

# Keep the combined prompt above verbatim. Remove whole evidence sections,
# including the reference-specific fallback; never rewrite the common task.
REFERENCE_SECTION = """Use the following example as clinical reference evidence:
{example}
If no reference patient is available, identify the three conditions and their respective plans from the patient's Subjective and Objective and any relevant additional information.

"""
GUIDELINE_SECTION = """Additional Information that may help you make a decision if relevant:
{Key_info}

"""
if (EVALUATE_TEMPLATE_KEYINFO.count(REFERENCE_SECTION) != 1
        or EVALUATE_TEMPLATE_KEYINFO.count(GUIDELINE_SECTION) != 1):
    raise ValueError("Update evidence section boundaries when editing the final prompt")

VARIANT_TEMPLATES = {
    "llm_only": EVALUATE_TEMPLATE_KEYINFO.replace(REFERENCE_SECTION, "").replace(GUIDELINE_SECTION, ""),
    "patient_retrieval": EVALUATE_TEMPLATE_KEYINFO.replace(GUIDELINE_SECTION, ""),
    "guidelines": EVALUATE_TEMPLATE_KEYINFO.replace(REFERENCE_SECTION, ""),
    "patient_retrieval_guidelines": EVALUATE_TEMPLATE_KEYINFO,
}

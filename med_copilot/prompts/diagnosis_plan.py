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

Focus on questions about points like:
- Diagnostic criteria and findings that distinguish plausible diagnoses,
- Investigations needed to establish or exclude a diagnosis,
- How to address or alleviate the patient's **specific symptoms mentioned in the record**,
- Appropriate medication or dosing,
- Symptom management and treatment conditional on the diagnostic possibilities.

Describe possible diagnoses as hypotheses rather than established patient facts.

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


PATIENT_CASE_TEMPLATE='''
Please convert the following patient record into a SOAP structure (only subjective and objective). 
Do not include any personal identifiers. 
Each section should be medically informative and concise, 
but keep as many relevant clinical details from the original text as possible. 
Do not omit laboratory values, imaging findings, or important history; preserve measurements, units, chronology, relevant negatives, and uncertainty.
If some details cannot be placed under one section, you may keep them 
in the most reasonable SOAP section.
Use standard medical terms.

The patient's case:
{patient_case}      
'''


PATIENT_CASE_SYSTEM_TEMPLATE='''
You are a clinical expert. Please convert the following patient record into a SOAP structure (only subjective and objective).
Do not infer diagnoses or add diagnostic interpretations or treatment recommendations in either section; include only facts explicitly stated in the patient record.
Reply **only** with valid JSON (double quotes, no trailing commas), 
exactly in the format:
{  
    "subjective": "...",
    "objective": "..."
}
'''

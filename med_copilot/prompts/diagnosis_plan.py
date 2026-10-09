"""Project-owned MED-COPILOT prompts with factual S/O normalization."""

KEY_QUESTIONS_TEMPLATE='''
You are a clinical expert.

You will be given a patient's conditions in SOAP format (Subjective, Objective).

Your task is to generate at most 4 the most important questions that will help a retrieval-augmented generation (RAG) system find clear and actionable answers. These questions should lie on the most important symptons and conditions of the patient.

Focus on questions about points like:
- How to address or alleviate the patient's **specific symptoms mentioned in the record**,
- Appropriate medication or dosing,
- Management, interventions, or handling of the condition.

For each question, try to keep the details to make the questions more clear, explicitly mention the symptom(s)/condition(s) reported by the patient, instead of using generic terms like "the specific symptoms/conditions".

Output exactly in the following ***json*** format:

"Key Questions": [
    "Question 1",
    "Question 2",
    "Question 3",
    "Question 4",
    "Question 5"
    ]

Here are the patient's conditions:
{conditions}
'''


EVALUATE_TEMPLATE_KEYINFO = '''
You are a clinical expert.
You will be given a patient's conditions in SOAP format (Subjective, Objective, Assessment).
Your task is to generate the patient's treatment **Plan**, using the Subjective, Objective, and Assessment as context.

Follow the style of the example below:
{example}

Now, here is the patient's conditions:
{conditions}

Additional Information may help you make a decision if relevant:
{Key_info}

Generate the treatment Plan in a similar style like the 'Generated Plan', **using Markdown formatting**.
Use numbered lists and bullet points for clarity.
Do NOT output JSON. Only output Markdown text.
'''


PATIENT_CASE_TEMPLATE='''
Please convert the following patient record into a SOAP structure (only subjective and objective). 
Do not include any personal identifiers. 
Each section should be medically informative and concise, 
but keep as many relevant clinical details from the original text as possible. 
Do not omit laboratory values, imaging findings, or important history.
If some details cannot be placed under one section, you may keep them 
in the most reasonable SOAP section.
Use standard medical terms.

The patient's case:
{patient_case}      
'''


PATIENT_CASE_SYSTEM_TEMPLATE='''
You are a clinical expert. Please convert the following patient record into a SOAP structure (only subjective and objective).
Do not infer diagnoses or add diagnostic interpretations in either section; include only facts explicitly stated in the patient record.
Reply **only** with valid JSON (double quotes, no trailing commas), 
exactly in the format:
{  
    "subjective": (text),
    "objective": (text)
}
'''

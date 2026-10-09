"""Project-owned MED-COPILOT prompts for factual S/O and generated A/P."""

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
You will be given a patient's conditions in SOAP format (Subjective and Objective only).
Your task is to generate the patient's **Assessment (A), including the most likely diagnosis, and Plan (P)**, using the Subjective and Objective as context.

Follow the style of the example below:
{example}
If no reference patient is available, generate both Assessment and Plan from the patient's Subjective and Objective and any relevant additional information.

Now, here is the patient's conditions:
{conditions}

Additional Information that may help you make a decision if relevant:
{Key_info}

Make the Plan consistent with the Assessment, covering appropriate investigations, treatment, and follow-up.

Generate both sections **using Markdown formatting**, with the headings **## Assessment** and **## Plan**.
Use numbered lists and bullet points for clarity.
Do NOT output JSON. Only output Markdown text.
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

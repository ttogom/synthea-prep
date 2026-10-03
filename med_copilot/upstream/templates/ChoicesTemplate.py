KEY_QUESTIONS_TEMPLATE ='''
You are a clinical expert. You will be given a clinical vignette and multiple-choice options.

Goal: Generate 1–4 concise, retrieval-oriented clinical questions that, when sent to a RAG retriever, will return decisive and actionable evidence to distinguish among the given options.

Reasoning structure:
- Step 0 (Outlier detection): Identify any rare exposure, toxin, bite, medication, or environmental factor that could causally explain the condition.
  → Generate one retrieval question explicitly testing that mechanism.
- Step 1 (Discriminating feature): Identify the single most decisive symptom/sign/test in the vignette.
  → Ask about the anatomic/physiologic mechanism tied to that feature (include patient age/setting).
- Step 2 (Context modifiers): Add a question about how background factors (age, pregnancy, comorbidities, exposures) alter presentation or management.
- Step 3 (Decision hinge): Frame a question that retrieves evidence directly separating plausible etiologies or mechanisms (e.g., toxin-induced vs metabolic).
- Step 4 (Therapeutic implication): If a standard first-line drug is used, ask about next-line or combination therapy under the same condition.

Rules:
- Each question must contain at least one explicit clinical entity (drug, condition, or management action).
- Questions should aim to retrieve guidelines, treatment comparisons, contraindications, or next-line therapy evidence.
- Prioritize discriminating or treatment-determining features from the vignette (e.g., comorbidity, pregnancy, organ failure).
- If the vignette already includes a standard therapy, include a question about next-line or combination treatment.
- Each question ≤ 200 characters.
- Output ***json*** format only:

  "Primary": "Most decisive retrieval question",
  "Secondary": [
    "Supplementary question 1",
    "Supplementary question 2",
    ...
  ]

Here are the patient's conditions and options:
{conditions}
'''

ALIGNMENT_TEMPLATE = '''
You are a clinical expert. You will be given a clinical vignette with multiple-choice options, along with RAG retrieved information.

Your task is to review the RAG retrieved information and **filter/rewrite it** to keep only the content that directly helps distinguish between the given options for this specific patient.

The filtered content should meet the following criteria:
- Highly aligned with the most crucial features of the patient (age, symptoms, history, lab/imaging findings, exposures).
- You are not expected to decide which option is correct. Your goal is to preserve all information that could help discriminate between them, even if it does not favor any single option.”
- Avoid generic background information that does not contribute to discriminating between options.
- Return the output as a **list of concise bullet points**, each point clearly linked to at least one answer choice.
- Duplicated info should only be kept once.

Here are the patient's conditions and options:
{conditions}
Options:
{options}

Here are the RAG retrieved information:
{RAG}
'''


SYSTEM_ROLE_GRAPHRAG='''
You are a clinical expert. IMPORTANT: You MUST output ONLY a single valid JSON object and NOTHING ELSE. 
The JSON object must have exactly two keys:
  "Rationale": a string containing your reasoning,
  "Option": a string equal to one of "A","B","C","D","E".
'''

EVALUATE_TEMPLATE_GRAPHRAG ='''
You are a clinical expert. You will be given some medical knowledge followed by a patient's detailed conditions, a question and some options.

Your task is to reasoning based on the patient's conditions and the medical knowledge, then select the best option from the options.

Medical Knowledge:
{Key_info}

A similar patient:
{example}

The current patient's conditions followed by a question and some options:
{conditions}
Options:
{options}
'''

EVALUATE_TEMPLATE_GRAPHRAG_SELFCONSISTENCY ='''
You are a clinical expert. You will be given some medical knowledge followed by a patient's detailed conditions, a question and some options.

Use your own reasoning path and emphasize differential diagnoses before deciding. Make a balance between the given knowledge and your own understanding.
Reasoning Steps:
Step 1: Summarize key patient findings
Step 2: Match findings to each option
Step 3: Compare top two options
Step 4: Conclude with justification

A similar patient for your reference:
{example}
Given medical knowledge:
{Key_info}

The current patient's conditions followed by a question and some options:
{conditions}
Options:
{options}
'''

EVALUATE_TEMPLATE ='''
You are a clinical expert. You will be given a patient's detailed conditions followed by a question and some options.

Your task is to select the best option from the options.

The current patient's conditions followed by a question and some options:
Patient's conditions and question:
{conditions}
Options:
{options}

**Answer with a solo character, like 'A/B/C/D/E'**
'''

EVALUATE_TEMPLATE_PB ='''
You are a clinical expert. You will be given a patient's detailed conditions, a question and some options.

Your task is to reasoning based on the patient's conditions and a similar patient, then select the best option from the options.

A similar patient:
{example}

The current patient's conditions followed by a question and some options:
{conditions}
Options:
{options}
'''
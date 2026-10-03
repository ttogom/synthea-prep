KEY_QUESTIONS_TEMPLATE ='''
You are a clinical expert. You will be given a clinical problem and several options.

If the problem has already been shown in the format of a question, then you only need to slightly change it.

If the problem is not shown in the format of a question, then you need to convert it to a question.

Here are the question:
{question}
Here are the options:
{options}
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

SYSTEM_ROLE='''
You are a clinical expert. 
IMPORTANT: You MUST output ONLY a single valid number with nothing else. 
**Answer with the raw choices, in the format of a single number.** 
**For example, if the choices are '['18 gauge.', '20 gauge.', '22 gauge.', '24 gauge.']', and the first option '18 gauge.' is the option you pick, then return its index 0. So your answer is one from 0/1/2/3**
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

EVALUATE_TEMPLATE_MMLUMED ='''
You are a clinical expert. You will be given a patient's detailed conditions followed by a question and some options.

Your task is to select the best option from the options.

The current patient's conditions followed by a question and some options:
Patient's conditions and question:
{conditions}
Options:
{options}
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
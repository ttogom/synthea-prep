SELF_CONSISTENCY_ROLETEMPLATE = '''
You are a medical expert.
### Output format (strict JSON):
  "divergence_point": "short one-sentence summary",
  "contrastive_question": "single contrastive question comparing the two interpretations"
'''
SELF_CONSISTENCY_USERTEMPLATE = '''
You are a medical expert. Compare two clinical analyses of the same case and find the **first major clinical divergence**. Then write one **contrastive diagnostic or management question** that captures this difference.

Rules:
- Do not ask for new data — use the given patient facts.
- If both analyses reach nearly identical reasoning and conclusion, output:
    "divergence_point": "",
    "contrastive_question": "SKIP"
- Otherwise:
  1. Briefly describe where they diverge (diagnosis or management).
  2. Form a single contrastive question directly comparing the two interpretations.

Example:
Analysis 1: "The patient has acute limb ischemia and should undergo surgical thrombectomy."
Analysis 2: "The patient has acute limb ischemia and should first receive heparin."
    "divergence_point": "Immediate surgery vs anticoagulation-first management.",
    "contrastive_question": "For a patient with sudden cold, pulseless leg, which initial treatment is more appropriate: start heparin or perform surgical thrombectomy?"

Now analyze the following pair:

Analysis 1:
{analysis1}

Analysis 2:
{analysis2}
'''

EVALUATE_TEMPLATE_EXTRA="""
You are a clinical expert. You will be given some medical knowledge followed by a patient's detailed conditions, a question and some options.

Use your own reasoning path and emphasize differential diagnoses before deciding. Make a balance between the given knowledge and your own understanding.
Reasoning Steps:
Step 1: Summarize key patient findings
Step 2: Match findings to each option
Step 3: Compare top two options
Step 4: Conclude with justification

Given medical knowledge:
{Extra_info}

The current patient's conditions followed by a question and some options:
{conditions}
Options:
{options}
'''
"""



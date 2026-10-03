EVALUATE_TEMPLATE ='''
You are a clinical expert.

You will be given a patient's conditions in SOAP format (Subjective, Objective, Assessment).

Your task is to generate the patient's treatment Plan, using the Subjective, Objective, and Assessment as context.

Generate the treatment Plan in a similar style like the 'Generated Plan', in the format of ***json***.

"Generated Plan":(text)

Now, here is the patient's conditions:
{conditions}
'''

RETRIEVED_INFO='''
"plan": "Discharge medications include Acetaminophen, Docusate Sodium, Enoxaparin, Nitrofurantoin, Oxycodone, Atorvastatin, Levothyroxine, and Losartan. Follow discharge instructions for urostomy care, medication management, and activities. Schedule follow-up appointments and arrange for visiting nurse services to assist with transition to home care. Patient advised to avoid heavy lifting and specific activities until cleared by PCP or urologist."
"subjective": "Patient presents with a chief complaint of bladder cancer. She underwent robotic anterior exenteration and open ileal conduit due to invasive bladder cancer, with a pelvic MRI indicating potential invasion into the anterior vaginal wall. Patient experienced nausea and several episodes of emesis post-operatively requiring NGT placement, which was subsequently self-removed.",
"objective": "Physical exam showed the patient alert and oriented. She was breathing comfortably on room air. Abdominal examination revealed appropriate postsurgical tenderness to palpation. Urostomy appeared pink and viable. Relevant laboratory results include: WBC 7.6, Hgb 10.6, Hct 32.5, Glucose 117, Urea Nitrogen 23, Creatinine 0.6, Sodium 136, Potassium 4.4, Calcium 7.9, and Phosphorus 3.4. At discharge, the abdomen was soft with tenderness noted along the incision, the stoma was well perfused, and urine color was yellow.",
"assessment": "Patient is a post-operative status after robotic anterior exenteration and ileal conduit. Diagnosis of bladder cancer confirmed. Post-operative course complicated by nausea/emesis but stabilized with diet advancement and successful ostomy management. Disposition to rehabilitation indicated for further recovery.",
'''

EVALUATE_TEMPLATE_PB='''
You are a clinical expert.

You will be given a patient's conditions in SOAP format (Subjective, Objective, Assessment).

Your task is to generate the patient's treatment Plan, using the Subjective, Objective, and Assessment as context.

Follow the style of the example below:
{example}

Generate the treatment Plan in a similar style like the 'Generated Plan', in the format of ***json***.

"Generated Plan":(text)

Now, here is the patient's conditions:
{conditions}
'''

KEY_QUESTIONS_TEMPLATE='''
You are a clinical expert.

You will be given a patient's conditions in SOAP format (Subjective, Objective, Assessment).

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
Please convert the following patient record into a SOAP structure(only subjective, objective and assessment). 
Do not include any personal identifiers. 
Each section should be medically informative and concise, 
but keep as many relevant clinical details from the original text as possible. 
Do not omit laboratory values, imaging findings, or important history, 
unless they are clearly irrelevant. 
If some details cannot be placed under one section, you may keep them 
in the most reasonable SOAP section.
Use standard medical terms.

The patient's case:
{patient_case}      
'''

PATIENT_CASE_SYSTEM_TEMPLATE='''
You are a clinical expert. Please convert the following patient record into a SOAP structure (only subjective, objective and assessment).
Reply **only** with valid JSON (double quotes, no trailing commas), 
exactly in the format:
{  
    "subjective": (text),
    "objective": (text),
    "assessment": (text)
}
'''
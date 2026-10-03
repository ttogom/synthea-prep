import pandas as pd
import os
from openai import OpenAI
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

ehr_path = 'discharge.csv'
ehrdata = pd.read_csv(ehr_path, encoding='utf-8')
ehrdata_sample = ehrdata.tail(100).copy()

extract_template = '''Please convert the following patient record into a SOAP structure 
(Subjective / Objective / Assessment / Plan). 
Do not include any personal identifiers. 
Each section should be medically informative and concise, 
but keep as many relevant clinical details from the original text as possible. 
Do not omit laboratory values, imaging findings, or important history, 
unless they are clearly irrelevant. 
If some details cannot be placed under one section, you may keep them 
in the most reasonable SOAP section.
Use standard medical terms.

Reply **only** with valid JSON (double quotes, no trailing commas), 
exactly in the format:
{
  "subjective": (text),
  "objective": (text),
  "assessment": (text),
  "plan": (text)
}                  
'''

results = []

for idx, row in ehrdata_sample.iterrows():
    record_text = row['text']
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        response_format={"type": "json_object"},
        messages=[
            {
                'role': 'system',
                'content': 'You are a clinical assistant, you need to extract SOAP content from raw EHR records'
            },
            {
                'role': 'user',
                'content': f'{extract_template} RECORD: {record_text}'
            }
        ]
    )

    soap_dict = json.loads(response.choices[0].message.content)
    row_dict = row.drop(labels=['text']).to_dict()
    row_dict.update(soap_dict)

    results.append(row_dict)
    print(f"Processed {idx+1}")

with open("soap_questions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
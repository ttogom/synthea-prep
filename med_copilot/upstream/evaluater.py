import pandas as pd
from create_embeddings import EmbeddingRetriever, HybridRetriever, CrossEncoderReranker
# from evaluate_metrics import Evaluator
import subprocess
import warnings
import json
import os
from openai import OpenAI
import pandas as pd
import sys
import io
from templates.SOA_P_TEMPLATE import KEY_QUESTIONS_TEMPLATE,EVALUATE_TEMPLATE_KEYINFO,EVALUATE_TEMPLATE_PB,EVALUATE_TEMPLATE,RETRIEVED_INFO,PATIENT_CASE_TEMPLATE,PATIENT_CASE_SYSTEM_TEMPLATE
warnings.filterwarnings("ignore")

from graphrag.cli.query import run_local_search
from graphrag.utils import storage
import lancedb
from pathlib import Path


class test:
    def __init__(self, OPENAI_API_KEY,model):
        self.json_path='soap_with_metadata.json'
        self.api_key=OPENAI_API_KEY
        print(f'api_key usde in evaluater.py: {self.api_key}')
        self.model=model #'gpt-5-nano'
        self.client = OpenAI(api_key=self.api_key)
        
        # self.api_key=os.getenv("DEEPSEEK_API_KEY")
        # self.model='deepseek-chat'
        # self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def query_graphrag(self, query: str, method='local'):
        return run_local_search(
        config_filepath=Path("guidelines/settings.yaml"), 
        data_dir=Path("guidelines/output"),
        root_dir=Path("guidelines"),
        community_level=2,
        response_type="Multiple Paragraphs",
        streaming=False,
        query=query,
        verbose=True
    )

    def evaluate_SOA_P_with_patientbase_graphrag(self, patient_case):
        key_questions_template = KEY_QUESTIONS_TEMPLATE
        patient_case_template = PATIENT_CASE_TEMPLATE
        evaluate_template = EVALUATE_TEMPLATE_KEYINFO
        
        df = pd.DataFrame(json.load(open(self.json_path, "r", encoding="utf-8")))
        retriever = HybridRetriever(df, alpha=0.5)
        generated_answer = []

        patient_case_instance=patient_case_template.format(
            patient_case=patient_case
        )

        patient_case_response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    'role': 'system',
                    'content': PATIENT_CASE_SYSTEM_TEMPLATE
                },
                {
                    'role': 'user',
                    'content': f'{patient_case_instance}'
                }
            ]
        ).choices[0].message.content

        print(f"\n=== Stage 1: Generate structured patient case ===")
        print(f"Your input is converted to: {patient_case_response}")

        patient=json.loads(patient_case_response)

        conditions = f'Subjective: {patient["subjective"]}\nObjective: {patient["objective"]}\n Assessment: {patient["assessment"]}'
        candidates = retriever.search(conditions, topk=20)
        top_candidate_original = candidates[0]
        reranker = CrossEncoderReranker()
        results = reranker.rerank(conditions, candidates, topk=5)
        print(f"\n=== Stage 2: Rerank most similar patients ===")
        
        k=1
        for candidate in results[:5]:
            print(f"No.{k} {candidate}")
            k+=1

        print(f"\n=== Top 1 candidate is selected from a 20 patients pool ===")
        retrieved_info = results[0]
        print(retrieved_info)
        
        key_questions_instance=key_questions_template.format(
            conditions=conditions
        )

        key_response=self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a clinical expert.'
                },
                {
                    'role': 'user',
                    'content': f'{key_questions_instance}'
                }
            ]
        )
        key_questions=key_response.choices[0].message.content
        print(f"\n=== Stage 3: Key questions are pointed out about the patient ===")
        print(key_questions)
        
        Key_info=self.query_graphrag(key_questions, method="local")
        print(f'Keyinfo: {Key_info}')

        print(f"\n=== Stage 4: Key Info about the patient is extracted from GraphRAG database ===")
        print(key_questions)

        evaluate_template_instance = evaluate_template.format(
            conditions=conditions,
            example=retrieved_info,
            Key_info=Key_info,
        )
        # print(f'\n=== Stage 5: Final input into LLMs ===')
        # print(f" Final input {evaluate_template_instance}")
        
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=[
        #         {
        #             'role': 'system',
        #             'content': 'You are a clinical expert.'
        #         },
        #         {
        #             'role': 'user',
        #             'content': f'{evaluate_template_instance}'
        #         }
        #     ]
        # )
        # generated_answer.append(response.choices[0].message.content)
        # print(f'generated_answer: {generated_answer}')
        # return retrieved_info, generated_answer[0], Key_info, top_candidate_original["dominant_terms"]

        display_text = (
            f"Subjective:\n{top_candidate_original['subjective']}\n\n"
            f"Objective:\n{top_candidate_original['objective']}\n\n"
            f"Assessment:\n{top_candidate_original['assessment']}\n\n"
            f"Plan:\n{top_candidate_original['plan']}"
        )
        
        return display_text, Key_info, top_candidate_original["dominant_terms"]


if __name__=="__main__":
    test=test()
    test.evaluate_SOA_P_with_patientbase_graphrag()
    print(test.avg_metrics_SOA_P_with_patientbase_graphrag)
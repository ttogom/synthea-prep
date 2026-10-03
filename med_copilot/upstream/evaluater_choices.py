import json
import pandas as pd
from openai import OpenAI
import os
from create_embeddings_choices import EmbeddingRetriever, BM25Retriever, HybridRetriever, CrossEncoderReranker
from evaluate_metrics import Evaluator
import subprocess
import warnings
from templates.ChoicesTemplate import KEY_QUESTIONS_TEMPLATE,EVALUATE_TEMPLATE_GRAPHRAG,EVALUATE_TEMPLATE,SYSTEM_ROLE_GRAPHRAG,ALIGNMENT_TEMPLATE,EVALUATE_TEMPLATE_PB,EVALUATE_TEMPLATE_GRAPHRAG_SELFCONSISTENCY
from templates.ChoicesTemplate_NEJM import EVALUATE_TEMPLATE_NEJM, EVALUATE_TEMPLATE_GRAPHRAG_NEJM, KEY_QUESTIONS_TEMPLATE_NEJM, ALIGNMENT_TEMPLATE_NEJM
from templates.Self_ConsistencyTemplate import SELF_CONSISTENCY_ROLETEMPLATE,SELF_CONSISTENCY_USERTEMPLATE,EVALUATE_TEMPLATE_EXTRA

warnings.filterwarnings("ignore")

from datasets import load_dataset
ds = load_dataset("nejm-ai-qa/exams")


class evaluater_choices:
    def __init__(self):
        self.json_path='create_datasets/medqa_10000.json'
        self.test_path='create_datasets/medqa_200.json'
        self.test_size=200
        self.testset=None
        with open(self.test_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.testset=data[:self.test_size]

        # self.testset=ds["general_surgery"]
        # self.test_size=len(self.testset)

        self.avg_metrics_with_patientbase_graphrag=None
        self.avg_metrics_with_patientbase=None
        self.avg_metrics_with_random_patientbase=None
        self.avg_metrics=None
        self.avg_metrics_with_patientbase_graphrag_self_consistency = None

        self.avg_metrics_nejm=None
        self.avg_metrics_with_patientbase_graphrag_nejm=None

        self.api_key=os.getenv("OPENAI_API_KEY")
        self.model='gpt-4.1-mini' #'gpt-5-nano'
        self.client = OpenAI(api_key=self.api_key)

        # self.api_key=os.getenv("DEEPSEEK_API_KEY")
        # self.model='deepseek-chat'
        # self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def query_graphrag(self, query: str, method='local'):
        result = subprocess.run(
            [
                "graphrag", "query",
                "--root", "./guidelines",
                "--method", method,
                "--query", query
            ],  # 把长文本通过 stdin 传递
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        return result.stdout

    def evaluate_with_patientbase_graphrag(self):
        self.avg_metrics_with_patientbase_graphrag = {
            "accuracy": 0
        }

        key_questions_template = KEY_QUESTIONS_TEMPLATE
        alignment_template=ALIGNMENT_TEMPLATE
        evaluate_template = EVALUATE_TEMPLATE_GRAPHRAG
        df = pd.DataFrame(json.load(open(self.json_path, "r", encoding="utf-8")))
        retriever = HybridRetriever(df, alpha=0.5)

        special_char_count=0
        wrong_answers= []

        for patient in self.testset:
            correct_answer=patient['answer_idx']
            conditions = f'{patient["question"]}'
            options=f'{patient["options"]}'
            candidates = retriever.search(conditions, topk=20)
            reranker = CrossEncoderReranker()
            results = reranker.rerank(conditions, candidates, topk=1)
            retrieved_info = results[0]

            key_questions_instance=key_questions_template.format(
                conditions=conditions,
                options=options
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
                ],
                temperature=0,
                timeout=30
            )
            key_questions=key_response.choices[0].message.content

            print(f'{key_questions}')

            Key_info=self.query_graphrag(key_questions, method="local")
            # print(f'Keyinfo: {Key_info}')

            alignment_instance=alignment_template.format(
                conditions=conditions,
                options=options,
                RAG=Key_info
            )


            Key_info=self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a clinical expert.'
                    },
                    {
                        'role': 'user',
                        'content': f'{alignment_instance}'
                    }


                ]
            ).choices[0].message.content

            evaluate_template_instance = evaluate_template.format(
                conditions=conditions,
                options=options,
                example=retrieved_info,
                Key_info=Key_info,
            )

            print(f'evaluate_template: {evaluate_template_instance}')

            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        'role': 'system',
                        'content': SYSTEM_ROLE_GRAPHRAG
                    },
                    {
                        'role': 'user',
                        'content': f'{evaluate_template_instance}'
                    }
                ],
                timeout=30,
                temperature=0
            )

            generated_answer=response.choices[0].message.content
            generated_answer=json.loads(generated_answer)
            print(f'{generated_answer}')
            option=generated_answer["Option"]

            if option not in ['A', 'B', 'C', 'D', 'E']:
                print(f'special answer: {generated_answer}')
                special_char_count+=1

            print(f'correct_option: {correct_answer}')
            print(f'option: {option}\n')

            if correct_answer != option:
                wrong_answer={}
                wrong_answer['evaluate_template']=evaluate_template_instance
                wrong_answer['option']=option
                wrong_answer['answer_idx'] = correct_answer
                wrong_answers.append(wrong_answer)
            self.avg_metrics_with_patientbase_graphrag["accuracy"] += (correct_answer==option) / self.test_size

        print(special_char_count)
        with open('log.json', 'w',encoding='utf-8') as f:
            f.write(json.dumps(wrong_answers, ensure_ascii=True, indent=4))
            print('error analysis saved!!')
        return self.avg_metrics_with_patientbase_graphrag

    def evaluate_with_patientbase_graphrag_self_consistency(self):
        self.avg_metrics_with_patientbase_graphrag_self_consistency = {
            "accuracy": 0
        }

        key_questions_template = KEY_QUESTIONS_TEMPLATE
        alignment_template=ALIGNMENT_TEMPLATE
        evaluate_template = EVALUATE_TEMPLATE_GRAPHRAG_SELFCONSISTENCY
        evaluate_template_extra=EVALUATE_TEMPLATE_EXTRA
        self_consistency_role_template=SELF_CONSISTENCY_ROLETEMPLATE
        self_consistency_user_template=SELF_CONSISTENCY_USERTEMPLATE
        df = pd.DataFrame(json.load(open(self.json_path, "r", encoding="utf-8")))
        retriever = HybridRetriever(df, alpha=0.5)

        special_char_count=0
        wrong_answers= []

        for patient in self.testset:
            correct_answer=patient['answer_idx']
            conditions = f'{patient["question"]}'
            options=f'{patient["options"]}'
            candidates = retriever.search(conditions, topk=20)
            reranker = CrossEncoderReranker()
            results = reranker.rerank(conditions, candidates, topk=1)
            retrieved_info = results[0]

            key_questions_instance=key_questions_template.format(
                conditions=conditions,
                options=options
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
                ],
                temperature=0,
                timeout=30
            )
            key_questions=key_response.choices[0].message.content

            print(f'{key_questions}')

            Key_info=self.query_graphrag(key_questions, method="local")
            # print(f'Keyinfo: {Key_info}')

            alignment_instance=alignment_template.format(
                conditions=conditions,
                options=options,
                RAG=Key_info
            )

            Aligned_Key_info=self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a clinical expert.'
                    },
                    {
                        'role': 'user',
                        'content': f'{alignment_instance}'
                    }
                ],
                timeout=30,
                temperature=0
            ).choices[0].message.content

            evaluate_template_instance = evaluate_template.format(
                conditions=conditions,
                options=options,
                Key_info=Aligned_Key_info,
                example=retrieved_info
            )

            print(f'evaluate_template: {evaluate_template_instance}')

            response1 = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        'role': 'system',
                        'content': SYSTEM_ROLE_GRAPHRAG
                    },
                    {
                        'role': 'user',
                        'content': f'{evaluate_template_instance}'
                    }
                ],
                timeout=30,
                temperature=0.7,
                top_p=0.7
            )
            response2 = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        'role': 'system',
                        'content': SYSTEM_ROLE_GRAPHRAG
                    },
                    {
                        'role': 'user',
                        'content': f'{evaluate_template_instance}'
                    }
                ],
                timeout=30,
                temperature=0.7,
                top_p=0.7
            )
            print(f'response1: {response1.choices[0].message.content}')
            print(f'response2: {response2.choices[0].message.content}')

            self_consistency_instance=self_consistency_user_template.format(
                analysis1=response1.choices[0].message.content,
                analysis2=response2.choices[0].message.content
            )
            # print(self_consistency_instance)

            self_consistency=self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {'role': 'system', 'content': self_consistency_role_template},
                    {'role': 'user', 'content': self_consistency_instance}
                ]
            )

            generated_divergence=json.loads(self_consistency.choices[0].message.content)

            if generated_divergence['contrastive_question']=="SKIP":
                print("no divergence")
                generated_answer = response1.choices[0].message.content
                generated_answer = json.loads(generated_answer)
            else:
                print(generated_divergence['divergence_point'])
                print(generated_divergence['contrastive_question'])
                extra_rag=self.query_graphrag(generated_divergence['contrastive_question'], method="local")
                Key_info=Key_info+'\n'+extra_rag
                extra_alignment_instance=alignment_template.format(
                    conditions=conditions,
                    options=options,
                    RAG=Key_info
                )
                aligned_extra_rag = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            'role': 'system',
                            'content': 'You are a clinical expert.'
                        },
                        {
                            'role': 'user',
                            'content': f'{extra_alignment_instance}'
                        }
                    ],
                    timeout=30,
                    temperature=0,
                    top_p=0.1
                ).choices[0].message.content

                extra_evaluate_template_instance = evaluate_template_extra.format(
                    conditions=conditions,
                    options=options,
                    Extra_info=aligned_extra_rag,
                    example=retrieved_info
                )

                print(f'extra_evaluate_template: {extra_evaluate_template_instance}')
                result = self.client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            'role': 'system',
                            'content': SYSTEM_ROLE_GRAPHRAG
                        },
                        {
                            'role': 'user',
                            'content': f'{extra_evaluate_template_instance}'
                        }
                    ],
                    timeout=30,
                    temperature=0,
                    top_p=0
                ).choices[0].message.content

                generated_answer = json.loads(result)
                print(generated_answer['Rationale'])

            option=generated_answer["Option"]

            if option not in ['A', 'B', 'C', 'D', 'E']:
                print(f'special answer: {generated_answer}')
                special_char_count+=1

            print(f'correct_option: {correct_answer}')
            print(f'option: {option}\n')

            if correct_answer != option:
                wrong_answer={}
                wrong_answer['evaluate_template']=evaluate_template_instance
                wrong_answer['option']=option
                wrong_answer['answer_idx'] = correct_answer
                wrong_answers.append(wrong_answer)
            self.avg_metrics_with_patientbase_graphrag_self_consistency["accuracy"] += (correct_answer==option) / self.test_size

        print(special_char_count)
        with open('log.json', 'w',encoding='utf-8') as f:
            f.write(json.dumps(wrong_answers, ensure_ascii=True, indent=4))
            print('error analysis saved!!')
        return self.avg_metrics_with_patientbase_graphrag_self_consistency

    def evaluate_with_patientbase(self):
        self.avg_metrics_with_patientbase = {
            "accuracy": 0
        }
        # self.avg_metrics_with_patientbase = {
        #     "BLEU": 0,
        #     "METEOR": 0,
        #     "ROUGE-1": 0,
        #     "ROUGE-2": 0,
        #     "ROUGE-L": 0,
        #     "BERTScore_F1": 0
        # }
        evaluate_template = EVALUATE_TEMPLATE_PB
        df = pd.DataFrame(json.load(open(self.json_path, "r", encoding="utf-8")))
        retriever = HybridRetriever(df, alpha=0.5)
        special_char_count=0
        wrong_answers= []

        for patient in self.testset:
            correct_answer=patient['answer_idx']
            conditions = f'{patient["question"]}'
            options=f'{patient["options"]}'
            candidates = retriever.search(conditions, topk=20)
            reranker = CrossEncoderReranker()
            results = reranker.rerank(conditions, candidates, topk=1)
            retrieved_info = results[0]

            evaluate_template_instance=evaluate_template.format(
                conditions=conditions,
                options=options,
                example=retrieved_info
            )

            print(f'evaluate_template: {evaluate_template_instance}')

            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        'role': 'system',
                        'content': SYSTEM_ROLE_GRAPHRAG
                    },
                    {
                        'role': 'user',
                        'content': f'{evaluate_template_instance}'
                    }
                ],
                timeout=30,
                temperature=0
            )

            generated_answer = response.choices[0].message.content
            generated_answer = json.loads(generated_answer)
            print(f'{generated_answer}')
            option = generated_answer["Option"]

            if option not in ['A', 'B', 'C', 'D', 'E']:
                print(f'special answer: {generated_answer}')
                special_char_count += 1

            print(f'correct_option: {correct_answer}')
            print(f'option: {option}\n')

            if correct_answer != option:
                wrong_answer = {}
                wrong_answer['evaluate_template'] = evaluate_template_instance
                wrong_answer['option'] = option
                wrong_answer['answer_idx'] = correct_answer
                wrong_answers.append(wrong_answer)

            self.avg_metrics_with_patientbase["accuracy"] += (correct_answer == option) / self.test_size

        print(special_char_count)
        with open('log_patientbase.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(wrong_answers, ensure_ascii=True, indent=4))
            print('error analysis saved!!')
        return self.avg_metrics_with_patientbase

    def evaluate_with_random_patientbase(self):
        self.avg_metrics_with_random_patientbase = {
            "accuracy": 0
        }
        evaluate_template = 'EVALUATE_TEMPLATE'
        correct_answer = []
        generated_answer = []

        for patient in self.testset:
            conditions = f'subjective: {patient["subjective"]}+{patient["objective"]}+{patient["assessment"]}'
            retrieved_info = 'RETRIEVED_INFO'
            evaluate_template_instance = evaluate_template.format(
                conditions=conditions,
                example=retrieved_info
            )

            correct_answer.append(f'Generated Plan: {patient["plan"]}')

            # print(evaluate_template)

            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a clinical expert.'
                    },
                    {
                        'role': 'user',
                        'content': f'{evaluate_template_instance}'
                    }
                ]
            )
            print(f'evaluate_template: {evaluate_template_instance}')
            generated_answer.append(response.choices[0].message.content)

            results = Evaluator.evaluate_metrics(correct_answer, generated_answer)
            for k, v in results.items():
                # print(f"{k}: {v:.4f}")
                self.avg_metrics_with_random_patientbase[k] += v / self.test_size

    def evaluate(self):
        self.avg_metrics = {
            "accuracy":0
        }
        evaluate_template = EVALUATE_TEMPLATE

        for patient in self.testset:
            conditions = patient['question']
            options=patient['options']

            evaluate_template_instance = evaluate_template.format(
                conditions=conditions,
                options=options
            )

            correct_answer=patient["answer_idx"]

            # print(evaluate_template)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a clinical expert.'
                    },
                    {
                        'role': 'user',
                        'content': f'{evaluate_template_instance}'
                    }
                ]
            )
            print(f'evaluate_template: {evaluate_template_instance}')
            generated_answer=response.choices[0].message.content

            print(f'correct_answer: {correct_answer}')
            print(f'generated_answer: {generated_answer}')

            if generated_answer == correct_answer:
                self.avg_metrics["accuracy"] += 1/self.test_size
        return self.avg_metrics

    def evaluate_with_patientbase_graphrag_nejm(self):
        self.avg_metrics_with_patientbase_graphrag_nejm = {
            "accuracy":0
        }
        count=0

        key_questions_template = KEY_QUESTIONS_TEMPLATE_NEJM
        alignment_template = ALIGNMENT_TEMPLATE_NEJM
        evaluate_template = EVALUATE_TEMPLATE_GRAPHRAG_NEJM

        for patient in self.testset:
            count+=1
            conditions = patient['question']

            correct_answer=patient["answer"]

            print(evaluate_template)
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
                ],
                temperature=0,
                timeout=30
            )
            key_questions=key_response.choices[0].message.content

            print(f'{key_questions}')

            key_questions=patient['question']

            Key_info=self.query_graphrag(key_questions, method="local")
            # print(f'Keyinfo: {Key_info}')

            alignment_instance=alignment_template.format(
                conditions=conditions,
                RAG=Key_info
            )

            Key_info=self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a clinical expert.'
                    },
                    {
                        'role': 'user',
                        'content': f'{alignment_instance}'
                    }


                ]
            ).choices[0].message.content

            evaluate_template_instance = evaluate_template.format(
                conditions=conditions,
                Key_info=Key_info
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a clinical expert.'
                    },
                    {
                        'role': 'user',
                        'content': f'{evaluate_template_instance}'
                    }
                ]
            )

            print(f'evaluate_template: {evaluate_template_instance}')
            generated_answer=response.choices[0].message.content

            print(f'correct_answer: {correct_answer}')
            print(f'generated_answer: {generated_answer}')

            if generated_answer == correct_answer:
                self.avg_metrics_with_patientbase_graphrag_nejm["accuracy"] += 1/self.test_size
                print(self.avg_metrics_with_patientbase_graphrag_nejm["accuracy"]*self.test_size/count)
        return self.avg_metrics_with_patientbase_graphrag_nejm

    def evaluate_nejm(self):
        self.avg_metrics_nejm = {
            "accuracy":0
        }
        evaluate_template = EVALUATE_TEMPLATE_NEJM

        for patient in self.testset:
            conditions = patient['question']

            evaluate_template_instance = evaluate_template.format(
                conditions=conditions
            )

            correct_answer=patient["answer"]

            # print(evaluate_template)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a clinical expert.'
                    },
                    {
                        'role': 'user',
                        'content': f'{evaluate_template_instance}'
                    }
                ]
            )
            print(f'evaluate_template: {evaluate_template_instance}')
            generated_answer=response.choices[0].message.content

            print(f'correct_answer: {correct_answer}')
            print(f'generated_answer: {generated_answer}')

            if generated_answer == correct_answer:
                self.avg_metrics_nejm["accuracy"] += 1/self.test_size
        return self.avg_metrics_nejm

if __name__=="__main__":
    test=evaluater_choices()
    print(f'Model under test: {test.model}')
    # test.evaluate_with_patientbase_graphrag()
    # test.evaluate_with_patientbase()
    # test.evaluate_SOA_P_with_random_patientbase()
    # test.evaluate()
    # test.evaluate_nejm()
    # test.evaluate_with_patientbase_graphrag_nejm()
    test.evaluate_with_patientbase_graphrag_self_consistency()


    # print(test.avg_metrics_with_patientbase_graphrag)
    # print(f'evaluate_with_patientbase: {test.avg_metrics_with_patientbase}')
    # print(f'evaluate_SOA_P_with_randompatientbase: {test.avg_metrics_with_random_patientbase}')
    # print(f'evaluate: {test.avg_metrics}')
    # print(test.avg_metrics_nejm)
    # print(test.avg_metrics_with_patientbase_graphrag_nejm)
    print(test.avg_metrics_with_patientbase_graphrag_self_consistency)
from pathlib import Path
import sys
import os
import json
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1] #project root dir
MED = ROOT / "med_copilot"
UPSTREAM = MED / "upstream" 
GUIDELINES = MED / "data" / "guidelines"
OUTPUT = ROOT / "data_run_med_copilot" / "output"

# Similar-patient corpus: change this path to select another JSON corpus.
# save new corpus in med_copilot/data/patient_corpus
PATIENT_CORPUS = UPSTREAM / "soap_with_metadata.json"

# sole input datapoint used: 
NOTE = ROOT / "data_run_med_copilot" / "input" / "Peter292_Gleichner915_0db4522b-544a-43b1-ad63-3caeb72be2ab.txt"


def main():

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(UPSTREAM))

    # ---------------------------------------------------------------------
    # Apple-Silicon compatibility workaround.
    # BioClinicalBERT must initialize before create_embeddings imports FAISS.
    # ---------------------------------------------------------------------

    from sentence_transformers import SentenceTransformer

    _preloaded_bioclinicalbert = SentenceTransformer(
        "emilyalsentzer/Bio_ClinicalBERT"
    )

    from openai import OpenAI
    from create_embeddings import HybridRetriever, CrossEncoderReranker
    from graphrag.cli.query import run_local_search
    from med_copilot.prompts.diagnosis_plan import (
        KEY_QUESTIONS_TEMPLATE,
        EVALUATE_TEMPLATE_KEYINFO,
        PATIENT_CASE_TEMPLATE,
        PATIENT_CASE_SYSTEM_TEMPLATE,
    )

    # ---------------------------------------------------------------------
    # Input
    # ---------------------------------------------------------------------
    OUTPUT.mkdir(parents=True, exist_ok=True)
    patient_case = NOTE.read_text(encoding="utf-8")

    api_key = os.environ["OPENAI_API_KEY"]
    model = "gpt-4o-mini"
    client = OpenAI(api_key=api_key)

    print("\n========== INPUT PATIENT CASE ==========\n")
    print(patient_case)

    # ---------------------------------------------------------------------
    # Stage 1: MED-COPILOT patient free-text -> Subjective / Objective
    # ---------------------------------------------------------------------

    patient_case_instance = PATIENT_CASE_TEMPLATE.format(
        patient_case=patient_case
    )

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": PATIENT_CASE_SYSTEM_TEMPLATE,
            },
            {
                "role": "user",
                "content": patient_case_instance,
            },
        ],
    )

    patient = json.loads(response.choices[0].message.content)
    conditions = (
        f'Subjective: {patient["subjective"]}\n'
        f'Objective: {patient["objective"]}'
    )

    print("\n========== STAGE 1: S/O ==========\n")
    print(conditions)
    with open(OUTPUT / "case_SOA.txt", "w") as f:
        f.write(conditions)

    # ---------------------------------------------------------------------
    # Stage 2: Similar-patient retrieval + reranking
    # ---------------------------------------------------------------------

    with open(PATIENT_CORPUS, "r", encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))

    # EmbeddingRetriever uses relative paths for these artifacts.
    old_cwd = Path.cwd()
    os.chdir(UPSTREAM)

    try:
        retriever = HybridRetriever(df, alpha=0.5)
        candidates = retriever.search(conditions, topk=20)

        reranker = CrossEncoderReranker()
        results = reranker.rerank(
            conditions,
            candidates,
            topk=5,
        )
    finally:
        os.chdir(old_cwd)

    retrieved_info = results[0]

    print("\n========== STAGE 2: TOP SIMILAR PATIENT ==========\n")
    print(retrieved_info)
    with open(OUTPUT / "top_similar_patient.txt", "w") as f:
        f.write(retrieved_info)
    # ---------------------------------------------------------------------
    # Stage 3: Key clinical questions
    # ---------------------------------------------------------------------

    key_questions_instance = KEY_QUESTIONS_TEMPLATE.format(
        conditions=conditions
    )

    key_response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a clinical expert.",
            },
            {
                "role": "user",
                "content": key_questions_instance,
            },
        ],
    )

    key_questions = key_response.choices[0].message.content

    print("\n========== STAGE 3: KEY QUESTIONS ==========\n")
    print(key_questions)
    with open(OUTPUT / "questions_for_guidelines_db.txt", "w") as f:
        f.write(key_questions)
    # ---------------------------------------------------------------------
    # Stage 4: GraphRAG guideline retrieval
    # ---------------------------------------------------------------------

    key_info, graphrag_context = run_local_search(
        config_filepath=GUIDELINES / "settings.yaml",
        data_dir=GUIDELINES / "output",
        root_dir=GUIDELINES,
        community_level=2,
        response_type="Multiple Paragraphs",
        streaming=False,
        query=key_questions,
        verbose=True,
    )

    print("\n========== STAGE 4: GUIDELINE EVIDENCE ==========\n")
    print(key_info)
    with open(OUTPUT / "answers_from_guidelines_db.txt", "w") as f:
        f.write(key_info)
    with open(OUTPUT / "graphrag_context.json", "w") as f:
        json.dump(
            graphrag_context, f,
            default=lambda table: json.loads(table.to_json(orient="split")),
            indent=2,
        )
    # ---------------------------------------------------------------------
    # Stage 5: Restore MED-COPILOT's commented-out final generation call
    # ---------------------------------------------------------------------

    final_prompt = EVALUATE_TEMPLATE_KEYINFO.format(
        conditions=conditions,
        example=retrieved_info,
        Key_info=key_info,
    )
    with open(OUTPUT / "final_prompt.txt", "w") as f:
        f.write(final_prompt)

    final_response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a clinical expert.",
            },
            {
                "role": "user",
                "content": final_prompt,
            },
        ],
    )

    generated_plan = final_response.choices[0].message.content

    print("\n========== MED-COPILOT FINAL OUTPUT ==========\n")
    print(generated_plan)
    with open(OUTPUT / "final_output.txt", "w") as f:
        f.write(str(generated_plan))


if __name__ == "__main__":
    main()

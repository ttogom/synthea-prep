from contextlib import chdir
import json
import os
from pathlib import Path
import sys

import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# These script-local paths must be available before importing project modules.
ROOT = Path(__file__).resolve().parents[1] #project root dir
MED = ROOT / "med_copilot"
UPSTREAM = MED / "upstream"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UPSTREAM))
GUIDELINES = MED / "data" / "guidelines"
OUTPUT = ROOT / "data_run_med_copilot" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)
# Similar-patient corpus: change this path to select another JSON corpus.
# TODO: save new corpus in med_copilot/data/patient_corpus
PATIENT_CORPUS = UPSTREAM / "soap_with_metadata.json"
# Prepared S/O input for the experiment.
PATIENT_INPUT = ROOT / "data_run_med_copilot" / "input" / "patient_records_so.jsonl"

# Apple-Silicon compatibility: initialize BioClinicalBERT before importing
# create_embeddings, whose imports load FAISS.
_preloaded_bioclinicalbert = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

from create_embeddings import HybridRetriever, CrossEncoderReranker
from graphrag.cli.query import run_local_search
from med_copilot.prompts.diagnosis_plan import (
    KEY_QUESTIONS_TEMPLATE,
    EVALUATE_TEMPLATE_KEYINFO,
)
from scripts.med_copilot_helpers import (
    JSONKeywordExtractor,
    load_patient_records,
    validate_guideline_tables,
)

# TODO: decide on API key to use for full experiment
api_key = os.environ["OPENAI_API_KEY"]
model = "gpt-4o-mini"
client = OpenAI(api_key=api_key)

def main():
# %% ------------------------------------------------------------------
#  Load input patient records in Subjective / Objective format
# ---------------------------------------------------------------------

    # Load all patient records before running the pipeline.
    patients = load_patient_records(PATIENT_INPUT)

    # TEMPORARY TEST RUN: select only the first patient.
    # TODO: replace this selection with a loop over patients and per-patient
    # outputs when running the full experiment.
    patient = patients[0]
    s_o = (
        f'Subjective: {patient["subjective"]}\n'
        f'Objective: {patient["objective"]}'
    )
# %% ------------------------------------------------------------------
# Similar-patient retrieval + reranking
# ---------------------------------------------------------------------
    with open(PATIENT_CORPUS, "r", encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))

    # EmbeddingRetriever uses relative paths for these artifacts.
    old_cwd = Path.cwd()
    os.chdir(UPSTREAM)

    try:
        results = []
        if not df.empty:
            # HybridRetriever combines embedding similarity and keyword overlap to find reference patients
            retriever = HybridRetriever(df, alpha=0.5)
            # Enforce JSON without changing the vendored retriever or its models.
            retriever.keyword_extractor = JSONKeywordExtractor(
                client=retriever.keyword_extractor.client,
                model=retriever.keyword_extractor.model,
            )
            candidates = retriever.search(s_o, topk=20)

            if candidates:
                #CrossEncoderReranker jointly scores the target patient and each similar-patient candidate, then reorders candidates by relevance
                reranker = CrossEncoderReranker()
                results = reranker.rerank(
                    s_o,
                    candidates,
                    topk=5,
                )
    finally:
        os.chdir(old_cwd)

    retrieved_info = results[0] if results else "No reference patient available."

    print("\n========== TOP SIMILAR PATIENT ==========\n")
    print(retrieved_info)
    with open(OUTPUT / "top_similar_patient.txt", "w") as f:
        f.write(retrieved_info)

# %% ------------------------------------------------------------------
# Generate key questions to query Clinical Guidelines graph db
# ---------------------------------------------------------------------
    key_questions_instance = KEY_QUESTIONS_TEMPLATE.format(
        conditions=s_o
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

    print("\n========== KEY QUESTIONS ==========\n")
    print(key_questions)
    with open(OUTPUT / "questions_for_guidelines_db.txt", "w") as f:
        f.write(key_questions)

# %% ------------------------------------------------------------------
# GraphRAG clinical guideline retrieval
# ---------------------------------------------------------------------

    guideline_output = GUIDELINES / "output"
    validate_guideline_tables(guideline_output)
    # The vendored table loader uses the working directory instead of storage's root.
    with chdir(guideline_output):
        key_info, graphrag_context = run_local_search(
            config_filepath=GUIDELINES / "settings.yaml",
            data_dir=guideline_output,
            root_dir=GUIDELINES,
            community_level=2,
            response_type="Multiple Paragraphs",
            streaming=False,
            query=key_questions,
            verbose=True,
        )

    print("\n========== CLINICAL GUIDELINE EVIDENCE ==========\n")
    print(key_info)
    with open(OUTPUT / "answers_from_guidelines_db.txt", "w") as f:
        f.write(key_info)
    with open(OUTPUT / "graphrag_context.json", "w") as f:
        json.dump(
            graphrag_context, f,
            default=lambda table: json.loads(table.to_json(orient="split")),
            indent=2,
        )

# %% ------------------------------------------------------------------
# Generate Assessment / Plan
# ---------------------------------------------------------------------

    final_prompt = EVALUATE_TEMPLATE_KEYINFO.format(
        conditions=s_o,
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

    print("\n========== ASSESSMENT AND PLAN ==========\n")
    print(generated_plan)
    with open(OUTPUT / "final_output.txt", "w") as f:
        f.write(str(generated_plan))


if __name__ == "__main__":
    main()

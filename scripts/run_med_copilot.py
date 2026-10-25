from contextlib import chdir
import json
import os
from pathlib import Path
import sys
import traceback

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
# Each execution creates its own run directory here (run.json + results.jsonl).
EXPERIMENT_OUTPUT = ROOT / "experiments" / "output"
# Similar-patient corpus: change this path to select another JSON corpus.
# TODO: save new corpus in med_copilot/data/patient_corpus
PATIENT_CORPUS = UPSTREAM / "soap_with_metadata.json"
# S/O-only artifacts are separate from the upstream S/O/A cache.
PATIENT_RETRIEVAL_ARTIFACTS_DIR = MED / "data" / "patient_retrieval_so"
# Prepared S/O input for the experiment.
PATIENT_INPUT = ROOT / "experiments" / "input" / "test_sample_so.jsonl"
# Matching experiment labels stay outside all retrieval and generation inputs.
PATIENT_LABELS = ROOT / "experiments" / "input" / "test_sample_labels.jsonl"

# Apple-Silicon compatibility: initialize BioClinicalBERT before importing
# create_embeddings, whose imports load FAISS.
_preloaded_bioclinicalbert = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

from scripts.med_copilot_retrieval import HybridRetriever, CrossEncoderReranker, matching_text
from scripts.med_copilot_observability import capture_graphrag_responses
from scripts.med_copilot_scoring import PatientScorer, parse_scoring_args
from graphrag.cli.query import run_local_search
from med_copilot.prompts.diagnosis_plan import (
    KEY_QUESTIONS_TEMPLATE,
    EVALUATE_TEMPLATE_KEYINFO,
)
from scripts.med_copilot_helpers import (
    ExperimentRun,
    JSONKeywordExtractor,
    git_branch,
    git_commit,
    graphrag_context_json,
    load_patient_records,
    model_response_metadata,
    new_patient_result,
    pacific_now,
    parse_guideline_questions,
    prepare_patient_output_capture,
    reference_data_metadata,
    selected_reference_patient,
    split_assessment_plan,
    validate_guideline_tables,
)

# TODO: decide on API key to use for full experiment
api_key = os.environ["OPENAI_API_KEY"]
model = "gpt-4o-mini"
client = OpenAI(api_key=api_key)

def process_patient(patient, result, corpus_records, retriever, reranker, guideline_output):
    """Run every pipeline stage for one patient, filling `result` as stages finish."""
    prepare_patient_output_capture(result, corpus_records, retriever, reranker)
    s_o = matching_text(patient)
# %% ------------------------------------------------------------------
# Similar-patient retrieval + reranking
# ---------------------------------------------------------------------
    candidates, results = [], []
    if retriever is not None:
        # EmbeddingRetriever uses relative paths for these artifacts.
        with chdir(PATIENT_RETRIEVAL_ARTIFACTS_DIR):
            candidates = retriever.search(s_o, topk=20)

            if candidates:
                #CrossEncoderReranker jointly scores the target patient and each similar-patient candidate, then reorders candidates by relevance
                results = reranker.rerank(
                    s_o,
                    candidates,
                    topk=5,
                )

    retrieved_info = results[0] if results else "No reference patient available."
    result["selected_reference_patient"] = selected_reference_patient(
        corpus_records, candidates, results,
    )

    print("\n========== TOP SIMILAR PATIENT ==========\n")
    print(retrieved_info)

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
    result["model_responses"]["guideline_questions"].append(model_response_metadata(key_response))

    key_questions = key_response.choices[0].message.content
    # The raw response remains the GraphRAG query; parsing only records it.
    result["guideline_questions"] = parse_guideline_questions(key_questions)

    print("\n========== KEY QUESTIONS ==========\n")
    print(key_questions)

# %% ------------------------------------------------------------------
# GraphRAG clinical guideline retrieval
# ---------------------------------------------------------------------

    # The vendored table loader uses the working directory instead of storage's root.
    with chdir(guideline_output), capture_graphrag_responses(result["model_responses"]):
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

    result["guideline_evidence"] = key_info
    result["graphrag_context"] = graphrag_context_json(graphrag_context)

    print("\n========== CLINICAL GUIDELINE EVIDENCE ==========\n")
    print(key_info)

# %% ------------------------------------------------------------------
# Generate Assessment / Plan
# ---------------------------------------------------------------------

    final_prompt = EVALUATE_TEMPLATE_KEYINFO.format(
        conditions=s_o,
        example=retrieved_info,
        Key_info=key_info,
    )
    final_messages = [
        {
            "role": "system",
            "content": "You are a clinical expert.",
        },
        {
            "role": "user",
            "content": final_prompt,
        },
    ]
    # Saved from the same values sent, before the request can fail.
    result["assessment_plan_prompts"] = {
        "system": final_messages[0]["content"],
        "user": final_messages[1]["content"],
    }

    final_response = client.chat.completions.create(
        model=model,
        messages=final_messages,
    )
    result["model_responses"]["assessment_plan"].append(model_response_metadata(final_response))

    generated_plan = final_response.choices[0].message.content
    result["assessment_plan_raw"] = generated_plan
    # Missing, empty, or repeated sections are saved as null; the raw reply keeps them.
    result.update(split_assessment_plan(generated_plan))

    print("\n========== ASSESSMENT AND PLAN ==========\n")
    print(generated_plan)


def main(argv=None):
# %% ------------------------------------------------------------------
#  Load input patient records in Subjective / Objective format
# ---------------------------------------------------------------------
    args = parse_scoring_args(argv)
    started_at = pacific_now()
    branch = git_branch(ROOT)
    commit = git_commit(ROOT)

    # Load and validate all patient records before creating a run.
    patients = load_patient_records(PATIENT_INPUT)
    scorer = PatientScorer(PATIENT_LABELS, patients, args.presenting_evidence_only)
    patients = scorer.patients

    run = ExperimentRun.start(
        EXPERIMENT_OUTPUT, root=ROOT, input_path=PATIENT_INPUT, started_at=started_at,
        branch=branch, commit=commit, patient_ids=[patient["id"] for patient in patients],
    )
    print(f"Run directory: {run.run_dir}")

    current = None
    try:
        run.update(evaluation=scorer.metadata(ROOT))
# %% ------------------------------------------------------------------
# Shared resources: similar-patient retriever, reranker, guideline tables
# ---------------------------------------------------------------------
        with open(PATIENT_CORPUS, "r", encoding="utf-8") as f:
            corpus_records = json.load(f)
        df = pd.DataFrame(corpus_records)

        retriever = reranker = None
        if not df.empty:
            PATIENT_RETRIEVAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            with chdir(PATIENT_RETRIEVAL_ARTIFACTS_DIR):
                retriever = HybridRetriever(df, alpha=0.5, encoder=_preloaded_bioclinicalbert)
                # Enforce JSON without changing the vendored retriever or its models.
                retriever.keyword_extractor = JSONKeywordExtractor(
                    client=retriever.keyword_extractor.client,
                    model=retriever.keyword_extractor.model,
                )
                reranker = CrossEncoderReranker()

        guideline_output = GUIDELINES / "output"
        validate_guideline_tables(guideline_output)

        # Fingerprint after initialization, which may regenerate the artifacts.
        run.update(reference_data=reference_data_metadata(
            ROOT,
            corpus_file=PATIENT_CORPUS,
            artifacts_directory=PATIENT_RETRIEVAL_ARTIFACTS_DIR,
            artifact_files=None if retriever is None else [
                PATIENT_RETRIEVAL_ARTIFACTS_DIR / retriever.embedder.emb_path,
                PATIENT_RETRIEVAL_ARTIFACTS_DIR / retriever.embedder.index_path,
                PATIENT_RETRIEVAL_ARTIFACTS_DIR / retriever.embedder.metadata_path,
            ],
            guideline_directory=GUIDELINES,
            guideline_settings=GUIDELINES / "settings.yaml",
            guideline_output=guideline_output,
        ))

# %% ------------------------------------------------------------------
# Process each patient; an isolated failure does not stop the batch
# ---------------------------------------------------------------------
        for number, patient in enumerate(patients, 1):
            print(f"\n========== PATIENT {number}/{len(patients)}: {patient['id']} ==========")
            current = new_patient_result(patient["id"])
            try:
                process_patient(patient, current, corpus_records, retriever, reranker,
                                guideline_output)
                scorer.score_result(current)
                current["status"] = "completed"
            except Exception:
                current["status"] = "failed_incomplete"
                print(f"Patient {patient['id']} failed:", file=sys.stderr)
                traceback.print_exc()
            run.save_result(current)
            current = None

        status = run.finish(pacific_now())
    except BaseException:
        # Record unfinished patients and the stopped run, then re-raise.
        run.stop(pacific_now(), current)
        raise

    print(f"\nRun {run.run_dir.name} finished with status: {status}")
    return status


if __name__ == "__main__":
    if main() != "completed":
        sys.exit(1)

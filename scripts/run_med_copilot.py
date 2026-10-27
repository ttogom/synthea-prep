from contextlib import chdir
import json
import os
from pathlib import Path
import sys
import traceback

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# These script-local paths must be available before importing project modules.
ROOT = Path(__file__).resolve().parents[1] #project root dir
# Load credentials before dependency/client initialization; exported values win.
load_dotenv(ROOT / ".env", override=False)
MED = ROOT / "med_copilot"
UPSTREAM = MED / "upstream"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UPSTREAM))

from scripts.med_copilot_config import ExperimentConfig, confirm_experiment
from scripts.med_copilot_startup import PatientRetrievalStartup

# Apple Silicon requires encoder initialization before retrieval imports FAISS.
retrieval_startup = PatientRetrievalStartup(ROOT / "experiments/configs/experiment.yaml")

from scripts.med_copilot_retrieval import HybridRetriever, CrossEncoderReranker, matching_text
from scripts.med_copilot_observability import capture_graphrag_responses
from scripts.med_copilot_scoring import PatientScorer, parse_scoring_args
from scripts.med_copilot_models import AssessmentPlanModels
from scripts.collate_med_copilot_results import generate_report, load_report_data, format_top_rank_summary
from graphrag.cli.query import run_local_search
from med_copilot.prompts.diagnosis_plan import (
    KEY_QUESTIONS_TEMPLATE,
    VARIANT_TEMPLATES,
)
from scripts.med_copilot_helpers import (
    ExperimentRun,
    capture_component,
    component_needed,
    prepare_variant_messages,
    JSONKeywordExtractor,
    format_assessment_plans,
    git_branch,
    git_commit,
    graphrag_context_json,
    load_patient_records,
    model_response_metadata,
    new_patient_result,
    pacific_now,
    parse_guideline_questions,
    reference_data_metadata,
    selected_reference_patient,
    validate_guideline_tables,
)

from scripts.med_copilot_variants import COMPONENTS

# The shared client is initialized only after startup validation and confirmation.
client = None

def process_patient(patient, result, corpus_records, retriever, reranker, guideline_output,
                    assessment_models, experiment=None, setup_errors=None):
    """Run every pipeline stage for one patient, filling `result` as stages finish."""
    s_o = matching_text(patient)
    experiment = experiment or retrieval_startup.experiment
    for component in result["components"].values():
        for resource in COMPONENTS[component["component"]].resources:
            if resource in (setup_errors or {}):
                component.update(status="failed_incomplete", error=setup_errors[resource])

# %% ------------------------------------------------------------------
# Similar-patient retrieval + reranking, once per distinct configuration
# ---------------------------------------------------------------------
    if experiment.needs("patient_retrieval"):
        for instance, settings in experiment.component_instances("patient_retrieval"):
            component = result["components"][instance]
            if component["status"] != "not_processed":
                continue
            with capture_component(component, error_metadata=assessment_models.error_metadata,
                                   corpus_records=corpus_records, retriever=retriever, reranker=reranker,
                                   credential_names=experiment.required_credentials) as capture:
                candidates, results = [], []
                if retriever is not None:
                    # The cache paths inherited from upstream are relative.
                    with chdir(experiment.patient_retrieval_artifacts_dir):
                        retriever.alpha = settings["alpha"]
                        candidates = retriever.search(s_o, topk=settings["top_k"])
                        if candidates:
                            results = reranker.rerank(s_o, candidates, topk=settings["rerank_top_k"])
                retrieved_info = results[0] if results else "No reference patient available."
                capture["selected_reference_patient"] = selected_reference_patient(
                    corpus_records, candidates, results)
                component["outputs"]["example"] = retrieved_info
                print("\n========== TOP SIMILAR PATIENT ==========\n")
                print(retrieved_info)

# %% ------------------------------------------------------------------
# Clinical guidelines: questions and GraphRAG, once per configuration
# ---------------------------------------------------------------------
    if experiment.needs("guidelines"):
        for instance, settings in experiment.component_instances("guidelines"):
            component = result["components"][instance]
            if component["status"] != "not_processed" or not component_needed(experiment, instance, result):
                continue
            with capture_component(component, error_metadata=assessment_models.error_metadata,
                                   credential_names=experiment.required_credentials) as capture:
                key_questions_instance = KEY_QUESTIONS_TEMPLATE.format(conditions=s_o)
                key_response = client.chat.completions.create(
                    model=settings["question_model"], response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are a clinical expert."},
                        {"role": "user", "content": key_questions_instance},
                    ],
                )
                capture["model_responses"]["guideline_questions"].append(model_response_metadata(key_response))
                key_questions = key_response.choices[0].message.content
                # Preserve the raw JSON as the GraphRAG query; parse only for capture.
                capture["guideline_questions"] = parse_guideline_questions(key_questions)
                print("\n========== KEY QUESTIONS ==========\n")
                print(key_questions)
                # Vendored table loading uses cwd instead of the storage root.
                with chdir(guideline_output), capture_graphrag_responses(capture["model_responses"]):
                    key_info, graphrag_context = run_local_search(
                        config_filepath=experiment.guidelines / "settings.yaml",
                        data_dir=guideline_output, root_dir=experiment.guidelines,
                        community_level=settings["community_level"], response_type=settings["response_type"],
                        streaming=False, query=key_questions, verbose=settings["verbose"],
                    )
                capture["guideline_evidence"] = key_info
                capture["graphrag_context"] = graphrag_context_json(graphrag_context)
                component["outputs"]["Key_info"] = key_info
                print("\n========== CLINICAL GUIDELINE EVIDENCE ==========\n")
                print(key_info)

# %% ------------------------------------------------------------------
# Render each variant from its component outputs; run all final models
# ---------------------------------------------------------------------
    for variant_name, output in result["variants"].items():
        messages = prepare_variant_messages(
            experiment, variant_name, s_o, result, output, templates=VARIANT_TEMPLATES,
            error_metadata=assessment_models.error_metadata)
        if messages is None:
            continue
        assessment_models.generate(messages, output)
        print(f"\n========== RANKED CONDITIONS AND PLANS: {variant_name} ==========\n")
        print(format_assessment_plans(output["assessment_plans"], assessment_models.metadata(ROOT)["models"]))
    result["status"] = ("completed" if all(v["status"] == "completed" for v in result["variants"].values())
                        else "failed_incomplete")


def main(argv=None):
# %% ------------------------------------------------------------------
#  Load assessment/plan generating models and input patient records
# ---------------------------------------------------------------------
    global client
    args = parse_scoring_args(argv)
    try:
        experiment = ExperimentConfig(ROOT / "experiments/configs/experiment.yaml")
    except (OSError, ValueError) as exc:
        print(f"Experiment configuration check failed: {exc}", file=sys.stderr)
        return "configuration_error"
    # Preserve the existing optional cohort CLI override; YAML supplies the default.
    if args.presenting_evidence_only is not None:
        experiment.presenting_evidence_only = args.presenting_evidence_only
    patients = load_patient_records(experiment.patient_input)
    scorer = PatientScorer(
        experiment.patient_labels, patients, experiment.presenting_evidence_only,
        max_patients=experiment.max_patients,
        exclude_none_recorded_plans=experiment.exclude_none_recorded_plans,
    )
    patients = scorer.patients
    try:
        assessment_models = AssessmentPlanModels(
            experiment.model_config, experiment.selected_models,
            temperature=experiment.temperature, max_retries=experiment.max_retries,
            required_shared_keys=experiment.required_credentials,
        )
    except (OSError, ValueError) as exc:
        print(f"Experiment configuration check failed:\n\n{exc}\n\n"
              "Run not started. Fix the errors above and restart the script.", file=sys.stderr)
        return "configuration_error"
    if not confirm_experiment(experiment, assessment_models.metadata(ROOT), ROOT, scorer.metadata(ROOT)):
        return "cancelled"
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) if experiment.needs("guidelines") else None
    started_at = pacific_now()
    branch, commit = git_branch(ROOT), git_commit(ROOT)

    run = ExperimentRun.start(
        experiment.experiment_output, root=ROOT, input_path=experiment.patient_input, started_at=started_at,
        branch=branch, commit=commit, patient_ids=[patient["id"] for patient in patients],
        model_ids=assessment_models.model_ids, variant_ids=experiment.selected_variants,
        component_plan=experiment.component_plan, variant_components=experiment.variant_components,
    )
    print(f"Run directory: {run.run_dir}")

    current = None
    try:
        run.update(experiment=experiment.metadata(ROOT), evaluation=scorer.metadata(ROOT),
                   assessment_plan_models=assessment_models.metadata(ROOT),
                   **experiment.variant_metadata(ROOT))
# %% ------------------------------------------------------------------
# Shared resources: similar-patient retriever, reranker, guideline tables
# ---------------------------------------------------------------------
        setup_errors = {}
        corpus_records, retriever, reranker = [], None, None
        if experiment.needs_patient_retrieval:
            try:
                encoder = retrieval_startup.encoder_for(experiment)
                with open(experiment.patient_corpus, "r", encoding="utf-8") as f:
                    corpus_records = json.load(f)
                df = pd.DataFrame(corpus_records)

                if not df.empty and experiment.needs("patient_retrieval"):
                    experiment.patient_retrieval_artifacts_dir.mkdir(parents=True, exist_ok=True)
                    with chdir(experiment.patient_retrieval_artifacts_dir):
                        retriever = HybridRetriever(
                            df, alpha=experiment.patient_retrieval_alpha, encoder=encoder,
                            model_name=experiment.patient_embedding_model,
                            llm_model=experiment.patient_keyword_model,
                        )
                        # Enforce JSON without changing the vendored retriever or its models.
                        retriever.keyword_extractor = JSONKeywordExtractor(
                            client=retriever.keyword_extractor.client,
                            model=retriever.keyword_extractor.model,
                        )
                        reranker = CrossEncoderReranker(model_name=experiment.patient_reranker_model)

            except Exception as exc:
                setup_errors["patient_retrieval"] = assessment_models.error_metadata(exc)
        guideline_output = experiment.guidelines / "output"
        if experiment.needs_guidelines:
            try:
                validate_guideline_tables(guideline_output)
            except Exception as exc:
                setup_errors["guidelines"] = assessment_models.error_metadata(exc)

        # Fingerprint after initialization, which may regenerate the artifacts.
        run.update(reference_data=reference_data_metadata(
            ROOT,
            corpus_file=(experiment.patient_corpus if experiment.needs_patient_retrieval
                         and "patient_retrieval" not in setup_errors else None),
            artifacts_directory=experiment.patient_retrieval_artifacts_dir,
            artifact_files=None if retriever is None else [
                experiment.patient_retrieval_artifacts_dir / retriever.embedder.emb_path,
                experiment.patient_retrieval_artifacts_dir / retriever.embedder.index_path,
                experiment.patient_retrieval_artifacts_dir / retriever.embedder.metadata_path,
            ],
            guideline_directory=(experiment.guidelines if experiment.needs_guidelines
                                 and "guidelines" not in setup_errors else None),
            guideline_settings=experiment.guidelines / "settings.yaml",
            guideline_output=guideline_output,
        ))

# %% ------------------------------------------------------------------
# Process each patient; an isolated failure does not stop the batch
# ---------------------------------------------------------------------
        for number, patient in enumerate(patients, 1):
            print(f"\n========== PATIENT {number}/{len(patients)}: {patient['id']} ==========")
            current = new_patient_result(patient["id"], model_ids=assessment_models.model_ids,
                                         variant_ids=experiment.selected_variants,
                                         component_plan=experiment.component_plan,
                                         variant_components=experiment.variant_components)
            try:
                process_patient(patient, current, corpus_records, retriever, reranker,
                                guideline_output, assessment_models, experiment, setup_errors)
                scorer.score_result(current)
                if current["status"] != "completed":
                    print(f"Patient {patient['id']} failed: one or more variants incomplete", file=sys.stderr)
            except Exception:
                current["status"] = "failed_incomplete"
                print(f"Patient {patient['id']} failed:", file=sys.stderr)
                traceback.print_exc()
            run.save_result(current)
            current = None

        status = run.finish(pacific_now())
# %% ------------------------------------------------------------------
# RESULTS: collate persisted outputs and summarize top-ranked BERTScore_F1
# ---------------------------------------------------------------------
        try:
            report = generate_report(run.run_dir, root=ROOT)
            print("\n========== RESULTS ==========\n")
            print(format_top_rank_summary(load_report_data(run.run_dir, root=ROOT)))
            print(f"\nHTML results: {report}")
            run.update(results_summary={"status": "completed", "file": report.name})
        except Exception as exc:
            # Saved generation/scoring artifacts remain valid if report writing fails.
            run.update(results_summary={"status": "failed_incomplete", "error": assessment_models.error_metadata(exc)})
            print("Results report failed; saved run.json/results.jsonl were preserved.", file=sys.stderr)
            return "report_error"
    except BaseException:
        # Record unfinished patients and the stopped run, then re-raise.
        run.stop(pacific_now(), current)
        raise

    print(f"\nRun {run.run_dir.name} finished with status: {status}")
    return status


if __name__ == "__main__":
    if main() not in ("completed", "cancelled"):
        sys.exit(1)

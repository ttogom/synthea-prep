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
    VARIANT_SOURCES,
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
    prepare_patient_output_capture,
    reference_data_metadata,
    selected_reference_patient,
    validate_guideline_tables,
)

# The shared client is initialized only after startup validation and confirmation.
client = None

def process_patient(patient, result, corpus_records, retriever, reranker, guideline_output,
                    assessment_models, experiment=None, setup_errors=None):
    """Run every pipeline stage for one patient, filling `result` as stages finish."""
    prepare_patient_output_capture(result, corpus_records, retriever, reranker)
    s_o = matching_text(patient)
    experiment = experiment or retrieval_startup.experiment
    setup_errors = setup_errors or {}
    variants = (result["variants"] if "variants" in result else
                {"patient_retrieval_guidelines": result})
    states = result.get("evidence_status", {
        source: {"status": "not_processed", "error": None}
        for source in ("patient_retrieval", "guidelines")
    })
    for source, error in setup_errors.items():
        states[source].update(status="failed_incomplete", error=error)
    retrieved_info, key_info = "No reference patient available.", None
# %% ------------------------------------------------------------------
# Similar-patient retrieval + reranking
# ---------------------------------------------------------------------
    if experiment.needs_patient_retrieval and states["patient_retrieval"]["status"] != "failed_incomplete":
        states["patient_retrieval"]["status"] = "failed_incomplete"
        try:
            candidates, results = [], []
            if retriever is not None:
                # EmbeddingRetriever uses relative paths for these artifacts.
                with chdir(experiment.patient_retrieval_artifacts_dir):
                    candidates = retriever.search(s_o, topk=experiment.patient_retrieval_top_k)

                    if candidates:
                        #CrossEncoderReranker jointly scores the target patient and each similar-patient candidate, then reorders candidates by relevance
                        results = reranker.rerank(
                            s_o,
                            candidates,
                            topk=experiment.patient_rerank_top_k,
                        )

            retrieved_info = results[0] if results else "No reference patient available."
            result["selected_reference_patient"] = selected_reference_patient(
                corpus_records, candidates, results,
            )

            print("\n========== TOP SIMILAR PATIENT ==========\n")
            print(retrieved_info)
            states["patient_retrieval"]["status"] = "completed"
        except Exception as exc:
            states["patient_retrieval"]["error"] = assessment_models.error_metadata(exc)
            error = states["patient_retrieval"]["error"]
            print(f"patient_retrieval failed ({error['type']}): {error['message']}", file=sys.stderr)
            if "variants" not in result:
                raise
        except BaseException as exc:
            states["patient_retrieval"]["error"] = {"type": type(exc).__name__, "message": "Evidence preparation interrupted"}
            raise

# %% ------------------------------------------------------------------
# Clinical guidelines: questions and GraphRAG, once per patient
# ---------------------------------------------------------------------
    # Avoid spending calls when the only guideline-dependent variant is already
    # blocked by failed patient retrieval. Standalone guidelines remain independent.
    guidelines_needed = experiment.needs_guidelines and (
        "guidelines" in variants or states["patient_retrieval"]["status"] != "failed_incomplete")
    if guidelines_needed and states["guidelines"]["status"] != "failed_incomplete":
        states["guidelines"]["status"] = "failed_incomplete"
        try:
            key_questions_instance = KEY_QUESTIONS_TEMPLATE.format(
                conditions=s_o
            )

            key_response = client.chat.completions.create(
                model=experiment.guideline_question_model,
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

            # GraphRAG clinical guideline retrieval

            # The vendored table loader uses the working directory instead of storage's root.
            with chdir(guideline_output), capture_graphrag_responses(result["model_responses"]):
                key_info, graphrag_context = run_local_search(
                    config_filepath=experiment.guidelines / "settings.yaml",
                    data_dir=guideline_output,
                    root_dir=experiment.guidelines,
                    community_level=experiment.guideline_community_level,
                    response_type=experiment.guideline_response_type,
                    streaming=False,
                    query=key_questions,
                    verbose=experiment.guideline_verbose,
                )

            result["guideline_evidence"] = key_info
            result["graphrag_context"] = graphrag_context_json(graphrag_context)

            print("\n========== CLINICAL GUIDELINE EVIDENCE ==========\n")
            print(key_info)
            states["guidelines"]["status"] = "completed"
        except Exception as exc:
            states["guidelines"]["error"] = assessment_models.error_metadata(exc)
            error = states["guidelines"]["error"]
            print(f"guidelines failed ({error['type']}): {error['message']}", file=sys.stderr)
            if "variants" not in result:
                raise
        except BaseException as exc:
            states["guidelines"]["error"] = {"type": type(exc).__name__, "message": "Evidence preparation interrupted"}
            raise

# %% ------------------------------------------------------------------
# Only evidence inclusion changes between final prompt variants
# ---------------------------------------------------------------------
    for variant_name, output in variants.items():
        failed = [source for source in VARIANT_SOURCES[variant_name]
                  if states[source]["status"] != "completed"]
        if failed:
            output.update(status="failed_incomplete", error={
                "type": "EvidenceUnavailable", "message": "Required evidence failed: " + ", ".join(failed),
            })
            continue
        final_prompt = VARIANT_TEMPLATES[variant_name].format(
            conditions=s_o, example=retrieved_info, Key_info=key_info,
        )
        final_messages = [
            {"role": "system", "content": "You are a clinical expert."},
            {"role": "user", "content": final_prompt},
        ]
        output["assessment_plan_prompts"] = {
            "system": final_messages[0]["content"], "user": final_messages[1]["content"],
        }
        output["status"] = "failed_incomplete"
        assessment_models.generate(final_messages, output)
        print(f"\n========== RANKED CONDITIONS AND PLANS: {variant_name} ==========\n")
        print(format_assessment_plans(output["assessment_plans"], assessment_models.metadata(ROOT)["models"]))
    result["status"] = ("completed" if all(v["status"] == "completed" for v in variants.values())
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
        shared_keys = set()
        if experiment.needs_patient_retrieval or experiment.needs_guidelines:
            shared_keys.add("OPENAI_API_KEY")
        if experiment.needs_guidelines:
            shared_keys.add("GRAPHRAG_API_KEY")
        assessment_models = AssessmentPlanModels(
            experiment.model_config, experiment.selected_models,
            temperature=experiment.temperature, max_retries=experiment.max_retries,
            required_shared_keys=sorted(shared_keys),
        )
    except (OSError, ValueError) as exc:
        print(f"Experiment configuration check failed:\n\n{exc}\n\n"
              "Run not started. Fix the errors above and restart the script.", file=sys.stderr)
        return "configuration_error"
    if not confirm_experiment(experiment, assessment_models.metadata(ROOT), ROOT, scorer.metadata(ROOT)):
        return "cancelled"
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) if experiment.needs_guidelines else None
    started_at = pacific_now()
    branch, commit = git_branch(ROOT), git_commit(ROOT)

    run = ExperimentRun.start(
        experiment.experiment_output, root=ROOT, input_path=experiment.patient_input, started_at=started_at,
        branch=branch, commit=commit, patient_ids=[patient["id"] for patient in patients],
        model_ids=assessment_models.model_ids, variant_ids=experiment.selected_variants,
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

                if not df.empty:
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
                                         variant_ids=experiment.selected_variants)
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

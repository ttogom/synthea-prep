"""Process-local evidence preparation and exact prompt rendering.

The launcher supplies retrieval factories only after loading its CPU encoder.
This module deliberately does not import FAISS or initialize a model at import.
"""

from contextlib import chdir
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

from med_copilot.prompts.diagnosis_plan import KEY_QUESTIONS_TEMPLATE, VARIANT_TEMPLATES
from scripts.med_copilot_config import ExperimentConfig
from scripts.med_copilot_helpers import (
    JSONKeywordExtractor, capture_component, component_needed, graphrag_context_json,
    new_patient_result, prepare_guideline_query, prepare_variant_messages, redact_secrets,
    reference_data_metadata, selected_reference_patient, validate_guideline_tables,
)
from scripts.med_copilot_observability import capture_graphrag_responses
from scripts.med_copilot_run_file import checkpoint, error_scope, pipeline_stage
from scripts.med_copilot_variants import COMPONENTS


ROOT = Path(__file__).resolve().parents[2]


def validate_preparation_result(experiment, result):
    """Reject malformed or mismatched recovery state before any paid work."""
    if not isinstance(result, dict) or not isinstance(result.get("id"), str) or not result["id"].strip():
        raise ValueError("Saved preparation must name a patient")
    expected = new_patient_result(
        result["id"], model_ids=experiment.selected_models,
        variant_ids=experiment.selected_variants, component_plan=experiment.component_plan,
        variant_components=experiment.variant_components,
    )
    statuses = ("completed", "failed_incomplete", "not_processed")
    if (result.get("status") not in (*statuses, None)
            or not isinstance(result.get("components"), dict)
            or set(result["components"]) != set(expected["components"])
            or not isinstance(result.get("variants"), dict)
            or set(result["variants"]) != set(expected["variants"])):
        raise ValueError("Saved preparation components or variants do not match the experiment")
    for name, component in result["components"].items():
        fresh = expected["components"][name]
        if (not isinstance(component, dict) or not set(fresh) <= set(component)
                or component.get("component") != fresh["component"]
                or component.get("settings") != fresh["settings"]
                or component.get("status") not in statuses
                or not isinstance(component.get("capture"), dict)
                or not isinstance(component.get("outputs"), dict)):
            raise ValueError("Invalid saved evidence component")
        responses = component["capture"].get("model_responses")
        if (not isinstance(responses, dict)
                or not set(fresh["capture"]["model_responses"]) <= set(responses)
                or any(not isinstance(items, list) or any(not isinstance(item, dict) for item in items)
                       for items in responses.values())):
            raise ValueError("Invalid saved evidence response capture")
        if component["status"] == "completed" and any(
                not isinstance(component["outputs"].get(key), str)
                for key in COMPONENTS[fresh["component"]].outputs):
            raise ValueError("Completed evidence is missing its outputs")
    for name, output in result["variants"].items():
        fresh = expected["variants"][name]
        if (not isinstance(output, dict) or not set(fresh) <= set(output)
                or output.get("components") != fresh["components"]
                or output.get("status") not in statuses
                or not isinstance(output.get("assessment_plans"), dict)
                or set(output["assessment_plans"]) != set(fresh["assessment_plans"])):
            raise ValueError("Invalid saved variant/model matrix")
        prompts = output.get("assessment_plan_prompts")
        if prompts is not None and (not isinstance(prompts, dict)
                or set(prompts) != {"system", "user"}
                or any(not isinstance(value, str) for value in prompts.values())):
            raise ValueError("Invalid saved exact prompt")
        for model, slot in output["assessment_plans"].items():
            if (not isinstance(slot, dict) or not set(fresh["assessment_plans"][model]) <= set(slot)
                    or slot.get("status") not in statuses
                    or not isinstance(slot.get("model_responses"), list)
                    or any(not isinstance(item, dict) for item in slot["model_responses"])
                    or slot.get("evaluation") is not None and not isinstance(slot["evaluation"], list)
                    or "_generation_reply_complete" in slot and type(slot["_generation_reply_complete"]) is not bool):
                raise ValueError("Invalid saved generation slot")
            predictions = slot.get("predictions")
            if predictions is not None and (not isinstance(predictions, list) or len(predictions) != 3
                    or any(not isinstance(item, dict) or set(item) - {"condition", "plan"}
                           or item.get("condition") is not None and not isinstance(item["condition"], str)
                           or item.get("plan") is not None and (
                               not isinstance(item["plan"], list) or len(item["plan"]) > 3
                               or any(not isinstance(action, str) for action in item["plan"]))
                           for item in predictions)):
                raise ValueError("Invalid saved ranked predictions")
            evaluation = slot.get("evaluation")
            if evaluation is not None and (len(evaluation) != 3 or any(
                    not isinstance(entry, dict) or type(entry.get("rank")) is not int
                    or entry["rank"] != index for index, entry in enumerate(evaluation, 1))):
                raise ValueError("Invalid saved ranked evaluation")


class _PreparationCheckpoint:
    """Route existing component checkpoint/error hooks to a durable parent owner."""

    def __init__(self, save, error_callback, credentials):
        self.save = save
        self.error_callback = error_callback
        self.credentials = credentials
        self.current = None
        self.metadata = {"errors": []}

    def redact(self, text):
        return redact_secrets(text, self.credentials)

    def checkpoint(self):
        if self.save is not None and self.current is not None:
            self.save(deepcopy(self.current))

    def add_error(self, entry):
        self.metadata["errors"].append(deepcopy(entry))
        if self.error_callback is not None:
            self.error_callback(deepcopy(entry))


class EvidencePreparer:
    """Reuse one worker's resources across patients, without generating final answers.

    ``checkpoint(result)`` and ``error_callback(error)`` must return only after
    their parent has durably saved the event. They can raise to stop execution.
    Retrieval factories and the preloaded encoder are injected by the ordered
    worker bootstrap; tests inject lightweight stand-ins without model/API calls.
    """

    def __init__(self, config_path, *, device="cpu", checkpoint=None, error_callback=None,
                 encoder=None, retriever_factory=None, reranker_factory=None,
                 guideline_search=None, client_factory=OpenAI, root=ROOT):
        self.experiment = ExperimentConfig(config_path)
        self.root = Path(root).resolve()
        self.device = device
        self.encoder = encoder
        self.guideline_search = guideline_search
        self.corpus_records = []
        self.retriever = self.reranker = None
        self.client = None
        self._owner = _PreparationCheckpoint(
            checkpoint, error_callback,
            [os.environ.get(name) for name in self.experiment.required_credentials],
        )
        experiment = self.experiment
        self.guideline_output = experiment.guidelines / "output"

        if experiment.needs_patient_retrieval:
            if encoder is None or retriever_factory is None or reranker_factory is None:
                raise ValueError("Patient retrieval requires a preloaded encoder and retrieval factories")
            self.corpus_records = json.loads(experiment.patient_corpus.read_text(encoding="utf-8"))
            if not isinstance(self.corpus_records, list) or not self.corpus_records:
                raise ValueError("Reference corpus must be a nonempty list")
            experiment.patient_retrieval_artifacts_dir.mkdir(parents=True, exist_ok=True)
            # Constructors validate/rebuild shared artifacts. Serialize that work
            # across spawned workers so metadata is never published out of order.
            lock_path = experiment.patient_retrieval_artifacts_dir / ".initialization.lock"
            with lock_path.open("a", encoding="utf-8") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                with chdir(experiment.patient_retrieval_artifacts_dir):
                    self.retriever = retriever_factory(
                        pd.DataFrame(self.corpus_records), alpha=experiment.patient_retrieval_alpha,
                        encoder=encoder, model_name=experiment.patient_embedding_model,
                        llm_model=experiment.patient_keyword_model,
                    )
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            self.retriever.keyword_extractor = JSONKeywordExtractor(
                client=self.retriever.keyword_extractor.client,
                model=self.retriever.keyword_extractor.model,
            )
            self.reranker = reranker_factory(model_name=experiment.patient_reranker_model)

        if experiment.needs_guidelines:
            if guideline_search is None:
                raise ValueError("Guidelines require a local-search callable")
            validate_guideline_tables(self.guideline_output)
            self.client = client_factory(api_key=os.environ["OPENAI_API_KEY"])

    @property
    def errors(self):
        """Detached errors from the latest patient; no credential values are returned."""
        return deepcopy(self._owner.metadata["errors"])

    def metadata(self, root=None):
        """Fingerprint shared evidence and report actual worker encoder placement."""
        experiment = self.experiment
        artifacts = None if self.retriever is None else [
            experiment.patient_retrieval_artifacts_dir / getattr(self.retriever.embedder, name)
            for name in ("emb_path", "index_path", "metadata_path")
        ]
        return {
            "reference_data": reference_data_metadata(
                self.root if root is None else root,
                corpus_file=experiment.patient_corpus if experiment.needs_patient_retrieval else None,
                artifacts_directory=experiment.patient_retrieval_artifacts_dir,
                artifact_files=artifacts,
                guideline_directory=experiment.guidelines if experiment.needs_guidelines else None,
                guideline_settings=experiment.guidelines / "settings.yaml",
                guideline_output=self.guideline_output,
            ),
            "devices": {
                "requested": self.device,
                "patient_encoder": None if self.encoder is None else str(getattr(self.encoder, "device", "unknown")),
                "reranker": None if self.reranker is None else str(getattr(self.reranker.model, "device", "unknown")),
            },
        }

    def prepare(self, patient, result):
        """Complete pending components and exact prompts, preserving received API work."""
        if not isinstance(result, dict) or result.get("id") != patient.get("id"):
            raise ValueError("Preparation result must belong to the supplied patient")
        for field in ("subjective", "objective"):
            if not isinstance(patient.get(field), str):
                raise ValueError(f"Patient {field} must be a string")
        self._validate_result(result)
        for component in result["components"].values():
            if component["status"] != "completed":
                component.update(status="not_processed", error=None)
        self._owner.current = result
        self._owner.metadata["errors"] = []
        try:
            with error_scope(self._owner, patient_id=patient["id"]):
                self._prepare_components(patient, result)
                s_o = f'Subjective: {patient["subjective"]}\nObjective: {patient["objective"]}'
                for variant, output in result["variants"].items():
                    prepare_variant_messages(
                        self.experiment, variant, s_o, result, output,
                        templates=VARIANT_TEMPLATES, error_metadata=self._error_metadata,
                    )
                checkpoint()
        finally:
            self._owner.current = None
        return result

    def _validate_result(self, result):
        validate_preparation_result(self.experiment, result)

    def _error_metadata(self, exc):
        return {"type": type(exc).__name__, "message": self._owner.redact(str(exc))}

    def _prepare_components(self, patient, result):
        experiment = self.experiment
        s_o = f'Subjective: {patient["subjective"]}\nObjective: {patient["objective"]}'

        # %% ------------------------------------------------------------------
        # Similar-patient retrieval and reranking, once per configuration
        # ---------------------------------------------------------------------
        for instance, settings in experiment.component_instances("patient_retrieval"):
            component = result["components"][instance]
            if component["status"] != "not_processed":
                continue
            with pipeline_stage("patient_retrieval"), capture_component(
                    component, error_metadata=self._error_metadata, corpus_records=self.corpus_records,
                    retriever=self.retriever, reranker=self.reranker,
                    credential_names=experiment.required_credentials) as capture:
                with chdir(experiment.patient_retrieval_artifacts_dir):
                    self.retriever.alpha = settings["alpha"]
                    candidates = self.retriever.search(s_o, topk=settings["top_k"])
                    results = []
                    if candidates:
                        with pipeline_stage("reranking"):
                            results = self.reranker.rerank(
                                s_o, candidates, topk=settings["rerank_top_k"], format_top_k=1)
                capture["selected_reference_patient"] = selected_reference_patient(
                    self.corpus_records, candidates, results)
                component["outputs"]["example"] = results[0] if results else "No reference patient available."

        # %% ------------------------------------------------------------------
        # Clinical guidelines: questions and GraphRAG, once per configuration
        # ---------------------------------------------------------------------
        for instance, settings in experiment.component_instances("guidelines"):
            component = result["components"][instance]
            if component["status"] != "not_processed" or not component_needed(experiment, instance, result):
                continue
            with pipeline_stage("guideline_retrieval"), capture_component(
                    component, error_metadata=self._error_metadata,
                    credential_names=experiment.required_credentials) as capture:
                key_questions = prepare_guideline_query(
                    self.client, settings, s_o, capture, template=KEY_QUESTIONS_TEMPLATE)
                with chdir(self.guideline_output), capture_graphrag_responses(capture["model_responses"]):
                    key_info, context = self.guideline_search(
                        config_filepath=experiment.guidelines / "settings.yaml",
                        data_dir=self.guideline_output, root_dir=experiment.guidelines,
                        community_level=settings["community_level"], response_type=settings["response_type"],
                        streaming=False, query=key_questions, verbose=settings["verbose"],
                    )
                capture["guideline_evidence"] = key_info
                capture["graphrag_context"] = graphrag_context_json(context)
                component["outputs"]["Key_info"] = format(key_info)

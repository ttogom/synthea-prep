"""Offline runner tests; activate an environment with pandas first.

Run directly: python3 tests/test_run_med_copilot.py
Or discover: python3 -m unittest discover -s tests -p 'test_*.py'
Models and API clients are mocked; selected upstream functions are loaded
without importing their heavyweight dependencies. Each fixture is a temporary
Git repository so run naming uses the real branch/commit lookup.
Prepared S/O text is trusted; the loader checks JSON objects, ids, and string fields.
"""

import ast
import asyncio
import builtins
from copy import deepcopy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import errno
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import yaml

from test_med_copilot_retrieval import FakeEncoder, load_retrieval_module
from test_med_copilot_observability import ProviderHarness


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_med_copilot.py"
HELPERS = ROOT / "scripts" / "med_copilot_helpers.py"
RETRIEVAL = ROOT / "scripts" / "med_copilot_retrieval.py"
OBSERVABILITY = ROOT / "scripts" / "med_copilot_observability.py"
SCORING = ROOT / "scripts" / "med_copilot_scoring.py"
MODELS = ROOT / "scripts" / "med_copilot_models.py"
CONFIG = ROOT / "scripts" / "med_copilot_config.py"
STARTUP = ROOT / "scripts/med_copilot_startup.py"
PIPELINE_DEFAULTS = ast.literal_eval(next(node.value for node in ast.parse(CONFIG.read_text()).body
    if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PIPELINE_DEFAULTS" for t in node.targets)))
EXPERIMENT_CONFIG = ROOT / "experiments/configs/experiment.yaml"
COLLATOR = ROOT / "scripts/collate_med_copilot_results.py"
MODEL_CONFIG = ROOT / "experiments" / "configs" / "condition_plan_models.yaml"
UPSTREAM_RETRIEVER = ROOT / "med_copilot" / "upstream" / "create_embeddings.py"
UPSTREAM_EVALUATOR = ROOT / "med_copilot" / "upstream" / "evaluate_metrics.py"
UPSTREAM_GRAPH_QUERY = ROOT / "med_copilot/upstream/graphrag/cli/query.py"
UPSTREAM_GRAPH_STORAGE = ROOT / "med_copilot/upstream/graphrag/utils/storage.py"
UPSTREAM_GRAPH_ENTITY_EXTRACTION = (
    ROOT / "med_copilot/upstream/graphrag/query/context_builder/entity_extraction.py"
)
GUIDELINE_SETTINGS = ROOT / "med_copilot/data/guidelines/settings.yaml"
GUIDELINE_TABLES = ("communities", "community_reports", "text_units", "relationships", "entities")
PROJECT_PROMPTS = ROOT / "med_copilot/prompts/diagnosis_plan.py"
INPUT_NAME = "test_sample_so.jsonl"
LABELS_NAME = "test_sample_labels.jsonl"
BRANCH = "feature/issue-9"
SAFE_BRANCH = "feature-issue-9"
RESULT_FIELDS = [
    "id", "status", "predictions", "assessment_plan_raw", "selected_reference_patient",
    "guideline_questions", "guideline_evidence", "graphrag_context", "assessment_plan_prompts",
    "keyword_extraction_raw", "retrieval_keywords", "retrieved_patient_ids",
    "reranked_patient_ids", "model_responses", "evaluation",
]
RUN_FIELDS = [
    "run_name", "started_at", "finished_at", "status", "branch", "input", "git_commit",
    "reference_data", "schema_version", "assessment_plan_models", "variant_definitions", "prompt_provenance", "experiment", "evaluation",
]
METRIC_NAMES = ("BLEU", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore_F1")
FINAL_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "condition_plan", "strict": True,
        "schema": {
            "type": "object",
            "properties": {"predictions": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "condition": {"type": "string"},
                        "plan": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                    },
                    "required": ["condition", "plan"], "additionalProperties": False,
                },
            }},
            "required": ["predictions"], "additionalProperties": False,
        },
    },
}


V2_RESULT_FIELDS = [field for field in RESULT_FIELDS
                    if field not in ("predictions", "assessment_plan_raw", "evaluation")] + ["assessment_plans"]


V3_RESULT_FIELDS = [field for field in V2_RESULT_FIELDS if field not in ("assessment_plans", "assessment_plan_prompts")] + ["evidence_status", "variants"]


def combined_variant_result(record):
    """View the combined variant for pre-existing pipeline regressions.

    New matrix tests inspect native v3 rows directly; this view keeps all earlier
    generation/retrieval/scoring assertions rather than deleting their coverage.
    """
    if "variants" not in record:
        return record
    variant = record["variants"]["patient_retrieval_guidelines"]
    return {field: (variant[field] if field in ("assessment_plans", "assessment_plan_prompts") else record[field])
            for field in V2_RESULT_FIELDS}


def legacy_patient_result(record):
    """Project the default model for the preserved single-model regression assertions.

    Native schema/selection/failure assertions live in MultiModelIntegrationTests.
    No production fallback fields are added to saved records.
    """
    record = combined_variant_result(record)
    if "assessment_plans" not in record:
        return record
    projected = deepcopy(record)
    output = projected.pop("assessment_plans")["gpt_4o_mini"]
    for key in ("predictions", "assessment_plan_raw", "evaluation"):
        projected[key] = output[key]
    projected["model_responses"]["assessment_plan"] = output["model_responses"]
    return {key: projected[key] for key in RESULT_FIELDS}


def prediction_pairs(condition="MOCK_DIAGNOSIS", plan=None):
    """Independent ranked mock answers, preserving each patient's marker."""
    return [{"condition": condition if rank == 1 else f"{condition} alternative {rank}",
             "plan": list(["MOCK_FINAL_PLAN"] if plan is None else plan)}
            for rank in (1, 2, 3)]

LEGACY_OUTPUTS = {
    "top_similar_patient.txt", "questions_for_guidelines_db.txt", "answers_from_guidelines_db.txt",
    "graphrag_context.json", "final_prompt.txt", "final_output.txt",
}
GUIDELINE_SETTINGS_TEXT = """\
local_search:
  prompt: "prompts/local_search_system_prompt.txt"
vector_store:
  default_vector_store:
    type: lancedb
    db_uri: output/lancedb
    container_name: default
"""
HELPER_NAMESPACE = runpy.run_path(str(HELPERS))
SHA256_BUNDLE = HELPER_NAMESPACE["sha256_bundle"]
PACIFIC_TIME = HELPER_NAMESPACE["pacific_time"]
PROMPT_NAMES = {"KEY_QUESTIONS_TEMPLATE", "EVALUATE_TEMPLATE_KEYINFO"}
def prompt_literals(path):
    # Extract only literals: importing the frozen upstream is unnecessary.
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in PROMPT_NAMES
    }


PROMPTS = prompt_literals(PROJECT_PROMPTS)


def frozen_keyword_extractor_class():
    # Execute only this dependency-free class from the unchanged upstream
    # source, so the regression exercises its real prompt and JSON parser.
    tree = ast.parse(UPSTREAM_RETRIEVER.read_text(encoding="utf-8"))
    extractor = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "LLMKeywordExtractor")
    namespace = {"json": json}
    exec(compile(ast.Module(body=[extractor], type_ignores=[]),
                 str(UPSTREAM_RETRIEVER), "exec"), namespace)
    return namespace["LLMKeywordExtractor"]


FROZEN_KEYWORD_EXTRACTOR = frozen_keyword_extractor_class()


def frozen_graphrag_query(load_config, create_storage, api):
    # Run the real vendored CLI/table-loading functions while replacing only
    # config, storage construction, and the external query API.
    storage_names = {"load_table_from_storage", "storage_has_table"}
    storage_namespace = {"pd": pd, "PipelineStorage": object}
    storage_tree = ast.parse(UPSTREAM_GRAPH_STORAGE.read_text(encoding="utf-8"))
    storage_functions = [node for node in storage_tree.body
                         if isinstance(node, ast.AsyncFunctionDef)
                         and node.name in storage_names]
    exec(compile(ast.Module(body=storage_functions, type_ignores=[]),
                 str(UPSTREAM_GRAPH_STORAGE), "exec"), storage_namespace)
    query_namespace = {
        "asyncio": asyncio, "sys": sys, "Path": Path,
        "GraphRagConfig": object, "Any": object, "api": api,
        "load_config": load_config, "create_storage_from_config": create_storage,
        **{name: storage_namespace[name] for name in storage_names},
    }
    query_tree = ast.parse(UPSTREAM_GRAPH_QUERY.read_text(encoding="utf-8"))
    query_functions = [node for node in query_tree.body
                       if isinstance(node, ast.FunctionDef)
                       and node.name in {"run_local_search", "_resolve_output_files"}]
    exec(compile(ast.Module(body=query_functions, type_ignores=[]),
                 str(UPSTREAM_GRAPH_QUERY), "exec"), query_namespace)
    return query_namespace["run_local_search"]


def keyword_response(content, finish_reason="stop", refusal=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, refusal=refusal),
    )])


def git(root, *args):
    # Isolate fixture commits from user hooks and signing configuration.
    return subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.com",
         "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def frozen_reranker_class(scores):
    # Preserve upstream scoring/order/formatting, replacing its model and adding
    # the project's capture contract immediately after its retained sort. This
    # test-only AST instrumentation does not change the vendored source file.
    tree = ast.parse(UPSTREAM_RETRIEVER.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if (isinstance(node, ast.ClassDef) and node.name == "CrossEncoderReranker")
             or (isinstance(node, ast.Assign)
                 and any(getattr(target, "id", None) == "CONTRIBUTION_THRESHOLD"
                         for target in node.targets))]
    reranker = next(node for node in nodes if isinstance(node, ast.ClassDef))
    rerank = next(node for node in reranker.body
                  if isinstance(node, ast.FunctionDef) and node.name == "rerank")
    retained_sort = next(index for index, node in enumerate(rerank.body)
                         if isinstance(node, ast.Assign)
                         and any(getattr(target, "id", None) == "reranked" for target in node.targets))
    rerank.body[retained_sort + 1:retained_sort + 1] = ast.parse(
        "self.ranked_candidates = reranked\n"
        "if getattr(self, 'on_ranking', None) is not None:\n"
        "    self.on_ranking(reranked)\n"
    ).body
    rerank.body.insert(0, ast.parse("self.ranked_candidates = None").body[0])
    namespace = {"CrossEncoder": lambda model_name: SimpleNamespace(
        predict=lambda pairs: list(scores[:len(pairs)]),
    )}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(UPSTREAM_RETRIEVER), "exec"), namespace)
    return namespace["CrossEncoderReranker"]


def sequential_clock(start=datetime(2026, 7, 4, 19, 0, tzinfo=timezone.utc)):
    # Each runner call advances one second, so repeated runs get distinct names.
    ticks = itertools.count()
    return lambda: PACIFIC_TIME(start + timedelta(seconds=next(ticks)))


_GIT_TEMPLATE = tempfile.TemporaryDirectory()


def template_git_dir():
    # One real repository is created per test session; fixtures copy its .git
    # so the runner still resolves branch and commit through git itself.
    template = Path(_GIT_TEMPLATE.name) / "repo"
    if not template.exists():
        template.mkdir()
        subprocess.run(["git", "init", "-q", "-b", BRANCH, str(template)], check=True, capture_output=True)
        git(template, "commit", "-q", "--allow-empty", "-m", "fixture")
    return template / ".git"


class PipelineFixture:
    def __init__(self, root, branch=BRANCH):
        root = root.resolve()
        self.root = root
        self.branch = branch
        self.safe_branch = SAFE_BRANCH if branch == BRANCH else branch
        self.upstream = root / "med_copilot/upstream"
        self.artifacts = root / "med_copilot/data/patient_retrieval_so"
        self.guidelines = root / "med_copilot/data/guidelines"
        self.output = root / "experiments/output"
        self.caller = root / "caller"
        self.script = root / "scripts/run_med_copilot.py"
        for directory in [self.upstream, self.artifacts, self.guidelines / "output", self.caller,
                          self.script.parent, root / "experiments/input"]:
            directory.mkdir(parents=True)
        for name in GUIDELINE_TABLES:
            (self.guidelines / "output" / f"{name}.parquet").write_text("MOCK_TABLE")
        self.write_guideline_settings(self.guidelines)
        for name, content in {"embeddings.npy": b"MOCK_EMBEDDINGS",
                              "index.faiss": b"MOCK_INDEX", "matching.json": b"MOCK_SO_METADATA"}.items():
            (self.artifacts / name).write_bytes(content)
        # Legacy files must remain untouched and must not enter run fingerprints.
        for name in ("embeddings.npy", "index.faiss"):
            (self.upstream / name).write_bytes(b"LEGACY_SOA")
        self.regenerated_artifacts = None
        # An unchanged copy lets runpy exercise the actual __main__ guard.
        self.script.write_text(RUNNER.read_text())
        self.helper_module = self.script.parent / "med_copilot_helpers.py"
        self.helper_module.write_text(HELPERS.read_text(encoding="utf-8"), encoding="utf-8")
        (self.script.parent / "med_copilot_retrieval.py").write_text(
            RETRIEVAL.read_text(encoding="utf-8"), encoding="utf-8")
        (self.script.parent / "med_copilot_observability.py").write_text(
            OBSERVABILITY.read_text(encoding="utf-8"), encoding="utf-8")
        if SCORING.exists():
            (self.script.parent / SCORING.name).write_text(
                SCORING.read_text(encoding="utf-8"), encoding="utf-8")
        (self.upstream / "evaluate_metrics.py").write_bytes(UPSTREAM_EVALUATOR.read_bytes())
        (self.script.parent / MODELS.name).write_bytes(MODELS.read_bytes())
        (self.script.parent / CONFIG.name).write_bytes(CONFIG.read_bytes())
        (self.script.parent / STARTUP.name).write_bytes(STARTUP.read_bytes())
        (self.script.parent / COLLATOR.name).write_bytes(COLLATOR.read_bytes())
        self.model_config = root / "experiments/configs/condition_plan_models.yaml"
        self.model_config.parent.mkdir()
        self.model_config.write_bytes(MODEL_CONFIG.read_bytes())
        self.experiment_config = self.model_config.with_name("experiment.yaml")
        self.experiment_config.write_bytes(EXPERIMENT_CONFIG.read_bytes())
        # Pipeline regressions use an explicit single-model, uncapped selection.
        experiment = yaml.safe_load(self.experiment_config.read_text())
        experiment["selected_models"] = ["gpt_4o_mini"]
        experiment["max_patients"] = None
        experiment["selected_variants"] = ["patient_retrieval_guidelines"]
        experiment["exclude_none_recorded_plans"] = False
        self.experiment_config.write_text(yaml.safe_dump(experiment, sort_keys=False))
        self.final_reply = None
        self.environment = {"OPENAI_API_KEY": "test-key", "GRAPHRAG_API_KEY": "test-key"}
        self.confirm_input = Mock(return_value="y")
        self.model_configs = []
        self.retrieval_settings = []
        self.reranker_models = []
        self.real_scoring = False
        self.scoring_calls = []
        self.scoring_reply = None
        self.metric_values = {name: (index + 1) / 10 for index, name in enumerate(METRIC_NAMES)}
        self.prompt_module = root / "med_copilot/prompts/diagnosis_plan.py"
        self.prompt_module.parent.mkdir(parents=True)
        self.prompt_module.write_text(PROJECT_PROMPTS.read_text(encoding="utf-8"), encoding="utf-8")
        self.prompts = PROMPTS.copy()
        self.corpus = [
            {"note_id": "NOTE-A", "subjective": "Corpus S", "objective": "Corpus O",
             "assessment": "Corpus A", "plan": "Corpus P"},
            {"note_id": "NOTE-B", "subjective": "Second corpus S", "objective": "Second corpus O",
             "assessment": "Second corpus A", "plan": "Second corpus P"},
        ]
        self.write_corpus()
        self.patient_id = "FIRST_PATIENT_ID"
        self.patient = {
            "subjective": "2000-07-20: Cough for 3 days; denies fever. café.",
            "objective": "Temperature 37.2 °C; SpO₂ 98%; weight 70 kg on 2000-07-20.",
        }
        self.input_path = root / "experiments/input" / INPUT_NAME
        self.labels_path = self.input_path.with_name(LABELS_NAME)
        self.write_patients()
        self.conditions = (
            f'Subjective: {self.patient["subjective"]}\n'
            f'Objective: {self.patient["objective"]}'
        )
        self.questions = '{"Key Questions": ["Question one?", "Question two?"]}'
        self.guideline_text = "GUIDELINE_RESPONSE_ONLY\nGuideline evidence with café."
        self.final_sections = {"predictions": prediction_pairs()}
        self.final_output = json.dumps(self.final_sections)
        self.candidates = [{"idx": 0, "id": "candidate-one"}, {"idx": 1, "id": "candidate-two"}]
        # The reranker moves the second candidate (corpus row 1) to the top.
        self.rerank_scores = [0.2, 0.9]
        self.reference = {"note_id": "NOTE-B"}
        self.results = [
            "SELECTED_SIMILAR_CASE\nAssessment: REFERENCE_DIAGNOSIS\nPlan: REFERENCE_PLAN",
            "UNSELECTED_SIMILAR_CASE",
        ]
        self.context = {
            "sources": pd.DataFrame({
                "id": list(range(135)),
                "text": [f"RAW_CONTEXT_ONLY_{i} café\nsource" for i in range(135)],
                "score": [i / 10 if i % 2 else float("nan") for i in range(135)],
                "in_context": [bool(i % 2) for i in range(135)],
            }, index=range(500, 635)),
            "claims": pd.DataFrame(columns=["id", "claim"]),
        }
        self.events = []
        self.calls = []
        self.keyword_calls = []
        self.keyword_response = None
        self.completion_responses = {}
        self.keyword_content = json.dumps({
            "symptoms": [" Cough ", "Cough", "denies fever"],
            "diagnoses": [], "medications": [],
        })
        self.extracted_keywords = None
        self.retrieval_calls = []
        self.rerank_calls = []
        self.graph_calls = []
        self.completion_cwds = []
        self.failure = None
        # When set, only a stage whose prompt/query contains this text fails.
        self.failure_marker = None
        # Optional reply(stage, text) overriding question/final/GraphRAG responses.
        self.reply = None
        self.error = RuntimeError("mock stage failure")
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.modules = self._modules()
        git_dir = template_git_dir()
        shutil.copytree(git_dir, root / ".git")
        self.commit = (git_dir / "refs/heads" / BRANCH).read_text().strip()
        if branch == "HEAD":
            (root / ".git/HEAD").write_text(self.commit + "\n")
        elif branch != BRANCH:
            ref = root / ".git/refs/heads" / branch
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(self.commit + "\n")
            (root / ".git/HEAD").write_text(f"ref: refs/heads/{branch}\n")

    def write_guideline_settings(self, directory):
        (directory / "prompts").mkdir(parents=True, exist_ok=True)
        (directory / "settings.yaml").write_text(GUIDELINE_SETTINGS_TEXT, encoding="utf-8")
        (directory / "prompts/local_search_system_prompt.txt").write_text("ACTIVE_LOCAL_PROMPT")
        # Present in the directory but not used by local search.
        (directory / "prompts/extract_graph.txt").write_text("UNUSED_INDEXING_PROMPT")
        vectors = directory / "output/lancedb/default-entity-description.lance/data"
        vectors.mkdir(parents=True, exist_ok=True)
        (vectors / "fragment.lance").write_bytes(b"MOCK_VECTORS")

    def write_corpus(self, path=None):
        path = self.upstream / "soap_with_metadata.json" if path is None else path
        path.write_text(json.dumps(self.corpus, ensure_ascii=False), encoding="utf-8")

    def write_patients(self, records=None):
        if records is None:
            records = [{"id": self.patient_id, **self.patient}]
        self.input_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def fail_at(self, stage, text=""):
        if self.failure == stage and (self.failure_marker is None or self.failure_marker in text):
            raise self.error

    def update_experiment(self, **settings):
        data = yaml.safe_load(self.experiment_config.read_text())
        data.update(settings)
        self.experiment_config.write_text(yaml.safe_dump(data, sort_keys=False))

    def write_labels(self, records):
        self.labels_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def _modules(self):
        fixture = self

        def preload(model_name):
            fixture.events.append(("preload", model_name))
            fixture.fail_at("preload")
            return object()

        def completion(_stage=None, **kwargs):
            fixture.calls.append(kwargs)
            fixture.completion_cwds.append(Path.cwd())
            stage = _stage or ("questions" if kwargs.get("response_format") == {"type": "json_object"} else "final")
            prompt = kwargs["messages"][-1]["content"]
            fixture.fail_at(stage, prompt)
            if stage == "final" and fixture.final_reply is not None:
                return fixture.final_reply(kwargs)
            content = fixture.questions if stage == "questions" else fixture.final_output
            if fixture.reply is not None:
                content = fixture.reply(stage, prompt)
            return fixture.completion_responses.get(stage, SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]))

        def client(api_key):
            fixture.events.append(("client", api_key))
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))

        def keyword_completion(**kwargs):
            fixture.keyword_calls.append(kwargs)
            fixture.fail_at("keywords", kwargs["messages"][-1]["content"])
            if fixture.keyword_response is not None:
                return (fixture.keyword_response(kwargs) if callable(fixture.keyword_response)
                        else fixture.keyword_response)
            # Reproduce the original failure: a perfectly valid model answer
            # wrapped in Markdown when JSON output was requested only in prose.
            content = fixture.keyword_content
            if kwargs.get("response_format") != {"type": "json_object"}:
                content = f"```json\n{content}\n```"
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content, refusal=None),
            )])

        class HybridRetriever:
            def __init__(self, df, alpha, encoder, model_name="emilyalsentzer/Bio_ClinicalBERT", llm_model="gpt-4o-mini"):
                fixture.retrieval_settings.append({"model_name": model_name, "llm_model": llm_model, "encoder": encoder})
                fixture.retrieval_calls.append(("init", df.copy(), alpha, Path.cwd()))
                fixture.fail_at("retriever_init")
                if fixture.regenerated_artifacts is not None:
                    # Like EmbeddingRetriever, rebuild artifacts relative to the cwd.
                    for name, content in fixture.regenerated_artifacts.items():
                        Path(name).write_bytes(content)
                self.embedder = SimpleNamespace(emb_path="embeddings.npy", index_path="index.faiss",
                                               metadata_path="matching.json")
                self.keyword_extractor = object.__new__(FROZEN_KEYWORD_EXTRACTOR)
                self.keyword_extractor.model = llm_model
                self.keyword_extractor.client = SimpleNamespace(chat=SimpleNamespace(
                    completions=SimpleNamespace(create=keyword_completion),
                ))

            def search(self, query, topk):
                self.ranked_candidates = None
                fixture.retrieval_calls.append(("search", query, topk, Path.cwd()))
                fixture.fail_at("retrieval", query)
                fixture.extracted_keywords = self.keyword_extractor.extract(query)
                self.ranked_candidates = [dict(candidate) for candidate in fixture.candidates[:topk]]
                if getattr(self, "on_ranking", None) is not None:
                    self.on_ranking(self.ranked_candidates)
                fixture.fail_at("retrieval_attribution", query)
                return self.ranked_candidates

        class CrossEncoderReranker:
            def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
                fixture.reranker_models.append(model_name)
                fixture.rerank_calls.append(("init", Path.cwd()))
                fixture.fail_at("reranker_init")

            def rerank(self, query, candidates, topk):
                self.ranked_candidates = None
                fixture.rerank_calls.append(
                    ("rerank", query, [dict(candidate) for candidate in candidates], topk, Path.cwd()))
                fixture.fail_at("reranking", query)
                # Like the upstream reranker, record scores on the candidates.
                for candidate, score in zip(candidates, fixture.rerank_scores):
                    candidate["rerank_score"] = score
                self.ranked_candidates = sorted(
                    candidates, key=lambda item: item["rerank_score"], reverse=True,
                )[:min(topk, len(fixture.results))]
                if getattr(self, "on_ranking", None) is not None:
                    self.on_ranking(self.ranked_candidates)
                fixture.fail_at("rerank_formatting", query)
                return list(fixture.results[:topk])

        def graph_search(**kwargs):
            fixture.graph_calls.append((kwargs, Path.cwd()))
            fixture.fail_at("graphrag", kwargs["query"])
            if fixture.reply is not None:
                return fixture.reply("graphrag", kwargs["query"])
            return fixture.guideline_text, fixture.context

        class ModelConfig:
            model_fields = {"type", "model", "model_provider", "temperature", "max_retries",
                            "retry_strategy", "api_base", "n", "api_key"}

            def __init__(self, **settings):
                self.settings = settings
                self.__dict__.update(settings)

            def model_dump(self, **kwargs):
                return {key: value for key, value in self.settings.items() if key != "api_key"}

        def create_chat_model(model_type, name, config):
            fixture.model_configs.append((name, config.settings.copy()))
            return SimpleNamespace(completion=lambda **kwargs: completion(
                _stage="final", model=f"{config.model_provider}/{config.model}", **kwargs))

        modules = {name: ModuleType(name) for name in [
            "sentence_transformers", "openai", "create_embeddings", "scripts.med_copilot_retrieval", "graphrag",
            "graphrag.cli", "graphrag.cli.query", "templates", "templates.SOA_P_TEMPLATE",
            "scripts.med_copilot_scoring", "med_copilot.upstream.evaluate_metrics",
            "graphrag.config.models.language_model_config", "graphrag.language_model.factory",
        ]}
        class NoopPatientScorer:
            # Legacy regressions retain every assertion about generation.
            # Dedicated integration tests below exercise the actual scorer.
            def __init__(self, labels_path, patients, presenting_evidence_only=True, *, max_patients=None, exclude_none_recorded_plans=False):
                self.patients = patients

            def metadata(self, root):
                return {"test_double": True, **{key: len(self.patients) for key in (
                    "input_patient_count", "after_presenting_evidence_count", "eligible_patient_count", "selected_patient_count")}}

            def score_result(self, result):
                pass

        def evaluate_metrics(references, candidates, lang="en"):
            fixture.scoring_calls.append((list(references), list(candidates), lang))
            if fixture.scoring_reply is not None:
                return fixture.scoring_reply(references, candidates, lang)
            return fixture.metric_values.copy()

        modules["graphrag.config.models.language_model_config"].LanguageModelConfig = ModelConfig
        modules["graphrag.language_model.factory"].ModelFactory = SimpleNamespace(create_chat_model=create_chat_model)
        report_module = ModuleType("scripts.collate_med_copilot_results")
        def fake_report(directory, **kwargs):
            path = directory / "results_summary.html"
            path.write_text("<html>mock report</html>")
            return path
        report_module.generate_report = fake_report
        report_module.load_report_data = lambda *args, **kwargs: {"patients": []}
        report_module.format_top_rank_summary = lambda data: "mock summary"
        modules["scripts.collate_med_copilot_results"] = report_module
        modules["scripts.med_copilot_scoring"].PatientScorer = NoopPatientScorer
        modules["scripts.med_copilot_scoring"].parse_scoring_args = lambda argv=None: SimpleNamespace(
            presenting_evidence_only=None)
        modules["med_copilot.upstream.evaluate_metrics"].Evaluator = SimpleNamespace(
            evaluate_metrics=evaluate_metrics)
        for name in ["graphrag", "graphrag.cli", "templates"]:
            modules[name].__path__ = []
        modules["sentence_transformers"].SentenceTransformer = preload
        modules["openai"].OpenAI = client
        modules["create_embeddings"].HybridRetriever = HybridRetriever
        modules["create_embeddings"].CrossEncoderReranker = CrossEncoderReranker
        modules["scripts.med_copilot_retrieval"].HybridRetriever = HybridRetriever
        modules["scripts.med_copilot_retrieval"].CrossEncoderReranker = CrossEncoderReranker
        # Exercise the real, dependency-free S/O formatter even in orchestration tests.
        tree = ast.parse(RETRIEVAL.read_text(encoding="utf-8"))
        formatter = next(node for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name == "matching_text")
        exec(compile(ast.Module(body=[formatter], type_ignores=[]), str(RETRIEVAL), "exec"),
             modules["scripts.med_copilot_retrieval"].__dict__)
        modules["graphrag.cli.query"].run_local_search = graph_search
        modules["templates.SOA_P_TEMPLATE"].__dict__.update(PROMPTS)
        return modules

    def assessment_models(self, runner):
        experiment = runner["ExperimentConfig"](self.experiment_config)
        return runner["AssessmentPlanModels"](
            experiment.model_config, experiment.selected_models,
            temperature=experiment.temperature, max_retries=experiment.max_retries)

    @contextmanager
    def runtime(self):
        original_import = builtins.__import__

        def tracked_import(name, *args, **kwargs):
            if name == "scripts.med_copilot_retrieval":
                self.events.append(("import", name))
                self.fail_at("upstream_import")
            return original_import(name, *args, **kwargs)

        old_cwd = Path.cwd()
        # Discovery adds the real checkout to sys.path. Keep fixture namespace
        # packages from falling back to real project files when one is absent.
        isolated_path = [entry for entry in sys.path if Path(entry).resolve() != ROOT]
        os.chdir(self.caller)
        try:
            # PATH is kept only so the runner can find git.
            selected_modules = {name: module for name, module in self.modules.items()
                                if not (self.real_scoring and name in ("scripts.med_copilot_scoring", "scripts.collate_med_copilot_results"))}
            with patch.dict(sys.modules, selected_modules), \
                    patch.dict(os.environ, {"PATH": os.environ.get("PATH", os.defpath),
                                            **self.environment}, clear=True), \
                    patch.object(sys, "path", isolated_path), \
                    patch.object(sys, "argv", [str(self.script)]), \
                    patch("builtins.input", self.confirm_input), \
                    patch("builtins.__import__", side_effect=tracked_import), \
                    redirect_stdout(self.stdout), redirect_stderr(self.stderr):
                # Exercise this fixture's real project prompts, even when a
                # caller has already imported the repository's package.
                for name in ["med_copilot.prompts.diagnosis_plan", "med_copilot.prompts", "med_copilot",
                             "scripts.med_copilot_helpers", "scripts.med_copilot_models", "scripts.med_copilot_config",
                             "scripts.med_copilot_observability", "scripts.med_copilot_startup", "scripts"]:
                    sys.modules.pop(name, None)
                if self.real_scoring:
                    sys.modules.pop("scripts.med_copilot_scoring", None)
                    sys.modules.pop("scripts.collate_med_copilot_results", None)
                yield
        finally:
            os.chdir(old_cwd)


class RunMedCopilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = PipelineFixture(Path(self.temporary.name))

    def load_runner(self):
        # Top-level imports initialize dependencies; keep them mocked even
        # when a test loads main() separately from calling it.
        with self.fixture.runtime():
            return runpy.run_path(str(self.fixture.script), run_name="runner_under_test")

    # ----------------------------------------------------------------
    # Saved-output helpers
    # ----------------------------------------------------------------

    def run_dirs(self):
        output = self.fixture.output
        return sorted(output.iterdir()) if output.exists() else []

    def read_run(self, run_dir=None):
        if run_dir is None:
            dirs = self.run_dirs()
            self.assertEqual(len(dirs), 1, dirs)
            run_dir = dirs[0]
        self.assertEqual({path.name for path in run_dir.iterdir()}, {"run.json", "results.jsonl"} | ({"results_summary.html"} if (run_dir / "results_summary.html").exists() else set()))
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        text = (run_dir / "results.jsonl").read_text(encoding="utf-8")
        self.assertTrue(text == "" or text.endswith("\n"))
        # JSONL lines are separated only by "\n"; other Unicode separators stay inside records.
        results = [json.loads(line) for line in text.split("\n")[:-1]]
        for record in results:
            self.assertEqual(list(record), V3_RESULT_FIELDS if "variants" in record else V2_RESULT_FIELDS)
            if "variants" in record:
                self.assertEqual(list(record["variants"]), ["patient_retrieval_guidelines"])
            self.assertEqual(list(combined_variant_result(record)["assessment_plans"]), ["gpt_4o_mini"])
            self.assertNotIn("assessment_plan", record["model_responses"])
        results = [legacy_patient_result(record) for record in results]
        self.assertEqual(len({record["id"] for record in results}), len(results))
        return run_dir, run, results

    def expected_context(self, context=None):
        context = self.fixture.context if context is None else context
        return {key: json.loads(table.to_json(orient="split")) for key, table in context.items()}

    def expected_prompt(self, reference_text=None, conditions=None, guideline_text=None):
        fixture = self.fixture
        return fixture.prompts["EVALUATE_TEMPLATE_KEYINFO"].format(
            conditions=fixture.conditions if conditions is None else conditions,
            example=fixture.results[0] if reference_text is None else reference_text,
            Key_info=fixture.guideline_text if guideline_text is None else guideline_text,
        )

    def expected_capture(self):
        fixture = self.fixture
        empty = {"reported_model": None, "response_id": None, "token_usage": None}
        metadata = {stage: [] for stage in HELPER_NAMESPACE["MODEL_RESPONSE_STAGES"]}
        if fixture.corpus and fixture.failure not in ("retrieval", "keywords"):
            metadata["keyword_extraction"] = [dict(empty)]
        if fixture.failure not in ("retrieval", "keywords", "reranking", "questions"):
            metadata["guideline_questions"] = [dict(empty)]
        if fixture.failure is None:
            metadata["assessment_plan"] = [dict(empty)]
        ids = []
        for candidate in fixture.candidates:
            row = candidate["idx"]
            note = fixture.corpus[row].get("note_id") if fixture.corpus else None
            identity = {"note_id": note} if isinstance(note, str) and note.strip() else {"corpus_row": row}
            if note and sum(record.get("note_id") == note for record in fixture.corpus) > 1:
                identity["corpus_row"] = row
            ids.append(identity)
        ran = bool(fixture.corpus) and fixture.failure not in ("retrieval", "keywords")
        scores = getattr(fixture, "expected_rerank_scores", fixture.rerank_scores)
        ranking = sorted([{**identity, "rerank_score": score} for identity, score in zip(ids, scores)],
                         key=lambda item: item["rerank_score"], reverse=True)[:len(fixture.results)]
        return {
            "keyword_extraction_raw": fixture.keyword_content if ran else None,
            "retrieval_keywords": fixture.extracted_keywords if ran else None,
            "retrieved_patient_ids": ids if ran else None,
            "reranked_patient_ids": ranking if ran and ids and fixture.failure != "reranking" else None,
            "model_responses": metadata,
        }

    def expected_result(self, reference_text=None, reference="default", questions="default"):
        fixture = self.fixture
        return {
            **self.expected_capture(),
            "id": fixture.patient_id,
            "status": "completed",
            **fixture.final_sections,
            "assessment_plan_raw": fixture.final_output,
            "selected_reference_patient": fixture.reference if reference == "default" else reference,
            "guideline_questions": (json.loads(fixture.questions)["Key Questions"]
                                    if questions == "default" else questions),
            "guideline_evidence": fixture.guideline_text,
            "graphrag_context": self.expected_context(),
            "assessment_plan_prompts": {
                "system": "You are a clinical expert.",
                "user": self.expected_prompt(reference_text),
            },
            "evaluation": None,
        }

    def relative(self, path):
        path = Path(path).resolve()
        return path.relative_to(self.fixture.root).as_posix() if path.is_relative_to(
            self.fixture.root) else str(path)

    def expected_reference_data(self, corpus=None, artifacts=None, guidelines=None,
                                retrieval_initialized=True, tables=GUIDELINE_TABLES):
        # Enumerate the expected files independently of the helper's resolver.
        fixture = self.fixture
        corpus = fixture.upstream / "soap_with_metadata.json" if corpus is None else corpus
        artifacts = fixture.artifacts if artifacts is None else artifacts
        guidelines = fixture.guidelines if guidelines is None else guidelines
        retrieval_entries = [("corpus", corpus)]
        if retrieval_initialized:
            retrieval_entries += [("embeddings.npy", artifacts / "embeddings.npy"),
                                  ("index.faiss", artifacts / "index.faiss"),
                                  ("matching.json", artifacts / "matching.json")]
        guideline_entries = [
            ("settings", guidelines / "settings.yaml"),
            ("local_search_prompt", guidelines / "prompts/local_search_system_prompt.txt"),
            *[(f"tables/{name}.parquet", guidelines / "output" / f"{name}.parquet") for name in tables],
            ("vector_store/default_vector_store/default-entity-description.lance",
             guidelines / "output/lancedb/default-entity-description.lance"),
        ]
        return {
            "patient_retrieval": {
                "corpus_file": self.relative(corpus),
                "artifacts_directory": self.relative(artifacts) if retrieval_initialized else None,
                "sha256": SHA256_BUNDLE(retrieval_entries),
            },
            "guidelines": {"directory": self.relative(guidelines),
                           "sha256": SHA256_BUNDLE(guideline_entries)},
        }

    def assert_no_legacy_outputs(self):
        root = self.fixture.root
        self.assertFalse((root / "data_run_med_copilot").exists())
        self.assertEqual([path for path in root.rglob("*") if path.name in LEGACY_OUTPUTS], [])

    def assert_run_metadata(self, run_dir, run, status, input_path=None, reference_data="default"):
        fixture = self.fixture
        input_path = fixture.input_path if input_path is None else input_path
        self.assertEqual(list(run), RUN_FIELDS + (["results_summary"] if "results_summary" in run else []))
        self.assertEqual(run["run_name"], run_dir.name)
        timestamp, stem, branch = run_dir.name.split("__")
        self.assertEqual((stem, branch), (input_path.stem, fixture.safe_branch))
        started = datetime.fromisoformat(run["started_at"])
        self.assertEqual(timestamp, started.strftime("%Y-%m-%d_%H%M%S")
                         + {-7: "PDT", -8: "PST"}[started.utcoffset() // timedelta(hours=1)])
        self.assertEqual(run["status"], status)
        if status == "running":
            self.assertIsNone(run["finished_at"])
        else:
            self.assertLessEqual(started, datetime.fromisoformat(run["finished_at"]))
            self.assertEqual(datetime.fromisoformat(run["finished_at"]).utcoffset(), started.utcoffset())
        self.assertEqual(run["branch"], fixture.branch)
        self.assertEqual(run["git_commit"], fixture.commit)
        self.assertEqual(run["input"], {
            "file": self.relative(input_path),
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        })
        self.assertEqual(run["reference_data"], self.expected_reference_data()
                         if reference_data == "default" else reference_data)
        self.assertEqual(run["evaluation"]["test_double"], True)

    def assert_outputs(self, reference_text=None, reference="default", input_path=None,
                       questions="default"):
        fixture = self.fixture
        expected = self.expected_result(reference_text, reference, questions)
        run_dir, run, results = self.read_run()
        self.assertEqual(results, [expected])
        self.assert_run_metadata(run_dir, run, "completed", input_path)
        # The saved prompts are exactly the messages sent to the final API call.
        saved_prompts = results[0]["assessment_plan_prompts"]
        self.assertEqual(fixture.calls[-1]["messages"], [
            {"role": "system", "content": saved_prompts["system"]},
            {"role": "user", "content": saved_prompts["user"]},
        ])
        context = results[0]["graphrag_context"]
        if "sources" in fixture.context and len(fixture.context["sources"]) == 135:
            self.assertEqual(len(context["sources"]["data"]), 135)
            self.assertEqual(context["sources"]["index"], list(range(500, 635)))
            self.assertEqual(context["sources"]["data"][-1][1], "RAW_CONTEXT_ONLY_134 café\nsource")
            self.assertIsNone(context["sources"]["data"][0][2])
            self.assertEqual(context["claims"], {"columns": ["id", "claim"], "index": [], "data": []})
        self.assertEqual(fixture.calls[0]["messages"][-1]["content"],
                         fixture.prompts["KEY_QUESTIONS_TEMPLATE"].format(conditions=fixture.conditions))
        self.assertNotIn("RAW_CONTEXT_ONLY", saved_prompts["user"])
        self.assertNotIn("UNSELECTED_SIMILAR_CASE", saved_prompts["user"])
        self.assertNotIn("RAW_CONTEXT_ONLY", fixture.stdout.getvalue())
        for call in fixture.calls + fixture.keyword_calls:
            self.assertNotIn(fixture.patient_id, call["messages"][-1]["content"])
        self.assert_no_legacy_outputs()

    def assert_stopped_run(self, results_statuses, reference_data=None):
        run_dir, run, results = self.read_run()
        self.assertEqual([(record["id"], record["status"]) for record in results], results_statuses)
        self.assert_run_metadata(run_dir, run, "failed_incomplete", reference_data=reference_data)
        self.assert_no_legacy_outputs()
        return results

    # ----------------------------------------------------------------
    # Import, startup, and entry point
    # ----------------------------------------------------------------

    def test_import_defers_shared_client_and_confirmation_without_processing_or_outputs(self):
        fixture = self.fixture
        original_path = sys.path.copy()
        original_cwd = Path.cwd()
        with fixture.runtime(), \
                patch.object(Path, "open", autospec=True, side_effect=lambda path, *a, **kw: (_ for _ in ()).throw(AssertionError("Patient file read on import")) if path == fixture.input_path else io.open(path, *a, **kw)):
            runner = runpy.run_path(str(fixture.script), run_name="runner_under_test")
            self.assertEqual(sys.path[:2], [str(fixture.upstream), str(fixture.root)])
        self.assertTrue(callable(runner["main"]))
        experiment = runner["retrieval_startup"].experiment
        self.assertEqual(experiment.patient_corpus, fixture.upstream / "soap_with_metadata.json")
        self.assertEqual(experiment.patient_retrieval_artifacts_dir, fixture.artifacts)
        for name in ("PATIENT_CORPUS", "PATIENT_RETRIEVAL_ARTIFACTS_DIR", "GUIDELINES", "EXPERIMENT_OUTPUT", "api_key"):
            self.assertNotIn(name, runner)
        self.assertNotIn("PATIENT_INPUT", runner)
        self.assertNotIn("PATIENT_LABELS", runner)
        self.assertEqual(experiment.experiment_output, fixture.output)
        self.assertNotIn("OUTPUT", runner)
        self.assertEqual(fixture.events, [
            ("preload", "emilyalsentzer/Bio_ClinicalBERT"), ("import", "scripts.med_copilot_retrieval"),
        ])
        self.assertIsNone(runner["client"])
        fixture.confirm_input.assert_not_called()
        self.assertEqual(fixture.calls, [])
        self.assertEqual(fixture.keyword_calls, [])
        self.assertEqual(fixture.retrieval_calls, [])
        self.assertEqual(fixture.rerank_calls, [])
        self.assertEqual(fixture.graph_calls, [])
        self.assertEqual(sys.path, original_path)
        self.assertEqual(Path.cwd(), original_cwd)
        self.assertFalse(fixture.output.exists())
        self.assertFalse((fixture.root / "experiments/output").exists())
        self.assert_no_legacy_outputs()

    def test_root_dotenv_loads_before_dependencies_and_clients_from_unrelated_cwd(self):
        fixture = self.fixture
        fixture.environment = {}
        source = fixture.root / ".env"
        contents = ('OPENAI_API_KEY="dotenv-openai"\n'
                    'GRAPHRAG_API_KEY=${OPENAI_API_KEY}\nGROQ_API_KEY=dotenv-groq\n')
        source.write_text(contents)
        (fixture.caller / ".env").write_text("OPENAI_API_KEY=wrong-cwd-key\n")
        (fixture.script.parent / ".env").write_text("OPENAI_API_KEY=wrong-script-key\n")
        original_preload = fixture.modules["sentence_transformers"].SentenceTransformer
        observed = []
        def preload(model_name):
            observed.append({key: os.environ.get(key) for key in
                             ("OPENAI_API_KEY", "GRAPHRAG_API_KEY", "GROQ_API_KEY")})
            return original_preload(model_name)
        fixture.modules["sentence_transformers"].SentenceTransformer = preload
        with fixture.runtime():
            runner = runpy.run_path(str(fixture.script), run_name="runner_under_test")
            self.assertNotIn("api_key", runner)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "dotenv-openai")
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(observed, [{"OPENAI_API_KEY": "dotenv-openai",
                                    "GRAPHRAG_API_KEY": "dotenv-openai", "GROQ_API_KEY": "dotenv-groq"}])
        self.assertFalse(any(event[0] == "client" for event in fixture.events))
        self.assertEqual(fixture.calls, [])
        self.assertEqual(fixture.retrieval_calls, [])
        self.assertFalse(fixture.output.exists())
        self.assertEqual(source.read_text(), contents)

    def test_exported_credentials_override_dotenv_and_interpolation_uses_effective_value(self):
        fixture = self.fixture
        (fixture.root / ".env").write_text(
            "OPENAI_API_KEY=file-openai\nGRAPHRAG_API_KEY=${OPENAI_API_KEY}\nGROQ_API_KEY=file-groq\n")
        for graph_key in (None, "exported-graph"):
            with self.subTest(graph_key=graph_key):
                fixture.environment = {"OPENAI_API_KEY": "exported-openai", "GROQ_API_KEY": "exported-groq"}
                if graph_key is not None:
                    fixture.environment["GRAPHRAG_API_KEY"] = graph_key
                with fixture.runtime():
                    runner = runpy.run_path(str(fixture.script), run_name="runner_under_test")
                    self.assertNotIn("api_key", runner)
                    self.assertEqual(os.environ["OPENAI_API_KEY"], "exported-openai")
                    self.assertEqual(os.environ["GROQ_API_KEY"], "exported-groq")
                    self.assertEqual(os.environ["GRAPHRAG_API_KEY"], graph_key or "exported-openai")
        self.assertEqual(fixture.calls, [])
        self.assertFalse(fixture.output.exists())

    def test_missing_required_key_does_not_fall_back_to_caller_dotenv_or_start_a_run(self):
        fixture = self.fixture
        fixture.environment = {}
        (fixture.caller / ".env").write_text("OPENAI_API_KEY=wrong-cwd-key\n")
        source = fixture.root / ".env"
        for contents in (None, "GRAPHRAG_API_KEY=only-graph\n"):
            with self.subTest(contents=contents):
                if contents is not None:
                    source.write_text(contents)
                with fixture.runtime():
                    runner = runpy.run_path(str(fixture.script), run_name="runner_under_test")
                    self.assertEqual(runner["main"](), "configuration_error")
                self.assertIn("OPENAI_API_KEY", fixture.stderr.getvalue())
                self.assertIn(".env in the repository root", fixture.stderr.getvalue())
                self.assertNotIn("Traceback", fixture.stderr.getvalue())
                fixture.confirm_input.assert_not_called()
                self.assertFalse(fixture.output.exists())
                self.assertFalse(any(event[0] == "client" for event in fixture.events))
                self.assertEqual(fixture.calls, [])

    def test_all_imports_are_in_the_header_before_function_or_class_definitions(self):
        for source in (RUNNER, HELPERS, RETRIEVAL, STARTUP):
            with self.subTest(file=source.name):
                tree = ast.parse(source.read_text(encoding="utf-8"))
                imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
                top_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
                self.assertEqual(set(imports), set(top_level_imports))
                first_definition = min(
                    node.lineno for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                )
                self.assertTrue(all(node.lineno < first_definition for node in imports))

    def test_runner_passes_configured_input_to_loader_and_uses_its_records(self):
        fixture = self.fixture
        runner = self.load_runner()
        self.assertEqual(Path(runner["load_patient_records"].__code__.co_filename), fixture.helper_module)
        # An alternate input argument must reach the helper; the default file
        # is absent so a hard-coded path inside the helper would fail.
        alternate_input = fixture.root / "alternate input" / "patients café.jsonl"
        alternate_input.parent.mkdir()
        alternate_input.write_bytes(fixture.input_path.read_bytes())
        fixture.input_path.unlink()
        config = yaml.safe_load(fixture.experiment_config.read_text())
        config["patient_input"] = str(alternate_input)
        fixture.experiment_config.write_text(yaml.safe_dump(config))
        with fixture.runtime(), \
                patch.dict(runner["main"].__globals__, {
                    "load_patient_records": Mock(wraps=runner["load_patient_records"]),
                }):
            loader = runner["main"].__globals__["load_patient_records"]
            runner["main"]()
            loader.assert_called_once_with(alternate_input)
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assert_outputs(input_path=alternate_input)
        self.assertTrue(self.run_dirs()[0].name.endswith(f"__patients café__{SAFE_BRANCH}"))

    def test_default_model_registry_comes_from_experiment_configs_from_unrelated_cwd(self):
        fixture = self.fixture
        old_path = fixture.root / "configs/condition_plan_models.yaml"
        old_path.parent.mkdir()
        old_path.write_text("OBSOLETE_CONFIG_DO_NOT_LOAD")
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
            self.assertEqual(Path.cwd(), fixture.caller)
        _, run, _ = self.read_run()
        self.assertEqual(run["assessment_plan_models"]["config_file"],
                         "experiments/configs/condition_plan_models.yaml")
        self.assertEqual(run["assessment_plan_models"]["config_sha256"],
                         hashlib.sha256(fixture.model_config.read_bytes()).hexdigest())
        # Keep this exhaustive as new experiment settings are introduced.
        for field, value in yaml.safe_load(fixture.experiment_config.read_text()).items():
            with self.subTest(config_field=field):
                self.assertIn(field, run["experiment"])
                self.assertEqual(run["experiment"][field], value)
        self.assertEqual(run["experiment"], {
            "config_file": "experiments/configs/experiment.yaml",
            "config_sha256": hashlib.sha256(fixture.experiment_config.read_bytes()).hexdigest(),
            "model_config": "experiments/configs/condition_plan_models.yaml",
            "selected_models": ["gpt_4o_mini"], "temperature": 1, "max_retries": 3, "max_patients": None,
            "selected_variants": ["patient_retrieval_guidelines"], "patient_input": "experiments/input/test_sample_so.jsonl",
            "patient_labels": "experiments/input/test_sample_labels.jsonl", "presenting_evidence_only": True,
            "exclude_none_recorded_plans": False, **PIPELINE_DEFAULTS,
        })
        self.assertEqual(old_path.read_text(), "OBSOLETE_CONFIG_DO_NOT_LOAD")
        self.assert_outputs()

    def test_missing_or_invalid_experiment_model_config_fails_without_legacy_fallback(self):
        fixture_root = self.fixture.root
        for case, error in (("missing", FileNotFoundError), ("invalid", ValueError)):
            with self.subTest(case=case):
                self.fixture = PipelineFixture(fixture_root / case)
                fixture = self.fixture
                old_path = fixture.root / "configs/condition_plan_models.yaml"
                old_path.parent.mkdir()
                old_path.write_bytes(MODEL_CONFIG.read_bytes())
                if case == "missing":
                    fixture.model_config.unlink()
                else:
                    fixture.model_config.write_text("models: [")
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](), "configuration_error")
                self.assertIn("condition_plan_models.yaml", fixture.stderr.getvalue())
                fixture.confirm_input.assert_not_called()
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.retrieval_calls, [])
                self.assertEqual(fixture.graph_calls, [])
                self.assertFalse(fixture.output.exists())

    def test_dependency_startup_failures_stop_before_processing_input(self):
        fixture_root = self.fixture.root
        for stage, expected_events in [("upstream_import", ["preload", "import"])]:
            with self.subTest(stage=stage):
                self.fixture = PipelineFixture(fixture_root / stage)
                fixture = self.fixture
                fixture.failure = stage
                original_path = sys.path.copy()
                with fixture.runtime(), \
                        patch.object(Path, "open", autospec=True, side_effect=lambda path, *a, **kw: (_ for _ in ()).throw(AssertionError("Patient read during startup")) if path == fixture.input_path else io.open(path, *a, **kw)):
                    with self.assertRaises(RuntimeError) as caught:
                        runpy.run_path(str(fixture.script), run_name="__main__")
                    self.assertIs(caught.exception, fixture.error)
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertEqual([event[0] for event in fixture.events], expected_events)
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.retrieval_calls, [])
                self.assertEqual(fixture.graph_calls, [])
                self.assertFalse(fixture.output.exists())
                self.assertEqual(sys.path, original_path)

    def test_full_main_preserves_pipeline_contract_and_saves_evidence(self):
        fixture = self.fixture
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(fixture.events[:2], [
            ("preload", "emilyalsentzer/Bio_ClinicalBERT"), ("import", "scripts.med_copilot_retrieval"),
        ])
        self.assertEqual(sum(event[0] == "preload" for event in fixture.events), 1)
        self.assertEqual(sum(event[0] == "import" for event in fixture.events), 1)
        self.assertIn(("client", "test-key"), fixture.events)
        self.assertEqual(len(fixture.calls), 2)
        self.assertEqual(len(fixture.keyword_calls), 1)
        keyword_request = fixture.keyword_calls[0]
        self.assertEqual(keyword_request["model"], "gpt-4o-mini")
        self.assertEqual(keyword_request["temperature"], 0)
        self.assertEqual(keyword_request["response_format"], {"type": "json_object"})
        self.assertIn(fixture.conditions, keyword_request["messages"][0]["content"])
        self.assertEqual(fixture.extracted_keywords, ["Cough", "denies fever"])
        for call in fixture.calls:
            self.assertEqual(call["model"], "gpt-4o-mini" if call is fixture.calls[0]
                             else "openai/gpt-4o-mini")
        self.assertEqual(fixture.calls[0]["messages"], [
            {"role": "system", "content": "You are a clinical expert."},
            {"role": "user", "content": PROMPTS["KEY_QUESTIONS_TEMPLATE"].format(conditions=fixture.conditions)},
        ])
        self.assertEqual(fixture.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(fixture.calls[1]["response_format"], FINAL_RESPONSE_FORMAT)
        init, search = fixture.retrieval_calls
        self.assertEqual(init[1].to_dict(orient="records"), fixture.corpus)
        self.assertEqual(init[2:], (0.5, fixture.artifacts))
        self.assertEqual(search, ("search", fixture.conditions, 20, fixture.artifacts))
        self.assertEqual(fixture.rerank_calls[0], ("init", fixture.artifacts))
        self.assertEqual(fixture.rerank_calls[1],
                         ("rerank", fixture.conditions, fixture.candidates, 5, fixture.artifacts))
        self.assertEqual(fixture.graph_calls, [({
            "config_filepath": fixture.guidelines / "settings.yaml",
            "data_dir": fixture.guidelines / "output", "root_dir": fixture.guidelines,
            "community_level": 2, "response_type": "Multiple Paragraphs",
            "streaming": False, "query": fixture.questions, "verbose": True,
        }, fixture.guidelines / "output")])
        self.assertEqual(fixture.completion_cwds, [fixture.caller, fixture.caller])
        self.assertNotIn("Assessment:", fixture.conditions)
        for field in fixture.patient.values():
            self.assertIn(field, fixture.retrieval_calls[1][1])
            self.assertIn(field, fixture.rerank_calls[1][1])
            self.assertIn(field, fixture.calls[0]["messages"][-1]["content"])
            self.assertIn(field, fixture.calls[1]["messages"][-1]["content"])
        self.assertNotIn("REFERENCE_DIAGNOSIS", fixture.retrieval_calls[1][1])
        self.assertNotIn("REFERENCE_DIAGNOSIS", fixture.rerank_calls[1][1])
        self.assertNotIn("REFERENCE_DIAGNOSIS", fixture.calls[0]["messages"][-1]["content"])
        self.assertIn("Assessment: REFERENCE_DIAGNOSIS", fixture.calls[1]["messages"][-1]["content"])
        self.assertIn("Plan: REFERENCE_PLAN", fixture.calls[1]["messages"][-1]["content"])
        self.assert_outputs()

    def test_script_entry_point_runs_the_full_pipeline(self):
        with self.fixture.runtime():
            runpy.run_path(str(self.fixture.script), run_name="__main__")
            self.assertEqual(Path.cwd(), self.fixture.caller)
        self.assertEqual(len(self.fixture.calls), 2)
        self.assert_outputs()

    def test_script_entry_point_exits_nonzero_unless_run_completed(self):
        fixture = self.fixture
        fixture.failure = "final"
        with fixture.runtime():
            with self.assertRaises(SystemExit) as caught:
                runpy.run_path(str(fixture.script), run_name="__main__")
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(caught.exception.code, 1)
        run_dir, run, results = self.read_run()
        self.assertEqual(run["status"], "completed_with_errors")
        self.assertEqual(results[0]["status"], "failed_incomplete")

    # ----------------------------------------------------------------
    # Run directory naming, Git identity, and launch location
    # ----------------------------------------------------------------

    def test_run_name_uses_pacific_time_with_pdt_or_pst_and_path_safe_branch(self):
        fixture_root = self.fixture.root
        for label, start, timestamp, offset in [
            ("summer", datetime(2026, 10, 17, 22, 30, 45, 987654, tzinfo=timezone.utc),
             "2026-10-17_153045PDT", "-07:00"),
            ("winter", datetime(2026, 1, 15, 17, 30, 0, tzinfo=timezone.utc),
             "2026-01-15_093000PST", "-08:00"),
        ]:
            with self.subTest(season=label):
                self.fixture = PipelineFixture(fixture_root / label)
                fixture = self.fixture
                runner = self.load_runner()
                with fixture.runtime(), \
                        patch.dict(runner["main"].__globals__, {"pacific_now": sequential_clock(start)}):
                    runner["main"]()
                run_dir, run, _ = self.read_run()
                self.assertEqual(run_dir.name, f"{timestamp}__{Path(INPUT_NAME).stem}__feature-issue-9")
                self.assertEqual(run["started_at"], f"{timestamp[:10]}T{timestamp[11:13]}:"
                                                    f"{timestamp[13:15]}:{timestamp[15:17]}{offset}")
                self.assertEqual(datetime.fromisoformat(run["finished_at"])
                                 - datetime.fromisoformat(run["started_at"]), timedelta(seconds=1))
                self.assertEqual(run["branch"], "feature/issue-9")
                self.assert_outputs()

    def test_detached_head_records_head_and_exact_commit(self):
        self.fixture = PipelineFixture(self.fixture.root / "detached", branch="HEAD")
        runner = self.load_runner()
        with self.fixture.runtime():
            runner["main"]()
        run_dir, run, _ = self.read_run()
        self.assertTrue(run_dir.name.endswith(f"__{Path(INPUT_NAME).stem}__HEAD"))
        self.assertEqual(run["branch"], "HEAD")
        self.assertRegex(run["git_commit"], r"^[0-9a-f]{40}$")
        self.assert_outputs()

    def test_missing_repository_or_git_fails_before_creating_a_run(self):
        fixture_root = self.fixture.root
        for case in ["no_repository", "no_git_executable"]:
            with self.subTest(case=case):
                self.fixture = PipelineFixture(fixture_root / case)
                fixture = self.fixture
                if case == "no_repository":
                    shutil.rmtree(fixture.root / ".git")
                runner = self.load_runner()
                with fixture.runtime():
                    environment = {"PATH": str(fixture.root / "no-git")} if case == "no_git_executable" else {}
                    with patch.dict(os.environ, environment), \
                            self.assertRaisesRegex(RuntimeError, "Git"):
                        runner["main"]()
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertFalse(fixture.output.exists())
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.retrieval_calls, [])

    def test_existing_run_directory_is_never_overwritten(self):
        fixture = self.fixture
        start = datetime(2026, 10, 17, 22, 30, 45, tzinfo=timezone.utc)
        existing = fixture.output / f"2026-10-17_153045PDT__{Path(INPUT_NAME).stem}__{SAFE_BRANCH}"
        existing.mkdir(parents=True)
        (existing / "run.json").write_text("PREVIOUS_RUN", encoding="utf-8")
        runner = self.load_runner()
        with fixture.runtime(), \
                patch.dict(runner["main"].__globals__, {"pacific_now": sequential_clock(start)}):
            with self.assertRaises(FileExistsError) as caught:
                runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(caught.exception.filename, str(existing))
        self.assertIn("already exists", str(caught.exception))
        self.assertEqual(self.run_dirs(), [existing])
        self.assertEqual([path.name for path in existing.iterdir()], ["run.json"])
        self.assertEqual((existing / "run.json").read_text(encoding="utf-8"), "PREVIOUS_RUN")
        self.assertEqual(fixture.calls, [])
        self.assertEqual(fixture.retrieval_calls, [])

    def test_run_metadata_paths_do_not_depend_on_launch_directory(self):
        fixture_root = self.fixture.root
        for location in ["caller", "scripts", "experiments"]:
            with self.subTest(cwd=location):
                self.fixture = PipelineFixture(fixture_root / location)
                fixture = self.fixture
                fixture.caller = {"caller": fixture.caller, "scripts": fixture.script.parent,
                                  "experiments": fixture.root / "experiments/input"}[location]
                runner = self.load_runner()
                with fixture.runtime():
                    runner["main"]()
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertEqual(self.run_dirs()[0].parent, fixture.output)
                self.assert_outputs()
                self.assertEqual({path.name for path in (fixture.root / "experiments").iterdir()},
                                 {"configs", "input", "output"})

    # ----------------------------------------------------------------
    # Reference-data metadata
    # ----------------------------------------------------------------

    def test_reference_metadata_follows_runtime_resources_and_regenerated_artifacts(self):
        fixture = self.fixture
        runner = self.load_runner()
        clock = sequential_clock()

        def run_once(**overrides):
            fixture.update_experiment(**overrides)
            with fixture.runtime(), \
                    patch.dict(runner["main"].__globals__, {"pacific_now": clock}):
                self.assertEqual(runner["main"](), "completed")
            return json.loads((self.run_dirs()[-1] / "run.json").read_text(encoding="utf-8"))

        baseline = run_once()["reference_data"]
        self.assertEqual(baseline, self.expected_reference_data())
        self.assertEqual(baseline["patient_retrieval"]["corpus_file"],
                         "med_copilot/upstream/soap_with_metadata.json")
        self.assertEqual(baseline["patient_retrieval"]["artifacts_directory"], "med_copilot/data/patient_retrieval_so")
        self.assertEqual(baseline["guidelines"]["directory"], "med_copilot/data/guidelines")

        (fixture.guidelines / "prompts/extract_graph.txt").write_text("CHANGED_UNUSED_PROMPT")
        self.assertEqual(run_once()["reference_data"], baseline)

        changes = {
            "active prompt": ("guidelines", lambda: (
                fixture.guidelines / "prompts/local_search_system_prompt.txt").write_text("NEW PROMPT")),
            "table": ("guidelines", lambda: (
                fixture.guidelines / "output/entities.parquet").write_text("NEW TABLE")),
            "vector index": ("guidelines", lambda: (
                fixture.guidelines / "output/lancedb/default-entity-description.lance/data/new.lance"
            ).write_bytes(b"NEW VECTORS")),
            "corpus contents": ("patient_retrieval", lambda: (
                fixture.corpus[0].update(plan="Changed plan"), fixture.write_corpus())),
            "regenerated artifacts": ("patient_retrieval", lambda: setattr(
                fixture, "regenerated_artifacts",
                {"embeddings.npy": b"REBUILT_EMBEDDINGS", "index.faiss": b"REBUILT_INDEX"})),
        }
        previous = baseline
        for name, (block, change) in changes.items():
            with self.subTest(change=name):
                change()
                current = run_once()["reference_data"]
                self.assertEqual(current, self.expected_reference_data())
                self.assertNotEqual(current[block]["sha256"], previous[block]["sha256"])
                other = "guidelines" if block == "patient_retrieval" else "patient_retrieval"
                self.assertEqual(current[other], previous[other])
                previous = current
        # Fingerprints describe the regenerated artifacts actually loaded.
        self.assertEqual((fixture.artifacts / "index.faiss").read_bytes(), b"REBUILT_INDEX")

        alternate = fixture.root / "alternate artifacts"
        alternate.mkdir()
        for name in ("embeddings.npy", "index.faiss", "matching.json"):
            (alternate / name).write_bytes(b"ALTERNATE_" + name.encode())
        fixture.regenerated_artifacts = None
        fixture.retrieval_calls.clear()
        current = run_once(patient_retrieval_artifacts_dir=str(alternate))["reference_data"]
        self.assertEqual(current, self.expected_reference_data(artifacts=alternate))
        self.assertEqual(current["patient_retrieval"]["artifacts_directory"], "alternate artifacts")
        self.assertEqual(fixture.retrieval_calls[0][3], alternate)
        self.assertEqual(fixture.retrieval_calls[1][3], alternate)

    # ----------------------------------------------------------------
    # Real vendored GraphRAG loader
    # ----------------------------------------------------------------

    def real_guideline_search(self):
        fixture = self.fixture
        state = SimpleNamespace(reads=[], api_calls=[], config_calls=[])

        class Storage:
            def __init__(self, directory):
                # Match FilePipelineStorage: it has _root_dir, not base_path.
                self._root_dir = directory

            async def has(self, filename):
                return (Path(self._root_dir) / filename).is_file()

        def load_config(root, config_filepath, overrides):
            state.config_calls.append((root, config_filepath, overrides, Path.cwd()))
            settings = yaml.safe_load(Path(config_filepath).read_text(encoding="utf-8"))
            return SimpleNamespace(outputs={}, output=SimpleNamespace(
                base_dir=overrides["output.base_dir"],
            ), local_search=SimpleNamespace(**settings["local_search"]))

        async def local_search(**kwargs):
            state.api_calls.append((kwargs, Path.cwd()))
            fixture.fail_at("graphrag_api")
            return fixture.guideline_text, fixture.context

        query = frozen_graphrag_query(
            load_config, lambda config: Storage(config.base_dir),
            SimpleNamespace(local_search=local_search),
        )

        def graph_search(**kwargs):
            fixture.graph_calls.append((kwargs, Path.cwd()))
            return query(**kwargs)

        def read_parquet(path):
            path = Path(path).resolve()
            state.reads.append(path)
            return pd.DataFrame({"origin": [path.read_text(encoding="utf-8")]})

        fixture.modules["graphrag.cli.query"].run_local_search = graph_search
        state.read_parquet = read_parquet
        return state

    def test_configured_entity_limit_reaches_every_query_and_saved_resource_fingerprint(self):
        fixture = self.fixture
        settings_path = fixture.guidelines / "settings.yaml"
        settings_path.write_bytes(GUIDELINE_SETTINGS.read_bytes())
        fixture.write_patients([
            {"id": fixture.patient_id, **fixture.patient},
            {"id": "SECOND_PATIENT", "subjective": "Sore throat", "objective": "Fever"},
        ])
        self.write_guideline_tables(fixture.guidelines / "output")
        state = self.real_guideline_search()
        runner = self.load_runner()
        with fixture.runtime(), patch("pandas.read_parquet", side_effect=state.read_parquet):
            self.assertEqual(runner["main"](), "completed")
        self.assertEqual(len(state.api_calls), 2)
        for kwargs, _ in state.api_calls:
            self.assertEqual(kwargs["config"].local_search.top_k_entities, 5)
            self.assertEqual(kwargs["query"], fixture.questions)
        _, run, results = self.read_run()
        self.assertEqual([result["id"] for result in results],
                         [fixture.patient_id, "SECOND_PATIENT"])
        for result in results:
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["guideline_evidence"], fixture.guideline_text)
            self.assertEqual(result["assessment_plan_raw"], fixture.final_output)
            self.assertIn(fixture.guideline_text, result["assessment_plan_prompts"]["user"])
        self.assertEqual(run["reference_data"], self.expected_reference_data())

    def test_configured_entity_limit_bounds_actual_upstream_oversampled_selection(self):
        # Execute the unchanged upstream mapper with a fake embedding store:
        # verify the observable ten-result selection, not just the YAML value.
        node = next(node for node in ast.parse(
            UPSTREAM_GRAPH_ENTITY_EXTRACTION.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name == "map_query_to_entities")
        namespace = {
            "Entity": object, "BaseVectorStore": object, "EmbeddingModel": object,
            "EntityVectorStoreKey": SimpleNamespace(ID="id"),
            "get_entity_by_id": lambda entities, identity: entities.get(identity),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]),
                     str(UPSTREAM_GRAPH_ENTITY_EXTRACTION), "exec"), namespace)
        entities = {str(i): SimpleNamespace(id=str(i), title=f"entity {i}") for i in range(30)}
        requested_k = yaml.safe_load(GUIDELINE_SETTINGS.read_text())["local_search"]["top_k_entities"]
        self.assertEqual(requested_k, 5)
        for available in (0, 3, 30):
            with self.subTest(available=available):
                store = Mock()
                store.similarity_search_by_text.side_effect = lambda **kwargs: [
                    SimpleNamespace(document=SimpleNamespace(id=str(i)))
                    for i in range(min(available, kwargs["k"]))
                ]
                mapped = namespace["map_query_to_entities"](
                    "clinical questions", store, Mock(), entities, k=requested_k)
                self.assertEqual([entity.id for entity in mapped],
                                 [str(i) for i in range(min(available, 10))])
                self.assertEqual(store.similarity_search_by_text.call_args.kwargs["k"], 10)

    def write_guideline_tables(self, directory, prefix="SELECTED", include_covariates=False):
        names = list(GUIDELINE_TABLES)
        if include_covariates:
            names.append("covariates")
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            (directory / f"{name}.parquet").write_text(f"{prefix}_{name}", encoding="utf-8")
        return names

    def test_real_guideline_loader_uses_selected_output_from_scripts_and_unrelated_cwds(self):
        fixture_root = self.fixture.root
        for location, optional in [("scripts", False), ("unrelated", True)]:
            with self.subTest(caller=location, covariates=optional):
                self.fixture = PipelineFixture(fixture_root / location)
                fixture = self.fixture
                if location == "scripts":
                    fixture.caller = fixture.script.parent
                output = fixture.guidelines / "output"
                names = self.write_guideline_tables(output, include_covariates=optional)
                # A caller/repository-wide search could select unrelated tables.
                self.write_guideline_tables(fixture.caller, prefix="WRONG_CALLER")
                self.write_guideline_tables(fixture.guidelines / "stale", prefix="WRONG_STALE")
                state = self.real_guideline_search()
                runner = self.load_runner()
                with fixture.runtime(), patch("pandas.read_parquet", side_effect=state.read_parquet):
                    runner["main"]()
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertEqual(state.reads, [output / f"{name}.parquet" for name in names])
                self.assertEqual(len(state.api_calls), 1)
                api_args, api_cwd = state.api_calls[0]
                self.assertEqual(api_cwd, output)
                for name in names:
                    self.assertEqual(api_args[name].to_dict(orient="records"),
                                     [{"origin": f"SELECTED_{name}"}])
                if not optional:
                    self.assertIsNone(api_args["covariates"])
                self.assertEqual(api_args["query"], fixture.questions)
                self.assertEqual(state.config_calls, [(
                    fixture.guidelines, fixture.guidelines / "settings.yaml",
                    {"output.base_dir": str(output)}, output,
                )])
                self.assertEqual(fixture.completion_cwds, [fixture.caller, fixture.caller])
                run_dir, run, results = self.read_run()
                self.assertEqual(results, [self.expected_result()])
                # The optional covariates table is fingerprinted only when it is loaded.
                self.assert_run_metadata(run_dir, run, "completed", reference_data=(
                    self.expected_reference_data(tables=names)))

    def test_real_guideline_loader_respects_configured_directory_outside_project(self):
        self.fixture = PipelineFixture(self.fixture.root / "project")
        fixture = self.fixture
        selected = fixture.root.parent / "alternate guidelines café"
        output = selected / "output"
        names = self.write_guideline_tables(output, prefix="ALTERNATE")
        fixture.write_guideline_settings(selected)
        self.write_guideline_tables(fixture.guidelines / "output", prefix="WRONG_DEFAULT")
        self.write_guideline_tables(fixture.caller, prefix="WRONG_CALLER")
        state = self.real_guideline_search()
        runner = self.load_runner()
        fixture.update_experiment(guidelines=str(selected))
        with fixture.runtime(), patch("pandas.read_parquet", side_effect=state.read_parquet):
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(state.reads, [output / f"{name}.parquet" for name in names])
        self.assertEqual(state.config_calls, [(
            selected, selected / "settings.yaml", {"output.base_dir": str(output)}, output,
        )])
        for name in names:
            self.assertEqual(state.api_calls[0][0][name].iloc[0]["origin"], f"ALTERNATE_{name}")
        self.assertEqual(fixture.graph_calls[0][1], output)
        run_dir, run, results = self.read_run()
        self.assertEqual(results, [self.expected_result()])
        # External resources are recorded by absolute path.
        self.assertEqual(run["reference_data"]["guidelines"]["directory"], str(selected.resolve()))
        self.assert_run_metadata(run_dir, run, "completed", reference_data=(
            self.expected_reference_data(guidelines=selected)))

    def test_missing_guideline_table_blocks_dependent_variants_and_restores_cwd(self):
        fixture_root = self.fixture.root
        for missing in GUIDELINE_TABLES:
            with self.subTest(missing=missing):
                self.fixture = PipelineFixture(fixture_root / missing)
                fixture = self.fixture
                output = fixture.guidelines / "output"
                missing_path = output / f"{missing}.parquet"
                missing_path.unlink()
                self.write_guideline_tables(fixture.caller, prefix="WRONG_CALLER")
                self.write_guideline_tables(fixture.guidelines / "stale", prefix="WRONG_SIBLING")
                self.write_guideline_tables(output / "stale", prefix="WRONG_NESTED")
                state = self.real_guideline_search()
                runner = self.load_runner()
                with fixture.runtime(), patch("pandas.read_parquet", side_effect=state.read_parquet):
                    self.assertEqual(runner["main"](), "completed_with_errors")
                    self.assertEqual(Path.cwd(), fixture.caller)
                record = json.loads((self.run_dirs()[0] / "results.jsonl").read_text())
                self.assertIn(str(missing_path), record["evidence_status"]["guidelines"]["error"]["message"])
                self.assertEqual(record["evidence_status"]["patient_retrieval"]["status"], "completed")
                self.assertEqual(record["variants"]["patient_retrieval_guidelines"]["error"]["type"], "EvidenceUnavailable")
                self.assertEqual(fixture.graph_calls, [])
                self.assertEqual(state.reads, [])
                self.assertEqual(state.api_calls, [])
                self.assertEqual(fixture.calls, [])
                self.assertEqual(len(fixture.keyword_calls), 1)
                self.assertEqual(record["status"], "failed_incomplete")

    def test_real_guideline_api_failure_fails_patient_and_stops_its_final_generation(self):
        fixture = self.fixture
        output = fixture.guidelines / "output"
        self.write_guideline_tables(output)
        state = self.real_guideline_search()
        fixture.failure = "graphrag_api"
        runner = self.load_runner()
        with fixture.runtime(), patch("pandas.read_parquet", side_effect=state.read_parquet):
            self.assertEqual(runner["main"](), "completed_with_errors")
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(len(state.api_calls), 1)
        self.assertEqual(len(fixture.calls), 1)
        run_dir, run, results = self.read_run()
        self.assertEqual(results, [{
            **self.expected_result(), "status": "failed_incomplete", "predictions": None,
            "assessment_plan_raw": None, "guideline_evidence": None,
            "graphrag_context": None, "assessment_plan_prompts": None,
        }])
        self.assert_run_metadata(run_dir, run, "completed_with_errors")
        self.assertIn("mock stage failure", fixture.stderr.getvalue())

    def test_missing_guideline_output_directory_blocks_without_search_fallback(self):
        fixture = self.fixture
        output = fixture.guidelines / "output"
        shutil.rmtree(output)
        self.write_guideline_tables(fixture.caller, prefix="WRONG_CALLER")
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed_with_errors")
            self.assertEqual(Path.cwd(), fixture.caller)
        record = json.loads((self.run_dirs()[0] / "results.jsonl").read_text())
        self.assertIn(str(output), record["evidence_status"]["guidelines"]["error"]["message"])
        self.assertEqual(record["status"], "failed_incomplete")
        self.assertEqual(fixture.graph_calls, [])
        self.assertEqual(fixture.calls, [])

    def test_guideline_vector_database_path_is_portable(self):
        # This project YAML has one db_uri; do not require a YAML parser just
        # to verify that it uses the existing output/lancedb layout on POSIX.
        entries = [line.split(":", 1)[1].strip().strip("\"'")
                   for line in GUIDELINE_SETTINGS.read_text(encoding="utf-8").splitlines()
                   if line.strip().startswith("db_uri:")]
        self.assertEqual(entries, ["output/lancedb"])
        self.assertEqual(Path(entries[0]).parts, ("output", "lancedb"))

    # ----------------------------------------------------------------
    # Reference patients and per-patient stage outcomes
    # ----------------------------------------------------------------

    def real_so_retrieval(self):
        """Use production matching/cache code, mocking only external dependencies."""
        fixture = self.fixture
        adapter = fixture.modules["scripts.med_copilot_retrieval"]
        stub = adapter.HybridRetriever(pd.DataFrame(fixture.corpus), alpha=0.5, encoder=None)
        fixture.retrieval_calls.clear()
        module = load_retrieval_module(keyword_client=stub.keyword_extractor.client)
        encoder = FakeEncoder()
        fixture.modules["sentence_transformers"].SentenceTransformer = lambda name: encoder
        # NLP attribution is downstream display data, not matching or selection.
        module.HybridRetriever.compute_case_phrase_contributions = Mock(return_value=([], []))
        fixture.modules["scripts.med_copilot_retrieval"] = module
        instances, rankers = [], []
        retriever_class, reranker_class = module.HybridRetriever, module.CrossEncoderReranker

        def retriever(*args, **kwargs):
            instance = retriever_class(*args, **kwargs)
            instance.search = Mock(wraps=instance.search)
            instances.append(instance)
            return instance

        def reranker(*args, **kwargs):
            instance = reranker_class(*args, **kwargs)
            instance.rerank = Mock(wraps=instance.rerank)
            rankers.append(instance)
            return instance

        module.HybridRetriever = retriever
        module.CrossEncoderReranker = reranker
        return SimpleNamespace(module=module, encoder=encoder, retrievers=instances, rankers=rankers)

    def test_real_so_matching_excludes_reference_ap_and_keeps_full_soap_in_final_request_and_outputs(self):
        fixture = self.fixture
        fixture.corpus = [
            {"note_id": "DISTRACTOR", "subjective": "headache", "objective": "normal",
             "assessment": "cough ONLY_IN_REFERENCE_A", "plan": "cough ONLY_IN_REFERENCE_P"},
            {"note_id": "MATCH", "subjective": "cough café\nDay 2", "objective": "SpO₂ 98%",
             "assessment": "MATCH_ASSESSMENT", "plan": "MATCH_PLAN\nfollow-up"},
        ]
        fixture.write_corpus()
        state = self.real_so_retrieval()
        runner = self.load_runner()
        clock = sequential_clock()
        with fixture.runtime(), patch.dict(runner["main"].__globals__, {"pacific_now": clock}):
            self.assertEqual(runner["main"](), "completed")
        first_dir, first_run, first_results = self.read_run()
        self.assertEqual(first_results[0]["selected_reference_patient"], {"note_id": "MATCH"})
        first_pairs = state.rankers[0].model.calls[0]
        self.assertEqual(first_pairs, [(fixture.conditions, runner["matching_text"](record))
                                      for record in reversed(fixture.corpus)])
        self.assertEqual(state.encoder.calls[0], [runner["matching_text"](record)
                                                for record in fixture.corpus])
        for field in ("subjective", "objective", "assessment", "plan"):
            self.assertIn(fixture.corpus[1][field], fixture.calls[-1]["messages"][-1]["content"])
        self.assertEqual(first_results[0]["assessment_plan_raw"], fixture.final_output)
        self.assertEqual(first_results[0]["assessment_plan_prompts"], {
            "system": fixture.calls[-1]["messages"][0]["content"],
            "user": fixture.calls[-1]["messages"][1]["content"],
        })
        self.assert_run_metadata(first_dir, first_run, "completed")
        cache_before = {path.name: path.read_bytes() for path in fixture.artifacts.iterdir()}
        # Read actual candidate scores from the downstream rerank input, captured
        # without replacing the production matching or reranking implementation.
        candidates_before = state.rankers[0].rerank.call_args.args[1]
        first_scores = [{name: item[name] for name in
                         ("idx", "embedding_score", "keyword_score", "hybrid_score", "rerank_score")}
                        for item in candidates_before]
        embedding_calls_before = len(state.encoder.calls)
        for number, record in enumerate(fixture.corpus):
            record.update(assessment=f"CHANGED_A_{number}", plan=f"CHANGED_P_{number}")
        fixture.write_corpus()
        with fixture.runtime(), patch.dict(runner["main"].__globals__, {"pacific_now": clock}):
            self.assertEqual(runner["main"](), "completed")
            self.assertEqual(Path.cwd(), fixture.caller)
        second_dir = self.run_dirs()[-1]
        _, second_run, second_results = self.read_run(second_dir)
        self.assertEqual(second_results[0]["selected_reference_patient"], {"note_id": "MATCH"})
        self.assertEqual(state.rankers[1].model.calls[0], first_pairs)
        self.assertEqual(cache_before, {path.name: path.read_bytes() for path in fixture.artifacts.iterdir()})
        self.assertEqual(len(state.encoder.calls), embedding_calls_before + 1)  # query only, cache reused
        self.assertIn("CHANGED_A_1", second_results[0]["assessment_plan_prompts"]["user"])
        self.assertIn("CHANGED_P_1", second_results[0]["assessment_plan_prompts"]["user"])
        candidates_after = state.rankers[1].rerank.call_args.args[1]
        self.assertEqual(first_scores, [{name: item[name] for name in first_scores[0]}
                                        for item in candidates_after])
        self.assertNotEqual(first_run["reference_data"]["patient_retrieval"]["sha256"],
                            second_run["reference_data"]["patient_retrieval"]["sha256"])
        self.assertEqual((fixture.upstream / "index.faiss").read_bytes(), b"LEGACY_SOA")
        self.assertEqual((fixture.upstream / "embeddings.npy").read_bytes(), b"LEGACY_SOA")

    def test_real_so_cache_write_and_reranker_failures_block_dependent_variants(self):
        fixture_root = self.fixture.root
        for stage in ("cache_write", "reranking"):
            with self.subTest(stage=stage):
                self.fixture = PipelineFixture(fixture_root / stage)
                fixture = self.fixture
                state = self.real_so_retrieval()
                runner = self.load_runner()
                if stage == "cache_write":
                    with fixture.runtime(), patch.object(state.module.faiss, "write_index",
                                                         side_effect=OSError("cache write failed")):
                        self.assertEqual(runner["main"](), "completed_with_errors")
                    record = json.loads((self.run_dirs()[0] / "results.jsonl").read_text())
                    self.assertEqual(record["evidence_status"]["patient_retrieval"]["error"],
                                     {"type": "OSError", "message": "cache write failed"})
                    self.assertEqual(record["status"], "failed_incomplete")
                    self.assertEqual(fixture.calls, [])
                else:
                    ranker = runner["CrossEncoderReranker"]()
                    ranker.model.predict = Mock(side_effect=RuntimeError("reranking failed"))
                    with fixture.runtime(), patch.dict(runner["main"].__globals__, {
                        "CrossEncoderReranker": lambda **kwargs: ranker,
                    }):
                        self.assertEqual(runner["main"](), "completed_with_errors")
                    _, _, results = self.read_run()
                    self.assertEqual(results[0]["status"], "failed_incomplete")
                    self.assertIsNone(results[0]["assessment_plan_raw"])
                    self.assertEqual(fixture.calls, [])

    def test_empty_reference_stages_continue_through_guidelines_and_final_generation(self):
        fixture_root = self.fixture.root
        for stage in ["corpus", "retrieval", "reranking"]:
            with self.subTest(stage=stage):
                self.fixture = PipelineFixture(fixture_root / stage)
                fixture = self.fixture
                if stage == "corpus":
                    fixture.corpus = []
                    fixture.write_corpus()
                elif stage == "retrieval":
                    fixture.candidates = []
                else:
                    fixture.results = []
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](), "completed")
                    self.assertEqual(Path.cwd(), fixture.caller)

                if stage == "corpus":
                    self.assertEqual(fixture.retrieval_calls, [])
                    self.assertEqual(fixture.keyword_calls, [])
                    self.assertEqual(fixture.rerank_calls, [])
                else:
                    self.assertEqual(fixture.retrieval_calls[1],
                                     ("search", fixture.conditions, 20, fixture.artifacts))
                    self.assertEqual(fixture.rerank_calls[0], ("init", fixture.artifacts))
                if stage == "reranking":
                    self.assertEqual(fixture.rerank_calls[1],
                                     ("rerank", fixture.conditions, fixture.candidates, 5, fixture.artifacts))
                else:
                    self.assertEqual([call for call in fixture.rerank_calls if call[0] == "rerank"], [])
                self.assertEqual(len(fixture.calls), 2)
                self.assertEqual(fixture.calls[0]["messages"][-1]["content"],
                                 fixture.prompts["KEY_QUESTIONS_TEMPLATE"].format(conditions=fixture.conditions))
                self.assertEqual(len(fixture.graph_calls), 1)
                self.assertEqual(fixture.graph_calls[0][0]["query"], fixture.questions)
                self.assertEqual(fixture.graph_calls[0][1], fixture.guidelines / "output")
                final_prompt = fixture.calls[1]["messages"][-1]["content"]
                for evidence in [fixture.conditions, fixture.guideline_text,
                                 "No reference patient available."]:
                    self.assertIn(evidence, final_prompt)
                self.assertIn("If no reference patient is available", final_prompt)
                self.assertIn("identify the three conditions and their respective plans from the patient's Subjective and Objective",
                              final_prompt)
                self.assertIn("any relevant additional information", final_prompt)
                self.assertNotIn("SELECTED_SIMILAR_CASE", final_prompt)
                run_dir, run, results = self.read_run()
                self.assertEqual(results, [self.expected_result(
                    reference_text="No reference patient available.", reference=None)])
                # Without a corpus the retriever and its artifacts are never loaded.
                self.assert_run_metadata(run_dir, run, "completed", reference_data=(
                    self.expected_reference_data(retrieval_initialized=stage != "corpus")))

    def test_selected_reference_identity_survives_real_upstream_reranking(self):
        fixture = self.fixture
        fixture.corpus = [
            {"note_id": f"NOTE-{row}", "subjective": f"Reference S{row}", "objective": f"Reference O{row}",
             "assessment": f"Reference A{row}", "plan": f"Reference P{row}"}
            for row in range(4)
        ]
        fixture.write_corpus()
        fixture.candidates = [
            {"idx": row, **{field: fixture.corpus[row][field]
                            for field in ("subjective", "objective", "assessment", "plan")},
             "dominant_terms": [{"original_text": f"term {row}", "contribution": 0.9}]}
            for row in (3, 1, 2)
        ]
        # Rows 1 and 2 tie; the upstream stable sort keeps retrieval order.
        scores = [0.5, 0.9, 0.9]
        fixture.expected_rerank_scores = scores
        reranker_class = frozen_reranker_class(scores)
        fixture.modules["scripts.med_copilot_retrieval"].CrossEncoderReranker = reranker_class
        fixture.results = reranker_class().rerank(
            "query", [dict(candidate) for candidate in fixture.candidates], topk=5)
        reference_text = fixture.results[0]
        self.assertIn("Reference P1", reference_text)
        fixture.reference = {"note_id": "NOTE-1"}
        runner = self.load_runner()
        with fixture.runtime():
            runner["main"]()
        self.assert_outputs(reference_text=reference_text)
        _, _, [record] = self.read_run()
        self.assertEqual(record["retrieved_patient_ids"], [
            {"note_id": "NOTE-3"}, {"note_id": "NOTE-1"}, {"note_id": "NOTE-2"}])
        self.assertEqual(record["reranked_patient_ids"], [
            {"note_id": "NOTE-1", "rerank_score": .9},
            {"note_id": "NOTE-2", "rerank_score": .9},
            {"note_id": "NOTE-3", "rerank_score": .5}])

    def test_corpus_without_note_ids_uses_corpus_row_and_duplicates_disambiguate(self):
        fixture_root = self.fixture.root
        for case, corpus, reference in [
            ("no_note_id", [{"subjective": "S0", "objective": "O0", "assessment": "A0", "plan": "P0"},
                            {"subjective": "S1", "objective": "O1", "assessment": "A1", "plan": "P1"}],
             {"corpus_row": 1}),
            ("duplicate_note_id", [{"note_id": "SAME", "subjective": "S0", "objective": "O0",
                                    "assessment": "A0", "plan": "P0"},
                                   {"note_id": "SAME", "subjective": "S1", "objective": "O1",
                                    "assessment": "A1", "plan": "P1"}],
             {"note_id": "SAME", "corpus_row": 1}),
        ]:
            with self.subTest(case=case):
                self.fixture = PipelineFixture(fixture_root / case)
                fixture = self.fixture
                fixture.corpus = corpus
                fixture.write_corpus()
                fixture.reference = reference
                runner = self.load_runner()
                with fixture.runtime():
                    runner["main"]()
                self.assert_outputs()

    def test_final_generation_failure_without_a_reference_keeps_prompts_and_partial_outputs(self):
        fixture = self.fixture
        fixture.candidates = []
        fixture.failure = "final"
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed_with_errors")
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual([call for call in fixture.rerank_calls if call[0] == "rerank"], [])
        self.assertEqual(len(fixture.graph_calls), 1)
        self.assertEqual(len(fixture.calls), 2)
        run_dir, run, results = self.read_run()
        self.assertEqual(results, [{
            **self.expected_result(reference_text="No reference patient available.", reference=None),
            "status": "failed_incomplete", "predictions": None,
            "assessment_plan_raw": None,
        }])
        self.assertIn("No reference patient available.",
                      results[0]["assessment_plan_prompts"]["user"])
        self.assertEqual(fixture.calls[-1]["messages"], [
            {"role": "system", "content": results[0]["assessment_plan_prompts"]["system"]},
            {"role": "user", "content": results[0]["assessment_plan_prompts"]["user"]},
        ])
        self.assert_run_metadata(run_dir, run, "completed_with_errors")

    def test_invalid_ranked_json_keeps_raw_reply_and_independent_pairs_and_fields(self):
        # Provider replies remain captured even when they violate the requested schema.
        fixture_root = self.fixture.root
        valid = prediction_pairs("Dx", ["Rx"])
        missing = {"condition": None, "plan": None}
        cases = [
            ("empty", "", None), ("whitespace", " \n\t", None), ("no_content", None, None),
            ("markdown", "## Assessment\nDx\n## Plan\nRx", None),
            ("malformed", '{"predictions": [', None),
            ("fenced", '```json\n' + json.dumps({"predictions": valid}) + '\n```', None),
            ("root_array", json.dumps(valid), None),
            ("legacy_shape", '{"condition":"Dx","plan":["Rx"]}', None),
            ("missing_predictions", '{}', None),
            ("too_few", json.dumps({"predictions": valid[:2]}), None),
            ("too_many", json.dumps({"predictions": valid + valid[:1]}), None),
            ("extra_root_key", json.dumps({"predictions": valid, "reason": "why"}), None),
            ("valid", json.dumps({"predictions": valid}), valid),
        ]
        for name, pair, expected in [
            ("condition_only", {"condition": "Dx café"}, {"condition": "Dx café", "plan": None}),
            ("plan_only", {"plan": ["Rx"]}, {"condition": None, "plan": ["Rx"]}),
            ("empty_plan", {"condition": "Dx", "plan": []}, {"condition": "Dx", "plan": []}),
            ("wrong_condition", {"condition": ["Dx"], "plan": ["Rx"]},
             {"condition": None, "plan": ["Rx"]}),
            ("wrong_plan", {"condition": "Dx", "plan": "Rx"}, {"condition": "Dx", "plan": None}),
            ("wrong_plan_item", {"condition": "Dx", "plan": ["Rx", 1]},
             {"condition": "Dx", "plan": None}),
            ("overlong_plan", {"condition": "Dx", "plan": ["Rx"] * 4},
             {"condition": "Dx", "plan": None}),
            ("extra_pair_key", {"condition": "Dx", "plan": ["Rx"], "reason": "why"}, missing),
            ("nonobject_pair", ["Dx", "Rx"], missing),
        ]:
            cases.append((name, json.dumps({"predictions": [pair, *valid[1:]]}),
                          [expected, *valid[1:]]))
        tail = ','.join(json.dumps(pair) for pair in valid[1:])
        for name, first in [
            ("duplicate_condition", '{"condition":"A","plan":["Rx"],"condition":"B"}'),
            ("nonfinite", '{"condition":"Dx","plan":[NaN]}'),
        ]:
            cases.append((name, '{"predictions":[' + first + ',' + tail + ']}', None))
        for name, content, predictions in cases:
            with self.subTest(case=name):
                self.fixture = PipelineFixture(fixture_root / name)
                fixture = self.fixture
                fixture.final_output = content
                fixture.final_sections = {"predictions": predictions}
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](), "completed")
                self.assert_outputs()

    def test_raw_json_and_exact_ranked_pairs_survive_unicode_and_whitespace(self):
        fixture = self.fixture
        fixture.final_sections = {"predictions": prediction_pairs(
            "  Viral URI café\u2028likely  ", [" Rest\t", "follow-up\r\nsoon", " Rest\t"])}
        fixture.final_output = " \r\n" + json.dumps(fixture.final_sections, ensure_ascii=False,
                                                   indent=2) + "\r\n\t"
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        self.assert_outputs()
        [run_dir] = self.run_dirs()
        raw_line = (run_dir / "results.jsonl").read_text(encoding="utf-8")
        self.assertEqual(combined_variant_result(json.loads(raw_line))["assessment_plans"]["gpt_4o_mini"]["assessment_plan_raw"], fixture.final_output)

    def test_empty_keyword_arrays_continue_through_full_pipeline(self):
        fixture = self.fixture
        fixture.keyword_content = '{"symptoms": [], "diagnoses": [], "medications": []}'
        runner = self.load_runner()
        with fixture.runtime():
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(fixture.extracted_keywords, [])
        self.assertEqual(len(fixture.keyword_calls), 1)
        self.assertEqual(len(fixture.rerank_calls), 2)
        self.assertEqual(len(fixture.graph_calls), 1)
        self.assertEqual(len(fixture.calls), 2)
        self.assert_outputs()

    def test_keyword_reply_failures_fail_patient_before_downstream_and_restore_cwd(self):
        fixture_root = self.fixture.root
        cases = {
            "fenced_json": {"content": '```json\n{"symptoms": [], "diagnoses": [], "medications": []}\n```'},
            "empty": {"content": ""},
            "missing_category": {"content": '{"symptoms": [], "diagnoses": []}'},
            "wrong_category_type": {"content": '{"symptoms": "cough", "diagnoses": [], "medications": []}'},
            "wrong_term_type": {"content": '{"symptoms": [3], "diagnoses": [], "medications": []}'},
            "refused": {"content": None, "refusal": "request refused"},
            "truncated": {"content": '{"symptoms": []}', "finish_reason": "length"},
            "no_choices": None,
        }
        for name, fields in cases.items():
            with self.subTest(case=name):
                self.fixture = PipelineFixture(fixture_root / name)
                fixture = self.fixture
                fixture.keyword_response = keyword_response(**fields) if fields else SimpleNamespace(choices=[])
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](), "completed_with_errors")
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertEqual(len(fixture.keyword_calls), 1)
                self.assertEqual(fixture.keyword_calls[0]["model"], "gpt-4o-mini")
                self.assertEqual([call for call in fixture.rerank_calls if call[0] == "rerank"], [])
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.graph_calls, [])
                _, _, results = self.read_run()
                self.assertEqual(results, [{"id": fixture.patient_id, "status": "failed_incomplete",
                                            **dict.fromkeys(RESULT_FIELDS[2:]),
                                      "model_responses": {stage: ([{"reported_model": None, "response_id": None, "token_usage": None}]
                                                       if stage == "keyword_extraction" else [])
                                                       for stage in HELPER_NAMESPACE["MODEL_RESPONSE_STAGES"]},
                                      "keyword_extraction_raw": fields.get("content") if fields else None}])
                self.assertIn("ValueError", fixture.stderr.getvalue())

    def test_patient_stage_errors_are_isolated_and_keep_partial_outputs(self):
        fixture_root = self.fixture.root
        unavailable = {
            "retrieval": RESULT_FIELDS[2:9], "keywords": RESULT_FIELDS[2:9],
            "reranking": RESULT_FIELDS[2:9],
            "questions": ["predictions", "assessment_plan_raw", "guideline_questions",
                          "guideline_evidence", "graphrag_context", "assessment_plan_prompts"],
            "graphrag": ["predictions", "assessment_plan_raw", "guideline_evidence",
                         "graphrag_context", "assessment_plan_prompts"],
            "final": ["predictions", "assessment_plan_raw"],
        }
        for stage, expected_calls in [("retrieval", 0), ("keywords", 0), ("reranking", 0),
                                      ("questions", 1), ("graphrag", 1), ("final", 2)]:
            with self.subTest(stage=stage):
                self.fixture = PipelineFixture(fixture_root / stage)
                fixture = self.fixture
                fixture.failure = stage
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](), "completed_with_errors")
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertEqual(len(fixture.calls), expected_calls)
                self.assertEqual(len(fixture.graph_calls), int(stage in {"graphrag", "final"}))
                run_dir, run, results = self.read_run()
                self.assertEqual(results, [{**self.expected_result(), "status": "failed_incomplete",
                                            **dict.fromkeys(unavailable[stage])}])
                self.assert_run_metadata(run_dir, run, "completed_with_errors")

    def test_shared_retrieval_initialization_failure_blocks_all_dependent_patients(self):
        fixture_root = self.fixture.root
        for stage in ["retriever_init", "reranker_init"]:
            with self.subTest(stage=stage):
                self.fixture = PipelineFixture(fixture_root / stage)
                fixture = self.fixture
                fixture.write_patients([{"id": "P1", **fixture.patient}, {"id": "P2", **fixture.patient}])
                fixture.failure = stage
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](), "completed_with_errors")
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.graph_calls, [])
                self.assertEqual(len([call for call in fixture.retrieval_calls if call[0] == "init"]), 1)
                records = [json.loads(line) for line in (self.run_dirs()[0] / "results.jsonl").read_text().splitlines()]
                self.assertEqual([(row["id"], row["status"]) for row in records],
                                 [("P1", "failed_incomplete"), ("P2", "failed_incomplete")])
                for row in records:
                    self.assertEqual(row["evidence_status"]["patient_retrieval"]["error"]["type"], "RuntimeError")
                    self.assertEqual(row["variants"]["patient_retrieval_guidelines"]["error"]["type"], "EvidenceUnavailable")

    def test_malformed_key_questions_are_null_and_raw_reply_still_queries_guidelines(self):
        fixture_root = self.fixture.root
        cases = {
            "not_json": "1. What is the cause?\n2. Which tests?",
            "missing_key": '{"Questions": ["What is the cause?"]}',
            "not_a_list": '{"Key Questions": "What is the cause?"}',
            "non_string_question": '{"Key Questions": ["What is the cause?", 7]}',
            "json_array": '["What is the cause?"]',
        }
        for name, reply in cases.items():
            with self.subTest(case=name):
                self.fixture = PipelineFixture(fixture_root / name)
                fixture = self.fixture
                fixture.questions = reply
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](), "completed")
                self.assertEqual(fixture.graph_calls[0][0]["query"], reply)
                self.assert_outputs(questions=None)

    def test_graphrag_context_keeps_every_returned_table_row_and_empty_table(self):
        fixture = self.fixture
        fixture.context = {
            "reports": pd.DataFrame({"id": ["26"], "title": ["Assessment"],
                                     "content": ["Full summary café\n" + "x" * 5000]}),
            "entities": pd.DataFrame({"id": ["1228"], "entity": ["DIAGNOSTIC ASSESSMENT"],
                                      "in_context": [True]}),
            "sources": pd.DataFrame({"id": ["157"], "text": ["Original excerpt line"]}),
            "empty_table": pd.DataFrame(),
            "additional_table": pd.DataFrame({"value": [1.5, float("nan")]}, index=["a", "b"]),
        }
        runner = self.load_runner()
        with fixture.runtime():
            runner["main"]()
        self.assert_outputs()
        _, _, results = self.read_run()
        self.assertEqual(list(results[0]["graphrag_context"]), list(fixture.context))
        self.assertEqual(results[0]["graphrag_context"]["empty_table"],
                         {"columns": [], "index": [], "data": []})
        self.assertEqual(results[0]["graphrag_context"]["additional_table"],
                         {"columns": ["value"], "index": ["a", "b"], "data": [[1.5], [None]]})
        self.assertEqual(len(results[0]["graphrag_context"]["reports"]["data"][0][2]), 5018)

    # ----------------------------------------------------------------
    # Multiple patients, statuses, and interruption
    # ----------------------------------------------------------------

    def write_marked_patients(self, markers=("ALPHA", "BRAVO", "CHARLIE")):
        fixture = self.fixture
        records = [{"id": f"id-{marker.lower()}-é",
                    "subjective": f"MARKER_{marker} reports cough.\nSecond line café",
                    "objective": f"MARKER_{marker} temperature 37.2 °C SpO₂ 98%"}
                   for marker in markers]
        fixture.write_patients(records)

        def reply(stage, text):
            # Each response echoes the patient marker found in its own request.
            marker = re.search(r"MARKER_[A-Z]+", text).group(0)
            if stage == "questions":
                return json.dumps({"Key Questions": [f"{marker} question?"]})
            if stage == "final":
                return json.dumps({"predictions": prediction_pairs(
                    f"{marker} diagnosis café", [f"{marker} plan end"])}, ensure_ascii=False)
            return (f"{marker} guideline evidence\nline two",
                    {"sources": pd.DataFrame({"id": [marker], "text": [f"{marker} source\n"]})})

        fixture.reply = reply
        return records

    def test_multiple_patients_are_processed_in_order_without_mixing_outputs(self):
        fixture = self.fixture
        records = self.write_marked_patients()
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        self.assertEqual(len(fixture.calls), 6)
        self.assertEqual(len([call for call in fixture.retrieval_calls if call[0] == "init"]), 1)
        self.assertEqual(len([call for call in fixture.rerank_calls if call[0] == "init"]), 1)
        run_dir, run, results = self.read_run()
        self.assert_run_metadata(run_dir, run, "completed")
        raw_lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").split("\n")
        self.assertEqual(len(raw_lines), 4)
        self.assertEqual([record["id"] for record in results], [record["id"] for record in records])
        for record, result in zip(records, results):
            marker = re.search(r"MARKER_[A-Z]+", record["subjective"]).group(0)
            conditions = f'Subjective: {record["subjective"]}\nObjective: {record["objective"]}'
            evidence = f"{marker} guideline evidence\nline two"
            self.assertEqual(result, {
                **self.expected_capture(),
                "id": record["id"],
                "status": "completed",
                "predictions": prediction_pairs(f"{marker} diagnosis café", [f"{marker} plan end"]),
                "assessment_plan_raw": json.dumps({"predictions": prediction_pairs(
                    f"{marker} diagnosis café", [f"{marker} plan end"])}, ensure_ascii=False),
                "selected_reference_patient": fixture.reference,
                "guideline_questions": [f"{marker} question?"],
                "guideline_evidence": evidence,
                "graphrag_context": {"sources": {"columns": ["id", "text"], "index": [0],
                                                 "data": [[marker, f"{marker} source\n"]]}},
                "assessment_plan_prompts": {
                    "system": "You are a clinical expert.",
                    "user": self.expected_prompt(conditions=conditions, guideline_text=evidence),
                },
                "evaluation": None,
            })
            others = [m for m in ("MARKER_ALPHA", "MARKER_BRAVO", "MARKER_CHARLIE") if m != marker]
            serialized = json.dumps(result, ensure_ascii=False)
            for other in others:
                self.assertNotIn(other, serialized)
        final_calls = [call for call in fixture.calls if call.get("response_format") == FINAL_RESPONSE_FORMAT]
        for call, result in zip(final_calls, results):
            self.assertEqual(call["messages"][-1]["content"], result["assessment_plan_prompts"]["user"])
        self.assert_no_legacy_outputs()

    def test_isolated_patient_failure_continues_batch_with_completed_with_errors(self):
        fixture = self.fixture
        records = self.write_marked_patients()
        fixture.failure = "graphrag"
        fixture.failure_marker = "MARKER_BRAVO"
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed_with_errors")
        run_dir, run, results = self.read_run()
        self.assert_run_metadata(run_dir, run, "completed_with_errors")
        self.assertEqual([(result["id"], result["status"]) for result in results], [
            (records[0]["id"], "completed"), (records[1]["id"], "failed_incomplete"),
            (records[2]["id"], "completed"),
        ])
        failed = results[1]
        self.assertEqual(failed["guideline_questions"], ["MARKER_BRAVO question?"])
        self.assertEqual(failed["selected_reference_patient"], fixture.reference)
        for field in ("guideline_evidence", "graphrag_context", "assessment_plan_prompts",
                      "predictions", "assessment_plan_raw"):
            self.assertIsNone(failed[field])
        self.assertIn("MARKER_CHARLIE", results[2]["predictions"][0]["condition"])
        self.assertIn("MARKER_CHARLIE", results[2]["predictions"][0]["plan"][0])
        self.assertIn(f"Patient {records[1]['id']} failed", fixture.stderr.getvalue())

    def test_all_patients_failing_still_finishes_the_batch(self):
        fixture = self.fixture
        self.write_marked_patients(("ALPHA", "BRAVO"))
        fixture.failure = "questions"
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed_with_errors")
        _, run, results = self.read_run()
        self.assertEqual(run["status"], "completed_with_errors")
        self.assertEqual([result["status"] for result in results], ["failed_incomplete"] * 2)

    def test_interruption_saves_earlier_results_and_marks_unfinished_patients(self):
        fixture = self.fixture
        records = self.write_marked_patients()
        fixture.failure = "final"
        fixture.failure_marker = "MARKER_BRAVO"
        fixture.error = KeyboardInterrupt()
        saved_before_second = []
        reply = fixture.reply

        def observing_reply(stage, text):
            if stage == "questions" and "MARKER_BRAVO" in text:
                [run_dir] = self.run_dirs()
                saved_before_second.extend(
                    (run_dir / "results.jsonl").read_text(encoding="utf-8").split("\n")[:-1])
                saved_before_second.append(json.loads((run_dir / "run.json").read_text())["status"])
            return reply(stage, text)

        fixture.reply = observing_reply
        runner = self.load_runner()
        with fixture.runtime():
            with self.assertRaises(KeyboardInterrupt):
                runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        # The first patient was persisted before the second began.
        self.assertEqual(json.loads(saved_before_second[0])["id"], records[0]["id"])
        self.assertEqual(saved_before_second[1], "running")
        results = self.assert_stopped_run([
            (records[0]["id"], "completed"), (records[1]["id"], "failed_incomplete"),
            (records[2]["id"], "not_processed"),
        ], reference_data=self.expected_reference_data())
        for field in ("predictions", "assessment_plan_raw"):
            self.assertIsNone(results[1][field])
        self.assertEqual(results[1]["selected_reference_patient"], fixture.reference)
        self.assertIn("MARKER_BRAVO", results[1]["assessment_plan_prompts"]["user"])
        self.assertEqual(results[2], {"id": records[2]["id"], "status": "not_processed",
                                      **dict.fromkeys(RESULT_FIELDS[2:]),
                                      "model_responses": {stage: [] for stage in HELPER_NAMESPACE["MODEL_RESPONSE_STAGES"]}})
        self.assertEqual(len(fixture.calls), 4)

    def test_results_write_failure_preserves_saved_results_and_stops_the_run(self):
        fixture = self.fixture
        records = self.write_marked_patients()
        original_open = Path.open
        appends = []

        def failing_open(path, mode="r", *args, **kwargs):
            if path.name == "results.jsonl" and mode == "a":
                appends.append(path)
                if len(appends) == 2:
                    raise OSError(errno.ENOSPC, "No space left on device", str(path))
            return original_open(path, mode, *args, **kwargs)

        runner = self.load_runner()
        with fixture.runtime(), patch.object(Path, "open", failing_open):
            with self.assertRaises(OSError) as caught:
                runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(caught.exception.errno, errno.ENOSPC)
        # No further appends after the failed write, so no partial or duplicate lines.
        self.assertEqual(len(appends), 2)
        self.assertEqual(len(fixture.calls), 4)
        results = self.assert_stopped_run([(records[0]["id"], "completed")],
                                          reference_data=self.expected_reference_data())
        self.assertIn("MARKER_ALPHA", results[0]["predictions"][0]["condition"])

    def test_run_json_is_running_until_the_batch_finishes(self):
        fixture = self.fixture
        observed = []

        def reply(stage, text):
            [run_dir] = self.run_dirs()
            observed.append(json.loads((run_dir / "run.json").read_text(encoding="utf-8")))
            return {"questions": fixture.questions, "final": fixture.final_output}.get(
                stage, (fixture.guideline_text, fixture.context))

        fixture.reply = reply
        runner = self.load_runner()
        with fixture.runtime():
            runner["main"]()
        self.assertEqual({(run["status"], run["finished_at"]) for run in observed}, {("running", None)})
        self.assertEqual(observed[0]["reference_data"], self.expected_reference_data())
        self.assert_outputs()

    # ----------------------------------------------------------------
    # Prepared input
    # ----------------------------------------------------------------

    def test_prepared_input_preserves_sections_dates_unicode_and_whitespace(self):
        fixture_root = self.fixture.root
        cases = {
            "single_encounter": " \n# Chief Complaint\nPrepared evidence café — 37.2 °C.\n\t ",
            "former_heading": "Initial findings.\n# Assessment and Plan\nSupplied evidence after the heading.\n",
            "former_date": "Initial findings.\n\n2000-07-02\nSupplied evidence in the later encounter.\n",
        }
        for name, patient_text in cases.items():
            with self.subTest(case=name):
                self.fixture = PipelineFixture(fixture_root / name)
                fixture = self.fixture
                fixture.patient["subjective"] = patient_text
                fixture.conditions = (
                    f'Subjective: {patient_text}\n'
                    f'Objective: {fixture.patient["objective"]}'
                )
                fixture.write_patients()
                runner = self.load_runner()
                with fixture.runtime():
                    runner["main"]()
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertEqual(len(fixture.calls), 2)
                self.assert_outputs()

    def test_prepared_so_preserves_empty_sections_and_field_text(self):
        fixture_root = self.fixture.root
        cases = {
            "no_subjective": {"subjective": "", "objective": self.fixture.patient["objective"]},
            "no_objective": {"subjective": self.fixture.patient["subjective"], "objective": ""},
            "whitespace_section": {"subjective": " \n\t", "objective": self.fixture.patient["objective"]},
            "empty_sections": {"subjective": "", "objective": ""},
            "preserved_whitespace": {
                "subjective": " \n2000-07-20: No fever; cough for 3 days. café.\t ",
                "objective": " Temperature 37.2 °C; SpO₂ 98%; weight 70 kg. \n",
            },
        }
        for name, patient in cases.items():
            with self.subTest(case=name):
                self.fixture = PipelineFixture(fixture_root / name)
                fixture = self.fixture
                fixture.patient = patient
                fixture.write_patients()
                fixture.conditions = (
                    f'Subjective: {patient["subjective"]}\n'
                    f'Objective: {patient["objective"]}'
                )
                runner = self.load_runner()
                with fixture.runtime():
                    runner["main"]()
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertEqual(len(fixture.calls), 2)
                self.assertEqual(fixture.retrieval_calls[1][1], fixture.conditions)
                self.assertEqual(fixture.rerank_calls[1][1], fixture.conditions)
                self.assertEqual(fixture.calls[0]["messages"][-1]["content"],
                                 PROMPTS["KEY_QUESTIONS_TEMPLATE"].format(conditions=fixture.conditions))
                self.assert_outputs()

    def test_only_so_fields_reach_downstream_and_outputs(self):
        fixture = self.fixture
        fixture.patient.update({
            "assessment": "UNUSED_TARGET_ASSESSMENT",
            "condition": "UNUSED_TARGET_CONDITION",
            "diagnosis": "UNUSED_TARGET_DIAGNOSIS",
            "plan": "UNUSED_TARGET_PLAN",
            "label": {"description": "UNUSED_TARGET_LABEL"},
            "metadata": {"source": "UNUSED_TARGET_SOURCE"},
        })
        fixture.write_patients()
        runner = self.load_runner()
        with fixture.runtime():
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(len(fixture.calls), 2)
        self.assertEqual(fixture.retrieval_calls[1][1], fixture.conditions)
        self.assertEqual(fixture.rerank_calls[1][1], fixture.conditions)
        for call in fixture.calls + fixture.keyword_calls:
            self.assertNotIn("UNUSED_TARGET", call["messages"][-1]["content"])
        self.assertNotIn("UNUSED_TARGET", fixture.stdout.getvalue())
        self.assert_outputs()
        [run_dir] = self.run_dirs()
        for output in run_dir.iterdir():
            self.assertNotIn("UNUSED_TARGET", output.read_text(encoding="utf-8"))

    def test_invalid_records_anywhere_in_jsonl_stop_before_pipeline_or_output(self):
        fixture_root = self.fixture.root
        valid = {"id": "VALID_FIRST", **self.fixture.patient}
        valid_later = {"id": "VALID_LATER", **self.fixture.patient}
        cases = {
            "malformed_json": ('{"subjective": "cough",', "invalid patient JSON"),
            "markdown_wrapped_json": ("```json", "invalid patient JSON"),
            "json_null": ("null", "patient must be a JSON object"),
            "json_array": (json.dumps([valid]), "patient must be a JSON object"),
            "json_string": ('"patient"', "patient must be a JSON object"),
            "json_number": ("123", "patient must be a JSON object"),
            "json_boolean": ("true", "patient must be a JSON object"),
            "empty_object": ("{}", "id must be a nonempty string"),
            "missing_id": (json.dumps(self.fixture.patient), "id must be a nonempty string"),
            "legacy_uuid_key": (json.dumps({"uuid": "LEGACY", **self.fixture.patient}),
                                "id must be a nonempty string"),
            "empty_id": (json.dumps({**valid, "id": ""}), "id must be a nonempty string"),
            "blank_id": (json.dumps({**valid, "id": " \t"}), "id must be a nonempty string"),
            "duplicate_id": (json.dumps({**valid_later, "id": "VALID_FIRST"}), "duplicate id 'VALID_FIRST'"),
            "missing_subjective": (json.dumps({"id": "X", "objective": "measured evidence"}),
                                   "subjective must be a string"),
            "missing_objective": (json.dumps({"id": "X", "subjective": "reported evidence"}),
                                  "objective must be a string"),
            "wrong_case": (json.dumps({"id": "X", "Subjective": "reported evidence",
                                       "objective": "measured evidence"}), "subjective must be a string"),
        }
        for value in (None, 123, True, [], {}):
            cases[f"id_{type(value).__name__}"] = (
                json.dumps({**valid_later, "id": value}), "id must be a nonempty string",
            )
        for field in ("subjective", "objective"):
            for value in (None, 123, True, [], {}):
                cases[f"{field}_{type(value).__name__}"] = (
                    json.dumps({**valid_later, field: value}), f"{field} must be a string",
                )
        for name, (invalid_line, message) in cases.items():
            for position in ("first", "later"):
                if name == "duplicate_id" and position == "first":
                    continue
                with self.subTest(case=name, position=position):
                    self.fixture = PipelineFixture(fixture_root / name / position)
                    fixture = self.fixture
                    prefix = "\n \t\n"
                    if position == "later" or name == "duplicate_id":
                        prefix += json.dumps(valid) + "\n"
                    invalid_line_number = prefix.count("\n") + 1
                    contents = prefix + invalid_line + "\n" + json.dumps(valid_later) + "\n"
                    fixture.input_path.write_text(contents, encoding="utf-8")
                    runner = self.load_runner()
                    with fixture.runtime():
                        with self.assertRaises(ValueError) as caught:
                            runner["main"]()
                        self.assertEqual(Path.cwd(), fixture.caller)
                    self.assertIn(f"{fixture.input_path}:{invalid_line_number}:", str(caught.exception))
                    self.assertIn(message, str(caught.exception))
                    if message == "invalid patient JSON":
                        self.assertIsInstance(caught.exception.__cause__, json.JSONDecodeError)
                    self.assertEqual([event[0] for event in fixture.events], ["preload", "import"])
                    self.assertEqual(fixture.calls, [])
                    self.assertEqual(fixture.retrieval_calls, [])
                    self.assertEqual(fixture.rerank_calls, [])
                    self.assertEqual(fixture.graph_calls, [])
                    self.assertFalse(fixture.output.exists())
                    self.assertEqual(fixture.input_path.read_text(encoding="utf-8"), contents)

    def test_empty_jsonl_stops_before_pipeline_or_output(self):
        fixture_root = self.fixture.root
        for name, contents in [("empty", ""), ("blank_lines", "\n \t\n\n")]:
            with self.subTest(case=name):
                self.fixture = PipelineFixture(fixture_root / name)
                fixture = self.fixture
                fixture.input_path.write_text(contents, encoding="utf-8")
                runner = self.load_runner()
                with fixture.runtime():
                    with self.assertRaisesRegex(ValueError, "no patient records found") as caught:
                        runner["main"]()
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertIn(str(fixture.input_path), str(caught.exception))
                self.assertEqual([event[0] for event in fixture.events], ["preload", "import"])
                self.assertEqual(fixture.calls, [])
                self.assertFalse(fixture.output.exists())

    def test_missing_jsonl_stops_without_falling_back_to_legacy_inputs(self):
        fixture = self.fixture
        fixture.input_path.unlink()
        legacy_note = fixture.input_path.parent / (
            "Peter292_Gleichner915_0db4522b-544a-43b1-ad63-3caeb72be2ab.txt"
        )
        legacy_note.write_text("LEGACY_NOTE_MUST_NOT_BE_USED", encoding="utf-8")
        legacy_input = fixture.root / "data_run_med_copilot/input" / INPUT_NAME
        legacy_input.parent.mkdir(parents=True)
        legacy_input.write_text(json.dumps({"id": "LEGACY", **fixture.patient}) + "\n", encoding="utf-8")
        runner = self.load_runner()
        with fixture.runtime():
            with self.assertRaises(FileNotFoundError) as caught:
                runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(caught.exception.filename, str(fixture.input_path))
        self.assertEqual([event[0] for event in fixture.events], ["preload", "import"])
        self.assertEqual(fixture.calls, [])
        self.assertFalse(fixture.output.exists())
        self.assertEqual(list((fixture.root / "data_run_med_copilot").rglob("*")),
                         [legacy_input.parent, legacy_input])

    def test_entire_jsonl_is_loaded_before_processing_every_patient(self):
        fixture = self.fixture
        patients = [
            {"id": "FIRST_PATIENT_ID", **fixture.patient},
            {"id": "SECOND_PATIENT_ID", "subjective": "Second S", "objective": "Second O"},
            {"id": "THIRD_PATIENT_ID", "subjective": "Third S", "objective": "Third O",
             "metadata": {"source": "café"}},
        ]
        fixture.input_path.write_text(
            "\n \t\n" + "\n\n".join(json.dumps(patient, ensure_ascii=False) for patient in patients)
            + "\n \t\n",
            encoding="utf-8",
        )
        original_input = fixture.input_path.read_bytes()
        original_loads = json.loads
        patient_ids = {patient["id"] for patient in patients}
        loaded_patients = []

        def track_patient_loads(value, *args, **kwargs):
            record = original_loads(value, *args, **kwargs)
            if isinstance(record, dict) and record.get("id") in patient_ids and "subjective" in record:
                # Dependencies load in the header, but all records must be
                # decoded before making an API request or starting a stage.
                self.assertEqual([event[0] for event in fixture.events], ["preload", "import"])
                self.assertEqual(fixture.keyword_calls, [])
                self.assertEqual(fixture.calls, [])
                self.assertFalse(fixture.output.exists())
                loaded_patients.append(record)
            return record

        runner = self.load_runner()
        with fixture.runtime(), patch("json.loads", side_effect=track_patient_loads):
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(loaded_patients, patients)
        self.assertEqual(len(fixture.calls), 6)
        self.assertEqual([call[1] for call in fixture.retrieval_calls if call[0] == "search"], [
            f'Subjective: {patient["subjective"]}\nObjective: {patient["objective"]}'
            for patient in patients
        ])
        self.assertEqual(fixture.input_path.read_bytes(), original_input)
        _, run, results = self.read_run()
        self.assertEqual(run["status"], "completed")
        self.assertEqual([result["id"] for result in results], [patient["id"] for patient in patients])

    def test_single_record_without_trailing_newline_runs_full_pipeline(self):
        fixture = self.fixture
        fixture.input_path.write_text(json.dumps({"id": fixture.patient_id, **fixture.patient}),
                                      encoding="utf-8")
        runner = self.load_runner()
        with fixture.runtime():
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(len(fixture.calls), 2)
        self.assert_outputs()

    def test_checked_in_sample_input_uses_id_and_unique_values(self):
        records = HELPER_NAMESPACE["load_patient_records"](
            ROOT / "experiments/input" / INPUT_NAME)
        self.assertEqual(len(records), 10)
        self.assertEqual(len({record["id"] for record in records}), len(records))
        self.assertTrue(all(list(record) == ["id", "subjective", "objective"] for record in records))
        self.assertFalse((ROOT / "data_run_med_copilot").exists())

    # ----------------------------------------------------------------
    # Prompts
    # ----------------------------------------------------------------

    def test_final_prompt_uses_so_and_preserves_reference_and_guideline_evidence(self):
        self.assertEqual(set(PROMPTS), PROMPT_NAMES)
        template = PROMPTS["EVALUATE_TEMPLATE_KEYINFO"]
        self.assertIn("(Subjective and Objective only)", template)
        rendered = template.format(
            conditions=self.fixture.conditions, example=self.fixture.results[0],
            Key_info=self.fixture.guideline_text,
        )
        for evidence in [self.fixture.conditions, self.fixture.results[0], self.fixture.guideline_text]:
            self.assertIn(evidence, rendered)
        self.assertIn("Assessment: REFERENCE_DIAGNOSIS", rendered)
        for requirement in [
            "top 3 most likely conditions", "ranked from most to least likely",
            "separate plan of at most 3 items for each condition",
            "using the Subjective and Objective as context",
            "tailor its own plan", "investigations, treatment, and follow-up",
            "ONLY one valid JSON object", 'single key "predictions"',
            "exactly three condition/plan pairs", "descending likelihood order",
            "Give distinct conditions", "array position determines rank",
            'Each "condition" must be a single concise diagnosis name',
            'Each "plan" must be a JSON array containing at most three',
            "one action per item", "empty array",
            "Do not include Markdown", "code fences", "any additional keys or text",
        ]:
            self.assertIn(requirement, rendered)
        self.assertLess(rendered.index("clinical reference evidence:"),
                        rendered.index(self.fixture.results[0]))
        output_example, _ = json.JSONDecoder().raw_decode(rendered[rendered.index('{\n  "predictions"'):])
        self.assertEqual(set(output_example), {"predictions"})
        self.assertEqual(len(output_example["predictions"]), 3)
        for pair in output_example["predictions"]:
            self.assertEqual(set(pair), {"condition", "plan"})
            self.assertIsInstance(pair["condition"], str)
            self.assertIsInstance(pair["plan"], list)
            self.assertLessEqual(len(pair["plan"]), 3)
            self.assertTrue(all(isinstance(item, str) for item in pair["plan"]))
        self.assertNotIn("Do NOT output JSON", rendered)
        self.assertLess(rendered.index("Now, here is the patient's conditions:"),
                        rendered.index(self.fixture.conditions))
        self.assertLess(rendered.index("Additional Information"),
                        rendered.index(self.fixture.guideline_text))

    def test_question_prompt_uses_so_for_diagnostic_and_management_evidence(self):
        rendered = PROMPTS["KEY_QUESTIONS_TEMPLATE"].format(conditions=self.fixture.conditions)
        self.assertIn("(Subjective and Objective only)", rendered)
        self.assertNotIn("assessment", rendered.lower())
        self.assertIn(self.fixture.conditions, rendered)
        for requirement in ["at most 4", "Diagnostic criteria", "distinguish plausible diagnoses",
                            "Investigations", "treatment conditional", "hypotheses"]:
            self.assertIn(requirement, rendered)
        example, _ = json.JSONDecoder().raw_decode(rendered[rendered.index("{"):])
        self.assertEqual(set(example), {"Key Questions"})
        self.assertEqual(len(example["Key Questions"]), 4)
        self.assertTrue(all(isinstance(question, str) and question.strip()
                            for question in example["Key Questions"]))

    def test_prepared_so_runs_without_normalization_prompts_or_calls(self):
        fixture = self.fixture
        # Only the active question/final prompts and response schema are needed.
        fixture.prompt_module.write_text(
            "\n".join(f"{name} = {value!r}" for name, value in fixture.prompts.items())
            + f"\nVARIANT_TEMPLATES = {{'patient_retrieval_guidelines': EVALUATE_TEMPLATE_KEYINFO}}\nCONDITION_PLAN_RESPONSE_FORMAT = {FINAL_RESPONSE_FORMAT!r}\n",
            encoding="utf-8",
        )
        original_input = fixture.input_path.read_bytes()
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(len(fixture.calls), 2)
        self.assertEqual(fixture.calls[0]["messages"][-1]["content"],
                         fixture.prompts["KEY_QUESTIONS_TEMPLATE"].format(
                             conditions=fixture.conditions))
        self.assertEqual(fixture.calls[1]["messages"][-1]["content"], self.expected_prompt())
        self.assertEqual(fixture.retrieval_calls[1][1], fixture.conditions)
        self.assertEqual(fixture.rerank_calls[1][1], fixture.conditions)
        self.assertEqual(fixture.input_path.read_bytes(), original_input)
        self.assert_outputs()

    def test_guideline_question_model_setting_controls_only_question_generation(self):
        fixture = self.fixture
        runner = self.load_runner()
        self.assertNotIn("GUIDELINE_QUESTION_MODEL", runner)
        fixture.update_experiment(guideline_question_model="question-model-only")
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        self.assertEqual(fixture.calls[0]["model"], "question-model-only")
        self.assertEqual(fixture.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(fixture.calls[1]["model"], "openai/gpt-4o-mini")
        self.assertEqual(fixture.keyword_calls[0]["model"], "gpt-4o-mini")
        self.assertEqual(fixture.graph_calls[0][0]["query"], fixture.questions)
        _, run, [record] = self.read_run()
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["experiment"]["selected_models"], ["gpt_4o_mini"])
        self.assertEqual(record["guideline_questions"], json.loads(fixture.questions)["Key Questions"])
        self.assertEqual(record["assessment_plan_raw"], fixture.final_output)

    def test_full_pipeline_uses_project_prompts_from_an_unrelated_cwd(self):
        fixture = self.fixture
        fixture.prompts = {name: f"PROJECT_{name} café\n{value}" for name, value in PROMPTS.items()}
        fixture.prompt_module.write_text(
            "\n".join(f"{name} = {value!r}" for name, value in fixture.prompts.items())
            + f"\nVARIANT_TEMPLATES = {{'patient_retrieval_guidelines': EVALUATE_TEMPLATE_KEYINFO}}\nCONDITION_PLAN_RESPONSE_FORMAT = {FINAL_RESPONSE_FORMAT!r}\n",
            encoding="utf-8",
        )
        with fixture.runtime():
            runner = runpy.run_path(str(fixture.script), run_name="runner_under_test")
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
            self.assertEqual(Path(sys.modules["med_copilot.prompts.diagnosis_plan"].__file__),
                             fixture.prompt_module)
        self.assertEqual(fixture.calls[0]["messages"], [
            {"role": "system", "content": "You are a clinical expert."},
            {"role": "user", "content": fixture.prompts["KEY_QUESTIONS_TEMPLATE"].format(
                conditions=fixture.conditions)},
        ])
        self.assertEqual(fixture.retrieval_calls[1][1], fixture.conditions)
        self.assertEqual(fixture.graph_calls[0][0]["query"], fixture.questions)
        self.assert_outputs()

    def test_missing_project_prompts_stops_without_using_upstream_templates(self):
        fixture = self.fixture
        fixture.prompt_module.unlink()
        # Reproduce discovery's real-checkout path even when run as a script.
        with patch.object(sys, "path", [str(ROOT), *sys.path]), fixture.runtime():
            with self.assertRaises(ImportError) as caught:
                runpy.run_path(str(fixture.script), run_name="runner_under_test")
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertIn("diagnosis_plan", str(caught.exception))
        self.assertEqual(fixture.calls, [])
        self.assertEqual(fixture.retrieval_calls, [])
        self.assertEqual(fixture.graph_calls, [])
        self.assertFalse(fixture.output.exists())

    # ----------------------------------------------------------------
    # Reference corpus selection
    # ----------------------------------------------------------------

    def test_configured_corpus_outside_upstream_runs_the_full_pipeline(self):
        # Nest the project so the external corpus stays inside the temporary directory.
        self.fixture = PipelineFixture(self.fixture.root / "project")
        fixture = self.fixture
        corpus_path = fixture.root.parent / "alternative corpus" / "patients café.json"
        corpus_path.parent.mkdir()
        fixture.corpus = [
            {"subjective": "Alternate S café", "objective": "Alternate O",
             "assessment": "Alternate A", "plan": "Alternate P"},
            {"subjective": "Second S", "objective": "Second O",
             "assessment": "Second A", "plan": "Second P"},
        ]
        fixture.write_corpus(corpus_path)
        (fixture.upstream / "soap_with_metadata.json").unlink()
        fixture.candidates = [{"idx": 1, "id": "alternate-patient", "display_text": "Second S"}]
        fixture.rerank_scores = [0.4]
        fixture.results = ["ALTERNATE_SIMILAR_CASE café\nSecond P"]
        # Without note ids, the zero-based corpus row identifies the reference.
        fixture.reference = {"corpus_row": 1}
        runner = self.load_runner()
        fixture.update_experiment(patient_corpus=str(corpus_path))
        with fixture.runtime():
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        init, search = fixture.retrieval_calls
        self.assertEqual(init[1].to_dict(orient="records"), fixture.corpus)
        self.assertEqual(init[2:], (0.5, fixture.artifacts))
        self.assertEqual(search, ("search", fixture.conditions, 20, fixture.artifacts))
        self.assertEqual(fixture.rerank_calls[1],
                         ("rerank", fixture.conditions, fixture.candidates, 5, fixture.artifacts))
        self.assertEqual(len(fixture.calls), 2)
        self.assertEqual(fixture.graph_calls[0][1], fixture.guidelines / "output")
        run_dir, run, results = self.read_run()
        self.assertEqual(results, [self.expected_result()])
        self.assertEqual(run["reference_data"]["patient_retrieval"]["corpus_file"],
                         str(corpus_path.resolve()))
        self.assert_run_metadata(run_dir, run, "completed", reference_data=(
            self.expected_reference_data(corpus=corpus_path)))

    def assert_corpus_load_failure(self, corpus_path, exception_type):
        fixture = self.fixture
        runner = self.load_runner()
        fixture.update_experiment(patient_corpus=str(corpus_path))
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed_with_errors")
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(len(fixture.calls), 0)
        self.assertEqual(fixture.retrieval_calls, [])
        self.assertEqual(fixture.rerank_calls, [])
        self.assertEqual(fixture.graph_calls, [])
        record = json.loads((self.run_dirs()[0] / "results.jsonl").read_text())
        error = record["evidence_status"]["patient_retrieval"]["error"]
        self.assertEqual(error["type"], exception_type.__name__)
        self.assertEqual(record["status"], "failed_incomplete")
        return error

    def test_missing_configured_corpus_blocks_pipeline_without_fallback(self):
        corpus_path = self.fixture.root / "missing-patients.json"
        error = self.assert_corpus_load_failure(corpus_path, FileNotFoundError)
        self.assertIn(str(corpus_path), error["message"])

    def test_invalid_configured_corpus_blocks_pipeline_without_fallback(self):
        corpus_path = self.fixture.root / "invalid-patients.json"
        corpus_path.write_text("not valid JSON", encoding="utf-8")
        error = self.assert_corpus_load_failure(corpus_path, json.JSONDecodeError)
        self.assertIn("Expecting value", error["message"])

    def provider_graph(self):
        harness = ProviderHarness()
        fixture = self.fixture
        def graph_search(**kwargs):
            fixture.graph_calls.append((kwargs, Path.cwd()))
            answer, _ = asyncio.run(harness.query())
            return answer, fixture.context
        fixture.modules["graphrag.cli.query"].run_local_search = graph_search
        return harness

    def test_persisted_capture_keeps_exact_raw_keywords_and_reported_models_for_all_stages(self):
        fixture = self.fixture
        raw = ' \n{ "medications": ["药", " café "], "diagnoses": ["asthma", "café"], "symptoms": [" café ", "cough", "cough", "左腕"] }\t '
        fixture.keyword_response = keyword_response(raw)
        fixture.keyword_response.model, fixture.keyword_response.id = "reported-keyword", "keyword-id"
        fixture.keyword_response.usage = SimpleNamespace(prompt_tokens=9, total_tokens=13)
        for stage, content in (("questions", fixture.questions), ("final", fixture.final_output)):
            fixture.completion_responses[stage] = SimpleNamespace(
                model=f"reported-{stage}", id=f"{stage}-id",
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=3, total_tokens=23),
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        harness = self.provider_graph()
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        _, _, [record] = self.read_run()
        self.assertEqual(record["keyword_extraction_raw"], raw)
        self.assertEqual(record["retrieval_keywords"], ["café", "cough", "左腕", "asthma", "药"])
        self.assertEqual(record["retrieval_keywords"], fixture.extracted_keywords)
        self.assertEqual(record["retrieved_patient_ids"], [{"note_id": "NOTE-A"}, {"note_id": "NOTE-B"}])
        self.assertEqual(record["reranked_patient_ids"], [
            {"note_id": "NOTE-B", "rerank_score": .9}, {"note_id": "NOTE-A", "rerank_score": .2}])
        self.assertEqual(record["model_responses"]["keyword_extraction"], [{
            "reported_model": "reported-keyword", "response_id": "keyword-id",
            "token_usage": {"prompt_tokens": 9, "completion_tokens": None, "total_tokens": 13}}])
        for stage, key in (("questions", "guideline_questions"), ("final", "assessment_plan")):
            self.assertEqual(record["model_responses"][key], [{
                "reported_model": f"reported-{stage}", "response_id": f"{stage}-id",
                "token_usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}}])
        self.assertEqual(record["model_responses"]["guideline_evidence"], [{
            "reported_model": "reported-chat-snapshot", "response_id": "stream-provider-id",
            "token_usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}}])
        self.assertEqual(record["model_responses"]["guideline_query_embedding"], [{
            "reported_model": "reported-embedding-snapshot", "response_id": None,
            "token_usage": {"prompt_tokens": 11, "completion_tokens": None, "total_tokens": 11}}])
        self.assertEqual(len(fixture.keyword_calls), 1)
        self.assertEqual(len(fixture.calls), 2)
        self.assertEqual(len(fixture.graph_calls), 1)
        harness.chat_create.assert_awaited_once()
        harness.embedding_create.assert_awaited_once()
        prompt = fixture.calls[-1]["messages"][-1]["content"]
        self.assertIn(fixture.results[0], prompt)
        for field in RESULT_FIELDS[RESULT_FIELDS.index("keyword_extraction_raw"):]:
            self.assertNotIn(field, prompt)
        self.assertNotIn(raw, prompt)
        self.assertEqual(fixture.graph_calls[0][0]["streaming"], False)

    def test_all_keyword_response_failures_persist_raw_and_provider_metadata(self):
        root = self.fixture.root
        cases = [("malformed", keyword_response('{"symptoms":')),
                 ("invalid-schema", keyword_response('{"symptoms": [], "diagnoses": []}')),
                 ("whitespace", keyword_response(' \n\t ')), ("empty", keyword_response('')),
                 ("truncated", keyword_response(' café {', finish_reason="length")),
                 ("refused", keyword_response('refused text', refusal="refusal")),
                 ("missing-choices", SimpleNamespace()), ("empty-choices", SimpleNamespace(choices=[])),
                 ("none", keyword_response(None)), ("object", keyword_response({"text": "not string"})),
                 ("missing-content", SimpleNamespace(choices=[SimpleNamespace(
                     finish_reason="stop", message=SimpleNamespace(refusal=None))]))]
        for name, response in cases:
            with self.subTest(case=name):
                self.fixture = PipelineFixture(root / name)
                fixture = self.fixture
                response.id, response.model = "provider-id", "reported"
                response.usage = SimpleNamespace(total_tokens=17)
                fixture.keyword_response = response
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](), "completed_with_errors")
                _, _, [record] = self.read_run()
                choices = getattr(response, "choices", None)
                raw = getattr(choices[0].message, "content", None) if choices else None
                self.assertEqual(record["keyword_extraction_raw"], raw if isinstance(raw, str) else None)
                self.assertIsNone(record["retrieval_keywords"])
                self.assertIsNone(record["retrieved_patient_ids"])
                self.assertIsNone(record["reranked_patient_ids"])
                self.assertEqual(record["model_responses"]["keyword_extraction"], [{
                    "reported_model": "reported", "response_id": "provider-id",
                    "token_usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": 17}}])
                self.assertTrue(all(value == [] for stage, value in record["model_responses"].items()
                                    if stage != "keyword_extraction"))
                self.assertEqual(len(fixture.keyword_calls), 1)
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.graph_calls, [])
                self.assertIn("ValueError" if name != "missing-content" else "AttributeError", fixture.stderr.getvalue())

    def test_real_pipeline_preserves_capture_before_embedding_attribution_and_format_failures(self):
        root = self.fixture.root
        for stage in ("embedding", "index", "attribution", "formatting"):
            with self.subTest(stage=stage):
                self.fixture = PipelineFixture(root / stage)
                fixture = self.fixture
                state = self.real_so_retrieval()
                constructor = state.module.HybridRetriever
                def initialize(*args, **kwargs):
                    instance = constructor(*args, **kwargs)
                    if stage == "embedding":
                        instance.build_query_embedding = Mock(side_effect=RuntimeError("embedding failed"))
                    elif stage == "index":
                        instance.embedder.index.search = Mock(side_effect=RuntimeError("index failed"))
                    elif stage == "attribution":
                        instance.compute_case_phrase_contributions = Mock(side_effect=RuntimeError("attribution failed"))
                    else:
                        instance.compute_case_phrase_contributions = Mock(return_value=([], [{}]))
                    return instance
                state.module.HybridRetriever = initialize
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](), "completed_with_errors")
                _, _, [record] = self.read_run()
                self.assertEqual(record["keyword_extraction_raw"], fixture.keyword_content)
                self.assertEqual(record["retrieval_keywords"], ["Cough", "denies fever"])
                self.assertEqual(len(record["model_responses"]["keyword_extraction"]), 1)
                if stage in ("embedding", "index"):
                    self.assertIsNone(record["retrieved_patient_ids"])
                else:
                    self.assertEqual(len(record["retrieved_patient_ids"]), 2)
                    self.assertIn("hybrid_score", record["retrieved_patient_ids"][0])
                if stage == "formatting":
                    self.assertEqual(len(record["reranked_patient_ids"]), 2)
                    self.assertIn("rerank_score", record["reranked_patient_ids"][0])
                else:
                    self.assertIsNone(record["reranked_patient_ids"])
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.graph_calls, [])
                self.assertEqual(legacy_patient_result(state.retrievers[0].keyword_extractor.capture_result), record)
                self.assertTrue(callable(state.retrievers[0].on_ranking))
                if state.rankers:
                    self.assertTrue(callable(state.rankers[0].on_ranking))

    def test_real_rankings_keep_nontrivial_rows_disambiguation_scores_ties_and_topfive(self):
        fixture = self.fixture
        fixture.corpus = [dict(note_id=f"N{row}", subjective="cough", objective="normal",
                              assessment=f"ASSESSMENT {row}", plan=f"PLAN {row}", subject_id="SAME")
                          for row in range(8)]
        fixture.corpus[2]["note_id"] = fixture.corpus[5]["note_id"] = "DUP"
        del fixture.corpus[6]["note_id"]
        fixture.write_corpus()
        state = self.real_so_retrieval()
        constructor, rank_constructor = state.module.HybridRetriever, state.module.CrossEncoderReranker
        order = [6, 2, 5, 1, 7, 0, 3, 4]
        def initialize(*args, **kwargs):
            instance = constructor(*args, **kwargs)
            instance.embedder.index.search = Mock(return_value=(
                state.module.np.array([[.9123456789] * 8]), state.module.np.array([order])))
            return instance
        def initialize_ranker(*args, **kwargs):
            instance = rank_constructor(*args, **kwargs)
            instance.model.predict = Mock(return_value=[.1, .987654321, .987654321, .3, .4, .5, .6, .7])
            return instance
        state.module.HybridRetriever, state.module.CrossEncoderReranker = initialize, initialize_ranker
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        _, _, [record] = self.read_run()
        hybrid, reranked = record["retrieved_patient_ids"], record["reranked_patient_ids"]
        expected = [{"corpus_row": 6}, {"note_id": "DUP", "corpus_row": 2},
                    {"note_id": "DUP", "corpus_row": 5}, {"note_id": "N1"}, {"note_id": "N7"},
                    {"note_id": "N0"}, {"note_id": "N3"}, {"note_id": "N4"}]
        def identity(item):
            return {key: value for key, value in item.items() if key in ("note_id", "corpus_row")}
        self.assertEqual([identity(item) for item in hybrid], expected)
        self.assertEqual([identity(item) for item in reranked], [expected[index] for index in (1, 2, 7, 6, 5)])
        self.assertEqual(reranked[0]["rerank_score"], .987654321)
        self.assertEqual(hybrid[0]["embedding_score"], .9123456789)
        self.assertEqual(record["selected_reference_patient"], identity(reranked[0]))
        prompt = fixture.calls[-1]["messages"][-1]["content"]
        self.assertIn("ASSESSMENT 2", prompt)
        self.assertIn("PLAN 2", prompt)
        self.assertNotIn("ASSESSMENT 6", prompt)
        for item in hybrid + reranked:
            self.assertFalse(set(item) - {"note_id", "corpus_row", "embedding_score", "keyword_score", "hybrid_score", "rerank_score"})
        self.assertEqual(state.retrievers[0].search.call_count, 1)
        self.assertEqual(state.rankers[0].rerank.call_count, 1)
        self.assertEqual(state.rankers[0].model.predict.call_count, 1)

    def test_empty_processed_keywords_persist_and_real_full_query_fallback_is_unchanged(self):
        fixture = self.fixture
        fixture.keyword_content = '{"symptoms": [], "diagnoses": [], "medications": []}'
        state = self.real_so_retrieval()
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        _, _, [record] = self.read_run()
        self.assertEqual(record["retrieval_keywords"], [])
        self.assertEqual(record["keyword_extraction_raw"], fixture.keyword_content)
        self.assertEqual(state.encoder.calls[-1], [fixture.conditions])
        self.assertEqual(len(fixture.keyword_calls), 1)
        self.assertEqual(len(fixture.calls), 2)

    def test_graph_stream_failure_preserves_provider_metadata_before_text_result_returns(self):
        fixture = self.fixture
        harness = self.provider_graph()
        harness.stream_error = RuntimeError("partial stream failed")
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed_with_errors")
        _, _, [record] = self.read_run()
        self.assertEqual(len(record["model_responses"]["guideline_evidence"]), 1)
        self.assertEqual(record["model_responses"]["guideline_evidence"][0]["response_id"], "stream-provider-id")
        self.assertEqual(len(record["model_responses"]["guideline_query_embedding"]), 1)
        self.assertEqual(record["model_responses"]["assessment_plan"], [])
        self.assertIsNone(record["guideline_evidence"])
        self.assertIsNone(record["graphrag_context"])
        self.assertIsNotNone(record["retrieved_patient_ids"])
        self.assertIsNotNone(record["reranked_patient_ids"])
        self.assertEqual(len(fixture.calls), 1)

    def test_sequential_patient_capture_does_not_inherit_on_failure_before_extract_or_rerank(self):
        fixture = self.fixture
        patients = [{"id": str(index), "subjective": f"PATIENT {index}", "objective": "normal"}
                    for index in range(4)]
        fixture.write_patients(patients)
        def keyword_reply(kwargs):
            prompt = kwargs["messages"][0]["content"]
            index = int(re.search(r"PATIENT (\d)", prompt).group(1))
            raw = json.dumps({"symptoms": [f"term {index}"], "diagnoses": [], "medications": []})
            response = keyword_response(raw)
            response.model, response.id = f"reported-{index}", f"response-{index}"
            return response
        fixture.keyword_response = keyword_reply
        retriever_class = fixture.modules["scripts.med_copilot_retrieval"].HybridRetriever
        ranker_class = fixture.modules["scripts.med_copilot_retrieval"].CrossEncoderReranker
        original_search, original_rerank = retriever_class.search, ranker_class.rerank
        def search(instance, query, topk):
            if "PATIENT 1" in query:
                raise RuntimeError("before extract")
            return original_search(instance, query, topk)
        def rerank(instance, query, candidates, topk):
            if "PATIENT 2" in query:
                raise RuntimeError("before rerank")
            return original_rerank(instance, query, candidates, topk)
        retriever_class.search, ranker_class.rerank = search, rerank
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed_with_errors")
        _, _, records = self.read_run()
        for index in (0, 2, 3):
            self.assertEqual(records[index]["retrieval_keywords"], [f"term {index}"])
            self.assertEqual(records[index]["model_responses"]["keyword_extraction"][0]["response_id"], f"response-{index}")
            self.assertEqual(json.loads(records[index]["keyword_extraction_raw"])["symptoms"], [f"term {index}"])
        for field in ("keyword_extraction_raw", "retrieval_keywords", "retrieved_patient_ids", "reranked_patient_ids"):
            self.assertIsNone(records[1][field])
        self.assertEqual(records[1]["model_responses"], {stage: [] for stage in HELPER_NAMESPACE["MODEL_RESPONSE_STAGES"]})
        self.assertIsNotNone(records[2]["retrieved_patient_ids"])
        self.assertIsNone(records[2]["reranked_patient_ids"])
        self.assertIsNotNone(records[3]["reranked_patient_ids"])
        self.assertEqual(len(fixture.keyword_calls), 3)

    def test_interrupt_after_graph_response_persists_capture_and_initializes_unstarted_patient(self):
        fixture = self.fixture
        fixture.write_patients([{"id": "first", **fixture.patient}, {"id": "second", **fixture.patient}])
        harness = self.provider_graph()
        harness.stream_error = KeyboardInterrupt()
        runner = self.load_runner()
        with fixture.runtime():
            with self.assertRaises(KeyboardInterrupt):
                runner["main"]()
        _, run, [first, second] = self.read_run()
        self.assertEqual(run["status"], "failed_incomplete")
        self.assertEqual(first["status"], "failed_incomplete")
        self.assertEqual(first["keyword_extraction_raw"], fixture.keyword_content)
        self.assertEqual(first["retrieval_keywords"], ["Cough", "denies fever"])
        self.assertIsNotNone(first["retrieved_patient_ids"])
        self.assertIsNotNone(first["reranked_patient_ids"])
        self.assertEqual(len(first["model_responses"]["guideline_evidence"]), 1)
        self.assertEqual(first["model_responses"]["assessment_plan"], [])
        self.assertEqual(second, HELPER_NAMESPACE["new_patient_result"]("second", "not_processed"))
        self.assertEqual(len(fixture.keyword_calls), 1)
        self.assertEqual(len(fixture.calls), 1)

    def test_direct_completion_metadata_survives_missing_choices_and_partial_usage(self):
        fixture = self.fixture
        fixture.completion_responses["questions"] = SimpleNamespace(
            id="provider-response", model="reported-not-requested", choices=[],
            usage=SimpleNamespace(prompt_tokens=8), _request_id="HTTP-REQUEST")
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed_with_errors")
        _, _, [record] = self.read_run()
        self.assertEqual(record["model_responses"]["guideline_questions"], [{
            "reported_model": "reported-not-requested", "response_id": "provider-response",
            "token_usage": {"prompt_tokens": 8, "completion_tokens": None, "total_tokens": None}}])
        self.assertIsNone(record["guideline_questions"])
        self.assertEqual(fixture.graph_calls, [])
        self.assertEqual(record["model_responses"]["assessment_plan"], [])

    def test_shared_graphrag_models_preserve_multiple_responses_and_do_not_leak_after_failure(self):
        fixture = self.fixture
        fixture.write_patients([{"id": str(index), "subjective": f"PATIENT {index}", "objective": "normal"}
                                for index in range(3)])
        harness = ProviderHarness()
        responses = itertools.count()
        original_stream = harness.new_stream
        async def stream(**kwargs):
            number = next(responses)
            if number == 2:
                raise RuntimeError("third chat request failed")
            chunks = [harness.chunk(f"evidence {number}"), harness.chunk(None, harness.usage)]
            for chunk in chunks:
                chunk.id, chunk.model = f"graph-id-{number}", f"reported-graph-{number}"
            harness.chunks = chunks
            return await original_stream(**kwargs)
        harness.chat_create.side_effect = stream
        def graph_search(**kwargs):
            fixture.graph_calls.append((kwargs, Path.cwd()))
            count = 2 if len(fixture.graph_calls) == 1 else 1
            answer = ""
            for _ in range(count):
                answer, _ = asyncio.run(harness.query())
            return answer, fixture.context
        fixture.modules["graphrag.cli.query"].run_local_search = graph_search
        fixture.failure, fixture.failure_marker = "retrieval", "PATIENT 2"
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed_with_errors")
        _, _, [first, second, third] = self.read_run()
        self.assertEqual([entry["response_id"] for entry in first["model_responses"]["guideline_evidence"]],
                         ["graph-id-0", "graph-id-1"])
        self.assertEqual(len(first["model_responses"]["guideline_query_embedding"]), 2)
        self.assertEqual(second["model_responses"]["guideline_evidence"], [])
        self.assertEqual(len(second["model_responses"]["guideline_query_embedding"]), 1)
        self.assertEqual(second["model_responses"]["assessment_plan"], [])
        self.assertEqual(third["model_responses"], {stage: [] for stage in HELPER_NAMESPACE["MODEL_RESPONSE_STAGES"]})
        self.assertEqual(harness.chat_create.await_count, 3)
        self.assertEqual(harness.embedding_create.await_count, 3)
        self.assertEqual(len(fixture.keyword_calls), 2)
        self.assertEqual(len(fixture.calls), 3)

    def test_final_response_metadata_is_saved_before_missing_content_raises(self):
        fixture = self.fixture
        fixture.completion_responses["final"] = SimpleNamespace(
            model="reported-final", usage=None, choices=[SimpleNamespace(message=SimpleNamespace())],
            _request_id="http-id-only")
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed_with_errors")
        _, _, [record] = self.read_run()
        self.assertEqual(record["model_responses"]["assessment_plan"], [{
            "reported_model": "reported-final", "response_id": None, "token_usage": None}])
        self.assertIsNotNone(record["assessment_plan_prompts"])
        self.assertIsNone(record["assessment_plan_raw"])
        self.assertEqual(len(fixture.calls), 2)

    def test_process_patient_rebinds_captures_after_success_failure_and_interruption(self):
        root = self.fixture.root
        for stage, error in ((None, None), ("keywords", RuntimeError("keyword failure")),
                             ("questions", RuntimeError("question failure")),
                             ("final", KeyboardInterrupt()), ("final", SystemExit(7))):
            with self.subTest(stage=stage):
                self.fixture = PipelineFixture(root / f"{stage or 'success'}-{type(error).__name__}")
                fixture = self.fixture
                state = self.real_so_retrieval()
                fixture.failure = stage
                if error is not None:
                    fixture.error = error
                runner = self.load_runner()
                with fixture.runtime():
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        with self.assertRaises(type(error)) as caught:
                            runner["main"]()
                        self.assertIs(caught.exception, error)
                    else:
                        self.assertEqual(runner["main"](), "completed" if stage is None
                                         else "completed_with_errors")
                _, _, [record] = self.read_run()
                self.assertEqual(record["status"], "completed" if stage is None else "failed_incomplete")
                retriever, reranker = state.retrievers[0], state.rankers[0]
                previous_result = retriever.keyword_extractor.capture_result
                # stop() persists a copy with failed status after BaseException;
                # the capture binding still points to the original in-flight record.
                expected_status = None if isinstance(error, (KeyboardInterrupt, SystemExit)) else record["status"]
                self.assertEqual(legacy_patient_result(previous_result), {**record, "status": expected_status})
                previous_snapshot = json.loads(json.dumps(previous_result))
                previous_callbacks = [component.on_ranking for component in (retriever, reranker)]
                self.assertTrue(all(callable(callback) for callback in previous_callbacks))
                if stage == "keywords":
                    self.assertIsNone(record["keyword_extraction_raw"])
                    self.assertIsNone(record["retrieval_keywords"])
                    self.assertIsNone(record["retrieved_patient_ids"])
                    self.assertIsNone(record["reranked_patient_ids"])
                    self.assertEqual(record["model_responses"], {
                        stage: [] for stage in HELPER_NAMESPACE["MODEL_RESPONSE_STAGES"]})
                else:
                    self.assertEqual(record["keyword_extraction_raw"], fixture.keyword_content)
                    self.assertEqual(record["retrieval_keywords"], ["Cough", "denies fever"])
                    self.assertEqual(len(record["retrieved_patient_ids"]), 2)
                    self.assertEqual(len(record["reranked_patient_ids"]), 2)
                    self.assertEqual(len(record["model_responses"]["keyword_extraction"]), 1)
                if stage in (None, "final"):
                    self.assertIsNotNone(record["assessment_plan_prompts"])
                    self.assertEqual(len(record["model_responses"]["guideline_questions"]), 1)
                if stage is None:
                    self.assertEqual(record["assessment_plan_raw"], fixture.final_output)
                    self.assertEqual(len(record["model_responses"]["assessment_plan"]), 1)
                else:
                    self.assertIsNone(record["assessment_plan_raw"])
                    self.assertEqual(record["model_responses"]["assessment_plan"], [])

                # The next call must replace every binding and clear both rankings
                # even when it fails before search/extract/rerank can reset anything.
                next_result = runner["new_patient_result"]("next-patient", model_ids=["gpt_4o_mini"])
                counts = (len(fixture.keyword_calls), len(fixture.calls), len(fixture.graph_calls))
                with fixture.runtime():
                    with self.assertRaises(ValueError):
                        runner["process_patient"](
                            {"subjective": "S", "objective": None}, next_result, fixture.corpus,
                            retriever, reranker, fixture.guidelines / "output", fixture.assessment_models(runner))
                self.assertIs(retriever.keyword_extractor.capture_result, next_result)
                self.assertEqual(next_result, runner["new_patient_result"]("next-patient", model_ids=["gpt_4o_mini"]))
                for component, previous_callback in zip((retriever, reranker), previous_callbacks):
                    self.assertIsNone(component.ranked_candidates)
                    self.assertIsNot(component.on_ranking, previous_callback)
                    component.on_ranking([])
                self.assertEqual(next_result["retrieved_patient_ids"], [])
                self.assertEqual(next_result["reranked_patient_ids"], [])
                self.assertEqual(previous_result, previous_snapshot)
                self.assertEqual(self.read_run()[2], [record])
                self.assertEqual((len(fixture.keyword_calls), len(fixture.calls), len(fixture.graph_calls)), counts)

                # Reuse the same retriever/reranker after that early failure and
                # even after a prior interrupt. Persist the successful new case.
                fixture.failure = None
                retry = runner["new_patient_result"]("retry-patient", model_ids=["gpt_4o_mini"])
                with fixture.runtime():
                    runner["process_patient"](
                        {"id": "retry-patient", **fixture.patient}, retry, fixture.corpus,
                        retriever, reranker, fixture.guidelines / "output", fixture.assessment_models(runner))
                    retry["status"] = "completed"
                    retry_run = runner["ExperimentRun"].start(
                        fixture.root / "recovery_output", root=fixture.root,
                        input_path=fixture.input_path, started_at=runner["pacific_now"](),
                        branch=fixture.branch, commit=fixture.commit, patient_ids=["retry-patient"])
                    retry_run.save_result(retry)
                    self.assertEqual(retry_run.finish(runner["pacific_now"]()), "completed")
                _, _, [saved_retry] = self.read_run(retry_run.run_dir)
                self.assertEqual(saved_retry, legacy_patient_result(retry))
                self.assertIs(retriever.keyword_extractor.capture_result, retry)
                self.assertEqual(retry["keyword_extraction_raw"], fixture.keyword_content)
                self.assertEqual(retry["retrieval_keywords"], ["Cough", "denies fever"])
                self.assertEqual(len(retry["retrieved_patient_ids"]), 2)
                self.assertEqual(len(retry["reranked_patient_ids"]), 2)
                self.assertEqual(len(retry["model_responses"]["keyword_extraction"]), 1)
                self.assertEqual(len(retry["model_responses"]["guideline_questions"]), 1)
                self.assertEqual(len(retry["assessment_plans"]["gpt_4o_mini"]["model_responses"]), 1)
                self.assertEqual(retry["assessment_plans"]["gpt_4o_mini"]["assessment_plan_raw"], fixture.final_output)
                self.assertEqual(previous_result, previous_snapshot)
                self.assertEqual(next_result["retrieved_patient_ids"], [])
                self.assertEqual(next_result["reranked_patient_ids"], [])
                self.assertEqual(self.read_run()[2], [record])

    def test_process_patient_resets_stale_bindings_before_matching_text_can_fail(self):
        fixture = self.fixture
        runner = self.load_runner()
        extractor = SimpleNamespace(capture_result="stale")
        retriever = SimpleNamespace(keyword_extractor=extractor, ranked_candidates=["stale"],
                                    on_ranking=Mock(), search=Mock())
        reranker = SimpleNamespace(ranked_candidates=["stale"], on_ranking=Mock(), rerank=Mock())
        previous_callbacks = [retriever.on_ranking, reranker.on_ranking]
        record = runner["new_patient_result"](fixture.patient_id)
        prepare = Mock(wraps=runner["prepare_patient_output_capture"])
        with fixture.runtime(), patch.dict(runner["process_patient"].__globals__, {
                "prepare_patient_output_capture": prepare}):
            with self.assertRaises(ValueError):
                runner["process_patient"](
                    {"subjective": "S", "objective": None}, record, fixture.corpus,
                    retriever, reranker, fixture.guidelines / "output", fixture.assessment_models(runner))
        prepare.assert_called_once_with(record, fixture.corpus, retriever, reranker)
        self.assertIs(extractor.capture_result, record)
        for component, previous_callback in zip((retriever, reranker), previous_callbacks):
            self.assertTrue(callable(component.on_ranking))
            self.assertIsNot(component.on_ranking, previous_callback)
            self.assertIsNone(component.ranked_candidates)
        retriever.search.assert_not_called()
        reranker.rerank.assert_not_called()
        self.assertEqual(record, runner["new_patient_result"](fixture.patient_id))
        self.assertEqual(fixture.keyword_calls, [])
        self.assertEqual(fixture.calls, [])
        self.assertEqual(fixture.graph_calls, [])

    def test_persisted_real_retained_ranking_excludes_larger_faiss_pool(self):
        fixture = self.fixture
        fixture.corpus = [dict(note_id=f"N{row}", subjective="cough", objective="normal",
                              assessment=f"A {row}", plan=f"P {row}") for row in range(35)]
        fixture.write_corpus()
        state = self.real_so_retrieval()
        constructor = state.module.HybridRetriever
        def initialize(*args, **kwargs):
            instance = constructor(*args, **kwargs)
            instance.embedder.index.search = Mock(wraps=instance.embedder.index.search)
            return instance
        state.module.HybridRetriever = initialize
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        _, _, [record] = self.read_run()
        self.assertEqual(len(record["retrieved_patient_ids"]), 20)
        self.assertEqual(len(record["reranked_patient_ids"]), 5)
        self.assertEqual([entry["note_id"] for entry in record["retrieved_patient_ids"]],
                         [f"N{row}" for row in range(20)])
        self.assertEqual([entry["note_id"] for entry in record["reranked_patient_ids"]],
                         [f"N{row}" for row in range(5)])
        self.assertEqual(state.retrievers[0].embedder.index.search.call_args.args[1], 35)
        self.assertEqual(record["selected_reference_patient"], {"note_id": "N0"})
        self.assertEqual(len(fixture.keyword_calls), 1)
        self.assertEqual(len(fixture.calls), 2)

    def test_completed_empty_real_hybrid_ranking_is_saved_and_reranking_stays_unattempted(self):
        fixture = self.fixture
        state = self.real_so_retrieval()
        constructor = state.module.HybridRetriever
        def initialize(*args, **kwargs):
            instance = constructor(*args, **kwargs)
            instance.embedder.index.search = Mock(return_value=(
                state.module.np.array([[1., 0.]]), state.module.np.array([[-1, 999]])))
            return instance
        state.module.HybridRetriever = initialize
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        _, _, [record] = self.read_run()
        self.assertEqual(record["retrieved_patient_ids"], [])
        self.assertIsNone(record["reranked_patient_ids"])
        self.assertIsNone(record["selected_reference_patient"])
        self.assertEqual(record["retrieval_keywords"], ["Cough", "denies fever"])
        self.assertIn("No reference patient available.", fixture.calls[-1]["messages"][-1]["content"])
        state.rankers[0].rerank.assert_not_called()

    def test_start_reset_isolates_shared_components_across_every_pipeline_failure_boundary(self):
        root = self.fixture.root
        boundaries = (
            "before_extract", "keyword_api", "keyword_schema", "embedding", "index",
            "attribution", "rerank_model", "formatting", "questions_api", "questions_response",
            "graph_embedding", "graph_stream", "graph_partial_stream", "graph_context",
            "final_api", "final_response",
        )
        for boundary in boundaries:
            with self.subTest(boundary=boundary):
                self.fixture = PipelineFixture(root / boundary)
                fixture = self.fixture
                patients = self.write_marked_patients()
                fixture.corpus = [
                    {"note_id": f"REF-{index}", "subjective": "reference cough", "objective": "normal",
                     "assessment": f"REFERENCE ASSESSMENT {index}", "plan": f"REFERENCE PLAN {index}"}
                    for index in range(3)
                ]
                fixture.write_corpus()
                state = self.real_so_retrieval()
                harness = ProviderHarness()
                active, seen_results, snapshots, callbacks = {}, [], [], []
                raw_replies = {}
                provider_chunks = harness.new_stream
                base_reply = fixture.reply
                create_retriever, create_reranker = state.module.HybridRetriever, state.module.CrossEncoderReranker

                def trip(stage):
                    if active["marker"] == "MARKER_BRAVO" and boundary == stage:
                        raise RuntimeError(f"injected {stage} failure")

                def keyword_reply(kwargs):
                    trip("keyword_api")
                    marker = active["marker"]
                    self.assertIn(marker, kwargs["messages"][0]["content"])
                    content = ' \n' + json.dumps({
                        "symptoms": [f" {marker} ", "cough", marker],
                        "diagnoses": ["cough"], "medications": ["药"],
                    }, ensure_ascii=False, indent=2) + '\t '
                    if marker == "MARKER_BRAVO" and boundary == "keyword_schema":
                        content = '{"symptoms": ["BRAVO malformed schema"]}'
                    raw_replies[marker] = content
                    response = keyword_response(content)
                    response.model, response.id = f"reported-keyword-{marker}", f"keyword-{marker}"
                    response.usage = SimpleNamespace(prompt_tokens=9, completion_tokens=4, total_tokens=13)
                    return response
                fixture.keyword_response = keyword_reply

                def retriever(*args, **kwargs):
                    instance = create_retriever(*args, **kwargs)
                    original_search, original_embedding = instance.search, instance.build_query_embedding
                    def search(query, topk):
                        self.assertIn(active["marker"], query)
                        self.assertIs(instance.keyword_extractor.capture_result, active["result"])
                        ranker = state.rankers[0]
                        for component in (instance, ranker):
                            self.assertIsNone(component.ranked_candidates)
                        current_callbacks = (instance.on_ranking, ranker.on_ranking)
                        if callbacks:
                            for current, previous in zip(current_callbacks, callbacks[-1]):
                                self.assertIsNot(current, previous)
                        callbacks.append(current_callbacks)
                        trip("before_extract")
                        return original_search(query, topk=topk)
                    def embed(keywords):
                        trip("embedding")
                        return original_embedding(keywords)
                    def index_search(query_vector, count):
                        trip("index")
                        first = active["index"]
                        order = [(first + offset) % 3 for offset in range(3)]
                        return (state.module.np.array([[.9123456789, .8123456789, .7123456789]]),
                                state.module.np.array([order]))
                    def attribute(**kwargs):
                        trip("attribution")
                        malformed = active["marker"] == "MARKER_BRAVO" and boundary == "formatting"
                        return [], ([{}] if malformed else [])
                    instance.search = Mock(side_effect=search)
                    instance.build_query_embedding = Mock(side_effect=embed)
                    instance.embedder.index.search = Mock(side_effect=index_search)
                    instance.compute_case_phrase_contributions = Mock(side_effect=attribute)
                    return instance
                def reranker(*args, **kwargs):
                    instance = create_reranker(*args, **kwargs)
                    def predict(pairs):
                        trip("rerank_model")
                        return [.12, .987654321, .12]
                    instance.model.predict = Mock(side_effect=predict)
                    return instance
                state.module.HybridRetriever, state.module.CrossEncoderReranker = retriever, reranker

                def reply(stage, text):
                    trip("questions_api" if stage == "questions" else "final_api")
                    marker = active["marker"]
                    content = base_reply(stage, text)
                    response = SimpleNamespace(
                        model=f"reported-{stage}-{marker}", id=f"{stage}-{marker}",
                        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=3, total_tokens=23),
                        choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
                    if marker == "MARKER_BRAVO" and boundary == f"{stage}_response":
                        response.choices = []
                    fixture.completion_responses[stage] = response
                    return content
                fixture.reply = reply

                async def embedding_response(**kwargs):
                    trip("graph_embedding")
                    harness.embedding_response.model = f'reported-embedding-{active["marker"]}'
                    return harness.embedding_raw
                async def stream_response(**kwargs):
                    trip("graph_stream")
                    marker = active["marker"]
                    harness.chunks = [harness.chunk(f"{marker} evidence"),
                                      harness.chunk(None, harness.usage), harness.chunk(None, harness.usage)]
                    for chunk in harness.chunks:
                        chunk.id, chunk.model = f"graph-{marker}", f"reported-graph-{marker}"
                    harness.stream_error = (RuntimeError("partial stream failed")
                                            if marker == "MARKER_BRAVO" and boundary == "graph_partial_stream"
                                            else None)
                    return await provider_chunks(**kwargs)
                harness.embedding_create.side_effect = embedding_response
                harness.chat_create.side_effect = stream_response
                def graph_search(**kwargs):
                    fixture.graph_calls.append((kwargs, Path.cwd()))
                    self.assertIn(active["marker"], kwargs["query"])
                    answer, _ = asyncio.run(harness.query())
                    context = (object() if active["marker"] == "MARKER_BRAVO" and boundary == "graph_context"
                               else fixture.context)
                    return answer, context
                fixture.modules["graphrag.cli.query"].run_local_search = graph_search

                runner = self.load_runner()
                process = runner["process_patient"]
                def observe_patient(patient, result, *args):
                    active.update(marker=re.search(r"MARKER_[A-Z]+", patient["subjective"]).group(0),
                                  index=len(seen_results), result=result)
                    seen_results.append(result)
                    try:
                        return process(patient, result, *args)
                    finally:
                        snapshots.append(json.loads(json.dumps(result)))
                with fixture.runtime(), patch.dict(runner["main"].__globals__, {"process_patient": observe_patient}):
                    self.assertEqual(runner["main"](), "completed_with_errors")
                _, _, records = self.read_run()
                self.assertEqual([record["id"] for record in records], [patient["id"] for patient in patients])
                self.assertEqual([record["status"] for record in records],
                                 ["completed", "failed_incomplete", "completed"])
                self.assertEqual(len(state.retrievers), 1)
                self.assertEqual(len(state.rankers), 1)
                self.assertEqual(len(callbacks), 3)
                # Later responses, ranking mutations, and rebindings cannot alter
                # either completed/failed in-memory captures or persisted JSONL.
                for index, (record, memory, snapshot) in enumerate(zip(records, seen_results, snapshots)):
                    self.assertEqual(legacy_patient_result(memory), record)
                    self.assertEqual({**legacy_patient_result(snapshot), "status": record["status"]}, record)
                    marker = ("MARKER_ALPHA", "MARKER_BRAVO", "MARKER_CHARLIE")[index]
                    level = boundaries.index(boundary) if index == 1 else len(boundaries)
                    if level >= 2:
                        self.assertEqual(record["keyword_extraction_raw"], raw_replies[marker])
                    else:
                        self.assertIsNone(record["keyword_extraction_raw"])
                    self.assertEqual(record["retrieval_keywords"],
                                     [marker, "cough", "药"] if level >= 3 else None)
                    if level >= 5:
                        self.assertEqual([item["note_id"] for item in record["retrieved_patient_ids"]],
                                         [f"REF-{(index + offset) % 3}" for offset in range(3)])
                        self.assertNotIn("rerank_score", record["retrieved_patient_ids"][0])
                    else:
                        self.assertIsNone(record["retrieved_patient_ids"])
                    if level >= 7:
                        self.assertEqual([item["note_id"] for item in record["reranked_patient_ids"]],
                                         [f"REF-{(index + offset) % 3}" for offset in (1, 0, 2)])
                        self.assertEqual(record["reranked_patient_ids"][0]["rerank_score"], .987654321)
                    else:
                        self.assertIsNone(record["reranked_patient_ids"])
                    expected_stages = {"keyword_extraction": level >= 2, "guideline_questions": level >= 9,
                                       "guideline_query_embedding": level >= 11,
                                       "guideline_evidence": level >= 12, "assessment_plan": level >= 15}
                    for stage, available in expected_stages.items():
                        entries = record["model_responses"][stage]
                        self.assertEqual(len(entries), int(available), (boundary, index, stage, entries))
                        if available:
                            self.assertIn(marker, entries[0]["reported_model"])
                            if stage == "guideline_query_embedding":
                                self.assertIsNone(entries[0]["response_id"])
                                self.assertEqual(entries[0]["token_usage"], {
                                    "prompt_tokens": 11, "completion_tokens": None, "total_tokens": 11})
                            else:
                                self.assertIn(marker, entries[0]["response_id"])
                            if index != 1:
                                self.assertNotIn("MARKER_BRAVO", json.dumps(entries))
                    if index != 1:
                        self.assertEqual(record["selected_reference_patient"], {"note_id": f"REF-{(index + 1) % 3}"})
                        self.assertIn(f"REFERENCE PLAN {(index + 1) % 3}", record["assessment_plan_prompts"]["user"])
                self.assertIs(state.retrievers[0].keyword_extractor.capture_result, seen_results[-1])
                self.assertEqual(len(fixture.keyword_calls), 3 - int(boundary == "before_extract"))
                self.assertEqual(state.rankers[0].model.predict.call_count, 2 + int(boundaries.index(boundary) >= 6))


    def test_reranker_callbacks_preserve_ranking_when_test_double_formatting_fails(self):
        root = self.fixture.root
        for implementation in ("stub", "frozen-upstream"):
            with self.subTest(implementation=implementation):
                self.fixture = PipelineFixture(root / implementation)
                fixture = self.fixture
                if implementation == "stub":
                    fixture.failure = "rerank_formatting"
                else:
                    fixture.candidates = [
                        {"idx": row, **record, "dominant_terms": [{}]}
                        for row, record in enumerate(fixture.corpus)
                    ]
                    ranker = frozen_reranker_class(fixture.rerank_scores)()
                    ranker.model.predict = Mock(wraps=ranker.model.predict)
                    fixture.modules["scripts.med_copilot_retrieval"].CrossEncoderReranker = lambda **kwargs: ranker
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](), "completed_with_errors")
                _, _, [record] = self.read_run()
                self.assertEqual(record["status"], "failed_incomplete")
                self.assertEqual(record["keyword_extraction_raw"], fixture.keyword_content)
                self.assertEqual(record["retrieval_keywords"], ["Cough", "denies fever"])
                self.assertEqual(record["retrieved_patient_ids"], [
                    {"note_id": "NOTE-A"}, {"note_id": "NOTE-B"}])
                self.assertEqual(record["reranked_patient_ids"], [
                    {"note_id": "NOTE-B", "rerank_score": .9},
                    {"note_id": "NOTE-A", "rerank_score": .2}])
                self.assertIsNone(record["selected_reference_patient"])
                self.assertIsNone(record["assessment_plan_prompts"])
                self.assertIsNone(record["assessment_plan_raw"])
                self.assertEqual(len(record["model_responses"]["keyword_extraction"]), 1)
                for stage in ("guideline_questions", "guideline_query_embedding", "guideline_evidence", "assessment_plan"):
                    self.assertEqual(record["model_responses"][stage], [])
                self.assertEqual(len(fixture.keyword_calls), 1)
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.graph_calls, [])
                if implementation == "stub":
                    self.assertEqual(len([call for call in fixture.rerank_calls if call[0] == "rerank"]), 1)
                else:
                    ranker.model.predict.assert_called_once()


    def test_retrieval_callback_preserves_retained_ranking_before_stub_attribution_failure(self):
        root = self.fixture.root
        for error in (RuntimeError("attribution failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.fixture = PipelineFixture(root / type(error).__name__)
                fixture = self.fixture
                fixture.failure, fixture.error = "retrieval_attribution", error
                fixture.corpus = [
                    {"note_id": f"NOTE-{row}", "subjective": "cough", "objective": "normal",
                     "assessment": f"ASSESSMENT {row}", "plan": f"PLAN {row}"}
                    for row in range(25)
                ]
                fixture.write_corpus()
                order = [24, 3, 18] + [row for row in range(25) if row not in (24, 3, 18)]
                # Equal scores retain this supplied hybrid order. Only the
                # requested 20 candidates are published, before attribution.
                fixture.candidates = [
                    {"idx": row, "embedding_score": .9123456789, "keyword_score": .5,
                     "hybrid_score": .70617283945, "display_text": "PRIVATE FULL SOAP"}
                    for row in order
                ]
                runner = self.load_runner()
                with fixture.runtime():
                    if isinstance(error, KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt) as caught:
                            runner["main"]()
                        self.assertIs(caught.exception, error)
                    else:
                        self.assertEqual(runner["main"](), "completed_with_errors")
                _, run, [record] = self.read_run()
                self.assertEqual(run["status"], "failed_incomplete" if isinstance(error, KeyboardInterrupt)
                                 else "completed_with_errors")
                self.assertEqual(record["status"], "failed_incomplete")
                self.assertEqual(record["keyword_extraction_raw"], fixture.keyword_content)
                self.assertEqual(record["retrieval_keywords"], ["Cough", "denies fever"])
                self.assertEqual(record["retrieved_patient_ids"], [
                    {"note_id": f"NOTE-{row}", "embedding_score": .9123456789,
                     "keyword_score": .5, "hybrid_score": .70617283945}
                    for row in order[:20]
                ])
                self.assertNotIn("PRIVATE FULL SOAP", json.dumps(record))
                self.assertIsNone(record["reranked_patient_ids"])
                self.assertIsNone(record["selected_reference_patient"])
                self.assertIsNone(record["assessment_plan_prompts"])
                self.assertIsNone(record["assessment_plan_raw"])
                self.assertEqual(len(record["model_responses"]["keyword_extraction"]), 1)
                for stage in ("guideline_questions", "guideline_query_embedding", "guideline_evidence", "assessment_plan"):
                    self.assertEqual(record["model_responses"][stage], [])
                self.assertEqual(len(fixture.keyword_calls), 1)
                self.assertEqual([call for call in fixture.rerank_calls if call[0] == "rerank"], [])
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.graph_calls, [])


class ScoringIntegrationTests(unittest.TestCase):
    """Real scoring wrapper, parser and labels; only metric/model runtimes are mocked."""

    load_runner = RunMedCopilotTests.load_runner
    run_dirs = RunMedCopilotTests.run_dirs
    read_run = RunMedCopilotTests.read_run

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = PipelineFixture(Path(self.temporary.name))
        self.fixture.real_scoring = True
        self.fixture.write_labels([self.label(self.fixture.patient_id)])

    def label(self, patient_id, *, evidence=True, condition="GOLD_ONLY_condition",
              plan=None):
        return {"id": patient_id, "condition": condition, "has_presenting_evidence": evidence,
                "plan": ["GOLD_ONLY drug", "GOLD_ONLY follow-up"] if plan is None else plan}

    def test_default_sample_paths_run_matched_files_with_seven_or_all_ten_patients(self):
        source_input = (ROOT / "experiments/input" / INPUT_NAME).read_bytes()
        source_labels = (ROOT / "experiments/input" / LABELS_NAME).read_bytes()
        patients = [json.loads(line) for line in source_input.decode("utf-8").split("\n")
                    if line.strip()]
        labels = {row["id"]: row for row in
                  (json.loads(line) for line in source_labels.decode("utf-8").split("\n")
                   if line.strip())}
        fixture_root = self.fixture.root
        for arguments, evidence_only, expected_count in (
            ([], True, 7), (["--presenting-evidence-only", "false"], False, 10),
        ):
            with self.subTest(evidence_only=evidence_only):
                self.fixture = PipelineFixture(fixture_root / str(evidence_only))
                fixture = self.fixture
                fixture.real_scoring = True
                fixture.input_path.write_bytes(source_input)
                fixture.labels_path.write_bytes(source_labels)
                runner = self.load_runner()
                self.assertNotIn("PATIENT_INPUT", runner)
                self.assertNotIn("PATIENT_LABELS", runner)
                with fixture.runtime():
                    self.assertEqual(runner["main"](arguments), "completed")
                run_dir, run, records = self.read_run()
                expected_ids = [patient["id"] for patient in patients
                                if not evidence_only or labels[patient["id"]]["has_presenting_evidence"]]
                self.assertEqual(len(expected_ids), expected_count)
                self.assertEqual([record["id"] for record in records], expected_ids)
                self.assertEqual(run["input"]["file"], f"experiments/input/{INPUT_NAME}")
                self.assertEqual(run["input"]["sha256"], hashlib.sha256(source_input).hexdigest())
                self.assertEqual(run["evaluation"]["labels_path"], f"experiments/input/{LABELS_NAME}")
                self.assertEqual(run["evaluation"]["labels_sha256"], hashlib.sha256(source_labels).hexdigest())
                self.assertEqual(run["evaluation"]["selected_patient_count"], expected_count)
                self.assertEqual(run["evaluation"]["presenting_evidence_only"], evidence_only)
                self.assertIn("__test_sample_so__", run_dir.name)
                expected_calls = []
                for record in records:
                    label = labels[record["id"]]
                    self.assertEqual(record["status"], "completed")
                    self.assertEqual(record["assessment_plan_raw"], fixture.final_output)
                    self.assert_field(record, "condition", "scored")
                    for pair in fixture.final_sections["predictions"]:
                        expected_calls.append(([label["condition"].replace("_", " ")],
                                               [pair["condition"]], "en"))
                        if label["plan"] == ["none recorded"]:
                            self.assert_field(record, "plan", "skipped_no_recorded_plan")
                        else:
                            self.assert_field(record, "plan", "scored")
                            expected_calls.append(([", ".join(label["plan"])],
                                                   [", ".join(pair["plan"])], "en"))
                self.assertEqual(fixture.scoring_calls, expected_calls)
                self.assertEqual(len(fixture.calls), 2 * expected_count)
                self.assertEqual(fixture.input_path.read_bytes(), source_input)
                self.assertEqual(fixture.labels_path.read_bytes(), source_labels)

    def assert_field(self, record, field, status, error_type=None, rank=None):
        self.assertIsInstance(record["evaluation"], list)
        self.assertEqual([entry["rank"] for entry in record["evaluation"]], [1, 2, 3])
        entries = record["evaluation"] if rank is None else [record["evaluation"][rank - 1]]
        for entry in entries:
            self.assertEqual(set(entry), {"rank", "condition", "plan"})
            score = entry[field]
            self.assertEqual(set(score), {"status", "metrics", "error"})
            self.assertEqual(score["status"], status)
            self.assertEqual(set(score["metrics"]), set(METRIC_NAMES))
            self.assertEqual(score["metrics"], self.fixture.metric_values if status == "scored"
                             else dict.fromkeys(METRIC_NAMES))
            if error_type is None:
                self.assertIsNone(score["error"])
            else:
                self.assertEqual(score["error"]["type"], error_type)
        return entries[-1][field]

    def assert_no_generation(self):
        fixture = self.fixture
        self.assertEqual(fixture.calls, [])
        self.assertEqual(fixture.keyword_calls, [])
        self.assertEqual(fixture.retrieval_calls, [])
        self.assertEqual(fixture.graph_calls, [])
        self.assertEqual(fixture.scoring_calls, [])
        self.assertFalse(fixture.output.exists())

    def test_real_wrapper_scores_each_saved_field_against_sidecar_and_never_leaks_labels(self):
        fixture = self.fixture
        input_bytes = fixture.input_path.read_bytes()
        label_bytes = fixture.labels_path.read_bytes()
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        run_dir, run, records = self.read_run()
        record = records[0]
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["assessment_plan_raw"], fixture.final_output)
        self.assertEqual(record["predictions"], fixture.final_sections["predictions"])
        self.assertEqual([entry["rank"] for entry in record["evaluation"]], [1, 2, 3])
        self.assert_field(record, "condition", "scored")
        self.assert_field(record, "plan", "scored")
        self.assertEqual(fixture.scoring_calls, [call
            for pair in fixture.final_sections["predictions"] for call in (
                (["GOLD ONLY condition"], [pair["condition"]], "en"),
                (["GOLD_ONLY drug, GOLD_ONLY follow-up"], [", ".join(pair["plan"])], "en"),
            )])
        # Retrieved case A/P is evidence, never the target patient's answer key.
        self.assertEqual(record["selected_reference_patient"], fixture.reference)
        self.assertIn("REFERENCE_DIAGNOSIS", record["assessment_plan_prompts"]["user"])
        for call in fixture.calls + fixture.keyword_calls:
            self.assertNotIn("GOLD_ONLY", json.dumps(call))
            self.assertNotIn("GOLD ONLY", json.dumps(call))
        for event in fixture.retrieval_calls + fixture.rerank_calls:
            self.assertNotIn("GOLD_ONLY", str(event))
        evaluation = run["evaluation"]
        self.assertEqual(evaluation["labels_path"], f"experiments/input/{LABELS_NAME}")
        self.assertEqual(evaluation["labels_sha256"], hashlib.sha256(label_bytes).hexdigest())
        self.assertEqual(evaluation["evaluator_path"], "med_copilot/upstream/evaluate_metrics.py")
        self.assertEqual(evaluation["evaluator_sha256"],
                         hashlib.sha256(UPSTREAM_EVALUATOR.read_bytes()).hexdigest())
        self.assertTrue(evaluation["presenting_evidence_only"])
        self.assertEqual(evaluation["input_patient_count"], 1)
        self.assertEqual(evaluation["selected_patient_count"], 1)
        self.assertEqual(evaluation["selected_patient_ids"], [fixture.patient_id])
        self.assertEqual(evaluation["excluded_patient_ids"], [])
        self.assertEqual(evaluation["label_count"], 1)
        self.assertEqual(set(evaluation["metric_packages"]),
                         {"sacrebleu", "nltk", "rouge-score", "bert-score", "transformers", "torch"})
        self.assertEqual(evaluation["settings"]["aggregation"], "per_patient_per_rank_per_field_singleton")
        self.assertEqual(fixture.input_path.read_bytes(), input_bytes)
        self.assertEqual(fixture.labels_path.read_bytes(), label_bytes)
        self.assertEqual({path.name for path in run_dir.iterdir()}, {"run.json", "results.jsonl"} | ({"results_summary.html"} if (run_dir / "results_summary.html").exists() else set()))

    def test_default_filter_joins_shuffled_superset_labels_by_id_and_keeps_input_order(self):
        fixture = self.fixture
        patients = [{"id": name, "subjective": f"S_{name}", "objective": f"O_{name}"}
                    for name in ("FIRST", "EXCLUDED", "LAST")]
        fixture.write_patients(patients)
        fixture.write_labels([self.label("LAST", condition="last_condition"),
                              self.label("EXTRA"), self.label("EXCLUDED", evidence=False),
                              self.label("FIRST", condition="first_condition")])
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        _, run, records = self.read_run()
        self.assertEqual([record["id"] for record in records], ["FIRST", "LAST"])
        self.assertEqual([call[0] for call in fixture.scoring_calls[::6]],
                         [["first condition"], ["last condition"]])
        self.assertEqual(len(fixture.calls), 4)
        self.assertNotIn("S_EXCLUDED", json.dumps(fixture.calls))
        self.assertEqual(fixture.retrieval_calls[0][1].to_dict(orient="records"), fixture.corpus)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["evaluation"]["input_patient_count"], 3)
        self.assertEqual(run["evaluation"]["selected_patient_ids"], ["FIRST", "LAST"])
        self.assertEqual(run["evaluation"]["excluded_patient_ids"], ["EXCLUDED"])
        self.assertEqual(run["evaluation"]["label_count"], 4)

    def test_three_ranked_pairs_roundtrip_and_each_plan_scores_exact_joined_items(self):
        fixture = self.fixture
        expected = [
            {"condition": "  FIRST_DIAGNOSIS café  ",
             "plan": ["Rx, 0.5 mg", "Follow-up\nsoon", "Rx, 0.5 mg"]},
            {"condition": "SECOND_DIAGNOSIS", "plan": ["Other investigation"]},
            {"condition": "THIRD_DIAGNOSIS", "plan": ["  café  ", "Last item"]},
        ]
        fixture.final_output = " \r\n" + json.dumps({"predictions": expected}, ensure_ascii=False,
                                                      indent=2) + "\t "
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        record = self.read_run()[2][0]
        self.assertEqual(fixture.calls[-1]["response_format"], FINAL_RESPONSE_FORMAT)
        self.assertEqual(record["predictions"], expected)
        self.assertEqual(record["assessment_plan_raw"], fixture.final_output)
        for legacy in ("assessment", "condition", "plan"):
            self.assertNotIn(legacy, record)
        self.assertEqual(fixture.scoring_calls, [call for pair in expected for call in (
            (["GOLD ONLY condition"], [pair["condition"]], "en"),
            (["GOLD_ONLY drug, GOLD_ONLY follow-up"], [", ".join(pair["plan"])], "en"),
        )])
        self.assert_field(record, "condition", "scored")
        self.assert_field(record, "plan", "scored")

    def test_ranked_pairs_keep_each_unweighted_metric_result_and_separate_plan(self):
        fixture = self.fixture
        pairs = [{"condition": f"DIAGNOSIS_{rank}", "plan": [f"PLAN_{rank}"]}
                 for rank in (1, 2, 3)]
        fixture.final_output = json.dumps({"predictions": pairs})
        values = {text: {name: position / 10 for name in METRIC_NAMES}
                  for position, text in enumerate(
                      ["DIAGNOSIS_1", "PLAN_1", "DIAGNOSIS_2", "PLAN_2", "DIAGNOSIS_3", "PLAN_3"], 1)}
        def evaluate(references, candidates, lang):
            self.assertEqual(lang, "en")
            self.assertEqual(references, ["GOLD ONLY condition"] if candidates[0].startswith("DIAGNOSIS")
                             else ["GOLD_ONLY drug, GOLD_ONLY follow-up"])
            return values[candidates[0]].copy()
        fixture.scoring_reply = evaluate
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        _, run, records = self.read_run()
        [record] = records
        self.assertEqual(record["predictions"], pairs)
        self.assertEqual(len(fixture.scoring_calls), 6)
        for rank, (pair, entry) in enumerate(zip(pairs, record["evaluation"]), 1):
            self.assertEqual(set(entry), {"rank", "condition", "plan"})
            self.assertEqual(entry["rank"], rank)
            self.assertEqual(entry["condition"]["metrics"], values[pair["condition"]])
            self.assertEqual(entry["plan"]["metrics"], values[pair["plan"][0]])
            for field in ("condition", "plan"):
                self.assertEqual(entry[field]["status"], "scored")
                self.assertIsNone(entry[field]["error"])
        self.assertIsNone(run["evaluation"]["settings"]["rank_weighting"])
        self.assertEqual(run["evaluation"]["settings"]["prediction_count"], 3)
        self.assertEqual(run["evaluation"]["settings"]["plan_item_limit"], 3)

    def test_one_second_rank_metric_failure_keeps_other_fields_and_ranks(self):
        fixture = self.fixture
        def evaluate(references, candidates, lang):
            if candidates == ["MOCK_DIAGNOSIS alternative 2"]:
                raise RuntimeError("only the second condition failed")
            return fixture.metric_values.copy()
        fixture.scoring_reply = evaluate
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        [record] = self.read_run()[2]
        self.assert_field(record, "plan", "scored")
        self.assert_field(record, "condition", "scored", rank=1)
        self.assert_field(record, "condition", "error", "RuntimeError", rank=2)
        self.assert_field(record, "condition", "scored", rank=3)
        self.assertEqual(record["predictions"], fixture.final_sections["predictions"])
        self.assertEqual(len(fixture.scoring_calls), 6)

    def test_false_flag_includes_no_evidence_records_and_explicit_true_filters_them(self):
        root = self.fixture.root
        for value, expected in (("false", ["YES", "NO"]), ("true", ["YES"])):
            with self.subTest(value=value):
                self.fixture = PipelineFixture(root / value)
                fixture = self.fixture
                fixture.real_scoring = True
                fixture.write_patients([{"id": name, "subjective": name, "objective": "O"}
                                        for name in ("YES", "NO")])
                fixture.write_labels([self.label("YES"), self.label("NO", evidence=False)])
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](["--presenting-evidence-only", value]), "completed")
                _, run, records = self.read_run()
                self.assertEqual([record["id"] for record in records], expected)
                self.assertEqual(run["evaluation"]["presenting_evidence_only"], value == "true")

    def test_entrypoint_honors_false_flag_and_rejects_invalid_flag_without_generation(self):
        fixture = self.fixture
        fixture.write_labels([self.label(fixture.patient_id, evidence=False)])
        with fixture.runtime(), patch.object(sys, "argv", [str(fixture.script),
                                                          "--presenting-evidence-only", "false"]):
            runpy.run_path(str(fixture.script), run_name="__main__")
        self.assertEqual(self.read_run()[2][0]["id"], fixture.patient_id)
        self.fixture = PipelineFixture(fixture.root / "invalid")
        self.fixture.real_scoring = True
        self.fixture.write_labels([self.label(self.fixture.patient_id)])
        runner = self.load_runner()
        with self.fixture.runtime(), self.assertRaises(SystemExit) as caught:
            runner["main"](["--presenting-evidence-only", "sometimes"])
        self.assertEqual(caught.exception.code, 2)
        self.assert_no_generation()

    def test_label_preflight_rejects_missing_duplicate_malformed_or_wrong_types_before_generation(self):
        root = self.fixture.root
        for case in ("missing_file", "missing_id", "duplicate", "malformed", "not_object",
                     "condition_type", "empty_condition", "plan_type", "plan_item_type", "evidence_type"):
            with self.subTest(case=case):
                self.fixture = PipelineFixture(root / case)
                fixture = self.fixture
                fixture.real_scoring = True
                label = self.label(fixture.patient_id)
                if case == "missing_id":
                    label["id"] = "OTHER"
                elif case == "condition_type":
                    label["condition"] = ["diagnosis"]
                elif case == "empty_condition":
                    label["condition"] = "___  "
                elif case == "plan_type":
                    label["plan"] = "drug"
                elif case == "plan_item_type":
                    label["plan"] = [1]
                elif case == "evidence_type":
                    label["has_presenting_evidence"] = "true"
                if case != "missing_file":
                    fixture.write_labels([label, label] if case == "duplicate" else [label])
                if case in ("malformed", "not_object"):
                    fixture.labels_path.write_text("not-json\n" if case == "malformed" else "[]\n")
                runner = self.load_runner()
                with fixture.runtime(), self.assertRaises((ValueError, FileNotFoundError)):
                    runner["main"]()
                self.assert_no_generation()

    def test_all_input_ids_and_filtered_out_label_schema_are_checked_before_filtering(self):
        root = self.fixture.root
        for case in ("missing_excluded", "invalid_excluded", "zero_eligible"):
            with self.subTest(case=case):
                self.fixture = PipelineFixture(root / case)
                fixture = self.fixture
                fixture.real_scoring = True
                fixture.write_patients([{"id": "IN", "subjective": "S", "objective": "O"},
                                        {"id": "OUT", "subjective": "S", "objective": "O"}])
                labels = [self.label("IN", evidence=case != "zero_eligible")]
                if case != "missing_excluded":
                    excluded = self.label("OUT", evidence=False)
                    if case == "invalid_excluded":
                        excluded["plan"] = [None]
                    labels.append(excluded)
                fixture.write_labels(labels)
                runner = self.load_runner()
                with fixture.runtime(), self.assertRaises(ValueError):
                    runner["main"]()
                self.assert_no_generation()

    def test_exact_no_recorded_sentinel_skips_all_three_plans_and_preserves_raw_and_status(self):
        fixture = self.fixture
        pairs = prediction_pairs("Condition café", ["Generated real treatment"])
        fixture.final_output = '  ' + json.dumps({"predictions": pairs}) + '\n  '
        fixture.write_labels([self.label(fixture.patient_id, plan=["none recorded"])])
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        record = self.read_run()[2][0]
        self.assertEqual(record["assessment_plan_raw"], fixture.final_output)
        self.assertEqual(record["predictions"], pairs)
        self.assert_field(record, "condition", "scored")
        self.assert_field(record, "plan", "skipped_no_recorded_plan")
        self.assertEqual(len(fixture.scoring_calls), 3)

    def test_nonexact_sentinel_and_record_artifacts_are_scored_as_recorded_without_cleaning(self):
        fixture = self.fixture
        fixture.write_labels([self.label(fixture.patient_id,
                                        plan=["none recorded", "management (record artifact)"])])
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        record = self.read_run()[2][0]
        self.assert_field(record, "plan", "scored")
        self.assertEqual(fixture.scoring_calls[1][0],
                         ["none recorded, management (record artifact)"])

    def test_invalid_or_partial_ranked_json_scores_other_fields_and_ranks_without_failure(self):
        root = self.fixture.root
        valid = prediction_pairs("A", ["P"])
        # First-rank field errors must not erase either valid lower-ranked answer.
        for case, pair, condition_status, plan_status in [
            ("only_condition", {"condition": "A"}, "scored", "missing_generation"),
            ("only_plan", {"plan": ["P"]}, "missing_generation", "scored"),
            ("wrong_condition", {"condition": ["A"], "plan": ["P"]}, "missing_generation", "scored"),
            ("wrong_plan", {"condition": "A", "plan": "P"}, "scored", "missing_generation"),
            ("empty_plan", {"condition": "A", "plan": []}, "scored", "missing_generation"),
            ("overlong_plan", {"condition": "A", "plan": ["P"] * 4}, "scored", "missing_generation"),
            ("extra_pair_key", {"condition": "A", "plan": ["P"], "reason": "why"},
             "missing_generation", "missing_generation"),
            ("nonobject_pair", "A", "missing_generation", "missing_generation"),
        ]:
            with self.subTest(case=case):
                self.fixture = PipelineFixture(root / case)
                fixture = self.fixture
                fixture.real_scoring = True
                fixture.final_output = json.dumps({"predictions": [pair, *valid[1:]]})
                fixture.write_labels([self.label(fixture.patient_id)])
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](), "completed")
                record = self.read_run()[2][0]
                self.assertEqual(record["status"], "completed")
                self.assertEqual(record["assessment_plan_raw"], fixture.final_output)
                self.assert_field(record, "condition", condition_status, rank=1)
                self.assert_field(record, "plan", plan_status, rank=1)
                for rank in (2, 3):
                    self.assert_field(record, "condition", "scored", rank=rank)
                    self.assert_field(record, "plan", "scored", rank=rank)
                self.assertEqual(len(fixture.scoring_calls), 4 + (condition_status == "scored")
                                 + (plan_status == "scored"))

    def test_globally_invalid_json_or_wrong_prediction_count_scores_no_candidates(self):
        root = self.fixture.root
        valid = prediction_pairs("A", ["P"])
        tail = ','.join(json.dumps(pair) for pair in valid[1:])
        for case, response in [
            ("too_few", json.dumps({"predictions": valid[:2]})),
            ("too_many", json.dumps({"predictions": valid + valid[:1]})),
            ("wrong_predictions_type", '{"predictions":{}}'),
            ("duplicate_condition", '{"predictions":[{"condition":"A","condition":"B",'
             '"plan":["P"]},' + tail + ']}'),
            ("extra_top_key", json.dumps({"predictions": valid, "reason": "why"})),
            ("malformed_json", '{"predictions":['),
            ("fenced_json", '```json\n' + json.dumps({"predictions": valid}) + '\n```'),
            ("nonfinite", '{"predictions":[{"condition":"A","plan":[NaN]},' + tail + ']}'),
            ("null_reply", None), ("nonstring_reply", 123),
        ]:
            with self.subTest(case=case):
                self.fixture = PipelineFixture(root / case)
                fixture = self.fixture
                fixture.real_scoring = True
                fixture.final_output = response
                fixture.write_labels([self.label(fixture.patient_id)])
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](), "completed")
                record = self.read_run()[2][0]
                self.assertEqual(record["assessment_plan_raw"], response)
                self.assertIsNone(record["predictions"])
                self.assert_field(record, "condition", "missing_generation")
                self.assert_field(record, "plan", "missing_generation")
                self.assertEqual(fixture.scoring_calls, [])

    def test_empty_reference_plan_is_missing_reference_and_is_not_scored(self):
        fixture = self.fixture
        fixture.write_labels([self.label(fixture.patient_id, plan=[])])
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        record = self.read_run()[2][0]
        self.assert_field(record, "condition", "scored")
        self.assert_field(record, "plan", "missing_reference")
        self.assertEqual(len(fixture.scoring_calls), 3)

    def test_metric_exception_is_field_local_preserves_generation_and_next_patient_scores(self):
        fixture = self.fixture
        fixture.write_patients([{"id": name, "subjective": name, "objective": "O"}
                                for name in ("FIRST", "SECOND")])
        fixture.write_labels([self.label("FIRST", condition="first_diagnosis"),
                              self.label("SECOND", condition="second_diagnosis")])
        def evaluate(references, candidates, lang):
            if references == ["first diagnosis"]:
                raise LookupError("offline metric resource unavailable")
            return fixture.metric_values.copy()
        fixture.scoring_reply = evaluate
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        _, run, records = self.read_run()
        self.assertEqual(run["status"], "completed")
        self.assertEqual([record["status"] for record in records], ["completed", "completed"])
        self.assert_field(records[0], "condition", "error", "LookupError")
        self.assert_field(records[0], "plan", "scored")
        self.assert_field(records[1], "condition", "scored")
        self.assert_field(records[1], "plan", "scored")
        self.assertEqual(len(fixture.scoring_calls), 12)
        for record in records:
            self.assertEqual(record["assessment_plan_raw"], fixture.final_output)
            self.assertEqual(len(record["model_responses"]["assessment_plan"]), 1)

    def test_plan_metric_failure_keeps_condition_score_and_raw_provider_output(self):
        fixture = self.fixture
        def evaluate(references, candidates, lang):
            if references == ["GOLD_ONLY drug, GOLD_ONLY follow-up"]:
                raise RuntimeError("plan scorer failure")
            return fixture.metric_values.copy()
        fixture.scoring_reply = evaluate
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        record = self.read_run()[2][0]
        self.assert_field(record, "condition", "scored")
        self.assert_field(record, "plan", "error", "RuntimeError")
        self.assertEqual(record["assessment_plan_raw"], fixture.final_output)

    def test_generation_failure_keeps_failure_capture_and_later_patient_uses_own_labels(self):
        fixture = self.fixture
        fixture.write_patients([{"id": name, "subjective": name, "objective": "O"}
                                for name in ("BAD", "GOOD")])
        fixture.write_labels([self.label("GOOD", condition="good_diagnosis"), self.label("BAD")])
        fixture.failure, fixture.failure_marker = "final", "BAD"
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed_with_errors")
        _, _, records = self.read_run()
        self.assertEqual([record["status"] for record in records], ["failed_incomplete", "completed"])
        self.assertIsNone(records[0]["assessment_plan_raw"])
        self.assertIsNone(records[0]["evaluation"])
        self.assertIsNotNone(records[0]["assessment_plan_prompts"])
        self.assert_field(records[1], "condition", "scored")
        self.assertEqual(fixture.scoring_calls[0][0], ["good diagnosis"])

    def test_scoring_interruption_preserves_finished_field_raw_output_and_unstarted_patients(self):
        fixture = self.fixture
        fixture.write_patients([{"id": name, "subjective": name, "objective": "O"}
                                for name in ("FIRST", "UNSTARTED")])
        fixture.write_labels([self.label("FIRST"), self.label("UNSTARTED")])
        def evaluate(references, candidates, lang):
            if references == ["GOLD_ONLY drug, GOLD_ONLY follow-up"]:
                raise KeyboardInterrupt()
            return fixture.metric_values.copy()
        fixture.scoring_reply = evaluate
        runner = self.load_runner()
        with fixture.runtime(), self.assertRaises(KeyboardInterrupt):
            runner["main"]()
        _, run, records = self.read_run()
        self.assertEqual(run["status"], "failed_incomplete")
        self.assertEqual([record["status"] for record in records], ["failed_incomplete", "not_processed"])
        self.assertEqual(records[0]["assessment_plan_raw"], fixture.final_output)
        self.assert_field(records[0], "condition", "scored", rank=1)
        self.assert_field(records[0], "plan", "interrupted", "KeyboardInterrupt", rank=1)
        for rank in (2, 3):
            self.assert_field(records[0], "condition", "pending", rank=rank)
            self.assert_field(records[0], "plan", "pending", rank=rank)
        self.assertIsNone(records[1]["evaluation"])
        self.assertEqual(len(fixture.calls), 2)

    def test_second_rank_plan_interruption_preserves_earlier_pair_and_later_pending_scores(self):
        fixture = self.fixture
        fixture.write_patients([{"id": name, "subjective": name, "objective": "O"}
                                for name in ("FIRST", "UNSTARTED")])
        fixture.write_labels([self.label("FIRST"), self.label("UNSTARTED")])
        pairs = [{"condition": f"DIAGNOSIS_{rank}", "plan": [f"PLAN_{rank}"]}
                 for rank in (1, 2, 3)]
        fixture.final_output = json.dumps({"predictions": pairs})
        def evaluate(references, candidates, lang):
            if candidates == ["PLAN_2"]:
                raise KeyboardInterrupt()
            return fixture.metric_values.copy()
        fixture.scoring_reply = evaluate
        runner = self.load_runner()
        with fixture.runtime(), self.assertRaises(KeyboardInterrupt):
            runner["main"]()
        _, run, records = self.read_run()
        self.assertEqual(run["status"], "failed_incomplete")
        self.assertEqual([row["status"] for row in records], ["failed_incomplete", "not_processed"])
        self.assertEqual(records[0]["predictions"], pairs)
        self.assertEqual(records[0]["assessment_plan_raw"], fixture.final_output)
        self.assert_field(records[0], "condition", "scored", rank=1)
        self.assert_field(records[0], "plan", "scored", rank=1)
        self.assert_field(records[0], "condition", "scored", rank=2)
        self.assert_field(records[0], "plan", "interrupted", "KeyboardInterrupt", rank=2)
        self.assert_field(records[0], "condition", "pending", rank=3)
        self.assert_field(records[0], "plan", "pending", rank=3)
        self.assertIsNone(records[1]["predictions"])
        self.assertIsNone(records[1]["evaluation"])
        self.assertEqual(len(fixture.scoring_calls), 4)
        self.assertEqual(len(fixture.calls), 2)



class MultiModelIntegrationTests(unittest.TestCase):
    """Check combined-variant compatibility with real config/parsing/scoring orchestration."""

    load_runner = RunMedCopilotTests.load_runner
    run_dirs = RunMedCopilotTests.run_dirs

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture = PipelineFixture(Path(temporary.name))
        self.fixture.real_scoring = True
        self.fixture.write_labels([self.label(self.fixture.patient_id)])
        self.experiment = {"selected_models": ["gpt_4o_mini", "llama"], "temperature": 1,
                           "max_retries": 3, "max_patients": None}
        self.config = {"models": {
            "gpt_4o_mini": {"type": "chat", "model_provider": "openai", "model": "gpt-4o-mini",
                            "api_key_env": "OPENAI_API_KEY"},
            "llama": {"type": "chat", "model_provider": "groq", "model": "llama-3.3-70b-versatile",
                       "api_key_env": "OPENAI_API_KEY", "output_mode": "json_object"},
            "third": {"type": "chat", "model_provider": "gemini", "model": "example-gemini",
                       "api_key_env": "OPENAI_API_KEY", "output_mode": "prompt"},
        }}
        self.write_config()
        self.fixture.final_reply = self.reply

    def label(self, patient_id):
        return {"id": patient_id, "condition": "GOLD_ONLY_condition", "plan": ["GOLD_ONLY_plan"],
                "has_presenting_evidence": True}

    def write_config(self):
        self.fixture.model_config.write_text(yaml.safe_dump(self.config, sort_keys=False))
        self.fixture.experiment_config.write_text(yaml.safe_dump(self.experiment, sort_keys=False))

    def reply(self, request):
        model = request["model"]
        raw = json.dumps({"predictions": prediction_pairs(model, [model + " action"])})
        return SimpleNamespace(model="reported:" + model, id="response:" + model,
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=raw))])

    def read_native(self):
        [directory] = self.run_dirs()
        run = json.loads((directory / "run.json").read_text())
        records = [json.loads(line) for line in (directory / "results.jsonl").read_text().split("\n") if line]
        self.assertEqual({file.name for file in directory.iterdir()}, {"run.json", "results.jsonl"} | ({"results_summary.html"} if (directory / "results_summary.html").exists() else set()))
        for record in records:
            self.assertEqual(list(record), V3_RESULT_FIELDS if "variants" in record else V2_RESULT_FIELDS)
            self.assertNotIn("assessment_plan", record["model_responses"])
            for output in combined_variant_result(record)["assessment_plans"].values():
                self.assertEqual(list(output), ["status", "error", "predictions", "assessment_plan_raw",
                                                "model_responses", "evaluation"])
        return run, [combined_variant_result(record) for record in records]

    def execute(self, args=None):
        runner = self.load_runner()
        with self.fixture.runtime():
            status = runner["main"](args)
        return status, *self.read_native()

    def test_confirmation_follows_cohort_validation_and_precedes_clients_requests_and_output(self):
        fixture = self.fixture
        runner = self.load_runner()
        load_patients = Mock(wraps=runner["load_patient_records"])
        def confirm(prompt):
            self.assertEqual(prompt, "Is this configuration correct? Start the run? [y/N]: ")
            self.assertFalse(fixture.output.exists())
            self.assertFalse(any(event[0] == "client" for event in fixture.events))
            self.assertEqual(fixture.calls, [])
            self.assertEqual(fixture.retrieval_calls, [])
            self.assertEqual(fixture.graph_calls, [])
            self.assertEqual(fixture.scoring_calls, [])
            load_patients.assert_called_once_with(fixture.input_path)
            text = fixture.stdout.getvalue()
            self.assertIn("Experiment: experiments/configs/experiment.yaml", text)
            self.assertIn("Model definitions: experiments/configs/condition_plan_models.yaml", text)
            self.assertIn("  1. gpt_4o_mini (openai/gpt-4o-mini)", text)
            self.assertIn("  2. llama (groq/llama-3.3-70b-versatile)", text)
            self.assertIn("API key configured: OPENAI_API_KEY", text)
            self.assertIn("edit experiments/configs/experiment.yaml and restart", text)
            for excluded in ("GRAPHRAG_API_KEY", "Shared API keys", "test-key"):
                self.assertNotIn(excluded, text)
            return " YeS "
        fixture.confirm_input.side_effect = confirm
        with fixture.runtime(), patch.dict(runner["main"].__globals__, load_patient_records=load_patients):
            self.assertEqual(runner["main"](), "completed")
        fixture.confirm_input.assert_called_once()
        load_patients.assert_called_once_with(fixture.input_path)
        run, [record] = self.read_native()
        self.assertEqual(run["experiment"]["temperature"], 1)
        self.assertEqual(run["experiment"]["max_retries"], 3)
        self.assertEqual(record["status"], "completed")
        self.assertTrue(any(event[0] == "client" for event in fixture.events))

    def test_cancellation_never_creates_client_reads_patient_or_starts_pipeline(self):
        fixture = self.fixture
        runner = self.load_runner()
        for answer in ("n", "", "maybe", EOFError(), KeyboardInterrupt()):
            with self.subTest(answer=answer):
                fixture.confirm_input.reset_mock(side_effect=True)
                if isinstance(answer, BaseException):
                    fixture.confirm_input.side_effect = answer
                else:
                    fixture.confirm_input.return_value = answer
                with fixture.runtime(), patch.dict(runner["main"].__globals__,
                        load_patient_records=Mock(wraps=runner["load_patient_records"])):
                    self.assertEqual(runner["main"](), "cancelled")
                fixture.confirm_input.assert_called_once()
                self.assertIn("Run cancelled. No experiment output was created.", fixture.stdout.getvalue())
                self.assertFalse(fixture.output.exists())
                self.assertFalse(any(event[0] == "client" for event in fixture.events))
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.retrieval_calls, [])
                self.assertEqual(fixture.scoring_calls, [])

    def test_unknown_alias_and_missing_keys_are_reported_together_before_any_clients(self):
        fixture = self.fixture
        self.experiment["selected_models"] = ["gpt_typo", "gpt_4o_mini", "llama"]
        self.config["models"]["llama"]["api_key_env"] = "GROQ_API_KEY"
        self.write_config()
        fixture.environment = {}
        with fixture.runtime():
            runner = runpy.run_path(str(fixture.script), run_name="runner_under_test")
            self.assertEqual(runner["main"](), "configuration_error")
        message = fixture.stderr.getvalue()
        for expected in ("Experiment configuration check failed", "Unknown model alias: gpt_typo",
                         "experiment.yaml", "condition_plan_models.yaml", "Available aliases",
                         "OPENAI_API_KEY", "GROQ_API_KEY", "GRAPHRAG_API_KEY", ".env in the repository root",
                         "Run not started."):
            self.assertIn(expected, message)
        self.assertNotIn("Traceback", message)
        fixture.confirm_input.assert_not_called()
        self.assertEqual(fixture.model_configs, [])
        self.assertEqual(fixture.calls, [])
        self.assertFalse(any(event[0] == "client" for event in fixture.events))
        self.assertFalse(fixture.output.exists())

    def test_shared_graph_key_is_required_before_confirmation_and_not_exposed(self):
        fixture = self.fixture
        fixture.environment = {"OPENAI_API_KEY": "private-openai", "GRAPHRAG_API_KEY": "  "}
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "configuration_error")
        message = fixture.stderr.getvalue()
        self.assertIn("Shared-stage API key is missing or empty: GRAPHRAG_API_KEY", message)
        self.assertIn("Add GRAPHRAG_API_KEY to .env in the repository root.", message)
        self.assertNotIn("private-openai", message + fixture.stdout.getvalue())
        fixture.confirm_input.assert_not_called()
        self.assertEqual(fixture.model_configs, [])
        self.assertEqual(fixture.calls, [])
        self.assertFalse(fixture.output.exists())

    def test_keyless_local_model_is_confirmed_and_runs_with_shared_credentials(self):
        self.experiment["selected_models"] = ["local"]
        self.config["models"]["local"] = {"type": "chat", "model_provider": "openai", "model": "llama-local",
            "api_base": "http://localhost:8000/v1", "output_mode": "prompt"}
        self.write_config()
        status, run, [record] = self.execute()
        self.assertEqual(status, "completed")
        self.assertEqual(list(record["assessment_plans"]), ["local"])
        self.assertEqual(run["assessment_plan_models"]["models"]["local"]["api_key_env"], None)
        self.assertIn("1. local (openai/llama-local)", self.fixture.stdout.getvalue())
        self.assertIn("API key not required (unauthenticated endpoint)", self.fixture.stdout.getvalue())
        self.assertNotIn("local-no-key", self.fixture.stdout.getvalue())

    def test_confirmation_occurs_once_for_the_whole_patient_batch(self):
        fixture = self.fixture
        patients = [{**fixture.patient, "id": str(index)} for index in range(2)]
        fixture.write_patients(patients)
        fixture.write_labels([self.label(patient["id"]) for patient in patients])
        status, run, records = self.execute()
        self.assertEqual(status, "completed")
        self.assertEqual([record["id"] for record in records], ["0", "1"])
        fixture.confirm_input.assert_called_once()

    def test_patient_limits_select_first_eligible_patients_and_record_exact_cohort(self):
        fixture_root = self.fixture.root
        patients = [{"id": name, "subjective": f"LIMIT_PATIENT_{name}_TEXT", "objective": "O"}
                    for name in ("EXCLUDED", "SECOND", "ANOTHER", "FINAL")]
        for evidence_only in (True, False):
            for limit in (None, 1, 2, 99):
                with self.subTest(evidence_only=evidence_only, max_patients=limit):
                    self.fixture = PipelineFixture(fixture_root / f"{evidence_only}-{limit}")
                    fixture = self.fixture
                    fixture.real_scoring = True
                    fixture.final_reply = self.reply
                    fixture.write_patients(patients)
                    labels = [{**self.label(patient["id"]),
                               "has_presenting_evidence": patient["id"] != "EXCLUDED"}
                              for patient in reversed(patients)]
                    fixture.write_labels(labels)
                    self.experiment["max_patients"] = limit
                    self.write_config()
                    eligible = patients[1:] if evidence_only else patients
                    expected_ids = [patient["id"] for patient in eligible[:limit]]
                    excluded_ids = [patient["id"] for patient in patients if patient["id"] not in expected_ids]
                    status, run, records = self.execute(["--presenting-evidence-only", str(evidence_only).lower()])
                    self.assertEqual(status, "completed")
                    self.assertEqual([row["id"] for row in records], expected_ids)
                    self.assertEqual(run["experiment"]["max_patients"], limit)
                    self.assertEqual(run["experiment"]["config_sha256"],
                                     hashlib.sha256(fixture.experiment_config.read_bytes()).hexdigest())
                    self.assertEqual(run["input"]["sha256"], hashlib.sha256(fixture.input_path.read_bytes()).hexdigest())
                    evaluation = run["evaluation"]
                    self.assertEqual(evaluation["max_patients"], limit)
                    self.assertEqual(evaluation["input_patient_count"], 4)
                    self.assertEqual(evaluation["selected_patient_count"], len(expected_ids))
                    self.assertEqual(evaluation["selected_patient_ids"], expected_ids)
                    self.assertEqual(evaluation["excluded_patient_ids"], excluded_ids)
                    self.assertEqual(len(fixture.calls), 3 * len(expected_ids))
                    self.assertEqual(len(fixture.keyword_calls), len(expected_ids))
                    self.assertEqual(len(fixture.graph_calls), len(expected_ids))
                    self.assertEqual(len(fixture.scoring_calls), 12 * len(expected_ids))
                    requests = json.dumps(fixture.calls)
                    for patient_id in excluded_ids:
                        self.assertNotIn(f"LIMIT_PATIENT_{patient_id}_TEXT", requests)
                        self.assertNotIn(f": {patient_id} ==========", fixture.stdout.getvalue())
                    for number, row in enumerate(records, 1):
                        self.assertEqual(list(row["assessment_plans"]), ["gpt_4o_mini", "llama"])
                        self.assertTrue(all(output["status"] == "completed"
                                            for output in row["assessment_plans"].values()))
                        self.assertIn(f"PATIENT {number}/{len(records)}: {row['id']}", fixture.stdout.getvalue())
                    fixture.confirm_input.assert_called_once()

    def test_invalid_patient_limit_stops_before_confirmation_clients_or_run_files(self):
        runner = self.load_runner()
        for value in (0, -1, True, False, 1.0, "1", [], {}):
            with self.subTest(max_patients=value):
                self.experiment["max_patients"] = value
                self.write_config()
                with self.fixture.runtime():
                    self.assertEqual(runner["main"](), "configuration_error")
                self.assertIn("max_patients", self.fixture.stderr.getvalue())
                self.assertIn("experiments/configs/experiment.yaml", self.fixture.stderr.getvalue())
                self.fixture.confirm_input.assert_not_called()
                self.assertEqual(self.fixture.model_configs, [])
                self.assertEqual(self.fixture.calls, [])
                self.assertFalse(self.fixture.output.exists())
                self.assertFalse(any(event[0] == "client" for event in self.fixture.events))

    def test_patient_limit_does_not_hide_invalid_input_or_labels_later_in_file(self):
        fixture_root = self.fixture.root
        self.experiment["max_patients"] = 1
        for case in ("malformed_input", "missing_label", "invalid_label", "no_eligible"):
            with self.subTest(case=case):
                self.fixture = PipelineFixture(fixture_root / case)
                fixture = self.fixture
                fixture.real_scoring = True
                fixture.write_patients([{"id": name, "subjective": name, "objective": "O"}
                                        for name in ("FIRST", "AFTER_LIMIT")])
                labels = [self.label("FIRST"), self.label("AFTER_LIMIT")]
                if case == "malformed_input":
                    with fixture.input_path.open("a") as source:
                        source.write("{bad json}\n")
                elif case == "missing_label":
                    labels.pop()
                elif case == "invalid_label":
                    labels[1]["plan"] = [None]
                else:
                    for label in labels:
                        label["has_presenting_evidence"] = False
                fixture.write_labels(labels)
                self.write_config()
                runner = self.load_runner()
                with fixture.runtime(), self.assertRaises(ValueError):
                    runner["main"]()
                self.assertFalse(fixture.output.exists())
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.keyword_calls, [])
                self.assertEqual(fixture.graph_calls, [])
                self.assertEqual(fixture.scoring_calls, [])

    def test_interrupted_limited_run_does_not_create_placeholders_for_excluded_patients(self):
        self.experiment["max_patients"] = 2
        self.write_config()
        fixture = self.fixture
        fixture.write_patients([{"id": name, "subjective": name, "objective": "O"}
                                for name in ("FIRST", "SECOND", "OUTSIDE_LIMIT")])
        fixture.write_labels([self.label(name) for name in ("FIRST", "SECOND", "OUTSIDE_LIMIT")])
        fixture.failure, fixture.error = "final", KeyboardInterrupt()
        runner = self.load_runner()
        with fixture.runtime(), self.assertRaises(KeyboardInterrupt):
            runner["main"]()
        run, records = self.read_native()
        self.assertEqual(run["status"], "failed_incomplete")
        self.assertEqual(run["experiment"]["max_patients"], 2)
        self.assertEqual(run["evaluation"]["selected_patient_ids"], ["FIRST", "SECOND"])
        self.assertEqual(run["evaluation"]["excluded_patient_ids"], ["OUTSIDE_LIMIT"])
        self.assertEqual([row["id"] for row in records], ["FIRST", "SECOND"])
        self.assertEqual([row["status"] for row in records], ["failed_incomplete", "not_processed"])

    def test_entrypoint_cancellation_exits_successfully_without_output(self):
        self.fixture.confirm_input.return_value = "n"
        with self.fixture.runtime():
            runpy.run_path(str(self.fixture.script), run_name="__main__")
        self.assertIn("Run cancelled.", self.fixture.stdout.getvalue())
        self.assertFalse(self.fixture.output.exists())
        self.assertEqual(self.fixture.calls, [])

    def test_entrypoint_configuration_failure_exits_nonzero_without_confirmation(self):
        self.experiment["selected_models"] = ["typo"]
        self.write_config()
        with self.fixture.runtime(), self.assertRaises(SystemExit) as failure:
            runpy.run_path(str(self.fixture.script), run_name="__main__")
        self.assertEqual(failure.exception.code, 1)
        self.fixture.confirm_input.assert_not_called()
        self.assertIn("Unknown model alias: typo", self.fixture.stderr.getvalue())
        self.assertFalse(self.fixture.output.exists())
        self.assertEqual(self.fixture.calls, [])

    def test_experiment_settings_are_passed_to_all_models_and_saved_with_both_fingerprints(self):
        self.experiment.update(temperature=0.25, max_retries=6)
        self.write_config()
        status, run, [record] = self.execute()
        self.assertEqual(status, "completed")
        self.assertEqual([(name, settings["temperature"], settings["max_retries"])
                          for name, settings in self.fixture.model_configs],
                         [("gpt_4o_mini", 0.25, 6), ("llama", 0.25, 6)])
        self.assertEqual(run["experiment"], {
            "config_file": "experiments/configs/experiment.yaml",
            "config_sha256": hashlib.sha256(self.fixture.experiment_config.read_bytes()).hexdigest(),
            "model_config": "experiments/configs/condition_plan_models.yaml",
            "selected_models": ["gpt_4o_mini", "llama"], "temperature": 0.25, "max_retries": 6, "max_patients": None,
            "selected_variants": ["patient_retrieval_guidelines"], "patient_input": "experiments/input/test_sample_so.jsonl",
            "patient_labels": "experiments/input/test_sample_labels.jsonl", "presenting_evidence_only": True,
            "exclude_none_recorded_plans": False, **PIPELINE_DEFAULTS,
        })
        self.assertEqual(run["assessment_plan_models"]["config_sha256"],
                         hashlib.sha256(self.fixture.model_config.read_bytes()).hexdigest())
        for settings in run["assessment_plan_models"]["models"].values():
            self.assertEqual(settings["config"]["temperature"], 0.25)
            self.assertEqual(settings["config"]["max_retries"], 6)
        self.assertEqual(len(self.fixture.graph_calls), 1)
        self.assertEqual(self.fixture.keyword_calls[0]["temperature"], 0)
        self.assertEqual(len(self.fixture.scoring_calls), 12)
        self.assertEqual(record["status"], "completed")

    def test_shipped_gpt_and_downloaded_ollama_selection_runs_and_records_all_models(self):
        fixture = self.fixture
        fixture.model_config.write_bytes(MODEL_CONFIG.read_bytes())
        fixture.experiment_config.write_bytes(EXPERIMENT_CONFIG.read_bytes())
        combined = yaml.safe_load(fixture.experiment_config.read_text())
        combined.update(selected_variants=["patient_retrieval_guidelines"], exclude_none_recorded_plans=False)
        fixture.experiment_config.write_text(yaml.safe_dump(combined))
        fixture.write_patients([{"id": name, "subjective": name, "objective": "O"}
                                for name in ("FIRST", "OUTSIDE_LIMIT")])
        fixture.write_labels([self.label(name) for name in ("FIRST", "OUTSIDE_LIMIT")])
        expected = {"gpt_4o_mini": "gpt-4o-mini", "ollama_healthcare": "Jayasimma/healthcare:latest",
                    "ollama_llama3": "llama3:latest", "ollama_gemma2_2b": "gemma2:2b"}
        # Only the shared stages/GPT have a key; no Groq or Ollama key is supplied.
        self.assertEqual(set(fixture.environment), {"OPENAI_API_KEY", "GRAPHRAG_API_KEY"})
        status, run, [record] = self.execute()
        self.assertEqual(status, "completed")
        self.assertEqual(record["id"], "FIRST")
        self.assertEqual(list(record["assessment_plans"]), list(expected))
        self.assertEqual(run["experiment"]["selected_models"], list(expected))
        self.assertEqual(run["experiment"]["max_patients"], 1)
        self.assertEqual(run["evaluation"]["selected_patient_ids"], ["FIRST"])
        self.assertEqual(run["evaluation"]["excluded_patient_ids"], ["OUTSIDE_LIMIT"])
        self.assertEqual(run["assessment_plan_models"]["selected_models"], list(expected))
        self.assertEqual(run["assessment_plan_models"]["config_sha256"],
                         hashlib.sha256(fixture.model_config.read_bytes()).hexdigest())
        self.assertEqual(run["experiment"]["config_sha256"],
                         hashlib.sha256(fixture.experiment_config.read_bytes()).hexdigest())
        self.assertEqual([request["model"] for request in fixture.calls[1:]],
                         ["openai/" + tag for tag in expected.values()])
        self.assertEqual(len(fixture.calls), 5)
        self.assertTrue(all(request["messages"] == fixture.calls[1]["messages"] for request in fixture.calls[1:]))
        self.assertEqual(len(fixture.graph_calls), 1)
        self.assertEqual(len(fixture.scoring_calls), 24)
        for alias, tag in expected.items():
            settings = run["assessment_plan_models"]["models"][alias]
            self.assertEqual(settings["config"]["model"], tag)
            self.assertNotIn("api_key", settings["config"])
            self.assertEqual(record["assessment_plans"][alias]["status"], "completed")
            if alias.startswith("ollama_"):
                self.assertIsNone(settings["api_key_env"])
                self.assertEqual(settings["config"]["api_base"], "http://localhost:11434/v1")
                self.assertIn(f"{alias} (openai/{tag})", fixture.stdout.getvalue())
        self.assertEqual(fixture.stdout.getvalue().count("API key not required (unauthenticated endpoint)"), 3)
        self.assertNotIn("test-key", json.dumps(run) + fixture.stdout.getvalue())
        fixture.confirm_input.assert_called_once()

    def test_removed_config_and_model_cli_overrides_are_rejected_before_confirmation(self):
        runner = self.load_runner()
        for option in ("--config", "--experiment-config", "--model-config", "--models"):
            with self.subTest(option=option), self.fixture.runtime():
                with self.assertRaises(SystemExit) as error:
                    runner["main"]([option, "override"])
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(self.fixture.output.exists())
                self.fixture.confirm_input.assert_not_called()
                self.assertEqual(self.fixture.calls, [])
                self.assertEqual(self.fixture.model_configs, [])

    def test_invalid_experiment_values_fail_before_model_initialization_or_pipeline_calls(self):
        runner = self.load_runner()
        for changes in ({"temperature": -1}, {"temperature": True}, {"temperature": float("nan")},
                        {"max_retries": 0}, {"max_retries": "3"}, {"selected_models": []},
                        {"selected_models": ["missing"]}, {"model_config": "missing.yaml"},
                        {"unexpected_setting": 1}):
            with self.subTest(changes=changes):
                self.fixture.experiment_config.write_text(yaml.safe_dump({**self.experiment, **changes}))
                with self.fixture.runtime():
                    self.assertEqual(runner["main"](), "configuration_error")
                self.fixture.confirm_input.assert_not_called()
                self.assertFalse(self.fixture.output.exists())
                self.assertEqual(self.fixture.model_configs, [])
                self.assertEqual(self.fixture.calls, [])
                self.assertEqual(self.fixture.retrieval_calls, [])

    def test_missing_or_malformed_master_cannot_fall_back_to_the_registry(self):
        runner = self.load_runner()
        for source in (None, "[", "temperature: 1\ntemperature: 2"):
            with self.subTest(source=source):
                if source is None:
                    self.fixture.experiment_config.unlink()
                else:
                    self.fixture.experiment_config.write_text(source)
                with self.fixture.runtime():
                    self.assertEqual(runner["main"](), "configuration_error")
                self.fixture.confirm_input.assert_not_called()
                self.assertFalse(self.fixture.output.exists())
                self.assertEqual(self.fixture.model_configs, [])
                self.assertEqual(self.fixture.calls, [])

    def test_registry_cannot_override_shared_experiment_temperature_or_retry_budget(self):
        runner = self.load_runner()
        for setting, value in (("temperature", 0.5), ("max_retries", 9)):
            with self.subTest(setting=setting):
                data = deepcopy(self.config)
                data["models"]["gpt_4o_mini"][setting] = value
                self.fixture.model_config.write_text(yaml.safe_dump(data))
                with self.fixture.runtime():
                    self.assertEqual(runner["main"](), "configuration_error")
                self.assertIn("experiment.yaml", self.fixture.stderr.getvalue())
                self.fixture.confirm_input.assert_not_called()
                self.assertFalse(self.fixture.output.exists())
                self.assertEqual(self.fixture.calls, [])
                self.assertEqual(self.fixture.model_configs, [])

    def test_dotenv_only_credentials_reach_shared_and_selected_models_without_leaking_to_artifacts(self):
        fixture = self.fixture
        fixture.environment = {}
        self.config["models"]["llama"]["api_key_env"] = "GROQ_API_KEY"
        self.write_config()
        source = fixture.root / ".env"
        contents = ("OPENAI_API_KEY=file-only-openai\nGRAPHRAG_API_KEY=${OPENAI_API_KEY}\n"
                    "GROQ_API_KEY=file-only-groq\n")
        source.write_text(contents)
        original_graph = fixture.modules["graphrag.cli.query"].run_local_search
        observed = []
        def graph_search(**kwargs):
            observed.append(os.environ["GRAPHRAG_API_KEY"])
            return original_graph(**kwargs)
        fixture.modules["graphrag.cli.query"].run_local_search = graph_search
        # Match a real invocation: import and main run within the same process environment.
        with fixture.runtime():
            runner = runpy.run_path(str(fixture.script), run_name="runner_under_test")
            self.assertEqual(runner["main"](), "completed")
        run, [record] = self.read_native()
        self.assertIn(("client", "file-only-openai"), fixture.events)
        self.assertEqual(observed, ["file-only-openai"])
        self.assertEqual([(name, config["api_key"]) for name, config in fixture.model_configs],
                         [("gpt_4o_mini", "file-only-openai"), ("llama", "file-only-groq")])
        self.assertEqual(list(record["assessment_plans"]), ["gpt_4o_mini", "llama"])
        self.assertEqual(len(fixture.calls), 3)
        self.assertEqual(len(fixture.scoring_calls), 12)
        captured = json.dumps([run, record]) + fixture.stdout.getvalue() + fixture.stderr.getvalue()
        for token in ("file-only-openai", "file-only-groq"):
            self.assertNotIn(token, captured)
        self.assertEqual(source.read_text(), contents)

    def test_missing_selected_key_in_dotenv_fails_preflight_but_unselected_key_is_optional(self):
        fixture = self.fixture
        fixture.environment = {}
        self.config["models"]["llama"]["api_key_env"] = "GROQ_API_KEY"
        self.write_config()
        (fixture.root / ".env").write_text("OPENAI_API_KEY=file-openai\nGRAPHRAG_API_KEY=${OPENAI_API_KEY}\n")
        with fixture.runtime():
            runner = runpy.run_path(str(fixture.script), run_name="runner_under_test")
            self.assertEqual(runner["main"](), "configuration_error")
            self.assertIn("GROQ_API_KEY", fixture.stderr.getvalue())
            fixture.confirm_input.assert_not_called()
            self.assertFalse(fixture.output.exists())
            self.assertEqual(fixture.calls, [])
            self.assertEqual(fixture.retrieval_calls, [])
            self.experiment["selected_models"] = ["gpt_4o_mini"]
            self.write_config()
            self.assertEqual(runner["main"](), "completed")
        run, [record] = self.read_native()
        self.assertEqual(list(record["assessment_plans"]), ["gpt_4o_mini"])
        self.assertEqual(len(fixture.calls), 2)

    def test_runner_prints_models_with_three_numbered_condition_and_three_step_plan_pairs(self):
        def ranked_reply(request):
            reply = self.reply(request)
            prefix = "First" if request["model"].startswith("openai/") else "Second"
            reply.choices[0].message.content = json.dumps({"predictions": [
                {"condition": f"{prefix} condition {letter}",
                 "plan": [f"{prefix} {letter} step {step}" for step in (1, 2, 3)]}
                for letter in ("A", "B", "C")
            ]})
            return reply
        self.fixture.final_reply = ranked_reply
        status, run, [record] = self.execute()
        self.assertEqual(status, "completed")
        stdout = self.fixture.stdout.getvalue()
        heading = "\n========== RANKED CONDITIONS AND PLANS: patient_retrieval_guidelines ==========\n"
        self.assertEqual(stdout.count(heading), 1)
        printed = stdout.split(heading, 1)[1].split("\n========== RESULTS", 1)[0].strip("\n")
        self.assertEqual(printed, "\n".join([
            "Model: gpt_4o_mini (openai/gpt-4o-mini)",
            "    1. Condition: First condition A", "        Plan:",
            "            1. First A step 1", "            2. First A step 2", "            3. First A step 3", "",
            "    2. Condition: First condition B", "        Plan:",
            "            1. First B step 1", "            2. First B step 2", "            3. First B step 3", "",
            "    3. Condition: First condition C", "        Plan:",
            "            1. First C step 1", "            2. First C step 2", "            3. First C step 3", "",
            "Model: llama (groq/llama-3.3-70b-versatile)",
            "    1. Condition: Second condition A", "        Plan:",
            "            1. Second A step 1", "            2. Second A step 2", "            3. Second A step 3", "",
            "    2. Condition: Second condition B", "        Plan:",
            "            1. Second B step 1", "            2. Second B step 2", "            3. Second B step 3", "",
            "    3. Condition: Second condition C", "        Plan:",
            "            1. Second C step 1", "            2. Second C step 2", "            3. Second C step 3",
        ]))
        self.assertNotIn('"predictions":', printed)
        self.assertNotIn("========== CONDITIONS AND PLANS:", stdout)
        self.assertEqual(len(self.fixture.calls), 3)
        self.assertEqual(len(self.fixture.scoring_calls), 12)
        for alias, prefix in (("gpt_4o_mini", "First"), ("llama", "Second")):
            saved = record["assessment_plans"][alias]
            self.assertEqual(saved["predictions"], [
                {"condition": f"{prefix} condition {letter}",
                 "plan": [f"{prefix} {letter} step {step}" for step in (1, 2, 3)]}
                for letter in ("A", "B", "C")])
            self.assertEqual(saved["predictions"], json.loads(saved["assessment_plan_raw"])["predictions"])
            self.assertTrue(all(item[field]["status"] == "scored" for item in saved["evaluation"]
                                for field in ("condition", "plan")))

    def test_runner_prints_failed_model_details_and_keeps_successful_sibling_pairs_separate(self):
        def fail_first(request):
            if request["model"].startswith("openai/"):
                raise RuntimeError("unavailable test-key\nretry later")
            return self.reply(request)
        self.fixture.final_reply = fail_first
        status, run, [record] = self.execute()
        self.assertEqual(status, "completed_with_errors")
        stdout = self.fixture.stdout.getvalue()
        printed = stdout.split("========== RANKED CONDITIONS AND PLANS: patient_retrieval_guidelines ==========\n", 1)[1]
        self.assertIn("Model: gpt_4o_mini (openai/gpt-4o-mini)\n"
                      "    Generation status: failed_incomplete\n"
                      "    Error (RuntimeError): unavailable [REDACTED]\n"
                      "    retry later\n\nModel: llama (groq/llama-3.3-70b-versatile)\n", printed)
        self.assertNotIn("test-key", printed)
        self.assertEqual(len(re.findall(r"^    [123]\. Condition:", printed, re.MULTILINE)), 3)
        self.assertIsNone(record["assessment_plans"]["gpt_4o_mini"]["evaluation"])
        self.assertEqual(record["assessment_plans"]["llama"]["evaluation"][0]["plan"]["status"], "scored")

    def test_two_models_share_exact_evidence_and_receive_independent_predictions_and_scores(self):
        status, run, [record] = self.execute()
        self.assertEqual(status, "completed")
        self.assertEqual(run["schema_version"], 3)
        self.assertEqual(run["assessment_plan_models"]["selected_models"], ["gpt_4o_mini", "llama"])
        self.assertEqual(run["assessment_plan_models"]["config_sha256"],
                         hashlib.sha256(self.fixture.model_config.read_bytes()).hexdigest())
        self.assertNotIn("test-key", json.dumps(run))
        self.assertEqual(len(self.fixture.graph_calls), 1)
        self.assertEqual(len(self.fixture.keyword_calls), 1)
        self.assertEqual(len(self.fixture.calls), 3)
        first, second = self.fixture.calls[1:]
        self.assertEqual(first["messages"], second["messages"])
        self.assertEqual(first["messages"], [
            {"role": "system", "content": record["assessment_plan_prompts"]["system"]},
            {"role": "user", "content": record["assessment_plan_prompts"]["user"]}])
        self.assertEqual(first["response_format"], FINAL_RESPONSE_FORMAT)
        self.assertEqual(second["response_format"], {"type": "json_object"})
        for alias, request in zip(("gpt_4o_mini", "llama"), (first, second)):
            output = record["assessment_plans"][alias]
            self.assertEqual(output["predictions"], prediction_pairs(request["model"], [request["model"] + " action"]))
            self.assertEqual(json.loads(output["assessment_plan_raw"])["predictions"], output["predictions"])
            self.assertEqual(output["model_responses"][0]["reported_model"], "reported:" + request["model"])
            self.assertEqual(output["model_responses"][0]["token_usage"]["total_tokens"], 5)
            self.assertEqual([entry["rank"] for entry in output["evaluation"]], [1, 2, 3])
            self.assertTrue(all(entry[field]["status"] == "scored" for entry in output["evaluation"]
                                for field in ("condition", "plan")))
            self.assertNotIn("GOLD_ONLY", json.dumps(request["messages"]))
        self.assertEqual(len(self.fixture.scoring_calls), 12)
        self.assertEqual(self.fixture.scoring_calls[0][0], ["GOLD ONLY condition"])
        self.assertEqual(self.fixture.scoring_calls[6][1], [second["model"]])

    def test_config_selection_order_and_shared_settings_are_recorded(self):
        self.experiment.update(selected_models=["third", "gpt_4o_mini"], temperature=0.5, max_retries=4)
        self.write_config()
        status, run, [record] = self.execute()
        self.assertEqual(status, "completed")
        self.assertEqual(list(record["assessment_plans"]), ["third", "gpt_4o_mini"])
        self.assertEqual(run["assessment_plan_models"]["config_file"], "experiments/configs/condition_plan_models.yaml")
        self.assertEqual(run["experiment"]["selected_models"], ["third", "gpt_4o_mini"])
        self.assertEqual(run["experiment"]["model_config"], "experiments/configs/condition_plan_models.yaml")
        self.assertEqual(run["experiment"]["temperature"], 0.5)
        self.assertEqual(run["experiment"]["max_retries"], 4)
        self.assertTrue(all(settings["temperature"] == 0.5 and settings["max_retries"] == 4
                            for _, settings in self.fixture.model_configs))
        self.assertNotIn("response_format", self.fixture.calls[1])
        self.assertEqual([call["model"] for call in self.fixture.calls[1:]],
                         ["gemini/example-gemini", "openai/gpt-4o-mini"])

    def test_failure_is_model_and_patient_local_and_successful_sibling_still_scores(self):
        fixture = self.fixture
        fixture.write_patients([{"id": name, "subjective": name, "objective": "O"} for name in ("BAD", "GOOD")])
        fixture.write_labels([self.label("GOOD"), self.label("BAD")])
        def fail_first(request):
            if request["model"].startswith("openai/") and "BAD" in request["messages"][-1]["content"]:
                raise RuntimeError("request failed test-key")
            return self.reply(request)
        fixture.final_reply = fail_first
        status, run, records = self.execute()
        self.assertEqual(status, "completed_with_errors")
        self.assertEqual([record["status"] for record in records], ["failed_incomplete", "completed"])
        failed = records[0]["assessment_plans"]["gpt_4o_mini"]
        self.assertEqual(failed["error"], {"type": "RuntimeError", "message": "request failed [REDACTED]"})
        self.assertIsNone(failed["evaluation"])
        self.assertIsNone(failed["predictions"])
        self.assertEqual(failed["model_responses"], [])
        self.assertEqual(records[0]["assessment_plans"]["llama"]["evaluation"][0]["condition"]["status"], "scored")
        self.assertEqual(len(fixture.scoring_calls), 18)
        self.assertEqual(len(fixture.graph_calls), 2)
        self.assertEqual(len(fixture.calls), 6)

    def test_shared_stage_failure_leaves_every_model_unattempted(self):
        self.fixture.failure = "graphrag"
        status, run, [record] = self.execute()
        self.assertEqual(status, "completed_with_errors")
        self.assertEqual(len(self.fixture.calls), 1)
        self.assertEqual(record["status"], "failed_incomplete")
        self.assertNotIn("========== RANKED CONDITIONS AND PLANS ==========", self.fixture.stdout.getvalue())
        self.assertTrue(all(output == HELPER_NAMESPACE["new_model_result"]()
                            for output in record["assessment_plans"].values()))

    def test_model_preflight_errors_make_no_calls_or_output_directory(self):
        runner = self.load_runner()
        for selection in (["unknown"], ["llama", "llama"]):
            with self.subTest(selection=selection):
                self.experiment["selected_models"] = selection
                self.write_config()
                with self.fixture.runtime():
                    self.assertEqual(runner["main"](), "configuration_error")
                self.assertFalse(self.fixture.output.exists())
                self.fixture.confirm_input.assert_not_called()
                self.assertEqual(self.fixture.calls, [])
                self.assertEqual(self.fixture.retrieval_calls, [])
        self.experiment["selected_models"] = ["llama"]
        self.config["models"]["llama"]["api_key_env"] = "UNCONFIGURED_TOKEN"
        self.write_config()
        with self.fixture.runtime():
            self.assertEqual(runner["main"](), "configuration_error")
        self.assertIn("UNCONFIGURED_TOKEN", self.fixture.stderr.getvalue())
        self.assertIn(".env in the repository root", self.fixture.stderr.getvalue())
        self.assertFalse(self.fixture.output.exists())

    def test_generation_interrupt_keeps_first_model_and_unattempted_slots_and_patients(self):
        self.experiment["selected_models"].append("third")
        self.write_config()
        self.fixture.write_patients([{"id": name, "subjective": name, "objective": "O"}
                                     for name in ("FIRST", "UNSTARTED")])
        self.fixture.write_labels([self.label("FIRST"), self.label("UNSTARTED")])
        def interrupt(request):
            if request["model"].startswith("groq/"):
                raise KeyboardInterrupt()
            return self.reply(request)
        self.fixture.final_reply = interrupt
        runner = self.load_runner()
        with self.fixture.runtime(), self.assertRaises(KeyboardInterrupt):
            runner["main"]()
        run, [first, second] = self.read_native()
        self.assertEqual(run["status"], "failed_incomplete")
        self.assertEqual(first["status"], "failed_incomplete")
        self.assertEqual(first["assessment_plans"]["gpt_4o_mini"]["status"], "completed")
        self.assertIsNotNone(first["assessment_plans"]["gpt_4o_mini"]["predictions"])
        self.assertEqual(first["assessment_plans"]["llama"]["error"]["type"], "KeyboardInterrupt")
        self.assertEqual(first["assessment_plans"]["third"]["status"], "not_processed")
        self.assertEqual(second["status"], "not_processed")
        self.assertEqual(list(second["assessment_plans"]), ["gpt_4o_mini", "llama", "third"])
        self.assertTrue(all(output == HELPER_NAMESPACE["new_model_result"]()
                            for output in second["assessment_plans"].values()))

    def test_malformed_output_and_field_errors_do_not_erase_sibling_predictions_or_metrics(self):
        def malformed(request):
            reply = self.reply(request)
            if request["model"].startswith("groq/"):
                reply.choices[0].message.content = "  malformed café\n"
            return reply
        self.fixture.final_reply = malformed
        status, run, [record] = self.execute()
        self.assertEqual(status, "completed")
        output = record["assessment_plans"]["llama"]
        self.assertEqual(output["assessment_plan_raw"], "  malformed café\n")
        self.assertIsNone(output["predictions"])
        self.assertEqual(output["evaluation"][0]["condition"]["status"], "missing_generation")
        self.assertEqual(record["assessment_plans"]["gpt_4o_mini"]["evaluation"][0]["plan"]["status"], "scored")
        self.assertEqual(len(self.fixture.scoring_calls), 6)
        stdout = self.fixture.stdout.getvalue()
        self.assertIn("Model: llama (groq/llama-3.3-70b-versatile)\n"
                      "    No valid ranked conditions and plans returned.", stdout)
        self.assertNotIn("  malformed café\n", stdout)

    def test_scoring_errors_are_local_to_model_rank_and_field(self):
        def metric(references, candidates, lang):
            if candidates[0].startswith("groq/") and "action" in candidates[0]:
                raise RuntimeError("metric failure")
            return self.fixture.metric_values.copy()
        self.fixture.scoring_reply = metric
        status, run, [record] = self.execute()
        self.assertEqual(status, "completed")
        first, second = record["assessment_plans"].values()
        self.assertTrue(all(entry["plan"]["status"] == "scored" for entry in first["evaluation"]))
        self.assertTrue(all(entry["plan"]["status"] == "error" and entry["condition"]["status"] == "scored"
                            for entry in second["evaluation"]))

    def test_scoring_interrupt_preserves_completed_sibling_scores(self):
        def metric(references, candidates, lang):
            if candidates[0].startswith("groq/"):
                raise KeyboardInterrupt()
            return self.fixture.metric_values.copy()
        self.fixture.scoring_reply = metric
        runner = self.load_runner()
        with self.fixture.runtime(), self.assertRaises(KeyboardInterrupt):
            runner["main"]()
        run, [record] = self.read_native()
        first, second = record["assessment_plans"].values()
        self.assertEqual(run["status"], "failed_incomplete")
        self.assertTrue(all(entry[field]["status"] == "scored" for entry in first["evaluation"]
                            for field in ("condition", "plan")))
        self.assertEqual(second["evaluation"][0]["condition"]["status"], "interrupted")
        self.assertEqual(second["evaluation"][0]["plan"]["status"], "pending")
        self.assertIsNotNone(second["assessment_plan_raw"])


class VariantIntegrationTests(unittest.TestCase):
    """Native v3 artifacts, real config/scoring/collator; only external dependencies mocked."""
    load_runner = RunMedCopilotTests.load_runner
    run_dirs = RunMedCopilotTests.run_dirs
    label = MultiModelIntegrationTests.label
    reply = MultiModelIntegrationTests.reply
    write_config = MultiModelIntegrationTests.write_config
    execute = MultiModelIntegrationTests.execute

    def setUp(self):
        MultiModelIntegrationTests.setUp(self)
        self.experiment.update(selected_variants=list(HELPER_NAMESPACE["VARIANT_SOURCES"]),
                               patient_input=f"experiments/input/{INPUT_NAME}",
                               patient_labels=f"experiments/input/{LABELS_NAME}",
                               presenting_evidence_only=True, exclude_none_recorded_plans=False)
        self.write_config()

    def read_native(self):
        [folder] = self.run_dirs()
        run = json.loads((folder / "run.json").read_text())
        records = [json.loads(line) for line in (folder / "results.jsonl").read_text().split("\n") if line]
        self.assertEqual(run["schema_version"], 3)
        for row in records:
            self.assertEqual(list(row), V3_RESULT_FIELDS)
            self.assertEqual(list(row["variants"]), self.experiment["selected_variants"])
            self.assertNotIn("assessment_plans", row)
            self.assertNotIn("assessment_plan_prompts", row)
            for variant, output in row["variants"].items():
                self.assertEqual(output["evidence_sources"], list(HELPER_NAMESPACE["VARIANT_SOURCES"][variant]))
                self.assertEqual(list(output["assessment_plans"]), self.experiment["selected_models"])
        return run, records

    def test_full_matrix_reuses_evidence_and_saves_independent_prompts_scores_and_report(self):
        fixture = self.fixture
        fixture.write_patients([dict(id=name, subjective=name, objective="O") for name in ("ONE", "TWO")])
        fixture.write_labels([self.label(name) for name in ("TWO", "ONE")])
        status, run, records = self.execute()
        self.assertEqual(status, "completed")
        self.assertEqual(len(fixture.keyword_calls), 2)
        self.assertEqual(len(fixture.graph_calls), 2)
        self.assertEqual(len(fixture.calls), 18)  # 2 shared questions + 2 patients * 4 variants * 2 models
        self.assertEqual(len(fixture.scoring_calls), 96)
        self.assertEqual(len(fixture.model_configs), 2)
        for key, value in self.experiment.items():
            self.assertEqual(run["experiment"][key], value, key)
        self.assertEqual(list(run["variant_definitions"]), self.experiment["selected_variants"])
        self.assertEqual(run["prompt_provenance"]["sha256"], hashlib.sha256(fixture.prompt_module.read_bytes()).hexdigest())
        for row in records:
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["evidence_status"], {source: {"status": "completed", "error": None}
                                                     for source in ("patient_retrieval", "guidelines")})
            for variant, output in row["variants"].items():
                prompt = output["assessment_plan_prompts"]["user"]
                sources = output["evidence_sources"]
                self.assertEqual(fixture.results[0] in prompt, "patient_retrieval" in sources)
                self.assertEqual(fixture.guideline_text in prompt, "guidelines" in sources)
                self.assertEqual("If no reference patient is available" in prompt, "patient_retrieval" in sources)
                self.assertIn(row["id"], prompt)
                self.assertIn("Return ONLY one valid JSON object", prompt)
                self.assertNotIn("GOLD_ONLY", prompt)
                for model, result in output["assessment_plans"].items():
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(len(result["evaluation"]), 3)
                    self.assertTrue(all(pair[field]["status"] == "scored" for pair in result["evaluation"] for field in ("condition", "plan")))
                sent = [request for request in fixture.calls if request["model"].startswith(("openai/", "groq/"))
                        and request["messages"][1]["content"] == prompt]
                self.assertEqual(len(sent), 2)
                self.assertEqual(sent[0]["messages"], sent[1]["messages"])
        [folder] = self.run_dirs()
        html = (folder / "results_summary.html").read_text()
        self.assertIn("Best evidence variant by model", html)
        self.assertIn("llm_only", html)
        self.assertIn("BERTScore_F1", fixture.stdout.getvalue())
        self.assertIn("population_sd=", fixture.stdout.getvalue())
        self.assertEqual(run["results_summary"], {"status": "completed", "file": "results_summary.html"})

    def test_each_variant_skips_unused_stages_and_resources(self):
        root = self.fixture.root
        for variant in HELPER_NAMESPACE["VARIANT_SOURCES"]:
            with self.subTest(variant=variant):
                self.fixture = PipelineFixture(root / variant)
                self.fixture.real_scoring = True
                self.fixture.write_labels([self.label(self.fixture.patient_id)])
                self.fixture.final_reply = self.reply
                self.experiment["selected_variants"] = [variant]
                self.write_config()
                sources = HELPER_NAMESPACE["VARIANT_SOURCES"][variant]
                if "patient_retrieval" not in sources:
                    (self.fixture.upstream / "soap_with_metadata.json").unlink()
                    shutil.rmtree(self.fixture.artifacts)
                if "guidelines" not in sources:
                    shutil.rmtree(self.fixture.guidelines)
                    self.fixture.environment.pop("GRAPHRAG_API_KEY")
                status, run, [row] = self.execute()
                self.assertEqual(status, "completed")
                self.assertEqual(len(self.fixture.keyword_calls), int("patient_retrieval" in sources))
                self.assertEqual(len(self.fixture.graph_calls), int("guidelines" in sources))
                self.assertEqual(len(self.fixture.calls), 2 + int("guidelines" in sources))
                self.assertEqual(sum(event[0] == "preload" for event in self.fixture.events), int("patient_retrieval" in sources))
                for source in ("patient_retrieval", "guidelines"):
                    self.assertEqual(row["evidence_status"][source]["status"], "completed" if source in sources else "not_requested")
                    self.assertEqual(run["reference_data"][source] is not None, source in sources)

    def test_llm_only_local_model_needs_no_shared_keys_encoder_or_artifacts(self):
        fixture = self.fixture
        self.config["models"]["local"] = dict(type="chat", model_provider="openai", model="local", api_base="http://localhost:11434/v1")
        self.experiment.update(selected_models=["local"], selected_variants=["llm_only"])
        fixture.environment = {}
        (fixture.upstream / "soap_with_metadata.json").unlink()
        shutil.rmtree(fixture.guidelines)
        self.write_config()
        status, run, [row] = self.execute()
        self.assertEqual(status, "completed")
        self.assertEqual(fixture.graph_calls, [])
        self.assertEqual(fixture.retrieval_calls, [])
        self.assertEqual(fixture.rerank_calls, [])
        self.assertFalse(any(event[0] in ("preload", "client") for event in fixture.events))
        self.assertEqual(run["reference_data"], {"patient_retrieval": None, "guidelines": None})
        self.assertEqual(row["variants"]["llm_only"]["assessment_plans"]["local"]["status"], "completed")

    def test_source_failures_only_block_dependent_variants_and_keep_capture(self):
        root = self.fixture.root
        for stage, source in (("keywords", "patient_retrieval"), ("graphrag", "guidelines")):
            with self.subTest(stage=stage):
                self.fixture = PipelineFixture(root / stage)
                self.fixture.real_scoring = True
                self.fixture.write_labels([self.label(self.fixture.patient_id)])
                self.fixture.final_reply = self.reply
                self.write_config()
                self.fixture.failure = stage
                status, run, [row] = self.execute()
                self.assertEqual(status, "completed_with_errors")
                self.assertEqual(row["evidence_status"][source]["error"]["type"], "RuntimeError")
                for variant, output in row["variants"].items():
                    blocked = source in output["evidence_sources"]
                    self.assertEqual(output["status"], "failed_incomplete" if blocked else "completed")
                    for slot in output["assessment_plans"].values():
                        self.assertEqual(slot["status"], "not_processed" if blocked else "completed")
                        if blocked:
                            self.assertIsNone(slot["evaluation"])
                self.assertTrue((self.run_dirs()[0] / "results_summary.html").exists())

    def test_source_initialization_failure_still_runs_baseline_and_saves_safe_errors(self):
        fixture = self.fixture
        (fixture.upstream / "soap_with_metadata.json").unlink()
        (fixture.guidelines / "output/entities.parquet").unlink()
        status, run, [row] = self.execute()
        self.assertEqual(status, "completed_with_errors")
        self.assertEqual(run["reference_data"], {"patient_retrieval": None, "guidelines": None})
        self.assertEqual(row["variants"]["llm_only"]["status"], "completed")
        for source in ("guidelines", "patient_retrieval"):
            self.assertEqual(row["evidence_status"][source]["error"]["type"], "FileNotFoundError")
        self.assertEqual(len(fixture.calls), 2)
        self.assertEqual(fixture.graph_calls, [])

    def test_config_paths_filters_limit_and_confirmation_match_saved_effective_cohort(self):
        fixture = self.fixture
        source_patients = (ROOT / "experiments/input" / INPUT_NAME).read_bytes()
        source_labels = (ROOT / "experiments/input" / LABELS_NAME).read_bytes()
        alternate_input, alternate_labels = fixture.root / "patients.jsonl", fixture.root / "labels.jsonl"
        alternate_input.write_bytes(source_patients)
        alternate_labels.write_bytes(source_labels)
        fixture.input_path.unlink()
        fixture.labels_path.unlink()
        self.experiment.update(patient_input=str(alternate_input), patient_labels=str(alternate_labels),
                               exclude_none_recorded_plans=True, max_patients=1)
        self.write_config()
        runner = self.load_runner()
        self.assertNotIn("PATIENT_INPUT", runner)
        self.assertNotIn("PATIENT_LABELS", runner)
        def confirm(prompt):
            self.assertFalse(fixture.output.exists())
            self.assertEqual(fixture.calls, [])
            self.assertEqual(fixture.retrieval_calls, [])
            self.assertEqual(fixture.scoring_calls, [])
            text = fixture.stdout.getvalue()
            self.assertIn("Planned final requests: 8", text)
            for key in self.experiment:
                if key not in ("selected_models", "selected_variants") and key not in PIPELINE_DEFAULTS:
                    self.assertIn(key + ":", text)
            return "y"
        fixture.confirm_input.side_effect = confirm
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        run, [row] = self.read_native()
        metadata = run["evaluation"]
        self.assertEqual([metadata[key] for key in ("input_patient_count", "after_presenting_evidence_count", "eligible_patient_count", "selected_patient_count")], [10, 7, 5, 1])
        self.assertEqual(run["input"]["sha256"], hashlib.sha256(source_patients).hexdigest())
        self.assertEqual(metadata["labels_sha256"], hashlib.sha256(source_labels).hexdigest())
        self.assertEqual(set(metadata["exclusion_reasons"]), set(metadata["excluded_patient_ids"]))
        for key, value in self.experiment.items():
            expected = Path(value).relative_to(fixture.root).as_posix() if key in ("patient_input", "patient_labels") else value
            self.assertEqual(run["experiment"][key], expected)
        self.assertEqual(metadata["selected_patient_ids"], [row["id"]])

    def test_generation_interrupt_preserves_earlier_slots_and_all_unstarted_combinations(self):
        fixture = self.fixture
        fixture.write_patients([dict(id=name, subjective=name, objective="O") for name in ("FIRST", "LATER")])
        fixture.write_labels([self.label(name) for name in ("FIRST", "LATER")])
        def interrupted(request):
            if request["model"].startswith("groq/"):
                raise KeyboardInterrupt()
            return self.reply(request)
        fixture.final_reply = interrupted
        runner = self.load_runner()
        with fixture.runtime(), self.assertRaises(KeyboardInterrupt):
            runner["main"]()
        run, [first, later] = self.read_native()
        self.assertEqual(run["status"], "failed_incomplete")
        slots = first["variants"]["llm_only"]["assessment_plans"]
        self.assertEqual(slots["gpt_4o_mini"]["status"], "completed")
        self.assertEqual(slots["llama"]["error"]["type"], "KeyboardInterrupt")
        self.assertEqual(first["variants"]["llm_only"]["status"], "failed_incomplete")
        for variant in list(first["variants"])[1:]:
            self.assertEqual(first["variants"][variant]["status"], "not_processed")
        for variant in later["variants"].values():
            self.assertTrue(all(slot["status"] == "not_processed" for slot in variant["assessment_plans"].values()))
        self.assertFalse((self.run_dirs()[0] / "results_summary.html").exists())

    def test_scoring_interrupt_preserves_each_variants_generation_and_prior_scores(self):
        fixture = self.fixture
        calls = []
        def interrupt(references, candidates, lang):
            calls.append(1)
            if len(calls) == 7:
                raise KeyboardInterrupt()
            return fixture.metric_values.copy()
        fixture.scoring_reply = interrupt
        runner = self.load_runner()
        with fixture.runtime(), self.assertRaises(KeyboardInterrupt):
            runner["main"]()
        run, [row] = self.read_native()
        slots = row["variants"]["llm_only"]["assessment_plans"]
        self.assertEqual(slots["gpt_4o_mini"]["evaluation"][2]["plan"]["status"], "scored")
        self.assertEqual(slots["llama"]["evaluation"][0]["condition"]["status"], "interrupted")
        self.assertEqual(row["variants"]["guidelines"]["assessment_plans"]["gpt_4o_mini"]["status"], "completed")
        self.assertIsNone(row["variants"]["guidelines"]["assessment_plans"]["gpt_4o_mini"]["evaluation"])

    def test_report_failure_preserves_finished_artifacts_and_is_explicit(self):
        runner = self.load_runner()
        with self.fixture.runtime(), patch.dict(runner["main"].__globals__, generate_report=Mock(side_effect=OSError("disk full"))):
            self.assertEqual(runner["main"](), "report_error")
        run, [row] = self.read_native()
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["results_summary"]["status"], "failed_incomplete")
        self.assertEqual(row["status"], "completed")
        self.assertIn("saved run.json/results.jsonl were preserved", self.fixture.stderr.getvalue())

    def test_encoder_load_failure_blocks_retrieval_variants_and_preserves_independent_generation(self):
        self.fixture.failure = "preload"
        status, run, [row] = self.execute()
        self.assertEqual(status, "completed_with_errors")
        self.assertIsNone(run["reference_data"]["patient_retrieval"])
        self.assertEqual(self.fixture.retrieval_calls, [])
        self.assertEqual(row["evidence_status"]["patient_retrieval"]["error"]["type"], "RuntimeError")
        self.assertEqual(row["variants"]["llm_only"]["status"], "completed")
        self.assertEqual(row["variants"]["guidelines"]["status"], "completed")
        self.assertEqual(row["variants"]["patient_retrieval"]["error"]["type"], "EvidenceUnavailable")
        self.assertEqual(len(self.fixture.calls), 5)

    def test_one_model_failure_in_one_variant_keeps_all_other_combinations_and_report(self):
        calls = []
        def reply(request):
            calls.append(request)
            if len(calls) == 3:
                raise RuntimeError("failure test-key")
            return self.reply(request)
        self.fixture.final_reply = reply
        status, run, [row] = self.execute()
        self.assertEqual(status, "completed_with_errors")
        slots = [(name, model, value) for name, variant in row["variants"].items()
                 for model, value in variant["assessment_plans"].items()]
        failed = [(name, model) for name, model, slot in slots if slot["status"] != "completed"]
        self.assertEqual(failed, [("patient_retrieval", "gpt_4o_mini")])
        self.assertEqual(len(calls), 8)
        self.assertEqual(len(self.fixture.scoring_calls), 42)
        self.assertEqual(run["results_summary"]["status"], "completed")
        self.assertNotIn("test-key", json.dumps(run) + json.dumps(row))
        self.assertEqual(row["variants"]["patient_retrieval"]["assessment_plans"]["gpt_4o_mini"]["error"]["message"],
                         "failure [REDACTED]")

    def test_invalid_variant_and_cohort_settings_fail_before_confirmation_or_outputs(self):
        runner = self.load_runner()
        for key, value in (("selected_variants", []), ("selected_variants", ["unknown"]),
                           ("selected_variants", ["llm_only"] * 2), ("selected_variants", [{}]),
                           ("patient_input", ""), ("patient_labels", None),
                           ("presenting_evidence_only", "true"), ("exclude_none_recorded_plans", 1)):
            with self.subTest(key=key, value=value):
                self.fixture.experiment_config.write_text(yaml.safe_dump({**self.experiment, key: value}))
                with self.fixture.runtime():
                    self.assertEqual(runner["main"](), "configuration_error")
                self.fixture.confirm_input.assert_not_called()
                self.assertFalse(self.fixture.output.exists())
                self.assertEqual(self.fixture.calls, [])
                self.assertEqual(self.fixture.model_configs, [])

    def test_plan_filter_validates_excluded_later_labels_and_rejects_empty_cohort_without_calls(self):
        self.experiment.update(exclude_none_recorded_plans=True, max_patients=1)
        self.write_config()
        self.fixture.write_patients([dict(id=name, subjective=name, objective="O") for name in ("A", "B")])
        runner = self.load_runner()
        for case in ("invalid_later", "no_eligible"):
            with self.subTest(case=case):
                labels = [self.label(name) for name in ("A", "B")]
                labels[1]["plan"] = [None] if case == "invalid_later" else ["none recorded"]
                if case == "no_eligible": labels[0]["plan"] = ["none recorded"]
                self.fixture.write_labels(labels)
                with self.fixture.runtime(), self.assertRaises(ValueError):
                    runner["main"]()
                self.assertFalse(self.fixture.output.exists())
                self.fixture.confirm_input.assert_not_called()
                self.assertEqual(self.fixture.calls, [])
                self.assertEqual(self.fixture.scoring_calls, [])

    def test_shipped_config_runs_four_models_by_four_variants_and_records_every_effective_setting(self):
        fixture = self.fixture
        self.config = yaml.safe_load(MODEL_CONFIG.read_text())
        self.experiment = yaml.safe_load(EXPERIMENT_CONFIG.read_text())
        fixture.write_patients([dict(id="eligible", subjective="S", objective="O")])
        fixture.write_labels([self.label("eligible")])
        self.write_config()
        status, run, [row] = self.execute()
        self.assertEqual(status, "completed")
        self.assertEqual(len(fixture.calls), 17)  # 16 final completions plus shared questions
        self.assertEqual(len(fixture.keyword_calls), 1)
        self.assertEqual(len(fixture.graph_calls), 1)
        self.assertEqual(len(fixture.scoring_calls), 96)
        self.assertEqual(len(fixture.model_configs), 4)
        for field, value in self.experiment.items():
            self.assertEqual(run["experiment"][field], value)
        for variant in row["variants"].values():
            self.assertEqual(list(variant["assessment_plans"]), self.experiment["selected_models"])
        self.assertEqual(fixture.stdout.getvalue().count("Best variant: llm_only"), 4)
        self.assertIn("Planned final requests: 16", fixture.stdout.getvalue())

    def test_prompt_variants_remove_only_evidence_sections_and_preserve_combined_verbatim(self):
        original = PROMPTS["EVALUATE_TEMPLATE_KEYINFO"]
        prompts = runpy.run_path(str(PROJECT_PROMPTS))
        self.assertEqual(prompts["VARIANT_TEMPLATES"]["patient_retrieval_guidelines"], original)
        for name, template in prompts["VARIANT_TEMPLATES"].items():
            expected = original
            if "patient_retrieval" not in HELPER_NAMESPACE["VARIANT_SOURCES"][name]:
                expected = expected.replace(prompts["REFERENCE_SECTION"], "")
            if "guidelines" not in HELPER_NAMESPACE["VARIANT_SOURCES"][name]:
                expected = expected.replace(prompts["GUIDELINE_SECTION"], "")
            self.assertEqual(template, expected)
            rendered = template.format(conditions="S/O {literal braces}", example="EXAMPLE", Key_info="GUIDE")
            self.assertIn("S/O {literal braces}", rendered)
            self.assertIn('"predictions": [', rendered)


class PipelineConfigurationIntegrationTests(unittest.TestCase):
    """Exercise configured paths/models/tuning through saved v3 outputs and real startup."""
    load_runner = RunMedCopilotTests.load_runner
    run_dirs = RunMedCopilotTests.run_dirs
    label = MultiModelIntegrationTests.label
    reply = MultiModelIntegrationTests.reply
    write_config = MultiModelIntegrationTests.write_config
    execute = MultiModelIntegrationTests.execute
    read_native = VariantIntegrationTests.read_native

    def setUp(self):
        VariantIntegrationTests.setUp(self)
        self.experiment.update(PIPELINE_DEFAULTS)
        self.write_config()

    def test_every_setting_controls_its_pipeline_stage_and_is_saved_without_expanding_confirmation(self):
        fixture = self.fixture
        corpus = fixture.root / "references café/patients.json"
        corpus.parent.mkdir()
        (fixture.upstream / "soap_with_metadata.json").rename(corpus)
        artifacts = fixture.root / "cache café"
        fixture.artifacts.rename(artifacts)
        fixture.artifacts = artifacts
        guidelines = fixture.root / "guidelines café"
        fixture.guidelines.rename(guidelines)
        fixture.guidelines = guidelines
        fixture.output = fixture.root / "custom results café"
        self.experiment.update(
            patient_corpus="references café/patients.json", patient_retrieval_artifacts_dir="cache café",
            guidelines="guidelines café", experiment_output="custom results café",
            patient_embedding_model="custom/biomedical-encoder", patient_keyword_model="custom-keyword-model",
            patient_reranker_model="custom/reranker", guideline_question_model="custom-question-model",
            patient_retrieval_alpha=0.25, patient_retrieval_top_k=1, patient_rerank_top_k=1,
            guideline_community_level=0, guideline_response_type="Single Sentence", guideline_verbose=False,
        )
        self.write_config()
        status, run, [row] = self.execute()
        self.assertEqual(status, "completed")
        self.assertEqual(fixture.events[0], ("preload", "custom/biomedical-encoder"))
        self.assertEqual(fixture.events[1], ("import", "scripts.med_copilot_retrieval"))
        self.assertEqual(fixture.retrieval_settings[0]["model_name"], "custom/biomedical-encoder")
        self.assertEqual(fixture.retrieval_settings[0]["llm_model"], "custom-keyword-model")
        self.assertEqual(fixture.retrieval_calls[0][2:], (0.25, artifacts))
        self.assertEqual(fixture.retrieval_calls[1][2:], (1, artifacts))
        self.assertEqual(fixture.reranker_models, ["custom/reranker"])
        self.assertEqual(fixture.rerank_calls[1][3:], (1, artifacts))
        self.assertEqual(fixture.keyword_calls[0]["model"], "custom-keyword-model")
        self.assertEqual(fixture.calls[0]["model"], "custom-question-model")
        graph, cwd = fixture.graph_calls[0]
        self.assertEqual(graph["root_dir"], guidelines)
        self.assertEqual(graph["config_filepath"], guidelines / "settings.yaml")
        self.assertEqual(graph["data_dir"], guidelines / "output")
        self.assertEqual(cwd, guidelines / "output")
        self.assertEqual(graph["community_level"], 0)
        self.assertEqual(graph["response_type"], "Single Sentence")
        self.assertIs(graph["verbose"], False)
        self.assertIs(graph["streaming"], False)
        for field, value in self.experiment.items():
            self.assertEqual(run["experiment"][field], value, field)
        self.assertEqual(len(run["experiment"]), len(self.experiment) + 3)
        self.assertEqual(run["reference_data"]["patient_retrieval"]["corpus_file"], "references café/patients.json")
        self.assertEqual(run["reference_data"]["patient_retrieval"]["artifacts_directory"], "cache café")
        self.assertEqual(run["reference_data"]["guidelines"]["directory"], "guidelines café")
        self.assertEqual(len(row["retrieved_patient_ids"]), 1)
        self.assertEqual(len(row["reranked_patient_ids"]), 1)
        self.assertEqual(len(fixture.calls), 9)  # one shared question call and 4 variants * 2 models
        self.assertEqual(len(fixture.scoring_calls), 48)
        self.assertTrue((self.run_dirs()[0] / "results_summary.html").is_file())
        self.assertFalse((fixture.root / "experiments/output").exists())
        confirmation = fixture.stdout.getvalue().split("Confirmed. Starting experiment run.", 1)[0]
        for field in PIPELINE_DEFAULTS:
            self.assertNotIn(field + ":", confirmation)
        self.assertIn("Planned final requests: 8", confirmation)

    def test_configured_embedding_name_matches_real_cache_manifest_and_preloaded_instance(self):
        fixture = self.fixture
        self.experiment.update(patient_embedding_model="custom/encoder", selected_variants=["patient_retrieval"],
                               patient_keyword_model="custom-keywords", patient_retrieval_alpha=1,
                               patient_retrieval_top_k=1, patient_rerank_top_k=1)
        self.write_config()
        state = RunMedCopilotTests.real_so_retrieval(self)
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        run, [row] = self.read_native()
        manifest = json.loads((fixture.artifacts / "matching.json").read_text())
        self.assertEqual(manifest["identity"]["model"], "custom/encoder")
        self.assertEqual(manifest["identity"]["representation"], "SO")
        self.assertIs(state.retrievers[0].embedder.model, runner["retrieval_startup"].encoder)
        self.assertIs(state.retrievers[0].embedder.model, state.encoder)
        self.assertEqual(fixture.keyword_calls[0]["model"], "custom-keywords")
        self.assertEqual(run["experiment"]["patient_embedding_model"], "custom/encoder")
        self.assertEqual(len(row["retrieved_patient_ids"]), 1)
        self.assertEqual(len(row["reranked_patient_ids"]), 1)

    def test_unused_configured_resource_paths_and_models_are_not_initialized(self):
        fixture = self.fixture
        self.experiment.update(selected_variants=["llm_only"], guidelines="absent/guidelines",
                               patient_corpus="absent/corpus.json", patient_retrieval_artifacts_dir="absent/cache",
                               patient_embedding_model="absent/encoder", patient_reranker_model="absent/reranker")
        self.write_config()
        status, run, [row] = self.execute()
        self.assertEqual(status, "completed")
        self.assertFalse(any(event[0] == "preload" for event in fixture.events))
        self.assertEqual(fixture.retrieval_calls, [])
        self.assertEqual(fixture.reranker_models, [])
        self.assertEqual(fixture.graph_calls, [])
        self.assertFalse((fixture.root / "absent").exists())
        self.assertEqual(run["reference_data"], {"patient_retrieval": None, "guidelines": None})
        for field in PIPELINE_DEFAULTS:
            self.assertEqual(run["experiment"][field], self.experiment[field])

    def test_encoder_setting_changed_after_import_cannot_silently_mislabel_cached_embeddings(self):
        runner = self.load_runner()
        self.experiment["patient_embedding_model"] = "changed/encoder"
        self.write_config()
        with self.fixture.runtime():
            self.assertEqual(runner["main"](), "completed_with_errors")
        run, [row] = self.read_native()
        self.assertIn("restart the script", row["evidence_status"]["patient_retrieval"]["error"]["message"])
        self.assertEqual(row["variants"]["llm_only"]["status"], "completed")
        self.assertEqual(row["variants"]["guidelines"]["status"], "completed")
        self.assertIsNone(run["reference_data"]["patient_retrieval"])
        self.assertEqual(self.fixture.retrieval_calls, [])
        self.assertEqual(sum(event[0] == "preload" for event in self.fixture.events), 1)

    def test_invalid_new_parameters_stop_before_confirmation_generation_or_output_creation(self):
        runner = self.load_runner()
        invalid = {"guidelines": None, "experiment_output": "", "patient_corpus": [],
                   "patient_retrieval_artifacts_dir": 1, "patient_embedding_model": " ",
                   "patient_keyword_model": None, "patient_reranker_model": False, "guideline_question_model": [],
                   "patient_retrieval_alpha": float("nan"), "patient_retrieval_top_k": 0,
                   "patient_rerank_top_k": True, "guideline_community_level": -1,
                   "guideline_response_type": "", "guideline_verbose": "false"}
        for field, value in invalid.items():
            with self.subTest(field=field):
                self.fixture.experiment_config.write_text(yaml.safe_dump({**self.experiment, field: value}))
                with self.fixture.runtime():
                    self.assertEqual(runner["main"](), "configuration_error")
                self.fixture.confirm_input.assert_not_called()
                self.assertEqual(self.fixture.calls, [])
                self.assertEqual(self.fixture.model_configs, [])
                self.assertFalse(self.fixture.output.exists())


if __name__ == "__main__":
    unittest.main()

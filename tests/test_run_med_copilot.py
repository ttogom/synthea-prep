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

from test_med_copilot_retrieval import FakeEncoder, load_retrieval_module


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_med_copilot.py"
HELPERS = ROOT / "scripts" / "med_copilot_helpers.py"
RETRIEVAL = ROOT / "scripts" / "med_copilot_retrieval.py"
UPSTREAM_RETRIEVER = ROOT / "med_copilot" / "upstream" / "create_embeddings.py"
UPSTREAM_GRAPH_QUERY = ROOT / "med_copilot/upstream/graphrag/cli/query.py"
UPSTREAM_GRAPH_STORAGE = ROOT / "med_copilot/upstream/graphrag/utils/storage.py"
GUIDELINE_SETTINGS = ROOT / "med_copilot/data/guidelines/settings.yaml"
GUIDELINE_TABLES = ("communities", "community_reports", "text_units", "relationships", "entities")
PROJECT_PROMPTS = ROOT / "med_copilot/prompts/diagnosis_plan.py"
INPUT_NAME = "patient_records_so.jsonl"
BRANCH = "feature/issue-9"
SAFE_BRANCH = "feature-issue-9"
RESULT_FIELDS = [
    "id", "status", "assessment", "plan", "assessment_plan_raw", "selected_reference_patient",
    "guideline_questions", "guideline_evidence", "graphrag_context", "assessment_plan_prompts",
]
RUN_FIELDS = [
    "run_name", "started_at", "finished_at", "status", "branch", "input", "git_commit",
    "reference_data",
]
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
PROMPT_NAMES = {
    "KEY_QUESTIONS_TEMPLATE", "EVALUATE_TEMPLATE_KEYINFO",
    "PATIENT_CASE_TEMPLATE", "PATIENT_CASE_SYSTEM_TEMPLATE",
}
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
    # Execute the unchanged upstream reranker with only its model replaced, so
    # reference selection is checked against the real ordering and formatting.
    tree = ast.parse(UPSTREAM_RETRIEVER.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if (isinstance(node, ast.ClassDef) and node.name == "CrossEncoderReranker")
             or (isinstance(node, ast.Assign)
                 and any(getattr(target, "id", None) == "CONTRIBUTION_THRESHOLD"
                         for target in node.targets))]
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
        self.write_patients()
        self.conditions = (
            f'Subjective: {self.patient["subjective"]}\n'
            f'Objective: {self.patient["objective"]}'
        )
        self.questions = '{"Key Questions": ["Question one?", "Question two?"]}'
        self.guideline_text = "GUIDELINE_RESPONSE_ONLY\nGuideline evidence with café."
        self.final_output = "## Assessment\nMOCK_DIAGNOSIS\n\n## Plan\n1. MOCK_FINAL_PLAN"
        self.final_sections = {"assessment": "MOCK_DIAGNOSIS", "plan": "1. MOCK_FINAL_PLAN"}
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

    def _modules(self):
        fixture = self

        def preload(model_name):
            fixture.events.append(("preload", model_name))
            fixture.fail_at("preload")
            return object()

        def completion(**kwargs):
            fixture.calls.append(kwargs)
            fixture.completion_cwds.append(Path.cwd())
            stage = "questions" if "response_format" in kwargs else "final"
            prompt = kwargs["messages"][-1]["content"]
            fixture.fail_at(stage, prompt)
            content = fixture.questions if stage == "questions" else fixture.final_output
            if fixture.reply is not None:
                content = fixture.reply(stage, prompt)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        def client(api_key):
            fixture.events.append(("client", api_key))
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))

        def keyword_completion(**kwargs):
            fixture.keyword_calls.append(kwargs)
            fixture.fail_at("keywords", kwargs["messages"][-1]["content"])
            if fixture.keyword_response is not None:
                return fixture.keyword_response
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
            def __init__(self, df, alpha, encoder):
                fixture.retrieval_calls.append(("init", df.copy(), alpha, Path.cwd()))
                fixture.fail_at("retriever_init")
                if fixture.regenerated_artifacts is not None:
                    # Like EmbeddingRetriever, rebuild artifacts relative to the cwd.
                    for name, content in fixture.regenerated_artifacts.items():
                        Path(name).write_bytes(content)
                self.embedder = SimpleNamespace(emb_path="embeddings.npy", index_path="index.faiss",
                                               metadata_path="matching.json")
                self.keyword_extractor = object.__new__(FROZEN_KEYWORD_EXTRACTOR)
                self.keyword_extractor.model = "retrieval-keyword-model"
                self.keyword_extractor.client = SimpleNamespace(chat=SimpleNamespace(
                    completions=SimpleNamespace(create=keyword_completion),
                ))

            def search(self, query, topk):
                fixture.retrieval_calls.append(("search", query, topk, Path.cwd()))
                fixture.fail_at("retrieval", query)
                fixture.extracted_keywords = self.keyword_extractor.extract(query)
                return [dict(candidate) for candidate in fixture.candidates]

        class CrossEncoderReranker:
            def __init__(self):
                fixture.rerank_calls.append(("init", Path.cwd()))
                fixture.fail_at("reranker_init")

            def rerank(self, query, candidates, topk):
                fixture.rerank_calls.append(
                    ("rerank", query, [dict(candidate) for candidate in candidates], topk, Path.cwd()))
                fixture.fail_at("reranking", query)
                # Like the upstream reranker, record scores on the candidates.
                for candidate, score in zip(candidates, fixture.rerank_scores):
                    candidate["rerank_score"] = score
                return list(fixture.results)

        def graph_search(**kwargs):
            fixture.graph_calls.append((kwargs, Path.cwd()))
            fixture.fail_at("graphrag", kwargs["query"])
            if fixture.reply is not None:
                return fixture.reply("graphrag", kwargs["query"])
            return fixture.guideline_text, fixture.context

        modules = {name: ModuleType(name) for name in [
            "sentence_transformers", "openai", "create_embeddings", "scripts.med_copilot_retrieval", "graphrag",
            "graphrag.cli", "graphrag.cli.query", "templates", "templates.SOA_P_TEMPLATE",
        ]}
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
            with patch.dict(sys.modules, self.modules), \
                    patch.dict(os.environ, {"OPENAI_API_KEY": "test-key",
                                            "PATH": os.environ.get("PATH", os.defpath)}, clear=True), \
                    patch.object(sys, "path", isolated_path), \
                    patch("builtins.__import__", side_effect=tracked_import), \
                    redirect_stdout(self.stdout), redirect_stderr(self.stderr):
                # Exercise this fixture's real project prompts, even when a
                # caller has already imported the repository's package.
                for name in ["med_copilot.prompts.diagnosis_plan", "med_copilot.prompts", "med_copilot",
                             "scripts.med_copilot_helpers", "scripts"]:
                    sys.modules.pop(name, None)
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
        self.assertEqual({path.name for path in run_dir.iterdir()}, {"run.json", "results.jsonl"})
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        text = (run_dir / "results.jsonl").read_text(encoding="utf-8")
        self.assertTrue(text == "" or text.endswith("\n"))
        # JSONL lines are separated only by "\n"; other Unicode separators stay inside records.
        results = [json.loads(line) for line in text.split("\n")[:-1]]
        for record in results:
            self.assertEqual(list(record), RESULT_FIELDS)
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

    def expected_result(self, reference_text=None, reference="default", questions="default"):
        fixture = self.fixture
        return {
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
        self.assertEqual(list(run), RUN_FIELDS)
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

    def test_import_initializes_client_without_processing_input_or_creating_outputs(self):
        fixture = self.fixture
        original_path = sys.path.copy()
        original_cwd = Path.cwd()
        with fixture.runtime(), \
                patch.object(Path, "open", side_effect=AssertionError("Patient file read on import")):
            runner = runpy.run_path(str(fixture.script), run_name="runner_under_test")
            self.assertEqual(sys.path[:2], [str(fixture.upstream), str(fixture.root)])
        self.assertTrue(callable(runner["main"]))
        self.assertEqual(runner["PATIENT_CORPUS"],
                         fixture.upstream / "soap_with_metadata.json")
        self.assertEqual(runner["PATIENT_RETRIEVAL_ARTIFACTS_DIR"], fixture.artifacts)
        self.assertEqual(runner["PATIENT_INPUT"], fixture.input_path)
        self.assertEqual(runner["EXPERIMENT_OUTPUT"], fixture.output)
        self.assertNotIn("OUTPUT", runner)
        self.assertEqual(fixture.events, [
            ("preload", "emilyalsentzer/Bio_ClinicalBERT"), ("import", "scripts.med_copilot_retrieval"),
            ("client", "test-key"),
        ])
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

    def test_all_imports_are_in_the_header_before_function_or_class_definitions(self):
        for source in (RUNNER, HELPERS, RETRIEVAL):
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
        with fixture.runtime(), \
                patch.dict(runner["main"].__globals__, {"PATIENT_INPUT": alternate_input}), \
                patch.dict(runner["main"].__globals__, {
                    "load_patient_records": Mock(wraps=runner["load_patient_records"]),
                }):
            loader = runner["main"].__globals__["load_patient_records"]
            runner["main"]()
            loader.assert_called_once_with(alternate_input)
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assert_outputs(input_path=alternate_input)
        self.assertTrue(self.run_dirs()[0].name.endswith(f"__patients café__{SAFE_BRANCH}"))

    def test_dependency_startup_failures_stop_before_processing_input(self):
        fixture_root = self.fixture.root
        for stage, expected_events in [("preload", ["preload"]),
                                       ("upstream_import", ["preload", "import"])]:
            with self.subTest(stage=stage):
                self.fixture = PipelineFixture(fixture_root / stage)
                fixture = self.fixture
                fixture.failure = stage
                original_path = sys.path.copy()
                with fixture.runtime(), \
                        patch.object(Path, "open", side_effect=AssertionError("Patient read during startup")):
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
        self.assertEqual(keyword_request["model"], "retrieval-keyword-model")
        self.assertEqual(keyword_request["temperature"], 0)
        self.assertEqual(keyword_request["response_format"], {"type": "json_object"})
        self.assertIn(fixture.conditions, keyword_request["messages"][0]["content"])
        self.assertEqual(fixture.extracted_keywords, ["Cough", "denies fever"])
        for call in fixture.calls:
            self.assertEqual(call["model"], "gpt-4o-mini")
        self.assertEqual(fixture.calls[0]["messages"], [
            {"role": "system", "content": "You are a clinical expert."},
            {"role": "user", "content": PROMPTS["KEY_QUESTIONS_TEMPLATE"].format(conditions=fixture.conditions)},
        ])
        self.assertEqual(fixture.calls[0]["response_format"], {"type": "json_object"})
        self.assertNotIn("response_format", fixture.calls[1])
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
                self.assertEqual(run_dir.name, f"{timestamp}__patient_records_so__feature-issue-9")
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
        self.assertTrue(run_dir.name.endswith("__patient_records_so__HEAD"))
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
        existing = fixture.output / f"2026-10-17_153045PDT__patient_records_so__{SAFE_BRANCH}"
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
                                 {"input", "output"})

    # ----------------------------------------------------------------
    # Reference-data metadata
    # ----------------------------------------------------------------

    def test_reference_metadata_follows_runtime_resources_and_regenerated_artifacts(self):
        fixture = self.fixture
        runner = self.load_runner()
        clock = sequential_clock()

        def run_once(**overrides):
            with fixture.runtime(), \
                    patch.dict(runner["main"].__globals__, {"pacific_now": clock, **overrides}):
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
        current = run_once(PATIENT_RETRIEVAL_ARTIFACTS_DIR=alternate)["reference_data"]
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
            return SimpleNamespace(outputs={}, output=SimpleNamespace(
                base_dir=overrides["output.base_dir"],
            ))

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
        with fixture.runtime(), \
                patch.dict(runner["main"].__globals__, {"GUIDELINES": selected}), \
                patch("pandas.read_parquet", side_effect=state.read_parquet):
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

    def test_missing_guideline_table_stops_the_run_before_any_patient_and_restores_cwd(self):
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
                    with self.assertRaises(FileNotFoundError) as caught:
                        runner["main"]()
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertIn(str(missing_path), str(caught.exception))
                self.assertEqual(fixture.graph_calls, [])
                self.assertEqual(state.reads, [])
                self.assertEqual(state.api_calls, [])
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.keyword_calls, [])
                self.assert_stopped_run([(fixture.patient_id, "not_processed")])

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
            **self.expected_result(), "status": "failed_incomplete", "assessment": None,
            "plan": None, "assessment_plan_raw": None, "guideline_evidence": None,
            "graphrag_context": None, "assessment_plan_prompts": None,
        }])
        self.assert_run_metadata(run_dir, run, "completed_with_errors")
        self.assertIn("mock stage failure", fixture.stderr.getvalue())

    def test_missing_guideline_output_directory_stops_without_search_fallback(self):
        fixture = self.fixture
        output = fixture.guidelines / "output"
        shutil.rmtree(output)
        self.write_guideline_tables(fixture.caller, prefix="WRONG_CALLER")
        runner = self.load_runner()
        with fixture.runtime():
            with self.assertRaises(FileNotFoundError) as caught:
                runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertIn(str(output), str(caught.exception))
        self.assertEqual(fixture.graph_calls, [])
        self.assertEqual(fixture.calls, [])
        self.assert_stopped_run([(fixture.patient_id, "not_processed")])

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

        def reranker():
            instance = reranker_class()
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

    def test_real_so_cache_write_failure_stops_before_patients_and_reranker_failure_is_isolated(self):
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
                        with self.assertRaisesRegex(OSError, "cache write failed"):
                            runner["main"]()
                    self.assert_stopped_run([(fixture.patient_id, "not_processed")])
                    self.assertEqual(fixture.calls, [])
                else:
                    ranker = runner["CrossEncoderReranker"]()
                    ranker.model.predict = Mock(side_effect=RuntimeError("reranking failed"))
                    with fixture.runtime(), patch.dict(runner["main"].__globals__, {
                        "CrossEncoderReranker": lambda: ranker,
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
                self.assertIn("generate both Assessment and Plan from the patient's Subjective and Objective",
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
        reranker_class = frozen_reranker_class(scores)
        fixture.modules["scripts.med_copilot_retrieval"].CrossEncoderReranker = reranker_class
        reference_text = reranker_class().rerank(
            "query", [dict(candidate) for candidate in fixture.candidates], topk=5)[0]
        self.assertIn("Reference P1", reference_text)
        fixture.reference = {"note_id": "NOTE-1"}
        runner = self.load_runner()
        with fixture.runtime():
            runner["main"]()
        self.assert_outputs(reference_text=reference_text)

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
            "status": "failed_incomplete", "assessment": None, "plan": None,
            "assessment_plan_raw": None,
        }])
        self.assertIn("No reference patient available.",
                      results[0]["assessment_plan_prompts"]["user"])
        self.assertEqual(fixture.calls[-1]["messages"], [
            {"role": "system", "content": results[0]["assessment_plan_prompts"]["system"]},
            {"role": "user", "content": results[0]["assessment_plan_prompts"]["user"]},
        ])
        self.assert_run_metadata(run_dir, run, "completed_with_errors")

    def test_unusable_assessment_plan_sections_are_null_and_the_raw_reply_is_kept(self):
        # Format validation is out of scope: a reply that cannot be split still completes,
        # and expected_result() requires assessment_plan_raw to equal the reply exactly.
        fixture_root = self.fixture.root
        for name, content, sections in [
            ("empty", "", (None, None)),
            ("whitespace", " \n\t", (None, None)),
            ("no_content", None, (None, None)),
            ("unrecognized_labels", "Impression:\nDx\nRecommendations:\nRx", (None, None)),
            ("combined_heading", "## Assessment and Plan\nDx; Rx", (None, None)),
            ("plain_labels", "Assessment:\nDx\n\nPlan:\nRx", ("Dx", "Rx")),
            ("assessment_only", "## Assessment\nDx café", ("Dx café", None)),
            ("empty_plan", "## Assessment\nDx\n## Plan\n \n", ("Dx", None)),
            ("repeated_assessment", "## Assessment\nA\n## Plan\nRx\n## Assessment\nB",
             (None, "Rx")),
        ]:
            with self.subTest(case=name):
                self.fixture = PipelineFixture(fixture_root / name)
                fixture = self.fixture
                fixture.final_output = content
                fixture.final_sections = dict(zip(("assessment", "plan"), sections))
                runner = self.load_runner()
                with fixture.runtime():
                    self.assertEqual(runner["main"](), "completed")
                self.assert_outputs()

    def test_raw_reply_is_saved_exactly_alongside_its_sections(self):
        fixture = self.fixture
        fixture.final_output = ("Here is the note.\r\n\r\n**Assessment (A):**\r\n  Viral URI caf\u00e9\u2028"
                                "likely  \r\n\r\nPlan:\r\n1. Rest\t\r\n\r\n")
        fixture.final_sections = {"assessment": "Viral URI caf\u00e9\u2028likely",
                                  "plan": "1. Rest"}
        runner = self.load_runner()
        with fixture.runtime():
            self.assertEqual(runner["main"](), "completed")
        self.assert_outputs()
        [run_dir] = self.run_dirs()
        raw_line = (run_dir / "results.jsonl").read_text(encoding="utf-8")
        self.assertEqual(json.loads(raw_line)["assessment_plan_raw"], fixture.final_output)

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
                self.assertEqual(fixture.keyword_calls[0]["model"], "retrieval-keyword-model")
                self.assertEqual([call for call in fixture.rerank_calls if call[0] == "rerank"], [])
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.graph_calls, [])
                _, _, results = self.read_run()
                self.assertEqual(results, [{"id": fixture.patient_id, "status": "failed_incomplete",
                                            **dict.fromkeys(RESULT_FIELDS[2:])}])
                self.assertIn("ValueError", fixture.stderr.getvalue())

    def test_patient_stage_errors_are_isolated_and_keep_partial_outputs(self):
        fixture_root = self.fixture.root
        unavailable = {
            "retrieval": RESULT_FIELDS[2:], "keywords": RESULT_FIELDS[2:],
            "reranking": RESULT_FIELDS[2:],
            "questions": ["assessment", "plan", "assessment_plan_raw", "guideline_questions",
                          "guideline_evidence", "graphrag_context", "assessment_plan_prompts"],
            "graphrag": ["assessment", "plan", "assessment_plan_raw", "guideline_evidence",
                         "graphrag_context", "assessment_plan_prompts"],
            "final": ["assessment", "plan", "assessment_plan_raw"],
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

    def test_shared_retrieval_initialization_failure_stops_the_run(self):
        fixture_root = self.fixture.root
        for stage in ["retriever_init", "reranker_init"]:
            with self.subTest(stage=stage):
                self.fixture = PipelineFixture(fixture_root / stage)
                fixture = self.fixture
                fixture.write_patients([{"id": "P1", **fixture.patient}, {"id": "P2", **fixture.patient}])
                fixture.failure = stage
                runner = self.load_runner()
                with fixture.runtime():
                    with self.assertRaises(RuntimeError) as caught:
                        runner["main"]()
                    self.assertIs(caught.exception, fixture.error)
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.graph_calls, [])
                self.assertEqual(len([call for call in fixture.retrieval_calls if call[0] == "init"]), 1)
                self.assert_stopped_run([("P1", "not_processed"), ("P2", "not_processed")])

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
                return f"## Assessment\n{marker} diagnosis café\n\n## Plan\n{marker} plan end"
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
                "id": record["id"],
                "status": "completed",
                "assessment": f"{marker} diagnosis café",
                "plan": f"{marker} plan end",
                "assessment_plan_raw": f"## Assessment\n{marker} diagnosis café\n\n## Plan\n{marker} plan end",
                "selected_reference_patient": fixture.reference,
                "guideline_questions": [f"{marker} question?"],
                "guideline_evidence": evidence,
                "graphrag_context": {"sources": {"columns": ["id", "text"], "index": [0],
                                                 "data": [[marker, f"{marker} source\n"]]}},
                "assessment_plan_prompts": {
                    "system": "You are a clinical expert.",
                    "user": self.expected_prompt(conditions=conditions, guideline_text=evidence),
                },
            })
            others = [m for m in ("MARKER_ALPHA", "MARKER_BRAVO", "MARKER_CHARLIE") if m != marker]
            serialized = json.dumps(result, ensure_ascii=False)
            for other in others:
                self.assertNotIn(other, serialized)
        final_calls = [call for call in fixture.calls if "response_format" not in call]
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
                      "assessment", "plan", "assessment_plan_raw"):
            self.assertIsNone(failed[field])
        self.assertIn("MARKER_CHARLIE", results[2]["assessment"])
        self.assertIn("MARKER_CHARLIE", results[2]["plan"])
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
        self.assertEqual([results[1][field] for field in RESULT_FIELDS[2:5]], [None] * 3)
        self.assertIn("MARKER_BRAVO", results[1]["assessment_plan_prompts"]["user"])
        self.assertEqual(results[2], {"id": records[2]["id"], "status": "not_processed",
                                      **dict.fromkeys(RESULT_FIELDS[2:])})
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
        self.assertIn("MARKER_ALPHA", results[0]["assessment"])

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
                    self.assertEqual([event[0] for event in fixture.events], ["preload", "import", "client"])
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
                self.assertEqual([event[0] for event in fixture.events], ["preload", "import", "client"])
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
        self.assertEqual([event[0] for event in fixture.events], ["preload", "import", "client"])
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
                self.assertEqual([event[0] for event in fixture.events], ["preload", "import", "client"])
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

    def test_checked_in_experiment_input_uses_id_and_unique_values(self):
        records = HELPER_NAMESPACE["load_patient_records"](ROOT / "experiments/input" / INPUT_NAME)
        self.assertEqual(len(records), 100)
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
        self.assertIn("Markdown formatting", rendered)
        for requirement in [
            "Assessment (A), including the most likely diagnosis, and Plan (P)",
            "using the Subjective and Objective as context",
            "Plan consistent with the Assessment",
            "investigations, treatment, and follow-up", "## Assessment", "## Plan",
            "Do NOT output JSON",
        ]:
            self.assertIn(requirement, rendered)
        self.assertLess(rendered.index("Follow the style of the example below:"),
                        rendered.index(self.fixture.results[0]))
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

    def test_normalization_prompts_request_so_without_diagnostic_interpretation(self):
        for name in ["PATIENT_CASE_TEMPLATE", "PATIENT_CASE_SYSTEM_TEMPLATE"]:
            with self.subTest(prompt=name):
                prompt = " ".join(PROMPTS[name].lower().split())
                self.assertIn("only subjective and objective", prompt)
        system_prompt = PROMPTS["PATIENT_CASE_SYSTEM_TEMPLATE"]
        instructions = " ".join(system_prompt.lower().split())
        self.assertIn("valid json", instructions)
        self.assertIn("do not infer diagnoses", instructions)
        self.assertIn("diagnostic interpretations", instructions)
        self.assertIn("treatment recommendations", instructions)
        self.assertIn("only facts explicitly stated in the patient record", instructions)
        for detail in ["measurements", "units", "chronology", "relevant negatives", "uncertainty"]:
            self.assertIn(detail, PROMPTS["PATIENT_CASE_TEMPLATE"])
        format_example = system_prompt[system_prompt.index("{"):system_prompt.rindex("}") + 1]
        schema_example = json.loads(format_example)
        self.assertEqual(set(schema_example), {"subjective", "objective"})
        self.assertTrue(all(isinstance(value, str) for value in schema_example.values()))

    def test_full_pipeline_uses_project_prompts_from_an_unrelated_cwd(self):
        fixture = self.fixture
        fixture.prompts = {name: f"PROJECT_{name} café\n{value}" for name, value in PROMPTS.items()}
        fixture.prompt_module.write_text(
            "\n".join(f"{name} = {value!r}" for name, value in fixture.prompts.items()),
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
            with self.assertRaises(ModuleNotFoundError) as caught:
                runpy.run_path(str(fixture.script), run_name="runner_under_test")
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(caught.exception.name, "med_copilot.prompts.diagnosis_plan")
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
        with patch.dict(runner["main"].__globals__, {"PATIENT_CORPUS": corpus_path}), \
                fixture.runtime():
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
        with patch.dict(runner["main"].__globals__, {"PATIENT_CORPUS": corpus_path}), \
                fixture.runtime():
            with self.assertRaises(exception_type) as caught:
                runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(len(fixture.calls), 0)
        self.assertEqual(fixture.retrieval_calls, [])
        self.assertEqual(fixture.rerank_calls, [])
        self.assertEqual(fixture.graph_calls, [])
        self.assert_stopped_run([(fixture.patient_id, "not_processed")])
        return caught.exception

    def test_missing_configured_corpus_stops_pipeline_without_fallback(self):
        corpus_path = self.fixture.root / "missing-patients.json"
        error = self.assert_corpus_load_failure(corpus_path, FileNotFoundError)
        self.assertEqual(error.filename, str(corpus_path))

    def test_invalid_configured_corpus_stops_pipeline_without_fallback(self):
        corpus_path = self.fixture.root / "invalid-patients.json"
        corpus_path.write_text("not valid JSON", encoding="utf-8")
        error = self.assert_corpus_load_failure(corpus_path, json.JSONDecodeError)
        self.assertEqual(error.doc, "not valid JSON")


if __name__ == "__main__":
    unittest.main()

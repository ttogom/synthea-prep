"""Offline runner tests; activate an environment with pandas first.

Run directly: python3 tests/test_run_med_copilot.py
Or discover: python3 -m unittest discover -s tests -p 'test_*.py'
Models and API clients are mocked; upstream modules are never imported.
Prepared S/O text is trusted; the loader checks JSON objects and string fields.
"""

import ast
import builtins
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_med_copilot.py"
HELPERS = ROOT / "scripts" / "med_copilot_helpers.py"
UPSTREAM_RETRIEVER = ROOT / "med_copilot" / "upstream" / "create_embeddings.py"
PROJECT_PROMPTS = ROOT / "med_copilot/prompts/diagnosis_plan.py"
INPUT_NAME = "patient_records_so.jsonl"
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


def keyword_response(content, finish_reason="stop", refusal=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, refusal=refusal),
    )])


class PipelineFixture:
    def __init__(self, root):
        root = root.resolve()
        self.root = root
        self.upstream = root / "med_copilot/upstream"
        self.guidelines = root / "med_copilot/data/guidelines"
        self.output = root / "data_run_med_copilot/output"
        self.caller = root / "caller"
        self.script = root / "scripts/run_med_copilot.py"
        for directory in [self.upstream, self.guidelines, self.caller,
                          self.script.parent, root / "data_run_med_copilot/input"]:
            directory.mkdir(parents=True)
        # An unchanged copy lets runpy exercise the actual __main__ guard.
        self.script.write_text(RUNNER.read_text())
        self.helper_module = self.script.parent / "med_copilot_helpers.py"
        self.helper_module.write_text(HELPERS.read_text(encoding="utf-8"), encoding="utf-8")
        self.prompt_module = root / "med_copilot/prompts/diagnosis_plan.py"
        self.prompt_module.parent.mkdir(parents=True)
        self.prompt_module.write_text(PROJECT_PROMPTS.read_text(encoding="utf-8"), encoding="utf-8")
        self.prompts = PROMPTS.copy()
        self.corpus = [{"subjective": "Corpus S", "objective": "Corpus O",
                        "assessment": "Corpus A", "plan": "Corpus P"}]
        (self.upstream / "soap_with_metadata.json").write_text(json.dumps(self.corpus))
        self.patient = {
            "subjective": "2000-07-20: Cough for 3 days; denies fever. café.",
            "objective": "Temperature 37.2 °C; SpO₂ 98%; weight 70 kg on 2000-07-20.",
        }
        self.second_patient = {
            "uuid": "SECOND_PATIENT_ID",
            "subjective": "SECOND_PATIENT_SUBJECTIVE_NOT_USED",
            "objective": "SECOND_PATIENT_OBJECTIVE_NOT_USED",
        }
        self.input_path = root / "data_run_med_copilot/input" / INPUT_NAME
        self.write_patients()
        self.conditions = (
            f'Subjective: {self.patient["subjective"]}\n'
            f'Objective: {self.patient["objective"]}'
        )
        self.questions = '{"Key Questions": ["Question one?", "Question two?"]}'
        self.guideline_text = "GUIDELINE_RESPONSE_ONLY\nGuideline evidence with café."
        self.final_output = "## Assessment\nMOCK_DIAGNOSIS\n\n## Plan\n1. MOCK_FINAL_PLAN"
        self.candidates = [{"id": "candidate-one"}, {"id": "candidate-two"}]
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
        self.failure = None
        self.error = RuntimeError("mock stage failure")
        self.stdout = io.StringIO()
        self.modules = self._modules()

    def write_patients(self, records=None):
        if records is None:
            records = [{"uuid": "FIRST_PATIENT_ID", **self.patient}, self.second_patient]
        self.input_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def fail_at(self, stage):
        if self.failure == stage:
            raise self.error

    def _modules(self):
        fixture = self

        def preload(model_name):
            fixture.events.append(("preload", model_name))
            fixture.fail_at("preload")
            return object()

        def completion(**kwargs):
            fixture.calls.append(kwargs)
            stage = ["questions", "final"][len(fixture.calls) - 1]
            fixture.fail_at(stage)
            content = [fixture.questions, fixture.final_output][len(fixture.calls) - 1]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        def client(api_key):
            fixture.events.append(("client", api_key))
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))

        def keyword_completion(**kwargs):
            fixture.keyword_calls.append(kwargs)
            fixture.fail_at("keywords")
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
            def __init__(self, df, alpha):
                fixture.retrieval_calls.append(("init", df.copy(), alpha, Path.cwd()))
                fixture.fail_at("retriever_init")
                self.keyword_extractor = object.__new__(FROZEN_KEYWORD_EXTRACTOR)
                self.keyword_extractor.model = "retrieval-keyword-model"
                self.keyword_extractor.client = SimpleNamespace(chat=SimpleNamespace(
                    completions=SimpleNamespace(create=keyword_completion),
                ))

            def search(self, query, topk):
                fixture.retrieval_calls.append(("search", query, topk, Path.cwd()))
                fixture.fail_at("retrieval")
                fixture.extracted_keywords = self.keyword_extractor.extract(query)
                return fixture.candidates

        class CrossEncoderReranker:
            def __init__(self):
                fixture.rerank_calls.append(("init", Path.cwd()))
                fixture.fail_at("reranker_init")

            def rerank(self, query, candidates, topk):
                fixture.rerank_calls.append(("rerank", query, candidates, topk, Path.cwd()))
                fixture.fail_at("reranking")
                return fixture.results

        def graph_search(**kwargs):
            fixture.graph_calls.append((kwargs, Path.cwd()))
            fixture.fail_at("graphrag")
            return fixture.guideline_text, fixture.context

        modules = {name: ModuleType(name) for name in [
            "sentence_transformers", "openai", "create_embeddings", "graphrag",
            "graphrag.cli", "graphrag.cli.query", "templates", "templates.SOA_P_TEMPLATE",
        ]}
        for name in ["graphrag", "graphrag.cli", "templates"]:
            modules[name].__path__ = []
        modules["sentence_transformers"].SentenceTransformer = preload
        modules["openai"].OpenAI = client
        modules["create_embeddings"].HybridRetriever = HybridRetriever
        modules["create_embeddings"].CrossEncoderReranker = CrossEncoderReranker
        modules["graphrag.cli.query"].run_local_search = graph_search
        modules["templates.SOA_P_TEMPLATE"].__dict__.update(PROMPTS)
        return modules

    @contextmanager
    def runtime(self):
        original_import = builtins.__import__

        def tracked_import(name, *args, **kwargs):
            if name == "create_embeddings":
                self.events.append(("import", name))
                self.fail_at("upstream_import")
            return original_import(name, *args, **kwargs)

        old_cwd = Path.cwd()
        os.chdir(self.caller)
        try:
            with patch.dict(sys.modules, self.modules), \
                    patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                    patch.object(sys, "path", sys.path.copy()), \
                    patch("builtins.__import__", side_effect=tracked_import), \
                    redirect_stdout(self.stdout):
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

    def test_import_initializes_client_and_output_without_processing_input_or_calling_apis(self):
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
        self.assertEqual(runner["PATIENT_INPUT"], fixture.input_path)
        self.assertEqual(fixture.events, [
            ("preload", "emilyalsentzer/Bio_ClinicalBERT"), ("import", "create_embeddings"),
            ("client", "test-key"),
        ])
        self.assertEqual(fixture.calls, [])
        self.assertEqual(fixture.keyword_calls, [])
        self.assertEqual(fixture.retrieval_calls, [])
        self.assertEqual(fixture.rerank_calls, [])
        self.assertEqual(fixture.graph_calls, [])
        self.assertEqual(sys.path, original_path)
        self.assertEqual(Path.cwd(), original_cwd)
        self.assertEqual(list(fixture.output.iterdir()), [])

    def test_all_imports_are_in_the_header_before_function_or_class_definitions(self):
        for source in (RUNNER, HELPERS):
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

    def test_patient_loader_returns_all_records_for_path_and_string_arguments(self):
        fixture = self.fixture
        records = [
            {"uuid": "FIRST_PATIENT_ID", **fixture.patient,
             "metadata": {"source": "café", "encounter": "2000-07-20"}},
            fixture.second_patient,
        ]
        fixture.write_patients(records)
        original_input = fixture.input_path.read_bytes()
        loader = runpy.run_path(str(HELPERS))["load_patient_records"]
        for argument in (fixture.input_path, str(fixture.input_path)):
            with self.subTest(argument_type=type(argument).__name__):
                self.assertEqual(loader(argument), records)
        self.assertEqual(fixture.input_path.read_bytes(), original_input)
        self.assertEqual(fixture.events, [])
        self.assertFalse(fixture.output.exists())

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
        self.assert_outputs()

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
                self.assertEqual(list(fixture.output.iterdir()), [])
                self.assertEqual(sys.path, original_path)

    def assert_outputs(self, reference_text=None):
        fixture = self.fixture
        if reference_text is None:
            reference_text = fixture.results[0]
        expected_prompt = fixture.prompts["EVALUATE_TEMPLATE_KEYINFO"].format(
            conditions=fixture.conditions, example=reference_text, Key_info=fixture.guideline_text,
        )
        expected_text = {
            "top_similar_patient.txt": reference_text,
            "questions_for_guidelines_db.txt": fixture.questions,
            "answers_from_guidelines_db.txt": fixture.guideline_text,
            "final_prompt.txt": expected_prompt,
            "final_output.txt": fixture.final_output,
        }
        self.assertEqual({path.name for path in fixture.output.iterdir()},
                         set(expected_text) | {"graphrag_context.json"})
        for name, content in expected_text.items():
            with self.subTest(output=name):
                self.assertEqual((fixture.output / name).read_text(), content)
        context = json.loads((fixture.output / "graphrag_context.json").read_text())
        self.assertEqual(context, {key: json.loads(table.to_json(orient="split"))
                                  for key, table in fixture.context.items()})
        self.assertEqual(len(context["sources"]["data"]), 135)
        self.assertEqual(context["sources"]["index"], list(range(500, 635)))
        self.assertEqual(context["sources"]["data"][-1][1], "RAW_CONTEXT_ONLY_134 café\nsource")
        self.assertIsNone(context["sources"]["data"][0][2])
        self.assertEqual(context["claims"], {"columns": ["id", "claim"], "index": [], "data": []})
        self.assertEqual(fixture.calls[0]["messages"][-1]["content"],
                         fixture.prompts["KEY_QUESTIONS_TEMPLATE"].format(conditions=fixture.conditions))
        self.assertEqual(fixture.calls[-1]["messages"][-1]["content"], expected_prompt)
        self.assertNotIn("RAW_CONTEXT_ONLY", expected_prompt)
        self.assertNotIn("UNSELECTED_SIMILAR_CASE", expected_prompt)
        self.assertNotIn("RAW_CONTEXT_ONLY", fixture.stdout.getvalue())
        for marker in ["FIRST_PATIENT_ID", "SECOND_PATIENT_ID", "SECOND_PATIENT_SUBJECTIVE_NOT_USED",
                       "SECOND_PATIENT_OBJECTIVE_NOT_USED"]:
            self.assertNotIn(marker, fixture.stdout.getvalue())
            for call in fixture.calls + fixture.keyword_calls:
                self.assertNotIn(marker, call["messages"][-1]["content"])
            for name in expected_text:
                self.assertNotIn(marker, (fixture.output / name).read_text())

    def test_full_main_preserves_pipeline_contract_and_saves_evidence(self):
        fixture = self.fixture
        runner = self.load_runner()
        with fixture.runtime():
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(fixture.events[:2], [
            ("preload", "emilyalsentzer/Bio_ClinicalBERT"), ("import", "create_embeddings"),
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
        self.assertEqual(init[2:], (0.5, fixture.upstream))
        self.assertEqual(search, ("search", fixture.conditions, 20, fixture.upstream))
        self.assertEqual(fixture.rerank_calls[0], ("init", fixture.upstream))
        self.assertEqual(fixture.rerank_calls[1],
                         ("rerank", fixture.conditions, fixture.candidates, 5, fixture.upstream))
        self.assertEqual(fixture.graph_calls, [({
            "config_filepath": fixture.guidelines / "settings.yaml",
            "data_dir": fixture.guidelines / "output", "root_dir": fixture.guidelines,
            "community_level": 2, "response_type": "Multiple Paragraphs",
            "streaming": False, "query": fixture.questions, "verbose": True,
        }, fixture.caller)])
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

    def test_empty_reference_stages_continue_through_guidelines_and_final_generation(self):
        fixture_root = self.fixture.root
        for stage in ["corpus", "retrieval", "reranking"]:
            with self.subTest(stage=stage):
                self.fixture = PipelineFixture(fixture_root / stage)
                fixture = self.fixture
                if stage == "corpus":
                    (fixture.upstream / "soap_with_metadata.json").write_text("[]")
                elif stage == "retrieval":
                    fixture.candidates = []
                else:
                    fixture.results = []
                # An absent reference must replace a previously saved selection.
                fixture.output.mkdir()
                (fixture.output / "top_similar_patient.txt").write_text("STALE_REFERENCE_CASE")
                runner = self.load_runner()
                with fixture.runtime():
                    runner["main"]()
                    self.assertEqual(Path.cwd(), fixture.caller)

                if stage == "corpus":
                    self.assertEqual(fixture.retrieval_calls, [])
                    self.assertEqual(fixture.keyword_calls, [])
                else:
                    self.assertEqual(fixture.retrieval_calls[1],
                                     ("search", fixture.conditions, 20, fixture.upstream))
                if stage == "reranking":
                    self.assertEqual(fixture.rerank_calls[1],
                                     ("rerank", fixture.conditions, fixture.candidates, 5, fixture.upstream))
                else:
                    self.assertEqual(fixture.rerank_calls, [])
                self.assertEqual(len(fixture.calls), 2)
                self.assertEqual(fixture.calls[0]["messages"][-1]["content"],
                                 fixture.prompts["KEY_QUESTIONS_TEMPLATE"].format(conditions=fixture.conditions))
                self.assertEqual(len(fixture.graph_calls), 1)
                self.assertEqual(fixture.graph_calls[0][0]["query"], fixture.questions)
                self.assertEqual(fixture.graph_calls[0][1], fixture.caller)
                final_prompt = fixture.calls[1]["messages"][-1]["content"]
                for evidence in [fixture.conditions, fixture.guideline_text,
                                 "No reference patient available."]:
                    self.assertIn(evidence, final_prompt)
                self.assertIn("If no reference patient is available", final_prompt)
                self.assertIn("generate both Assessment and Plan from the patient's Subjective and Objective",
                              final_prompt)
                self.assertIn("any relevant additional information", final_prompt)
                self.assertNotIn("SELECTED_SIMILAR_CASE", final_prompt)
                self.assertNotIn("STALE_REFERENCE_CASE", final_prompt)
                self.assert_outputs(reference_text="No reference patient available.")

    def test_final_generation_failure_without_a_reference_propagates(self):
        fixture = self.fixture
        fixture.candidates = []
        fixture.failure = "final"
        runner = self.load_runner()
        with fixture.runtime():
            with self.assertRaises(RuntimeError) as caught:
                runner["main"]()
            self.assertIs(caught.exception, fixture.error)
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(fixture.rerank_calls, [])
        self.assertEqual(len(fixture.graph_calls), 1)
        self.assertEqual(len(fixture.calls), 2)
        self.assertEqual((fixture.output / "top_similar_patient.txt").read_text(),
                         "No reference patient available.")
        self.assertIn("No reference patient available.",
                      (fixture.output / "final_prompt.txt").read_text())
        self.assertFalse((fixture.output / "final_output.txt").exists())

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

    def test_keyword_reply_failures_stop_before_downstream_and_restore_cwd(self):
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
                    with self.assertRaises(ValueError):
                        runner["main"]()
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertEqual(len(fixture.keyword_calls), 1)
                self.assertEqual(fixture.keyword_calls[0]["model"], "retrieval-keyword-model")
                self.assertEqual(fixture.rerank_calls, [])
                self.assertEqual(fixture.calls, [])
                self.assertEqual(fixture.graph_calls, [])
                self.assertEqual(list(fixture.output.iterdir()), [])

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

    def test_only_so_fields_from_first_record_reach_downstream(self):
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
        for output in fixture.output.iterdir():
            self.assertNotIn("UNUSED_TARGET", output.read_text())

    def test_invalid_records_anywhere_in_jsonl_stop_before_pipeline_or_output(self):
        fixture_root = self.fixture.root
        valid = self.fixture.patient
        cases = {
            "malformed_json": ('{"subjective": "cough",', "invalid patient JSON"),
            "markdown_wrapped_json": ("```json", "invalid patient JSON"),
            "json_null": ("null", "patient must be a JSON object"),
            "json_array": (json.dumps([valid]), "patient must be a JSON object"),
            "json_string": ('"patient"', "patient must be a JSON object"),
            "json_number": ("123", "patient must be a JSON object"),
            "json_boolean": ("true", "patient must be a JSON object"),
            "empty_object": ("{}", "subjective must be a string"),
            "missing_subjective": (json.dumps({"objective": "measured evidence"}), "subjective must be a string"),
            "missing_objective": (json.dumps({"subjective": "reported evidence"}), "objective must be a string"),
            "wrong_case": (json.dumps({"Subjective": "reported evidence", "objective": "measured evidence"}), "subjective must be a string"),
        }
        for field in ("subjective", "objective"):
            for value in (None, 123, True, [], {}):
                cases[f"{field}_{type(value).__name__}"] = (
                    json.dumps({**valid, field: value}), f"{field} must be a string",
                )
        for name, (invalid_line, message) in cases.items():
            for position in ("first", "later"):
                with self.subTest(case=name, position=position):
                    self.fixture = PipelineFixture(fixture_root / name / position)
                    fixture = self.fixture
                    prefix = "\n \t\n"
                    if position == "later":
                        prefix += json.dumps(valid) + "\n"
                    invalid_line_number = prefix.count("\n") + 1
                    contents = prefix + invalid_line + "\n" + json.dumps(valid) + "\n"
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
                    self.assertEqual(list(fixture.output.iterdir()), [])
                    self.assertEqual(fixture.input_path.read_text(encoding="utf-8"), contents)
                    self.assertNotIn("STAGE 1: S/O", fixture.stdout.getvalue())

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
                self.assertEqual(list(fixture.output.iterdir()), [])

    def test_missing_jsonl_stops_without_falling_back_to_legacy_note(self):
        fixture = self.fixture
        fixture.input_path.unlink()
        legacy_note = fixture.input_path.parent / (
            "Peter292_Gleichner915_0db4522b-544a-43b1-ad63-3caeb72be2ab.txt"
        )
        legacy_note.write_text("LEGACY_NOTE_MUST_NOT_BE_USED", encoding="utf-8")
        runner = self.load_runner()
        with fixture.runtime():
            with self.assertRaises(FileNotFoundError) as caught:
                runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(caught.exception.filename, str(fixture.input_path))
        self.assertEqual([event[0] for event in fixture.events], ["preload", "import", "client"])
        self.assertEqual(fixture.calls, [])
        self.assertEqual(list(fixture.output.iterdir()), [])

    def test_entire_jsonl_is_loaded_before_processing_only_first_patient(self):
        fixture = self.fixture
        patients = [
            {"uuid": "FIRST_PATIENT_ID", **fixture.patient},
            fixture.second_patient,
            {"uuid": "THIRD_PATIENT_ID", "subjective": "THIRD_PATIENT_SUBJECTIVE_NOT_USED",
             "objective": "THIRD_PATIENT_OBJECTIVE_NOT_USED", "metadata": {"source": "café"}},
        ]
        fixture.input_path.write_text(
            "\n \t\n" + "\n\n".join(json.dumps(patient, ensure_ascii=False) for patient in patients)
            + "\n \t\n",
            encoding="utf-8",
        )
        original_input = fixture.input_path.read_bytes()
        original_loads = json.loads
        patient_ids = {patient["uuid"] for patient in patients}
        loaded_patients = []

        def track_patient_loads(value, *args, **kwargs):
            record = original_loads(value, *args, **kwargs)
            if isinstance(record, dict) and record.get("uuid") in patient_ids:
                # Dependencies load in the header, but all records must be
                # decoded before making an API request or starting a stage.
                self.assertEqual([event[0] for event in fixture.events], ["preload", "import", "client"])
                self.assertEqual(fixture.keyword_calls, [])
                self.assertEqual(fixture.calls, [])
                loaded_patients.append(record)
            return record

        runner = self.load_runner()
        with fixture.runtime(), patch("json.loads", side_effect=track_patient_loads):
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(loaded_patients, patients)
        self.assertEqual(len(fixture.calls), 2)
        self.assertEqual(fixture.retrieval_calls[1][1], fixture.conditions)
        self.assertEqual(fixture.input_path.read_bytes(), original_input)
        self.assert_outputs()
        self.assertNotIn("THIRD_PATIENT", fixture.stdout.getvalue())

    def test_single_record_without_trailing_newline_runs_full_pipeline(self):
        fixture = self.fixture
        fixture.input_path.write_text(json.dumps(fixture.patient), encoding="utf-8")
        runner = self.load_runner()
        with fixture.runtime():
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(len(fixture.calls), 2)
        self.assert_outputs()

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
        with fixture.runtime():
            with self.assertRaises(ModuleNotFoundError) as caught:
                runpy.run_path(str(fixture.script), run_name="runner_under_test")
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(caught.exception.name, "med_copilot.prompts.diagnosis_plan")
        self.assertEqual(fixture.calls, [])
        self.assertEqual(fixture.retrieval_calls, [])
        self.assertEqual(fixture.graph_calls, [])
        self.assertEqual(list(fixture.output.iterdir()), [])

    def test_configured_corpus_outside_upstream_runs_the_full_pipeline(self):
        fixture = self.fixture
        corpus_path = fixture.root / "alternative corpus" / "patients café.json"
        corpus_path.parent.mkdir()
        fixture.corpus = [
            {"subjective": "Alternate S café", "objective": "Alternate O",
             "assessment": "Alternate A", "plan": "Alternate P"},
            {"subjective": "Second S", "objective": "Second O",
             "assessment": "Second A", "plan": "Second P"},
        ]
        corpus_path.write_text(json.dumps(fixture.corpus, ensure_ascii=False),
                               encoding="utf-8")
        (fixture.upstream / "soap_with_metadata.json").unlink()
        fixture.candidates = [{"id": "alternate-patient", "display_text": "Alternate S café"}]
        fixture.results = ["ALTERNATE_SIMILAR_CASE café\nAlternate P"]
        runner = self.load_runner()
        with patch.dict(runner["main"].__globals__, {"PATIENT_CORPUS": corpus_path}), \
                fixture.runtime():
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        init, search = fixture.retrieval_calls
        self.assertEqual(init[1].to_dict(orient="records"), fixture.corpus)
        self.assertEqual(init[2:], (0.5, fixture.upstream))
        self.assertEqual(search, ("search", fixture.conditions, 20, fixture.upstream))
        self.assertEqual(fixture.rerank_calls[1],
                         ("rerank", fixture.conditions, fixture.candidates, 5, fixture.upstream))
        self.assertEqual(len(fixture.calls), 2)
        self.assertEqual(fixture.graph_calls[0][1], fixture.caller)
        self.assert_outputs()

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
        self.assertEqual(list(fixture.output.iterdir()), [])
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

    def test_stage_errors_propagate_stop_generation_and_restore_cwd(self):
        runner = self.load_runner()
        fixture = self.fixture
        for stage, expected_calls in [
            ("retriever_init", 0), ("retrieval", 0), ("keywords", 0),
            ("reranker_init", 0), ("reranking", 0), ("questions", 1),
            ("graphrag", 1), ("final", 2),
        ]:
            with self.subTest(stage=stage):
                fixture.failure = stage
                fixture.calls.clear()
                fixture.graph_calls.clear()
                with fixture.runtime():
                    with self.assertRaises(RuntimeError) as caught:
                        runner["main"]()
                    self.assertIs(caught.exception, fixture.error)
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertEqual(len(fixture.calls), expected_calls)
                self.assertEqual(len(fixture.graph_calls), int(stage in {"graphrag", "final"}))
                self.assertFalse((fixture.output / "final_output.txt").exists())
                if stage == "final":
                    self.assertTrue((fixture.output / "final_prompt.txt").is_file())


class JSONKeywordExtractorTests(unittest.TestCase):
    def setUp(self):
        self.extractor_class = runpy.run_path(str(HELPERS))["JSONKeywordExtractor"]
        self.client = Mock()
        self.create = self.client.chat.completions.create
        self.valid_content = json.dumps({
            "symptoms": [" cough ", "fever", "cough"],
            "diagnoses": ["fever", "Asthma"],
            "medications": [" albuterol ", "Asthma", "cough"],
        })
        self.create.return_value = keyword_response(self.valid_content)
        self.extractor = self.extractor_class(self.client, "selected-keyword-model")

    def test_json_mode_preserves_upstream_prompt_model_temperature_and_client(self):
        text = "Subjective: Cough café.\nObjective: Temperature 37.2 °C."
        self.assertEqual(self.extractor.extract(text), ["cough", "fever", "Asthma", "albuterol"])
        self.create.assert_called_once()
        actual = self.create.call_args.kwargs
        legacy_client = Mock()
        legacy_client.chat.completions.create.return_value = keyword_response(self.valid_content)
        legacy = object.__new__(FROZEN_KEYWORD_EXTRACTOR)
        legacy.client, legacy.model = legacy_client, "selected-keyword-model"
        legacy.extract(text)
        original = legacy_client.chat.completions.create.call_args.kwargs
        self.assertEqual(actual, {**original, "response_format": {"type": "json_object"}})
        self.assertIs(self.extractor.client, self.client)
        self.assertEqual(actual["model"], "selected-keyword-model")
        self.assertEqual(actual["temperature"], 0)
        self.assertIn(text, actual["messages"][0]["content"])

    def test_frozen_original_parser_reproduces_markdown_json_failure(self):
        legacy = object.__new__(FROZEN_KEYWORD_EXTRACTOR)
        legacy.client, legacy.model = self.client, "selected-keyword-model"
        self.create.return_value = keyword_response(f"```json\n{self.valid_content}\n```")
        with self.assertRaises(json.JSONDecodeError):
            legacy.extract("Patient evidence")
        self.assertNotIn("response_format", self.create.call_args.kwargs)

    def test_deduplication_is_stable_preserves_case_and_resets_per_patient(self):
        for _ in range(3):
            self.assertEqual(self.extractor.extract("Patient evidence"),
                             ["cough", "fever", "Asthma", "albuterol"])
        self.create.return_value = keyword_response(json.dumps({
            "symptoms": ["Cough", "cough"], "diagnoses": [], "medications": [],
        }))
        self.assertEqual(self.extractor.extract("Another patient"), ["Cough", "cough"])
        self.assertEqual(self.create.call_count, 4)

    def test_empty_categories_are_valid_and_extra_fields_do_not_affect_terms(self):
        self.create.return_value = keyword_response(json.dumps({
            "symptoms": [], "diagnoses": [], "medications": [],
            "comment": "ignored", "extra_terms": ["not a keyword"],
        }))
        self.assertEqual(self.extractor.extract("No clinical entities"), [])

    def test_missing_empty_and_incomplete_model_responses_raise_without_retry(self):
        responses = {
            "no_choices": SimpleNamespace(choices=[]),
            "none_choices": SimpleNamespace(choices=None),
            "missing_choices": SimpleNamespace(),
            "none_content": keyword_response(None),
            "empty_content": keyword_response(""),
            "blank_content": keyword_response(" \n\t "),
            "nonstring_content": keyword_response({"symptoms": []}),
            "refusal": keyword_response(self.valid_content, refusal="request refused"),
        }
        for reason in ("length", "content_filter", "tool_calls", None):
            responses[f"finish_{reason}"] = keyword_response(self.valid_content, finish_reason=reason)
        for name, response in responses.items():
            with self.subTest(case=name):
                self.create.reset_mock()
                self.create.return_value = response
                with self.assertRaises(ValueError):
                    self.extractor.extract("Patient evidence")
                self.create.assert_called_once()

    def test_malformed_json_raises_with_original_parse_error_and_without_raw_clinical_text(self):
        for content in ("PRIVATE_CLINICAL_TEXT", f"```json\n{self.valid_content}\n```", '{"symptoms":'):
            with self.subTest(content_type=content[:8]):
                self.create.reset_mock()
                self.create.return_value = keyword_response(content)
                with self.assertRaises(ValueError) as caught:
                    self.extractor.extract("Patient evidence")
                self.assertIsInstance(caught.exception.__cause__, json.JSONDecodeError)
                self.assertNotIn("PRIVATE_CLINICAL_TEXT", str(caught.exception))
                self.create.assert_called_once()

    def test_valid_json_with_invalid_keyword_schema_is_rejected(self):
        valid = {"symptoms": [], "diagnoses": [], "medications": []}
        payloads = [None, [], True, 3, "text", {}]
        for field in valid:
            payloads.append({key: value for key, value in valid.items() if key != field})
            for value in (None, "cough", {}, 12, True):
                payloads.append({**valid, field: value})
            for value in (None, [], {}, 12, True):
                payloads.append({**valid, field: ["cough", value]})
        for payload in payloads:
            with self.subTest(payload=payload):
                self.create.reset_mock()
                self.create.return_value = keyword_response(json.dumps(payload))
                with self.assertRaises(ValueError):
                    self.extractor.extract("Patient evidence")
                self.create.assert_called_once()

    def test_api_errors_propagate_unchanged_without_retry(self):
        error = RuntimeError("mock keyword API failure")
        self.create.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            self.extractor.extract("Patient evidence")
        self.assertIs(caught.exception, error)
        self.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""Offline runner tests; activate an environment with pandas first.

Run directly: python3 tests/test_run_med_copilot.py
Or discover: python3 -m unittest discover -s tests -p 'test_*.py'
Models and API clients are mocked; upstream modules are never imported.
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
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_med_copilot.py"
NOTE_NAME = "Peter292_Gleichner915_0db4522b-544a-43b1-ad63-3caeb72be2ab.txt"
PROMPT_NAMES = {
    "KEY_QUESTIONS_TEMPLATE", "EVALUATE_TEMPLATE_KEYINFO",
    "PATIENT_CASE_TEMPLATE", "PATIENT_CASE_SYSTEM_TEMPLATE",
}
# Extract only literals: importing the frozen upstream is unnecessary.
PROMPTS = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in ast.parse(
        (ROOT / "med_copilot/upstream/templates/SOA_P_TEMPLATE.py").read_text()
    ).body
    if isinstance(node, ast.Assign)
    and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id in PROMPT_NAMES
}


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
        self.patient_input = "2000-07-20\n# Chief Complaint\nClinical evidence only."
        note = (self.patient_input + "\n# Assessment and Plan\nHIDDEN_TARGET_PLAN"
                + "\n\n2000-07-02\nEXCLUDED_LATER_ENCOUNTER")
        (root / "data_run_med_copilot/input" / NOTE_NAME).write_text(note)
        self.corpus = [{"subjective": "Corpus S", "objective": "Corpus O",
                        "assessment": "Corpus A", "plan": "Corpus P"}]
        (self.upstream / "soap_with_metadata.json").write_text(json.dumps(self.corpus))
        self.patient = {"subjective": "Normalized S", "objective": "Normalized O",
                        "assessment": "Inferred A"}
        self.conditions = "Subjective: Normalized S\nObjective: Normalized O\nAssessment: Inferred A"
        self.questions = '{"Key Questions": ["Question one?", "Question two?"]}'
        self.guideline_text = "GUIDELINE_RESPONSE_ONLY\nGuideline evidence with café."
        self.plan = "## Plan\n1. MOCK_FINAL_PLAN"
        self.candidates = [{"id": "candidate-one"}, {"id": "candidate-two"}]
        self.results = ["SELECTED_SIMILAR_CASE_AND_PLAN", "UNSELECTED_SIMILAR_CASE"]
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
        self.retrieval_calls = []
        self.rerank_calls = []
        self.graph_calls = []
        self.failure = None
        self.error = RuntimeError("mock stage failure")
        self.stdout = io.StringIO()
        self.modules = self._modules()

    def fail_at(self, stage):
        if self.failure == stage:
            raise self.error

    def _modules(self):
        fixture = self

        def preload(model_name):
            fixture.events.append(("preload", model_name))
            return object()

        def completion(**kwargs):
            fixture.calls.append(kwargs)
            stage = ["normalization", "questions", "final"][len(fixture.calls) - 1]
            fixture.fail_at(stage)
            content = [json.dumps(fixture.patient), fixture.questions, fixture.plan][len(fixture.calls) - 1]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        def client(api_key):
            fixture.events.append(("client", api_key))
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))

        class HybridRetriever:
            def __init__(self, df, alpha):
                fixture.retrieval_calls.append(("init", df.copy(), alpha, Path.cwd()))
                fixture.fail_at("retriever_init")

            def search(self, query, topk):
                fixture.retrieval_calls.append(("search", query, topk, Path.cwd()))
                fixture.fail_at("retrieval")
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
            return original_import(name, *args, **kwargs)

        old_cwd = Path.cwd()
        os.chdir(self.caller)
        try:
            with patch.dict(sys.modules, self.modules), \
                    patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                    patch.object(sys, "path", sys.path.copy()), \
                    patch("builtins.__import__", side_effect=tracked_import), \
                    redirect_stdout(self.stdout):
                yield
        finally:
            os.chdir(old_cwd)


class RunMedCopilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = PipelineFixture(Path(self.temporary.name))

    def load_runner(self):
        return runpy.run_path(str(self.fixture.script), run_name="runner_under_test")

    def test_import_is_inert_without_api_key_or_model_dependencies(self):
        original_import = builtins.__import__

        def forbid_models(name, *args, **kwargs):
            if name.split(".")[0] in {"sentence_transformers", "openai", "create_embeddings", "graphrag", "templates"}:
                raise AssertionError(f"Model dependency imported during import: {name}")
            return original_import(name, *args, **kwargs)

        original_path = sys.path.copy()
        original_cwd = Path.cwd()
        with patch.dict(os.environ, {}, clear=True), \
                patch("builtins.__import__", side_effect=forbid_models), \
                patch.object(Path, "read_text", side_effect=AssertionError("Patient file read on import")), \
                patch.object(Path, "mkdir", side_effect=AssertionError("Output created on import")):
            runner = self.load_runner()
        self.assertTrue(callable(runner["main"]))
        self.assertEqual(runner["PATIENT_CORPUS"],
                         self.fixture.upstream / "soap_with_metadata.json")
        self.assertEqual(sys.path, original_path)
        self.assertEqual(Path.cwd(), original_cwd)
        self.assertFalse(self.fixture.output.exists())

    def assert_outputs(self):
        fixture = self.fixture
        expected_prompt = PROMPTS["EVALUATE_TEMPLATE_KEYINFO"].format(
            conditions=fixture.conditions, example=fixture.results[0], Key_info=fixture.guideline_text,
        )
        expected_text = {
            "case_truncated_freetext.txt": fixture.patient_input,
            "case_SOA.txt": fixture.conditions,
            "top_similar_patient.txt": fixture.results[0],
            "questions_for_guidelines_db.txt": fixture.questions,
            "answers_from_guidelines_db.txt": fixture.guideline_text,
            "final_prompt.txt": expected_prompt,
            "final_output.txt": fixture.plan,
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
        self.assertEqual(fixture.calls[-1]["messages"][-1]["content"], expected_prompt)
        self.assertNotIn("RAW_CONTEXT_ONLY", expected_prompt)
        self.assertNotIn("UNSELECTED_SIMILAR_CASE", expected_prompt)
        self.assertNotIn("RAW_CONTEXT_ONLY", fixture.stdout.getvalue())

    def test_full_main_preserves_pipeline_contract_and_saves_evidence(self):
        fixture = self.fixture
        runner = self.load_runner()
        with fixture.runtime():
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(fixture.events[:2], [
            ("preload", "emilyalsentzer/Bio_ClinicalBERT"), ("import", "create_embeddings"),
        ])
        self.assertIn(("client", "test-key"), fixture.events)
        self.assertEqual(len(fixture.calls), 3)
        for call in fixture.calls:
            self.assertEqual(call["model"], "gpt-4o-mini")
        self.assertEqual(fixture.calls[0]["messages"], [
            {"role": "system", "content": PROMPTS["PATIENT_CASE_SYSTEM_TEMPLATE"]},
            {"role": "user", "content": PROMPTS["PATIENT_CASE_TEMPLATE"].format(patient_case=fixture.patient_input)},
        ])
        self.assertNotIn("HIDDEN_TARGET_PLAN", fixture.calls[0]["messages"][-1]["content"])
        self.assertNotIn("EXCLUDED_LATER_ENCOUNTER", fixture.calls[0]["messages"][-1]["content"])
        self.assertEqual(fixture.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(fixture.calls[1]["response_format"], {"type": "json_object"})
        self.assertEqual(fixture.calls[1]["messages"][-1]["content"],
                         PROMPTS["KEY_QUESTIONS_TEMPLATE"].format(conditions=fixture.conditions))
        self.assertNotIn("response_format", fixture.calls[2])
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
        self.assert_outputs()

    def test_script_entry_point_runs_the_full_pipeline(self):
        with self.fixture.runtime():
            runpy.run_path(str(self.fixture.script), run_name="__main__")
            self.assertEqual(Path.cwd(), self.fixture.caller)
        self.assertEqual(len(self.fixture.calls), 3)
        self.assert_outputs()

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
        self.assertEqual(len(fixture.calls), 3)
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
        self.assertEqual(len(fixture.calls), 1)
        self.assertEqual(fixture.retrieval_calls, [])
        self.assertEqual(fixture.rerank_calls, [])
        self.assertEqual(fixture.graph_calls, [])
        self.assertEqual({path.name for path in fixture.output.iterdir()},
                         {"case_truncated_freetext.txt", "case_SOA.txt"})
        self.assertEqual((fixture.output / "case_truncated_freetext.txt").read_text(),
                         fixture.patient_input)
        self.assertEqual((fixture.output / "case_SOA.txt").read_text(), fixture.conditions)
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
            ("normalization", 1), ("retriever_init", 1), ("retrieval", 1),
            ("reranker_init", 1), ("reranking", 1), ("questions", 2),
            ("graphrag", 2), ("final", 3),
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


if __name__ == "__main__":
    unittest.main()

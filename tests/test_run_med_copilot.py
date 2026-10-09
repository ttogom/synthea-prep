"""Offline runner tests; activate an environment with pandas first.

Run directly: python3 tests/test_run_med_copilot.py
Or discover: python3 -m unittest discover -s tests -p 'test_*.py'
Models and API clients are mocked; upstream modules are never imported.
Normalized S/O is trusted; schema and factual validation are outside this runner.
"""

import ast
import builtins
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_med_copilot.py"
PROJECT_PROMPTS = ROOT / "med_copilot/prompts/diagnosis_plan.py"
NOTE_NAME = "Peter292_Gleichner915_0db4522b-544a-43b1-ad63-3caeb72be2ab.txt"
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
DEFAULT_NORMALIZATION_RESPONSE = object()


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
        self.prompt_module = root / "med_copilot/prompts/diagnosis_plan.py"
        self.prompt_module.parent.mkdir(parents=True)
        self.prompt_module.write_text(PROJECT_PROMPTS.read_text(encoding="utf-8"), encoding="utf-8")
        self.prompts = PROMPTS.copy()
        # Prepared input is consumed in full, including text after the old
        # smoke-test heading/date boundaries and its surrounding whitespace.
        self.patient_input = (
            " \n2000-07-20\n# Chief Complaint\nCough for 3 days; denies fever. café."
            "\n# Objective\nTemperature 37.2 °C; SpO₂ 98%; weight 70 kg on 2000-07-20."
            "\n# Assessment and Plan\nSUPPLIED_SECTION_EVIDENCE café"
            "\n\n2000-07-02\nLATER_ENCOUNTER_EVIDENCE"
            "\n\n2001-01-03\nEND_OF_RECORD_EVIDENCE\n \n"
        )
        self.note = root / "data_run_med_copilot/input" / NOTE_NAME
        self.note.write_text(self.patient_input, encoding="utf-8")
        self.corpus = [{"subjective": "Corpus S", "objective": "Corpus O",
                        "assessment": "Corpus A", "plan": "Corpus P"}]
        (self.upstream / "soap_with_metadata.json").write_text(json.dumps(self.corpus))
        self.patient = {
            "subjective": "2000-07-20: Cough for 3 days; denies fever. café.",
            "objective": "Temperature 37.2 °C; SpO₂ 98%; weight 70 kg on 2000-07-20.",
        }
        self.conditions = (
            f'Subjective: {self.patient["subjective"]}\n'
            f'Objective: {self.patient["objective"]}'
        )
        self.normalization_response = DEFAULT_NORMALIZATION_RESPONSE
        self.questions = '{"Key Questions": ["Question one?", "Question two?"]}'
        self.guideline_text = "GUIDELINE_RESPONSE_ONLY\nGuideline evidence with café."
        self.plan = "## Plan\n1. MOCK_FINAL_PLAN"
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
            if stage == "normalization" and fixture.normalization_response is not DEFAULT_NORMALIZATION_RESPONSE:
                content = fixture.normalization_response
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
                # Exercise this fixture's real project prompts, even when a
                # caller has already imported the repository's package.
                for name in ["med_copilot.prompts.diagnosis_plan", "med_copilot.prompts", "med_copilot"]:
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
        return runpy.run_path(str(self.fixture.script), run_name="runner_under_test")

    def test_import_is_inert_without_api_key_or_model_dependencies(self):
        original_import = builtins.__import__

        def forbid_models(name, *args, **kwargs):
            if name.split(".")[0] in {"sentence_transformers", "openai", "create_embeddings", "graphrag", "templates", "med_copilot"}:
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
        expected_prompt = fixture.prompts["EVALUATE_TEMPLATE_KEYINFO"].format(
            conditions=fixture.conditions, example=fixture.results[0], Key_info=fixture.guideline_text,
        )
        expected_text = {
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
        self.assertEqual(fixture.calls[0]["messages"][-1]["content"],
                         fixture.prompts["PATIENT_CASE_TEMPLATE"].format(patient_case=fixture.patient_input))
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
        for evidence in ["SUPPLIED_SECTION_EVIDENCE café", "LATER_ENCOUNTER_EVIDENCE", "END_OF_RECORD_EVIDENCE"]:
            self.assertIn(evidence, fixture.calls[0]["messages"][-1]["content"])
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
        self.assertIn("STAGE 1: S/O", fixture.stdout.getvalue())
        self.assertNotIn("STAGE 1: S/O/A", fixture.stdout.getvalue())
        self.assertNotIn("Assessment:", fixture.conditions)
        for field in fixture.patient.values():
            self.assertIn(field, fixture.retrieval_calls[1][1])
            self.assertIn(field, fixture.rerank_calls[1][1])
            self.assertIn(field, fixture.calls[1]["messages"][-1]["content"])
            self.assertIn(field, fixture.calls[2]["messages"][-1]["content"])
        self.assertNotIn("REFERENCE_DIAGNOSIS", fixture.retrieval_calls[1][1])
        self.assertNotIn("REFERENCE_DIAGNOSIS", fixture.rerank_calls[1][1])
        self.assertNotIn("REFERENCE_DIAGNOSIS", fixture.calls[1]["messages"][-1]["content"])
        self.assertIn("Assessment: REFERENCE_DIAGNOSIS", fixture.calls[2]["messages"][-1]["content"])
        self.assertIn("Plan: REFERENCE_PLAN", fixture.calls[2]["messages"][-1]["content"])
        self.assert_outputs()

    def test_script_entry_point_runs_the_full_pipeline(self):
        with self.fixture.runtime():
            runpy.run_path(str(self.fixture.script), run_name="__main__")
            self.assertEqual(Path.cwd(), self.fixture.caller)
        self.assertEqual(len(self.fixture.calls), 3)
        self.assert_outputs()

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
                fixture.patient_input = patient_text
                fixture.note.write_text(patient_text, encoding="utf-8")
                runner = self.load_runner()
                with fixture.runtime():
                    runner["main"]()
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertEqual(len(fixture.calls), 3)
                self.assertIn(patient_text, fixture.stdout.getvalue())
                self.assert_outputs()

    def test_trusted_so_preserves_empty_sections_and_field_text(self):
        fixture_root = self.fixture.root
        cases = {
            "no_subjective": {"subjective": "", "objective": self.fixture.patient["objective"]},
            "no_objective": {"subjective": self.fixture.patient["subjective"], "objective": ""},
            "whitespace_section": {"subjective": " \n\t", "objective": self.fixture.patient["objective"]},
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
                fixture.conditions = (
                    f'Subjective: {patient["subjective"]}\n'
                    f'Objective: {patient["objective"]}'
                )
                runner = self.load_runner()
                with fixture.runtime():
                    runner["main"]()
                    self.assertEqual(Path.cwd(), fixture.caller)
                self.assertEqual(len(fixture.calls), 3)
                self.assertEqual(fixture.retrieval_calls[1][1], fixture.conditions)
                self.assertEqual(fixture.rerank_calls[1][1], fixture.conditions)
                self.assertEqual(fixture.calls[1]["messages"][-1]["content"],
                                 PROMPTS["KEY_QUESTIONS_TEMPLATE"].format(conditions=fixture.conditions))
                self.assert_outputs()

    def test_only_so_fields_from_trusted_response_reach_downstream(self):
        fixture = self.fixture
        fixture.patient.update({
            "assessment": "UNUSED_TARGET_ASSESSMENT",
            "diagnosis": "UNUSED_TARGET_DIAGNOSIS",
            "plan": "UNUSED_TARGET_PLAN",
            "metadata": {"source": "external normalization"},
        })
        runner = self.load_runner()
        with fixture.runtime():
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(len(fixture.calls), 3)
        self.assertEqual(fixture.retrieval_calls[1][1], fixture.conditions)
        self.assertEqual(fixture.rerank_calls[1][1], fixture.conditions)
        for call in fixture.calls[1:]:
            self.assertNotIn("UNUSED_TARGET", call["messages"][-1]["content"])
        self.assertNotIn("UNUSED_TARGET", fixture.stdout.getvalue())
        self.assert_outputs()

    def test_undecodable_or_unusable_so_response_stops_before_retrieval(self):
        fixture_root = self.fixture.root
        valid = self.fixture.patient
        # These are ordinary decoding/key-access failures, not schema checks.
        cases = {
            "no_content": (None, TypeError),
            "nonstring_content": (valid, TypeError),
            "blank_content": ("", json.JSONDecodeError),
            "whitespace_content": (" \n\t", json.JSONDecodeError),
            "malformed_json": ('{"subjective": "cough",', json.JSONDecodeError),
            "markdown_wrapped_json": ("```json\n" + json.dumps(valid) + "\n```", json.JSONDecodeError),
            "json_null": ("null", TypeError),
            "json_array": (json.dumps([valid]), TypeError),
            "empty_object": ("{}", KeyError),
            "missing_subjective": (json.dumps({"objective": "measured evidence"}), KeyError),
            "missing_objective": (json.dumps({"subjective": "reported evidence"}), KeyError),
            "wrong_case": (json.dumps({"Subjective": "reported evidence", "objective": "measured evidence"}), KeyError),
        }
        for name, (raw_response, exception_type) in cases.items():
            with self.subTest(case=name):
                self.fixture = PipelineFixture(fixture_root / name)
                fixture = self.fixture
                fixture.normalization_response = raw_response
                runner = self.load_runner()
                with fixture.runtime():
                    with self.assertRaises(exception_type) as caught:
                        runner["main"]()
                    self.assertEqual(Path.cwd(), fixture.caller)
                if exception_type is KeyError:
                    self.assertEqual(caught.exception.args[0],
                                     "objective" if name == "missing_objective" else "subjective")
                self.assertEqual(len(fixture.calls), 1)
                self.assertEqual(fixture.retrieval_calls, [])
                self.assertEqual(fixture.rerank_calls, [])
                self.assertEqual(fixture.graph_calls, [])
                self.assertEqual(list(fixture.output.iterdir()), [])
                self.assertEqual(fixture.note.read_text(encoding="utf-8"), fixture.patient_input)
                self.assertNotIn("STAGE 1: S/O", fixture.stdout.getvalue())

    def test_final_prompt_remains_verbatim_copy_of_upstream(self):
        upstream_prompts = prompt_literals(ROOT / "med_copilot/upstream/templates/SOA_P_TEMPLATE.py")
        self.assertEqual(set(PROMPTS), PROMPT_NAMES)
        self.assertEqual(PROMPTS["EVALUATE_TEMPLATE_KEYINFO"], upstream_prompts["EVALUATE_TEMPLATE_KEYINFO"])

    def test_question_prompt_describes_so_input_without_assessment(self):
        rendered = PROMPTS["KEY_QUESTIONS_TEMPLATE"].format(conditions=self.fixture.conditions)
        self.assertRegex(rendered.lower(), r"\(subjective,\s*objective\)")
        self.assertNotIn("assessment", rendered.lower())
        self.assertIn(self.fixture.conditions, rendered)
        self.assertIn('"Key Questions"', rendered)

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
        self.assertIn("only facts explicitly stated in the patient record", instructions)
        # The prompt uses schematic (text) placeholders, not literal JSON values.
        format_example = system_prompt[system_prompt.index("{"):system_prompt.rindex("}") + 1]
        self.assertEqual(re.findall(r'"([^"]+)"\s*:', format_example), ["subjective", "objective"])

    def test_full_pipeline_uses_project_prompts_from_an_unrelated_cwd(self):
        fixture = self.fixture
        fixture.prompts = {name: f"PROJECT_{name} café\n{value}" for name, value in PROMPTS.items()}
        fixture.prompt_module.write_text(
            "\n".join(f"{name} = {value!r}" for name, value in fixture.prompts.items()),
            encoding="utf-8",
        )
        runner = self.load_runner()
        with fixture.runtime():
            runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
            self.assertEqual(Path(sys.modules["med_copilot.prompts.diagnosis_plan"].__file__),
                             fixture.prompt_module)
        self.assertEqual(fixture.calls[0]["messages"], [
            {"role": "system", "content": fixture.prompts["PATIENT_CASE_SYSTEM_TEMPLATE"]},
            {"role": "user", "content": fixture.prompts["PATIENT_CASE_TEMPLATE"].format(
                patient_case=fixture.patient_input)},
        ])
        self.assertEqual(fixture.calls[1]["messages"][-1]["content"],
                         fixture.prompts["KEY_QUESTIONS_TEMPLATE"].format(conditions=fixture.conditions))
        self.assertEqual(fixture.retrieval_calls[1][1], fixture.conditions)
        self.assertEqual(fixture.graph_calls[0][0]["query"], fixture.questions)
        self.assert_outputs()

    def test_missing_project_prompts_stops_without_using_upstream_templates(self):
        fixture = self.fixture
        fixture.prompt_module.unlink()
        runner = self.load_runner()
        with fixture.runtime():
            with self.assertRaises(ModuleNotFoundError) as caught:
                runner["main"]()
            self.assertEqual(Path.cwd(), fixture.caller)
        self.assertEqual(caught.exception.name, "med_copilot.prompts.diagnosis_plan")
        self.assertEqual(fixture.calls, [])
        self.assertEqual(fixture.retrieval_calls, [])
        self.assertEqual(fixture.graph_calls, [])
        self.assertFalse(fixture.output.exists())

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
                         {"case_SOA.txt"})
        self.assertEqual(fixture.calls[0]["messages"][-1]["content"],
                         fixture.prompts["PATIENT_CASE_TEMPLATE"].format(patient_case=fixture.patient_input))
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

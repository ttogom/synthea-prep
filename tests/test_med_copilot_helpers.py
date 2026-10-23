"""Offline unit tests for every project-owned MED-COPILOT helper.

Run directly: python tests/test_med_copilot_helpers.py
Or discover: python -m unittest discover -s tests -p 'test_*.py'
Only temporary files and mocked API responses are used; no model/runtime
initialization is needed. Keep this suite updated when helpers are added.
"""

import ast
import builtins
from datetime import datetime, timedelta, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "scripts/med_copilot_helpers.py"
UPSTREAM_RETRIEVER = ROOT / "med_copilot/upstream/create_embeddings.py"
GUIDELINE_TABLES = ("communities", "community_reports", "text_units", "relationships", "entities")
RESULT_FIELDS = [
    "id", "status", "assessment", "plan", "assessment_plan_raw", "selected_reference_patient",
    "guideline_questions", "guideline_evidence", "graphrag_context", "assessment_plan_prompts",
]
RUN_FIELDS = [
    "run_name", "started_at", "finished_at", "status", "branch", "input", "git_commit",
    "reference_data",
]


def frozen_keyword_extractor_class():
    # Execute only the dependency-free class, preserving its real prompt/parser
    # without loading the upstream model clients and runtime dependencies.
    tree = ast.parse(UPSTREAM_RETRIEVER.read_text(encoding="utf-8"))
    extractor = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "LLMKeywordExtractor")
    namespace = {"json": json}
    exec(compile(ast.Module(body=[extractor], type_ignores=[]),
                 str(UPSTREAM_RETRIEVER), "exec"), namespace)
    return namespace["LLMKeywordExtractor"]


FROZEN_KEYWORD_EXTRACTOR = frozen_keyword_extractor_class()
HELPER_NAMESPACE = runpy.run_path(str(HELPERS))
H = HELPER_NAMESPACE
LOAD_PATIENT_RECORDS = HELPER_NAMESPACE["load_patient_records"]
VALIDATE_GUIDELINE_TABLES = HELPER_NAMESPACE["validate_guideline_tables"]


def keyword_response(content, finish_reason="stop", refusal=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, refusal=refusal),
    )])


class LoadPatientRecordsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "patients café.jsonl"
        self.records = [
            {
                "id": "patient-1",
                "subjective": "  Cough café.\nDenies fever.\t ",
                "objective": "SpO₂ 98%; 37.2 °C; 左腕\n ",
                "metadata": {"encounter": 12, "tags": ["prepared", None]},
            },
            {"id": "patient-2 café", "subjective": "SECOND S", "objective": "SECOND O"},
        ]

    def write_records(self, records):
        self.input.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n"
                                      for record in records), encoding="utf-8")

    def assert_invalid(self, text, line_number, message, parse_error=False):
        self.input.write_text(text, encoding="utf-8")
        original = self.input.read_bytes()
        with self.assertRaises(ValueError) as caught:
            LOAD_PATIENT_RECORDS(self.input)
        error = caught.exception
        self.assertEqual(str(error), f"{self.input}:{line_number}: {message}")
        self.assertNotIn("PRIVATE_CLINICAL_TEXT", str(error))
        if parse_error:
            self.assertIsInstance(error.__cause__, json.JSONDecodeError)
        else:
            self.assertIsNone(error.__cause__)
        self.assertEqual(self.input.read_bytes(), original)
        self.assertEqual(set(self.root.iterdir()), {self.input})
        return error

    def test_loads_all_records_in_order_preserving_ids_metadata_and_clinical_text(self):
        self.write_records(self.records)
        original = self.input.read_bytes()
        patients = LOAD_PATIENT_RECORDS(self.input)
        self.assertEqual(patients, self.records)
        self.assertEqual(len(patients), 2)
        self.assertEqual(patients[0]["subjective"], self.records[0]["subjective"])
        self.assertEqual(patients[0]["objective"], self.records[0]["objective"])
        self.assertEqual([patient["id"] for patient in patients], ["patient-1", "patient-2 café"])
        self.assertEqual(self.input.read_bytes(), original)
        self.assertEqual(set(self.root.iterdir()), {self.input})

    def test_accepts_path_and_string_input_and_loads_fresh_records_each_time(self):
        self.write_records(self.records)
        for path in (self.input, str(self.input)):
            with self.subTest(path_type=type(path).__name__):
                self.assertEqual(LOAD_PATIENT_RECORDS(path), self.records)
        changed = [{"id": "updated", "subjective": "UPDATED S", "objective": "UPDATED O"}]
        self.write_records(changed)
        self.assertEqual(LOAD_PATIENT_RECORDS(self.input), changed)

    def test_relative_input_resolves_from_caller_and_does_not_change_cwd(self):
        self.write_records(self.records)
        original_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            self.assertEqual(LOAD_PATIENT_RECORDS(self.input.name), self.records)
            self.assertEqual(Path.cwd(), self.root.resolve())
        finally:
            os.chdir(original_cwd)

    def test_blank_lines_crlf_no_final_newline_and_json_indentation_are_supported(self):
        lines = ["", " \t", "  " + json.dumps(self.records[0], ensure_ascii=False) + " \t",
                 "", json.dumps(self.records[1], ensure_ascii=False)]
        self.input.write_bytes("\r\n".join(lines).encode("utf-8"))
        original = self.input.read_bytes()
        self.assertEqual(LOAD_PATIENT_RECORDS(self.input), self.records)
        self.assertEqual(self.input.read_bytes(), original)

    def test_empty_so_sections_are_valid_prepared_records(self):
        records = [{"id": "empty-sections", "subjective": "", "objective": " \n\t "}]
        self.write_records(records)
        self.assertEqual(LOAD_PATIENT_RECORDS(self.input), records)

    def test_empty_and_blank_only_files_report_no_records(self):
        for text in ("", "\n", " \t\r\n\n\t"):
            with self.subTest(contents=repr(text)):
                self.input.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError) as caught:
                    LOAD_PATIENT_RECORDS(self.input)
                self.assertEqual(str(caught.exception), f"{self.input}: no patient records found")
                self.assertEqual(self.input.read_text(encoding="utf-8"), text.replace("\r\n", "\n"))

    def test_missing_input_propagates_file_error_without_creating_files(self):
        with self.assertRaises(FileNotFoundError) as caught:
            LOAD_PATIENT_RECORDS(self.input)
        self.assertEqual(caught.exception.errno, errno.ENOENT)
        self.assertEqual(caught.exception.filename, str(self.input))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_directory_input_propagates_directory_error(self):
        self.input.mkdir()
        with self.assertRaises(IsADirectoryError) as caught:
            LOAD_PATIENT_RECORDS(self.input)
        self.assertEqual(caught.exception.filename, str(self.input))
        self.assertEqual(list(self.input.iterdir()), [])

    def test_read_permission_error_propagates_unchanged(self):
        error = PermissionError(errno.EACCES, "Permission denied", str(self.input))
        # Mock permissions instead of chmod: this remains reliable for root users.
        with patch.object(Path, "open", side_effect=error) as open_file:
            with self.assertRaises(PermissionError) as caught:
                LOAD_PATIENT_RECORDS(self.input)
        self.assertIs(caught.exception, error)
        open_file.assert_called_once_with("r", encoding="utf-8")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_utf8_propagates_decode_error_and_preserves_source(self):
        original = b'{"subjective": "\xff", "objective": "O"}\n'
        self.input.write_bytes(original)
        with self.assertRaises(UnicodeDecodeError):
            LOAD_PATIENT_RECORDS(self.input)
        self.assertEqual(self.input.read_bytes(), original)

    def test_malformed_first_record_reports_path_physical_line_and_json_cause(self):
        malformed = '{"subjective": "PRIVATE_CLINICAL_TEXT", "objective": }'
        parse_error = None
        try:
            json.loads(malformed)
        except json.JSONDecodeError as exc:
            parse_error = exc
        error = self.assert_invalid(
            " \n\t\n" + malformed + "\n", 3,
            f"invalid patient JSON: {parse_error.msg}", parse_error=True,
        )
        self.assertEqual(error.__cause__.doc, malformed + "\n")
        self.assertEqual(error.__cause__.pos, parse_error.pos)

    def test_malformed_later_record_does_not_return_partial_patients(self):
        first = json.dumps(self.records[0], ensure_ascii=False)
        malformed = '{"subjective":"PRIVATE_CLINICAL_TEXT","objective":'
        parse_error = None
        try:
            json.loads(malformed)
        except json.JSONDecodeError as exc:
            parse_error = exc
        self.assert_invalid(first + "\n\n" + malformed, 3,
                            f"invalid patient JSON: {parse_error.msg}", parse_error=True)

    def test_json_values_other_than_objects_are_rejected_including_patient_arrays(self):
        for value in (None, [], self.records, "PRIVATE_CLINICAL_TEXT", 7, 2.5, True):
            with self.subTest(value_type=type(value).__name__):
                self.assert_invalid("\n" + json.dumps(value) + "\n", 2,
                                    "patient must be a JSON object")

    def test_each_required_so_field_rejects_missing_and_nonstring_values(self):
        valid = {"id": "patient", "subjective": "PRIVATE_CLINICAL_TEXT",
                 "objective": "PRIVATE_CLINICAL_TEXT"}
        for field in ("subjective", "objective"):
            payloads = [{key: value for key, value in valid.items() if key != field}]
            payloads.extend({**valid, field: value} for value in (None, [], {}, 3, 2.5, True))
            for record in payloads:
                with self.subTest(field=field, value_type=type(record.get(field)).__name__):
                    self.assert_invalid(json.dumps(record) + "\n", 1,
                                        f"{field} must be a string")

    def test_invalid_later_schema_is_validated_before_any_records_are_returned(self):
        first = json.dumps(self.records[0], ensure_ascii=False)
        self.assert_invalid(first + '\n\n{"id": "later", "subjective": "PRIVATE_CLINICAL_TEXT"}\n', 3,
                            "objective must be a string")

    def test_id_must_be_a_nonempty_string(self):
        valid = {"subjective": "PRIVATE_CLINICAL_TEXT", "objective": "PRIVATE_CLINICAL_TEXT"}
        payloads = [valid, {"uuid": "legacy-key", **valid}]
        payloads.extend({"id": value, **valid} for value in ("", " \t\n", None, 7, 2.5, True, [], {}))
        for record in payloads:
            with self.subTest(id=repr(record.get("id", "<missing>"))):
                self.assert_invalid("\n" + json.dumps(record) + "\n", 2, "id must be a nonempty string")

    def test_id_is_validated_before_so_fields(self):
        self.assert_invalid('{"objective": 3}\n', 1, "id must be a nonempty string")

    def test_duplicate_ids_report_both_lines_and_return_no_records(self):
        first = json.dumps(self.records[0], ensure_ascii=False)
        second = json.dumps({**self.records[1], "id": "patient-1"}, ensure_ascii=False)
        self.assert_invalid(first + "\n\n" + json.dumps(self.records[1], ensure_ascii=False)
                            + "\n" + second + "\n", 4,
                            "duplicate id 'patient-1' (first used on line 1)")

    def test_ids_are_compared_exactly(self):
        records = [{**self.records[0], "id": "Patient"}, {**self.records[1], "id": "patient"},
                   {**self.records[1], "id": " patient"}]
        self.write_records(records)
        self.assertEqual([patient["id"] for patient in LOAD_PATIENT_RECORDS(self.input)],
                         ["Patient", "patient", " patient"])


class ValidateGuidelineTablesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "selected guidelines" / "output"
        self.output.mkdir(parents=True)

    def write_tables(self, directory=None):
        directory = self.output if directory is None else directory
        directory.mkdir(parents=True, exist_ok=True)
        for name in GUIDELINE_TABLES:
            # The helper checks presence; parquet parsing belongs to GraphRAG.
            (directory / f"{name}.parquet").touch()

    def assert_missing(self, path, table):
        with self.assertRaises(FileNotFoundError) as caught:
            VALIDATE_GUIDELINE_TABLES(path)
        self.assertEqual(caught.exception.errno, errno.ENOENT)
        self.assertEqual(caught.exception.filename, str(Path(path) / f"{table}.parquet"))
        self.assertEqual(caught.exception.strerror, "Required guideline table not found")

    def test_all_five_tables_are_accepted_for_path_and_string_inputs(self):
        self.write_tables()
        for path in (self.output, str(self.output)):
            with self.subTest(path_type=type(path).__name__):
                self.assertIsNone(VALIDATE_GUIDELINE_TABLES(path))

    def test_each_required_table_is_independently_required(self):
        self.write_tables()
        for name in GUIDELINE_TABLES:
            with self.subTest(table=name):
                missing = self.output / f"{name}.parquet"
                missing.unlink()
                self.assert_missing(self.output, name)
                self.assertFalse(missing.exists())
                missing.touch()

    def test_directory_named_like_a_required_table_is_rejected(self):
        self.write_tables()
        for name in GUIDELINE_TABLES:
            with self.subTest(table=name):
                table = self.output / f"{name}.parquet"
                table.unlink()
                table.mkdir()
                self.assert_missing(self.output, name)
                self.assertEqual(list(table.iterdir()), [])
                table.rmdir()
                table.touch()

    def test_missing_output_directory_is_not_created(self):
        missing = self.root / "does not exist"
        self.assert_missing(missing, "communities")
        self.assertFalse(missing.exists())

    def test_file_in_place_of_output_directory_is_rejected(self):
        output_file = self.root / "output-file"
        output_file.write_text("unchanged", encoding="utf-8")
        self.assert_missing(output_file, "communities")
        self.assertEqual(output_file.read_text(encoding="utf-8"), "unchanged")

    def test_nested_sibling_and_caller_tables_cannot_mask_a_missing_selected_table(self):
        self.write_tables()
        self.write_tables(self.output / "nested")
        self.write_tables(self.output.parent / "other output")
        caller = self.root / "caller"
        self.write_tables(caller)
        missing = self.output / "entities.parquet"
        missing.unlink()
        original_cwd = Path.cwd()
        try:
            os.chdir(caller)
            self.assert_missing(self.output, "entities")
            self.assertEqual(Path.cwd(), caller.resolve())
        finally:
            os.chdir(original_cwd)
        self.assertFalse(missing.exists())

    def test_optional_covariates_are_not_required_and_unrelated_files_are_untouched(self):
        self.write_tables()
        unrelated = self.output / "README.txt"
        unrelated.write_text("Keep this content", encoding="utf-8")
        self.assertFalse((self.output / "covariates.parquet").exists())
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        original_cwd = Path.cwd()
        self.assertIsNone(VALIDATE_GUIDELINE_TABLES(self.output))
        self.assertEqual(Path.cwd(), original_cwd)
        self.assertEqual({path.name: path.read_bytes() for path in self.output.iterdir()}, before)

    def test_relative_output_uses_selected_directory_and_preserves_caller_cwd(self):
        self.write_tables()
        original_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            self.assertIsNone(VALIDATE_GUIDELINE_TABLES(self.output.relative_to(self.root)))
            self.assertEqual(Path.cwd(), self.root.resolve())
        finally:
            os.chdir(original_cwd)


class HelperImportTests(unittest.TestCase):
    def test_loading_helpers_does_not_import_model_clients_or_heavy_runtime_dependencies(self):
        original_import = builtins.__import__
        blocked = {"openai", "create_embeddings", "graphrag", "transformers",
                   "sentence_transformers", "spacy", "torch", "faiss", "pandas", "numpy"}

        def offline_import(name, *args, **kwargs):
            if name.split(".")[0] in blocked:
                raise AssertionError(f"Helper import must remain offline: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=offline_import):
            namespace = runpy.run_path(str(HELPERS))
        public = [name for name, value in namespace.items()
                  if callable(value) and not name.startswith("_")
                  and getattr(value, "__module__", None) == "<run_path>"]
        self.assertEqual(sorted(public), sorted([
            "ExperimentRun", "JSONKeywordExtractor", "create_run_dir", "git_branch", "git_commit",
            "graphrag_context_json", "guideline_resource_files", "load_patient_records",
            "make_run_name", "new_patient_result", "pacific_now", "pacific_time",
            "parse_guideline_questions", "path_safe_branch", "reference_data_metadata",
            "repo_path", "run_status", "selected_reference_patient", "sha256_bundle",
            "sha256_file", "split_assessment_plan", "validate_guideline_tables",
            "write_run_json",
        ]), "Add direct tests when adding a public helper")

    def test_all_helper_imports_are_in_the_module_header(self):
        tree = ast.parse(HELPERS.read_text(encoding="utf-8"))
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(len(imports), len(top_imports), "Keep imports out of functions/classes")
        definitions = [node.lineno for node in tree.body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        self.assertTrue(all(node.lineno < min(definitions) for node in imports),
                        "Keep imports before helper definitions")


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


def run_git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.com",
         "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


class TemporaryRootTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()


class RunNamingTests(unittest.TestCase):
    def test_pacific_time_uses_pdt_and_pst_and_drops_microseconds(self):
        for instant, expected, suffix in [
            (datetime(2026, 10, 17, 22, 30, 45, 999999, tzinfo=timezone.utc),
             "2026-10-17T15:30:45-07:00", "PDT"),
            (datetime(2026, 1, 15, 17, 30, 0, tzinfo=timezone.utc), "2026-01-15T09:30:00-08:00", "PST"),
            # Around the 2026-03-08 and 2026-11-01 transitions.
            (datetime(2026, 3, 8, 9, 59, 59, tzinfo=timezone.utc), "2026-03-08T01:59:59-08:00", "PST"),
            (datetime(2026, 3, 8, 10, 0, 0, tzinfo=timezone.utc), "2026-03-08T03:00:00-07:00", "PDT"),
            (datetime(2026, 11, 1, 8, 59, 59, tzinfo=timezone.utc), "2026-11-01T01:59:59-07:00", "PDT"),
            (datetime(2026, 11, 1, 9, 0, 0, tzinfo=timezone.utc), "2026-11-01T01:00:00-08:00", "PST"),
        ]:
            with self.subTest(instant=instant):
                local = H["pacific_time"](instant)
                self.assertEqual(local.isoformat(), expected)
                self.assertEqual(local.tzname(), suffix)
        tokyo = datetime(2026, 7, 1, 9, 0, tzinfo=timezone(timedelta(hours=9)))
        self.assertEqual(H["pacific_time"](tokyo).isoformat(), "2026-06-30T17:00:00-07:00")

    def test_pacific_now_is_current_whole_second_los_angeles_time(self):
        before = datetime.now(timezone.utc).replace(microsecond=0)
        now = H["pacific_now"]()
        after = datetime.now(timezone.utc)
        self.assertEqual(str(now.tzinfo), "America/Los_Angeles")
        self.assertEqual(now.microsecond, 0)
        self.assertTrue(before <= now <= after)

    def test_path_safe_branch_keeps_safe_characters_and_replaces_others(self):
        for branch, expected in [
            ("integrate-medcopilot", "integrate-medcopilot"),
            ("feature/issue-9", "feature-issue-9"),
            ("user/feature//nested", "user-feature-nested"),
            ("release_1.2", "release_1.2"),
            ("fix: spaces & café", "fix-spaces-caf"),
            ("../escape", "escape"),
            ("HEAD", "HEAD"),
        ]:
            with self.subTest(branch=branch):
                safe = H["path_safe_branch"](branch)
                self.assertEqual(safe, expected)
                self.assertEqual(Path(safe).parts, (safe,))
        for branch in ("", "/", "..", "///"):
            with self.subTest(branch=branch), self.assertRaises(ValueError):
                H["path_safe_branch"](branch)

    def test_make_run_name_joins_pacific_timestamp_input_stem_and_branch(self):
        pdt = H["pacific_time"](datetime(2026, 10, 17, 22, 30, 45, tzinfo=timezone.utc))
        pst = H["pacific_time"](datetime(2026, 1, 15, 17, 30, 0, tzinfo=timezone.utc))
        self.assertEqual(H["make_run_name"](pdt, Path("experiments/input/patient_records_so.jsonl"),
                                            "integrate-medcopilot"),
                         "2026-10-17_153045PDT__patient_records_so__integrate-medcopilot")
        self.assertEqual(H["make_run_name"](pst, "/data/batch.v2.jsonl", "feature/issue-9"),
                         "2026-01-15_093000PST__batch.v2__feature-issue-9")


class GitIdentityTests(TemporaryRootTestCase):
    def make_repo(self, branch="feature/issue-9", commit=True):
        repo = self.root / "repo"
        subprocess.run(["git", "init", "-q", "-b", branch, str(repo)], check=True, capture_output=True)
        if commit:
            run_git(repo, "commit", "-q", "--allow-empty", "-m", "fixture")
        return repo

    def test_branch_and_commit_come_from_the_selected_repository(self):
        repo = self.make_repo()
        original_cwd = Path.cwd()
        try:
            os.chdir(self.root)  # Not the repository: lookups must not use the cwd.
            self.assertEqual(H["git_branch"](repo), "feature/issue-9")
            self.assertEqual(H["git_branch"](str(repo)), "feature/issue-9")
            self.assertEqual(H["git_commit"](repo), run_git(repo, "rev-parse", "HEAD"))
            self.assertRegex(H["git_commit"](repo), r"^[0-9a-f]{40}$")
        finally:
            os.chdir(original_cwd)

    def test_detached_head_is_reported_as_head(self):
        repo = self.make_repo()
        run_git(repo, "checkout", "-q", "--detach")
        self.assertEqual(H["git_branch"](repo), "HEAD")
        self.assertEqual(H["git_commit"](repo), run_git(repo, "rev-parse", "HEAD"))

    def test_unborn_branch_has_a_name_but_no_commit(self):
        repo = self.make_repo(commit=False)
        self.assertEqual(H["git_branch"](repo), "feature/issue-9")
        self.assertIsNone(H["git_commit"](repo))

    def test_non_repository_and_repository_subdirectory_fail_clearly(self):
        outside = self.root / "outside"
        outside.mkdir()
        with self.assertRaisesRegex(RuntimeError, "Cannot identify the Git repository"):
            H["git_branch"](outside)
        self.assertIsNone(H["git_commit"](outside))
        nested = self.make_repo() / "scripts"
        nested.mkdir()
        with self.assertRaisesRegex(RuntimeError, "Cannot identify the Git repository"):
            H["git_branch"](nested)

    def test_missing_git_executable_fails_branch_lookup_and_nulls_commit(self):
        repo = self.make_repo()
        with patch.dict(os.environ, {"PATH": str(self.root / "no-git")}):
            with self.assertRaisesRegex(RuntimeError, "git executable was not found"):
                H["git_branch"](repo)
            self.assertIsNone(H["git_commit"](repo))


class CreateRunDirTests(TemporaryRootTestCase):
    def test_creates_missing_output_root_and_a_new_run_directory(self):
        output = self.root / "experiments/output"
        run_dir = H["create_run_dir"](output, "2026-10-17_153045PDT__in__main")
        self.assertEqual(run_dir, output / "2026-10-17_153045PDT__in__main")
        self.assertTrue(run_dir.is_dir())
        self.assertEqual(list(run_dir.iterdir()), [])

    def test_existing_run_directory_is_reported_and_left_unchanged(self):
        output = self.root / "output"
        existing = output / "same-name"
        existing.mkdir(parents=True)
        (existing / "run.json").write_text("PREVIOUS", encoding="utf-8")
        with self.assertRaises(FileExistsError) as caught:
            H["create_run_dir"](output, "same-name")
        self.assertEqual(caught.exception.filename, str(existing))
        self.assertIn("refusing to overwrite", str(caught.exception))
        self.assertEqual((existing / "run.json").read_text(encoding="utf-8"), "PREVIOUS")
        self.assertEqual([path.name for path in existing.iterdir()], ["run.json"])


class FingerprintTests(TemporaryRootTestCase):
    def test_repo_path_is_relative_inside_root_and_absolute_outside(self):
        repo = self.root / "repo"
        (repo / "med_copilot").mkdir(parents=True)
        self.assertEqual(H["repo_path"](repo / "med_copilot/corpus café.json", repo),
                         "med_copilot/corpus café.json")
        self.assertEqual(H["repo_path"](repo, repo), ".")
        self.assertEqual(H["repo_path"](self.root / "external/data", repo),
                         str(self.root / "external/data"))
        self.assertEqual(H["repo_path"](repo / "med_copilot/../../outside", repo), str(self.root / "outside"))

    def test_sha256_file_matches_hashlib(self):
        path = self.root / "data.bin"
        path.write_bytes("\x00binary café".encode() * 200000)
        self.assertEqual(H["sha256_file"](path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_bundle_changes_with_contents_labels_members_and_order(self):
        file_a, directory = self.root / "a.txt", self.root / "index"
        file_a.write_text("A")
        (directory / "nested").mkdir(parents=True)
        (directory / "nested/part.bin").write_bytes(b"P")
        entries = [("a", file_a), ("index", directory)]
        baseline = H["sha256_bundle"](entries)
        self.assertRegex(baseline, r"^[0-9a-f]{64}$")
        self.assertEqual(H["sha256_bundle"](entries), baseline)
        self.assertNotEqual(H["sha256_bundle"]([("renamed", file_a), ("index", directory)]), baseline)
        self.assertNotEqual(H["sha256_bundle"](list(reversed(entries))), baseline)
        changes = [
            lambda: file_a.write_text("A2"),
            lambda: (directory / "nested/part.bin").write_bytes(b"P2"),
            lambda: (directory / "new.bin").write_bytes(b"N"),
            lambda: (directory / "new.bin").rename(directory / "renamed.bin"),
        ]
        previous = baseline
        for change in changes:
            change()
            current = H["sha256_bundle"](entries)
            self.assertNotEqual(current, previous)
            previous = current

    def test_bundle_is_null_when_any_file_is_missing(self):
        present = self.root / "present"
        present.write_text("x")
        self.assertIsNone(H["sha256_bundle"]([("present", present), ("missing", self.root / "missing")]))
        self.assertEqual(H["sha256_bundle"]([]), hashlib.sha256().hexdigest())

    def write_guidelines(self, directory, settings=None, covariates=False):
        if settings is None:
            settings = {
                "local_search": {"prompt": "prompts/local.txt"},
                "global_search": {"map_prompt": "prompts/unused.txt"},
                "vector_store": {"default_vector_store": {
                    "type": "lancedb", "db_uri": "output/lancedb", "container_name": "default"}},
            }
        (directory / "prompts").mkdir(parents=True, exist_ok=True)
        (directory / "output/lancedb/default-entity-description.lance").mkdir(parents=True, exist_ok=True)
        (directory / "settings.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
        (directory / "prompts/local.txt").write_text("LOCAL")
        (directory / "prompts/unused.txt").write_text("UNUSED")
        (directory / "output/lancedb/default-entity-description.lance/f.lance").write_text("V")
        (directory / "output/lancedb/other-table.lance").mkdir(exist_ok=True)
        for name in GUIDELINE_TABLES + (("covariates",) if covariates else ()):
            (directory / f"output/{name}.parquet").write_text(name)
        return directory

    def test_guideline_files_follow_local_search_configuration_only(self):
        guidelines = self.write_guidelines(self.root / "guidelines")
        files = H["guideline_resource_files"](guidelines, guidelines / "settings.yaml", guidelines / "output")
        self.assertEqual(files, [
            ("settings", guidelines / "settings.yaml"),
            ("local_search_prompt", guidelines / "prompts/local.txt"),
            *[(f"tables/{name}.parquet", guidelines / f"output/{name}.parquet") for name in GUIDELINE_TABLES],
            ("vector_store/default_vector_store/default-entity-description.lance",
             guidelines / "output/lancedb/default-entity-description.lance"),
        ])
        self.write_guidelines(guidelines, covariates=True)
        files = H["guideline_resource_files"](guidelines, guidelines / "settings.yaml", guidelines / "output")
        self.assertIn(("tables/covariates.parquet", guidelines / "output/covariates.parquet"), files)

    def test_guideline_files_resolve_configured_prompt_store_and_output_locations(self):
        guidelines = self.root / "guidelines"
        self.write_guidelines(guidelines, settings={
            "local_search": {"prompt": "custom/answer.txt"},
            "vector_store": {"b_store": {"type": "lancedb", "db_uri": "vectors", "container_name": "proj"},
                             "a_store": {"type": "lancedb"}},
        })
        (guidelines / "custom").mkdir()
        (guidelines / "custom/answer.txt").write_text("CUSTOM")
        selected_output = self.root / "selected output"
        files = dict(H["guideline_resource_files"](guidelines, guidelines / "settings.yaml", selected_output))
        self.assertEqual(files["local_search_prompt"], guidelines / "custom/answer.txt")
        self.assertEqual(files["tables/entities.parquet"], selected_output / "entities.parquet")
        self.assertEqual(list(files)[-2:], ["vector_store/a_store/default-entity-description.lance",
                                            "vector_store/b_store/proj-entity-description.lance"])
        self.assertEqual(files["vector_store/b_store/proj-entity-description.lance"],
                         guidelines / "vectors/proj-entity-description.lance")

    def test_guideline_files_skip_absent_prompt_and_reject_unresolvable_configuration(self):
        guidelines = self.write_guidelines(self.root / "guidelines")
        (guidelines / "prompts/local.txt").unlink()
        files = H["guideline_resource_files"](guidelines, guidelines / "settings.yaml", guidelines / "output")
        self.assertNotIn("local_search_prompt", dict(files))
        for name, contents in [
            ("missing", None),
            ("invalid_yaml", "local_search: [unclosed"),
            ("not_mapping", "- item"),
            ("multi_index", yaml.safe_dump({"outputs": {"a": {"base_dir": "x"}}})),
            ("remote_store", yaml.safe_dump({"vector_store": {"s": {"type": "azure_ai_search"}}})),
        ]:
            with self.subTest(settings=name):
                settings = self.root / f"{name}.yaml"
                if contents is not None:
                    settings.write_text(contents, encoding="utf-8")
                self.assertIsNone(H["guideline_resource_files"](guidelines, settings, guidelines / "output"))

    def test_reference_data_metadata_records_runtime_paths_and_fingerprints(self):
        repo = self.root / "repo"
        upstream = repo / "med_copilot/upstream"
        upstream.mkdir(parents=True)
        corpus = upstream / "soap_with_metadata.json"
        corpus.write_text("[]")
        (upstream / "embeddings.npy").write_bytes(b"E")
        (upstream / "index.faiss").write_bytes(b"I")
        guidelines = self.write_guidelines(repo / "med_copilot/data/guidelines")
        arguments = dict(
            corpus_file=corpus, artifacts_directory=upstream,
            artifact_files=[upstream / "embeddings.npy", upstream / "index.faiss"],
            guideline_directory=guidelines, guideline_settings=guidelines / "settings.yaml",
            guideline_output=guidelines / "output",
        )
        metadata = H["reference_data_metadata"](repo, **arguments)
        self.assertEqual(metadata, {
            "patient_retrieval": {
                "corpus_file": "med_copilot/upstream/soap_with_metadata.json",
                "artifacts_directory": "med_copilot/upstream",
                "sha256": H["sha256_bundle"]([("corpus", corpus), ("embeddings.npy", upstream / "embeddings.npy"),
                                              ("index.faiss", upstream / "index.faiss")]),
            },
            "guidelines": {
                "directory": "med_copilot/data/guidelines",
                "sha256": H["sha256_bundle"](H["guideline_resource_files"](
                    guidelines, guidelines / "settings.yaml", guidelines / "output")),
            },
        })
        # Unused prompts do not affect the guideline fingerprint; used files do.
        (guidelines / "prompts/unused.txt").write_text("CHANGED")
        self.assertEqual(H["reference_data_metadata"](repo, **arguments), metadata)
        (guidelines / "prompts/local.txt").write_text("CHANGED")
        changed = H["reference_data_metadata"](repo, **arguments)
        self.assertNotEqual(changed["guidelines"]["sha256"], metadata["guidelines"]["sha256"])
        (upstream / "index.faiss").write_bytes(b"REGENERATED")
        regenerated = H["reference_data_metadata"](repo, **arguments)
        self.assertNotEqual(regenerated["patient_retrieval"]["sha256"],
                            metadata["patient_retrieval"]["sha256"])

    def test_reference_data_metadata_handles_external_uninitialized_disabled_and_missing(self):
        repo = self.root / "repo"
        repo.mkdir()
        external = self.root / "external corpus.json"
        external.write_text("[]")
        metadata = H["reference_data_metadata"](
            repo, corpus_file=external, artifacts_directory=repo / "artifacts", artifact_files=None,
            guideline_directory=None, guideline_settings=None, guideline_output=None,
        )
        self.assertEqual(metadata, {
            "patient_retrieval": {"corpus_file": str(external), "artifacts_directory": None,
                                  "sha256": H["sha256_bundle"]([("corpus", external)])},
            "guidelines": None,
        })
        missing = H["reference_data_metadata"](
            repo, corpus_file=None, artifacts_directory=None, artifact_files=None,
            guideline_directory=repo / "guidelines", guideline_settings=repo / "guidelines/settings.yaml",
            guideline_output=repo / "guidelines/output",
        )
        self.assertEqual(missing, {"patient_retrieval": None,
                                   "guidelines": {"directory": "guidelines", "sha256": None}})
        artifacts = repo / "artifacts"
        artifacts.mkdir()
        unavailable = H["reference_data_metadata"](
            repo, corpus_file=external, artifacts_directory=artifacts,
            artifact_files=[artifacts / "embeddings.npy"], guideline_directory=None,
            guideline_settings=None, guideline_output=None,
        )
        self.assertEqual(unavailable["patient_retrieval"]["artifacts_directory"], "artifacts")
        self.assertIsNone(unavailable["patient_retrieval"]["sha256"])


class PatientOutputHelperTests(unittest.TestCase):
    def test_new_patient_result_has_every_field_in_order_with_null_outputs(self):
        self.assertEqual(list(H["new_patient_result"]("p")), RESULT_FIELDS)
        self.assertEqual(H["new_patient_result"]("p"), {"id": "p", "status": None,
                                                        **dict.fromkeys(RESULT_FIELDS[2:])})
        self.assertEqual(H["new_patient_result"]("p", "not_processed")["status"], "not_processed")

    def test_selected_reference_patient_follows_rerank_scores_not_candidate_order(self):
        corpus = [{"note_id": f"N{row}"} for row in range(4)]
        candidates = [{"idx": 3, "rerank_score": 0.1}, {"idx": 0, "rerank_score": 0.7},
                      {"idx": 2, "rerank_score": 0.7}, {"idx": 1, "rerank_score": -2.0}]
        # Ties keep retrieval order, matching the upstream stable sort.
        self.assertEqual(H["selected_reference_patient"](corpus, candidates, ["formatted"]),
                         {"note_id": "N0"})

    def test_selected_reference_patient_identifier_rules(self):
        candidates = [{"idx": 1, "rerank_score": 1.0}]
        for corpus, expected in [
            ([{"note_id": "A"}, {"note_id": "B"}], {"note_id": "B"}),
            ([{}, {}], {"corpus_row": 1}),
            ([{"note_id": "A"}, {"note_id": ""}], {"corpus_row": 1}),
            ([{"note_id": "A"}, {"note_id": " "}], {"corpus_row": 1}),
            ([{"note_id": "A"}, {"note_id": 12}], {"corpus_row": 1}),
            ([{"note_id": "A"}, {"note_id": None}], {"corpus_row": 1}),
            ([{"note_id": "DUP"}, {"note_id": "DUP"}], {"note_id": "DUP", "corpus_row": 1}),
        ]:
            with self.subTest(corpus=corpus):
                self.assertEqual(H["selected_reference_patient"](corpus, candidates, ["formatted"]), expected)

    def test_selected_reference_patient_is_null_without_reranked_results(self):
        self.assertIsNone(H["selected_reference_patient"]([{"note_id": "A"}], [], []))
        self.assertIsNone(H["selected_reference_patient"]([{"note_id": "A"}], [{"idx": 0}], []))

    def test_selected_reference_patient_accepts_numpy_like_indexes(self):
        class Index(int):
            pass
        self.assertEqual(H["selected_reference_patient"](
            [{}, {}], [{"idx": Index(1), "rerank_score": 0.0}], ["x"]), {"corpus_row": 1})
        self.assertIs(type(H["selected_reference_patient"](
            [{}, {}], [{"idx": Index(1), "rerank_score": 0.0}], ["x"])["corpus_row"]), int)

    def test_parse_guideline_questions_preserves_text_and_order(self):
        questions = [" Second? ", "First café?\nmore", ""]
        self.assertEqual(H["parse_guideline_questions"](json.dumps({"Key Questions": questions, "x": 1})),
                         questions)
        self.assertEqual(H["parse_guideline_questions"]('{"Key Questions": []}'), [])
        for response in [None, "", "not json", '["Q?"]', '"Q?"', '{"Questions": ["Q?"]}',
                         '{"Key Questions": "Q?"}', '{"Key Questions": ["Q?", null]}',
                         '```json\n{"Key Questions": ["Q?"]}\n```']:
            with self.subTest(response=response):
                self.assertIsNone(H["parse_guideline_questions"](response))

    def test_split_assessment_plan_keeps_each_section_verbatim(self):
        response = ("## Assessment\n1. **Most likely:** café\u2028pneumonia\n\n"
                    "### Differential\n- Bronchitis\n  - less likely\n\n"
                    "## Plan\n1. Chest X-ray\n   - PA/lateral\n\n2. Follow-up in 48 h\n")
        self.assertEqual(H["split_assessment_plan"](response), {
            "assessment": ("1. **Most likely:** café\u2028pneumonia\n\n"
                           "### Differential\n- Bronchitis\n  - less likely"),
            "plan": "1. Chest X-ray\n   - PA/lateral\n\n2. Follow-up in 48 h",
        })

    def test_split_assessment_plan_splits_plain_labels_like_the_reference_patient(self):
        response = "Assessment:\nLikely viral URI café\n- low risk\n\nPlan:\n1. Fluids\n2. Return if worse"
        self.assertEqual(H["split_assessment_plan"](response), {
            "assessment": "Likely viral URI café\n- low risk",
            "plan": "1. Fluids\n2. Return if worse",
        })

    def test_split_assessment_plan_tolerates_heading_variants_and_order(self):
        for response in [
            "# Assessment\nA\n# Plan\nP",
            "### ASSESSMENT\nA\n###### plan\nP",
            "## Assessment (A):\nA\n## Plan (P)\nP",
            "## **Assessment**\nA\n## __Plan:__\nP",
            "   ## Assessment ##\nA\n## Plan\t\nP",
            "## Assessment\r\nA\r\n## Plan\r\nP\r\n",
            "## Plan\nP\n\n## Assessment\nA",
            "Here is the note.\n\n## Assessment\nA\n## Plan\nP",
            "assessment:\nA\nPLAN:\nP",
            "Assessment (A):\nA\nPlan (P):\nP",
            "**Assessment**\nA\n**Plan:**\nP",
            "__Assessment (A)__:\nA\n**plan**\nP",
            "## Assessment\nA\nPlan:\nP",
            "Plan:\r\nP\r\n  Assessment:  \r\nA",
        ]:
            with self.subTest(response=response):
                self.assertEqual(H["split_assessment_plan"](response),
                                 {"assessment": "A", "plan": "P"})

    def test_split_assessment_plan_ignores_text_that_is_not_a_section_heading(self):
        for response in [
            "Assessment\nA\nPlan\nP",
            "Assessment: pneumonia\nPlan: antibiotics",
            "The assessment: pneumonia.\nOur plan: rest.",
            "- Assessment:\nA\n1. Plan:\nP",
            "Assessment and Plan:\nA\nP",
            "**Assessment and Plan**\nA\nP",
            "    Assessment:\nA\n    Plan:\nP",
            "##Assessment\nA\n##Plan\nP",
            "    ## Assessment\nA\n    ## Plan\nP",
            "## Assessment and Plan\nA\nP",
            "## Assessment: pneumonia\n## Plan: antibiotics",
        ]:
            with self.subTest(response=response):
                self.assertEqual(H["split_assessment_plan"](response),
                                 {"assessment": None, "plan": None})
        self.assertEqual(H["split_assessment_plan"]("## Assessment\nA\nPlan: rest\n#### Plan details\nx"),
                         {"assessment": "A\nPlan: rest\n#### Plan details\nx", "plan": None})

    def test_split_assessment_plan_keeps_an_available_section_when_the_other_is_unusable(self):
        for response, expected in [
            ("## Assessment\nA", {"assessment": "A", "plan": None}),
            ("## Plan\nP", {"assessment": None, "plan": "P"}),
            ("Assessment:\nA\nPlan:", {"assessment": "A", "plan": None}),
            ("## Assessment\n \n\t\n## Plan\nP", {"assessment": None, "plan": "P"}),
            ("## Assessment\nA\n## Plan\n", {"assessment": "A", "plan": None}),
            ("## Assessment\nA1\n## Plan\nP\n## Assessment\nA2", {"assessment": None, "plan": "P"}),
            ("## Assessment\nA\n## Plan\nP1\n## Plan\nP2", {"assessment": "A", "plan": None}),
        ]:
            with self.subTest(response=response):
                self.assertEqual(H["split_assessment_plan"](response), expected)

    def test_split_assessment_plan_is_null_without_text(self):
        for response in [None, "", " \n\t", 3, ["## Assessment\nA"]]:
            with self.subTest(response=response):
                self.assertEqual(H["split_assessment_plan"](response),
                                 {"assessment": None, "plan": None})

    def test_graphrag_context_json_converts_every_table_with_split_orientation(self):
        class Table:
            def __init__(self, split):
                self.split = split

            def to_json(self, orient):
                self.orient = orient
                return json.dumps(self.split)

        tables = {
            "reports": Table({"columns": ["id"], "index": [0], "data": [["26"]]}),
            "empty": Table({"columns": [], "index": [], "data": []}),
            "extra": Table({"columns": ["text", "score"], "index": ["a"], "data": [["café\n", None]]}),
        }
        self.assertEqual(H["graphrag_context_json"](tables),
                         {name: table.split for name, table in tables.items()})
        self.assertTrue(all(table.orient == "split" for table in tables.values()))
        self.assertEqual(H["graphrag_context_json"]({"plain": [1, "two"]}), {"plain": [1, "two"]})
        with self.assertRaises(AttributeError):
            H["graphrag_context_json"]({"unknown": object()})

    def test_run_status_aggregates_patient_statuses(self):
        for statuses, stopped, expected in [
            (["completed"], False, "completed"),
            (["completed", "completed"], False, "completed"),
            (["completed", "failed_incomplete"], False, "completed_with_errors"),
            (["failed_incomplete"], False, "completed_with_errors"),
            (["completed", "not_processed"], False, "failed_incomplete"),
            (["completed"], True, "failed_incomplete"),
            ([], True, "failed_incomplete"),
        ]:
            with self.subTest(statuses=statuses, stopped=stopped):
                self.assertEqual(H["run_status"](statuses, stopped=stopped), expected)


class RunFileTests(TemporaryRootTestCase):
    def test_write_run_json_replaces_atomically_and_leaves_no_temporary_file(self):
        H["write_run_json"](self.root, {"status": "running", "name": "café"})
        H["write_run_json"](self.root, {"status": "completed"})
        self.assertEqual([path.name for path in self.root.iterdir()], ["run.json"])
        text = (self.root / "run.json").read_text(encoding="utf-8")
        self.assertEqual(json.loads(text), {"status": "completed"})
        self.assertTrue(text.endswith("\n"))

    def test_write_run_json_failure_keeps_previous_file(self):
        H["write_run_json"](self.root, {"status": "running"})
        for bad in [{"value": float("nan")}, {"value": object()}]:
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                H["write_run_json"](self.root, bad)
            self.assertEqual(json.loads((self.root / "run.json").read_text()), {"status": "running"})
            self.assertEqual([path.name for path in self.root.iterdir()], ["run.json"])
        with patch.object(os, "replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                H["write_run_json"](self.root, {"status": "completed"})
        self.assertEqual(json.loads((self.root / "run.json").read_text()), {"status": "running"})
        self.assertEqual([path.name for path in self.root.iterdir()], ["run.json"])


class ExperimentRunTests(TemporaryRootTestCase):
    def setUp(self):
        super().setUp()
        self.repo = self.root / "repo"
        self.input = self.repo / "experiments/input/patient_records_so.jsonl"
        self.input.parent.mkdir(parents=True)
        self.input.write_text('{"id": "a"}\n', encoding="utf-8")
        self.started = H["pacific_time"](datetime(2026, 10, 17, 22, 30, 45, tzinfo=timezone.utc))
        self.finished = H["pacific_time"](datetime(2026, 10, 17, 22, 48, 12, tzinfo=timezone.utc))

    def start(self, patient_ids=("a", "b", "c")):
        return H["ExperimentRun"].start(
            self.repo / "experiments/output", root=self.repo, input_path=self.input,
            started_at=self.started, branch="feature/issue-9", commit="0" * 40,
            patient_ids=list(patient_ids),
        )

    def read(self, run):
        text = (run.run_dir / "results.jsonl").read_text(encoding="utf-8")
        return (json.loads((run.run_dir / "run.json").read_text(encoding="utf-8")),
                [json.loads(line) for line in text.split("\n")[:-1]], text)

    def result(self, patient_id, status="completed", **fields):
        return {**H["new_patient_result"](patient_id, status), **fields}

    def test_start_creates_run_directory_with_running_metadata_and_empty_results(self):
        run = self.start()
        self.assertEqual(run.run_dir, self.repo / "experiments/output"
                         / "2026-10-17_153045PDT__patient_records_so__feature-issue-9")
        self.assertEqual({path.name for path in run.run_dir.iterdir()}, {"run.json", "results.jsonl"})
        metadata, results, _ = self.read(run)
        self.assertEqual(metadata, {
            "run_name": "2026-10-17_153045PDT__patient_records_so__feature-issue-9",
            "started_at": "2026-10-17T15:30:45-07:00",
            "finished_at": None,
            "status": "running",
            "branch": "feature/issue-9",
            "input": {"file": "experiments/input/patient_records_so.jsonl",
                      "sha256": hashlib.sha256(b'{"id": "a"}\n').hexdigest()},
            "git_commit": "0" * 40,
            "reference_data": None,
        })
        self.assertEqual(list(metadata), RUN_FIELDS)
        self.assertEqual(results, [])

    def test_start_never_reuses_an_existing_run_directory(self):
        first = self.start()
        (first.run_dir / "results.jsonl").write_text("KEEP\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.start()
        self.assertEqual((first.run_dir / "results.jsonl").read_text(), "KEEP\n")

    def test_update_rewrites_metadata_fields(self):
        run = self.start()
        run.update(reference_data={"patient_retrieval": None, "guidelines": None})
        self.assertEqual(self.read(run)[0]["reference_data"], {"patient_retrieval": None, "guidelines": None})

    def test_save_result_appends_one_physical_line_per_patient(self):
        run = self.start(["a", "b"])
        first = self.result("a", assessment="line café x", plan="- y")
        run.save_result(first)
        _, results, text = self.read(run)
        self.assertEqual(results, [first])
        self.assertEqual(text.count("\n"), 1)
        self.assertIn("café x", text)
        run.save_result(self.result("b", "failed_incomplete"))
        _, results, text = self.read(run)
        self.assertEqual([record["id"] for record in results], ["a", "b"])
        self.assertEqual(text.count("\n"), 2)
        self.assertEqual(run.statuses, ["completed", "failed_incomplete"])

    def test_save_result_rejects_invalid_status_unknown_and_duplicate_patients(self):
        run = self.start(["a"])
        for record, message in [
            (self.result("a", None), "Invalid patient status"),
            (self.result("a", "running"), "Invalid patient status"),
            (self.result("z"), "not awaiting a result"),
        ]:
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, message):
                run.save_result(record)
        run.save_result(self.result("a"))
        with self.assertRaisesRegex(ValueError, "not awaiting a result"):
            run.save_result(self.result("a"))
        self.assertEqual(len(self.read(run)[1]), 1)

    def test_unserializable_result_is_not_written_and_writing_continues(self):
        run = self.start(["a", "b"])
        with self.assertRaises(ValueError):
            run.save_result(self.result("a", assessment=float("nan")))
        self.assertTrue(run.results_writable)
        run.save_result(self.result("a"))
        self.assertEqual([record["id"] for record in self.read(run)[1]], ["a"])

    def test_write_failure_disables_later_appends(self):
        run = self.start(["a", "b"])
        run.save_result(self.result("a"))
        with patch.object(Path, "open", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                run.save_result(self.result("b"))
        self.assertFalse(run.results_writable)
        with self.assertRaisesRegex(RuntimeError, "no longer writable"):
            run.save_result(self.result("b"))
        self.assertEqual([record["id"] for record in self.read(run)[1]], ["a"])

    def test_finish_sets_completed_or_completed_with_errors(self):
        for statuses, expected in [(["completed", "completed"], "completed"),
                                   (["completed", "failed_incomplete"], "completed_with_errors")]:
            with self.subTest(statuses=statuses):
                run = self.start(["a", "b"])
                for patient_id, status in zip(["a", "b"], statuses):
                    run.save_result(self.result(patient_id, status))
                self.assertEqual(run.finish(self.finished), expected)
                metadata = self.read(run)[0]
                self.assertEqual((metadata["status"], metadata["finished_at"]),
                                 (expected, "2026-10-17T15:48:12-07:00"))
                shutil.rmtree(run.run_dir)

    def test_finish_refuses_when_patients_have_no_result(self):
        run = self.start(["a", "b"])
        run.save_result(self.result("a"))
        with self.assertRaisesRegex(RuntimeError, "1 patients have no saved result"):
            run.finish(self.finished)
        self.assertEqual(self.read(run)[0]["status"], "running")

    def test_stop_marks_current_failed_keeps_partial_outputs_and_rest_not_processed(self):
        run = self.start(["a", "b", "c"])
        run.save_result(self.result("a"))
        current = self.result("b", None, guideline_questions=["Q?"], assessment="A only")
        run.stop(self.finished, current)
        metadata, results, _ = self.read(run)
        self.assertEqual(results, [
            self.result("a"),
            self.result("b", "failed_incomplete", guideline_questions=["Q?"], assessment="A only"),
            self.result("c", "not_processed"),
        ])
        self.assertEqual((metadata["status"], metadata["finished_at"]),
                         ("failed_incomplete", "2026-10-17T15:48:12-07:00"))
        self.assertIsNone(current["status"])  # The caller's in-progress record is not mutated.

    def test_stop_without_current_patient_marks_all_remaining_not_processed(self):
        run = self.start(["a", "b"])
        run.stop(self.finished)
        self.assertEqual([(r["id"], r["status"]) for r in self.read(run)[1]],
                         [("a", "not_processed"), ("b", "not_processed")])

    def test_stop_after_every_patient_was_saved_only_updates_run_status(self):
        run = self.start(["a"])
        run.save_result(self.result("a"))
        run.stop(self.finished, self.result("a", "completed"))
        metadata, results, _ = self.read(run)
        self.assertEqual(results, [self.result("a")])
        self.assertEqual(metadata["status"], "failed_incomplete")

    def test_stop_replaces_unserializable_current_outputs_with_a_null_record(self):
        run = self.start(["a", "b"])
        run.stop(self.finished, self.result("a", None, graphrag_context={"bad": float("nan")}))
        self.assertEqual(self.read(run)[1], [self.result("a", "failed_incomplete"),
                                              self.result("b", "not_processed")])

    def test_stop_after_write_failure_preserves_saved_lines_and_finalizes_metadata(self):
        run = self.start(["a", "b", "c"])
        run.save_result(self.result("a"))
        with patch.object(Path, "open", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                run.save_result(self.result("b"))
        run.stop(self.finished, self.result("b"))
        metadata, results, _ = self.read(run)
        self.assertEqual(results, [self.result("a")])
        self.assertEqual(metadata["status"], "failed_incomplete")
        self.assertEqual({path.name for path in run.run_dir.iterdir()}, {"run.json", "results.jsonl"})

    def test_stop_tolerates_a_run_json_write_failure(self):
        run = self.start(["a"])
        with patch.object(os, "replace", side_effect=OSError("disk full")):
            run.stop(self.finished)
        metadata, results, _ = self.read(run)
        self.assertEqual(results, [self.result("a", "not_processed")])
        self.assertEqual(metadata["status"], "running")
        self.assertEqual({path.name for path in run.run_dir.iterdir()}, {"run.json", "results.jsonl"})


if __name__ == "__main__":
    unittest.main()

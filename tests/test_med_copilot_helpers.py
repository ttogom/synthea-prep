"""Offline unit tests for every project-owned MED-COPILOT helper.

Run directly: python tests/test_med_copilot_helpers.py
Or discover: python -m unittest discover -s tests -p 'test_*.py'
Only temporary files and mocked API responses are used; no model/runtime
initialization is needed. Keep this suite updated when helpers are added.
"""

import ast
import builtins
import errno
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "scripts/med_copilot_helpers.py"
UPSTREAM_RETRIEVER = ROOT / "med_copilot/upstream/create_embeddings.py"
GUIDELINE_TABLES = ("communities", "community_reports", "text_units", "relationships", "entities")


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
                "uuid": "same-patient-id",
                "subjective": "  Cough café.\nDenies fever.\t ",
                "objective": "SpO₂ 98%; 37.2 °C; 左腕\n ",
                "metadata": {"encounter": 12, "tags": ["prepared", None]},
            },
            {"uuid": "same-patient-id", "subjective": "SECOND S", "objective": "SECOND O"},
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

    def test_loads_all_records_in_order_preserving_metadata_duplicates_and_clinical_text(self):
        self.write_records(self.records)
        original = self.input.read_bytes()
        patients = LOAD_PATIENT_RECORDS(self.input)
        self.assertEqual(patients, self.records)
        self.assertEqual(len(patients), 2)
        self.assertEqual(patients[0]["subjective"], self.records[0]["subjective"])
        self.assertEqual(patients[0]["objective"], self.records[0]["objective"])
        self.assertEqual(patients[0]["uuid"], patients[1]["uuid"])
        self.assertEqual(self.input.read_bytes(), original)
        self.assertEqual(set(self.root.iterdir()), {self.input})

    def test_accepts_path_and_string_input_and_loads_fresh_records_each_time(self):
        self.write_records(self.records)
        for path in (self.input, str(self.input)):
            with self.subTest(path_type=type(path).__name__):
                self.assertEqual(LOAD_PATIENT_RECORDS(path), self.records)
        changed = [{"subjective": "UPDATED S", "objective": "UPDATED O"}]
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

    def test_empty_so_and_missing_patient_identifier_are_valid_prepared_records(self):
        records = [{"subjective": "", "objective": " \n\t "}]
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
        valid = {"subjective": "PRIVATE_CLINICAL_TEXT", "objective": "PRIVATE_CLINICAL_TEXT"}
        for field in ("subjective", "objective"):
            payloads = [{key: value for key, value in valid.items() if key != field}]
            payloads.extend({**valid, field: value} for value in (None, [], {}, 3, 2.5, True))
            for record in payloads:
                with self.subTest(field=field, value_type=type(record.get(field)).__name__):
                    self.assert_invalid(json.dumps(record) + "\n", 1,
                                        f"{field} must be a string")

    def test_invalid_later_schema_is_validated_before_any_records_are_returned(self):
        first = json.dumps(self.records[0], ensure_ascii=False)
        self.assert_invalid(first + '\n\n{"subjective": "PRIVATE_CLINICAL_TEXT"}\n', 3,
                            "objective must be a string")


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
        for name in ("load_patient_records", "validate_guideline_tables", "JSONKeywordExtractor"):
            self.assertTrue(callable(namespace[name]))

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


if __name__ == "__main__":
    unittest.main()

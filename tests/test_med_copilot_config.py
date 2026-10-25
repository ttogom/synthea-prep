"""Offline tests for the master experiment configuration and shared YAML reader."""

import ast
import builtins
from contextlib import chdir, redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

# Direct execution puts tests/ on sys.path; project imports need the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.med_copilot_config import (
    ExperimentConfig, configured_model_credentials, confirm_experiment, read_yaml_config,
)


class ExperimentConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "configs/experiment.yaml"
        self.path.parent.mkdir()
        self.data = {"selected_models": ["mini"],
                     "temperature": 1, "max_retries": 3, "max_patients": None}
        self.write()

    def write(self):
        self.path.write_text(yaml.safe_dump(self.data, sort_keys=False), encoding="utf-8")

    def test_shipped_experiment_is_the_only_source_of_selection_temperature_and_retries(self):
        path = ROOT / "experiments/configs/experiment.yaml"
        config = ExperimentConfig(path)
        self.assertEqual(config.selected_models, ["gpt_4o_mini", "ollama_healthcare", "ollama_llama3", "ollama_gemma2_2b"])
        self.assertEqual(config.temperature, 1)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.max_patients, 1)
        self.assertEqual(config.model_config, ROOT / "experiments/configs/condition_plan_models.yaml")
        metadata = config.metadata(ROOT)
        for field, value in yaml.safe_load(path.read_text()).items():
            with self.subTest(config_field=field):
                self.assertIn(field, metadata)
                self.assertEqual(metadata[field], value)
        registry, _ = read_yaml_config(config.model_config)
        self.assertEqual(set(registry), {"models"})
        self.assertTrue(all(not ({"temperature", "max_retries", "selected_models"} & set(model))
                            for model in registry["models"].values()))

    def test_values_metadata_fingerprint_and_caller_lists_are_independent(self):
        self.data.update(temperature=0.35, max_retries=5, selected_models=["llama", "mini"])
        self.write()
        config = ExperimentConfig(self.path)
        self.data["selected_models"].reverse()
        metadata = config.metadata(self.root)
        self.assertEqual(metadata, {
            "config_file": "configs/experiment.yaml",
            "config_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "model_config": "configs/condition_plan_models.yaml", "selected_models": ["llama", "mini"],
            "temperature": 0.35, "max_retries": 5, "max_patients": None,
        })
        metadata["selected_models"].clear()
        self.assertEqual(config.metadata(self.root)["selected_models"], ["llama", "mini"])
        self.assertEqual(config.metadata(self.root / "other")["config_file"], str(self.path))
        original_hash = config.config_sha256
        self.path.write_text("changed after initialization")
        self.assertEqual(config.metadata(self.root)["config_sha256"], original_hash)

    def test_shipped_ollama_entries_match_downloaded_tags_and_need_no_provider_keys(self):
        config = ExperimentConfig(ROOT / "experiments/configs/experiment.yaml")
        registry, _ = read_yaml_config(config.model_config)
        expected = {"ollama_healthcare": "Jayasimma/healthcare:latest",
                    "ollama_llama3": "llama3:latest", "ollama_gemma2_2b": "gemma2:2b"}
        self.assertEqual(config.selected_models, ["gpt_4o_mini", *expected])
        for alias, tag in expected.items():
            with self.subTest(model=alias):
                self.assertEqual(registry["models"][alias], {
                    "type": "chat", "model_provider": "openai", "model": tag,
                    "api_base": "http://localhost:11434/v1", "output_mode": "json_schema",
                })
        with patch.dict(os.environ, {}, clear=True):
            credentials = configured_model_credentials(registry["models"], list(expected))
        self.assertEqual(credentials, {alias: (None, "local-no-key") for alias in expected})

    def test_registry_reference_uses_experiment_directory_from_any_cwd(self):
        caller = self.root / "unrelated"
        caller.mkdir()
        with chdir(caller):
            config = ExperimentConfig(self.path)
            self.assertEqual(config.model_config, self.path.with_name("condition_plan_models.yaml"))
            self.assertEqual(Path.cwd(), caller)

    def test_constructor_rejects_selection_and_registry_override_arguments(self):
        for override in ({"selected_models": ["other"]}, {"model_config": "other.yaml"}):
            with self.subTest(override=override), self.assertRaises(TypeError):
                ExperimentConfig(self.path, **override)

    def test_zero_and_boundary_temperatures_are_valid(self):
        for value in (0, 0.0, 0.2, 1, 2, 2.0):
            with self.subTest(value=value):
                self.data["temperature"] = value
                self.write()
                self.assertEqual(ExperimentConfig(self.path).temperature, value)

    def test_invalid_temperatures_fail_without_creating_artifacts(self):
        for value in (True, False, None, "1", -0.1, 2.1, float("nan"), float("inf"), 10 ** 400):
            with self.subTest(value=value):
                self.data["temperature"] = value
                self.write()
                with self.assertRaisesRegex(ValueError, "temperature"):
                    ExperimentConfig(self.path)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_retries_must_match_upstream_positive_integer_contract(self):
        for value in (0, -1, True, False, None, 1.5, 3.0, "3"):
            with self.subTest(value=value):
                self.data["max_retries"] = value
                self.write()
                with self.assertRaisesRegex(ValueError, "max_retries"):
                    ExperimentConfig(self.path)
        self.data["max_retries"] = 1
        self.write()
        self.assertEqual(ExperimentConfig(self.path).max_retries, 1)

    def test_configured_selections_require_unique_nonblank_strings(self):
        original = deepcopy(self.data)
        for value in ([], "mini", None, [1], [""], [" "], ["mini", "mini"], [["mini"]]):
            with self.subTest(value=value):
                self.data = deepcopy(original)
                self.data["selected_models"] = value
                self.write()
                with self.assertRaises(ValueError):
                    ExperimentConfig(self.path)

    def test_patient_limit_accepts_null_and_positive_integers_and_is_recorded(self):
        for value in (None, 1, 2, 10000):
            with self.subTest(value=value):
                self.data["max_patients"] = value
                self.write()
                config = ExperimentConfig(self.path)
                self.assertEqual(config.max_patients, value)
                self.assertEqual(config.metadata(self.root)["max_patients"], value)

    def test_invalid_patient_limits_name_the_setting_and_experiment_file(self):
        for value in (0, -1, True, False, 1.0, 1.5, "1", "all", "", [], {}, float("inf")):
            with self.subTest(value=value):
                self.data["max_patients"] = value
                self.write()
                with self.assertRaisesRegex(ValueError, "max_patients.*experiment.yaml"):
                    ExperimentConfig(self.path)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_missing_extra_and_wrong_root_fields_are_rejected(self):
        original = deepcopy(self.data)
        for value in (None, [], {}, {**original, "temperatur": 1},
                      *[{key: value for key, value in original.items() if key != missing}
                        for missing in original]):
            with self.subTest(value=value):
                self.path.write_text(yaml.safe_dump(value))
                with self.assertRaisesRegex(ValueError, "Experiment config"):
                    ExperimentConfig(self.path)

    def test_registry_override_field_is_rejected(self):
        self.data["model_config"] = "other.yaml"
        self.write()
        with self.assertRaisesRegex(ValueError, "Experiment config"):
            ExperimentConfig(self.path)

    def test_yaml_reader_preserves_unicode_crlf_and_exact_fingerprint_without_writing(self):
        source = 'label: "café"\r\nitems: [1, 2]\r\n'.encode("utf-8")
        self.path.write_bytes(source)
        parsed, fingerprint = read_yaml_config(str(self.path))
        self.assertEqual(parsed, {"label": "café", "items": [1, 2]})
        self.assertEqual(fingerprint, hashlib.sha256(source).hexdigest())
        self.assertEqual(self.path.read_bytes(), source)

    def test_yaml_reader_rejects_malformed_duplicate_unsafe_and_invalid_utf8_input(self):
        for source in (b"[", b"a: 1\na: 2\n", b"a: {b: 1, b: 2}",
                       b"!!python/object:builtins.object {}", b"\xff"):
            with self.subTest(source=source):
                self.path.write_bytes(source)
                with self.assertRaises(ValueError):
                    read_yaml_config(self.path)
        self.path.unlink()
        for loader in (read_yaml_config, ExperimentConfig):
            with self.assertRaises(FileNotFoundError):
                loader(self.path)

    def test_config_imports_are_in_header_and_do_not_load_model_clients(self):
        path = ROOT / "scripts/med_copilot_config.py"
        tree = ast.parse(path.read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertTrue(all(node in tree.body for node in imports))
        first_definition = min(node.lineno for node in tree.body
                               if isinstance(node, (ast.ClassDef, ast.FunctionDef)))
        self.assertTrue(all(node.lineno < first_definition for node in imports))
        original_import = builtins.__import__
        def import_checked(name, *args, **kwargs):
            self.assertFalse(name.startswith(("openai", "graphrag", "litellm", "sentence_transformers", "faiss")))
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=import_checked):
            namespace = runpy.run_path(str(path))
        self.assertTrue(callable(namespace["ExperimentConfig"]))


class CredentialConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "mini": {"api_key_env": "OPENAI_API_KEY"},
            "llama": {"api_key_env": "GROQ_API_KEY"},
            "local": {"api_base": "http://localhost:8000/v1"},
            "azure": {"auth_type": "azure_managed_identity"},
        }

    def test_resolves_only_selected_credentials_in_selection_order(self):
        original = deepcopy(self.registry)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "private-key"}, clear=True):
            self.assertEqual(configured_model_credentials(self.registry, ["local", "mini", "azure"]), {
                "local": (None, "local-no-key"), "mini": ("OPENAI_API_KEY", "private-key"),
                "azure": (None, None),
            })
        self.assertEqual(self.registry, original)

    def test_collects_unknown_aliases_and_missing_selected_and_shared_keys(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError) as error:
            configured_model_credentials(self.registry, ["typo", "mini", "llama"],
                                         require_shared_credentials=True)
        message = str(error.exception)
        for expected in ("Unknown model alias: typo", "selected_models", "experiment.yaml",
                         "condition_plan_models.yaml", "Available aliases", "mini, llama, local, azure",
                         "OPENAI_API_KEY", "GROQ_API_KEY", "GRAPHRAG_API_KEY", ".env in the repository root"):
            self.assertIn(expected, message)
        self.assertEqual(message.count("Shared-stage API key is missing"), 1)

    def test_missing_empty_and_whitespace_values_are_errors(self):
        for value in (None, "", " ", "\t\n"):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["OPENAI_API_KEY"] = value
                with self.assertRaisesRegex(ValueError, "API key for mini is missing or empty"):
                    configured_model_credentials(self.registry, ["mini"])

    def test_missing_or_malformed_key_mapping_names_both_config_and_dotenv(self):
        for entry in ({}, {"api_key_env": ""}, {"api_key_env": "KEY WITH SPACE"},
                      {"api_key_env": "key=value"}, {"api_key_env": 1}):
            with self.subTest(entry=entry), self.assertRaises(ValueError) as error:
                configured_model_credentials({"broken": entry}, ["broken"])
            self.assertIn("api_key_env", str(error.exception))
            self.assertIn("condition_plan_models.yaml", str(error.exception))
            self.assertIn(".env in the repository root", str(error.exception))

    def test_explicit_endpoint_key_mapping_is_required_even_for_local_models(self):
        self.registry["local"]["api_key_env"] = "LOCAL_API_KEY"
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "LOCAL_API_KEY"):
            configured_model_credentials(self.registry, ["local"])

    def test_shared_credentials_are_required_even_with_keyless_final_model(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "private", "GRAPHRAG_API_KEY": "graph"}, clear=True):
            self.assertEqual(configured_model_credentials(self.registry, ["local"],
                             require_shared_credentials=True), {"local": (None, "local-no-key")})
            os.environ["GRAPHRAG_API_KEY"] = " "
            with self.assertRaisesRegex(ValueError, "GRAPHRAG_API_KEY") as error:
                configured_model_credentials(self.registry, ["local"], require_shared_credentials=True)
            self.assertNotIn("private", str(error.exception))


class ExperimentConfirmationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        path = self.root / "experiments/configs/experiment.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"selected_models": ["mini", "local", "azure"],
                                        "temperature": 0.25, "max_retries": 7, "max_patients": None}))
        self.experiment = ExperimentConfig(path)
        self.metadata = {"models": {
            "mini": {"api_key_env": "OPENAI_API_KEY", "config": {
                "model_provider": "openai", "model": "gpt-4o-mini", "temperature": 0.25, "max_retries": 7, "max_patients": None}},
            "local": {"api_key_env": None, "config": {"model_provider": "openai", "model": "llama"}},
            "azure": {"api_key_env": None, "config": {"model_provider": "azure", "model": "base",
                "deployment_name": "my-deployment", "auth_type": "azure_managed_identity"}},
        }}

    def test_exact_display_shows_ordered_models_auth_and_edit_instructions_only(self):
        original = deepcopy(self.metadata)
        with redirect_stdout(StringIO()) as stdout, patch("builtins.input", return_value="yes") as prompt:
            self.assertTrue(confirm_experiment(self.experiment, self.metadata, self.root))
        prompt.assert_called_once_with("Is this correct? Start the run? [y/N]: ")
        self.assertEqual(stdout.getvalue(), "\n========== EXPERIMENT CONFIGURATION ==========\n\n"
            "Experiment: experiments/configs/experiment.yaml\n"
            "Model definitions: experiments/configs/condition_plan_models.yaml\n\n"
            "Selected models:\n"
            "  1. mini (openai/gpt-4o-mini)\n     API key configured: OPENAI_API_KEY\n"
            "  2. local (openai/llama)\n     API key not required (unauthenticated endpoint)\n"
            "  3. azure (azure/my-deployment)\n     Authentication: Azure managed identity\n\n"
            "To change the selection, edit selected_models in\n"
            "experiments/configs/experiment.yaml and restart the script.\n\n"
            "These are the models selected for this experiment run.\n"
            "Confirmed. Starting experiment run.\n")
        self.assertEqual(self.metadata, original)
        self.assertNotIn("temperature", stdout.getvalue())
        self.assertNotIn("max_retries", stdout.getvalue())
        self.assertNotIn("GRAPHRAG_API_KEY", stdout.getvalue())

    def test_only_y_and_yes_are_affirmative(self):
        for answer in ("y", "Y", "yes", "YES", " YeS \n"):
            with self.subTest(answer=answer), redirect_stdout(StringIO()), patch("builtins.input", return_value=answer):
                self.assertTrue(confirm_experiment(self.experiment, self.metadata, self.root))

    def test_negative_empty_invalid_eof_and_interrupt_cancel(self):
        for answer in ("n", "no", "", " ", "maybe", "true", EOFError(), KeyboardInterrupt()):
            prompt = Mock(side_effect=answer) if isinstance(answer, BaseException) else Mock(return_value=answer)
            with self.subTest(answer=answer), redirect_stdout(StringIO()) as stdout, patch("builtins.input", prompt):
                self.assertFalse(confirm_experiment(self.experiment, self.metadata, self.root))
            self.assertIn("Run cancelled. No experiment output was created.", stdout.getvalue())
            self.assertNotIn("Starting experiment", stdout.getvalue())
        self.assertEqual(sorted(file.name for file in self.experiment.config_path.parent.iterdir()), ["experiment.yaml"])


if __name__ == "__main__":
    unittest.main()

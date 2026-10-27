"""Offline tests for the master experiment configuration and shared YAML reader."""

import ast
import builtins
from contextlib import chdir, redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import json
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
    PATH_SETTINGS, PIPELINE_DEFAULTS, ExperimentConfig, configured_model_credentials, confirm_experiment, read_yaml_config,
)
from scripts.med_copilot_variants import COMPONENTS, VARIANTS, ComponentSpec, Parameter, VariantSpec
from med_copilot.prompts.diagnosis_plan import VARIANT_TEMPLATES


DEFAULT_COMPONENT_SETTINGS = {
    "patient_retrieval": {"alpha": 0.5, "top_k": 20, "rerank_top_k": 5},
    "guidelines": {"question_model": "gpt-4o-mini", "community_level": 2,
                   "response_type": "Multiple Paragraphs", "verbose": True},
}


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
        self.assertEqual(config.max_patients, yaml.safe_load(path.read_text())["max_patients"])
        self.assertEqual(config.model_config, ROOT / "experiments/configs/condition_plan_models.yaml")
        metadata = config.metadata(ROOT)
        for field, value in yaml.safe_load(path.read_text()).items():
            with self.subTest(config_field=field):
                self.assertIn(field, metadata)
                if field != "variant_settings":
                    self.assertEqual(metadata[field], value)
        self.assertEqual(metadata["variant_settings"], {
            "patient_retrieval": {"patient_retrieval": {"alpha": 0.5, "top_k": 20, "rerank_top_k": 5}},
            "guidelines": {"guidelines": {"question_model": "gpt-4o-mini", "community_level": 2,
                                           "response_type": "Multiple Paragraphs", "verbose": False}},
            "patient_retrieval_guidelines": {
                "patient_retrieval": {"alpha": 0.5, "top_k": 20, "rerank_top_k": 5},
                "guidelines": {"question_model": "gpt-4o-mini", "community_level": 2,
                               "response_type": "Multiple Paragraphs", "verbose": False}},
        })
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
            "variant_settings": {"patient_retrieval_guidelines": deepcopy(DEFAULT_COMPONENT_SETTINGS)},
            **{key: str(ROOT / value) if key in PATH_SETTINGS else value for key, value in PIPELINE_DEFAULTS.items()},
            **{key: metadata[key] for key in ("selected_variants", "patient_input", "patient_labels",
                                              "presenting_evidence_only", "exclude_none_recorded_plans")},
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
        prompt.assert_called_once_with("Is this configuration correct? Start the run? [y/N]: ")
        text = stdout.getvalue()
        for expected in ("Selected models:", "Run settings:", "Evidence variants:", "Dataset and cohort:",
                         "temperature: 0.25", "max_retries: 7", "max_patients: null",
                         "mini (openai/gpt-4o-mini)", "local (openai/llama)", "azure (azure/my-deployment)",
                         "API key configured: OPENAI_API_KEY", "API key not required", "Azure managed identity",
                         "exclude_none_recorded_plans: false", '- Excluded label plan value is exactly ["none recorded"]',
                         "edit experiments/configs/experiment.yaml and restart"):
            self.assertIn(expected, text)
        self.assertEqual(self.metadata, original)
        self.assertIn("temperature", stdout.getvalue())
        self.assertIn("max_retries", stdout.getvalue())
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


class VariantConfigTests(unittest.TestCase):
    setUp = ExperimentConfigTests.setUp
    write = ExperimentConfigTests.write

    def test_defaults_selection_flags_paths_and_complete_metadata(self):
        config = ExperimentConfig(self.path)
        self.assertEqual(config.selected_variants, ["patient_retrieval_guidelines"])
        self.assertTrue(config.needs_patient_retrieval)
        self.assertTrue(config.needs_guidelines)
        self.assertFalse(config.exclude_none_recorded_plans)
        for variants, retrieval, guidelines in ((["llm_only"], False, False),
                (["patient_retrieval"], True, False), (["guidelines"], False, True),
                (["llm_only", "patient_retrieval_guidelines"], True, True)):
            self.data.update(selected_variants=variants, patient_input="experiments/input/test_sample_so.jsonl",
                             patient_labels="experiments/input/test_sample_labels.jsonl",
                             presenting_evidence_only=False, exclude_none_recorded_plans=True)
            self.write()
            config = ExperimentConfig(self.path)
            self.assertEqual((config.needs_patient_retrieval, config.needs_guidelines), (retrieval, guidelines))
            metadata = config.metadata(ROOT)
            for key, value in self.data.items():
                self.assertEqual(metadata[key], value, key)
            self.assertEqual(config.patient_input, ROOT / self.data["patient_input"])
            metadata["selected_variants"].clear()
            self.assertEqual(config.selected_variants, variants)

    def test_invalid_variants_booleans_paths_and_unknown_fields(self):
        fields = {
            "selected_variants": [[], ["bogus"], ["llm_only", "llm_only"], "llm_only", [1], [None]],
            "patient_input": [None, "", "  ", 7, []], "patient_labels": [False, "", {}],
            "presenting_evidence_only": [None, 0, 1, "true"],
            "exclude_none_recorded_plans": [None, 0, 1, "false"],
        }
        original = dict(self.data)
        for field, values in fields.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.data = {**original, field: value}
                    self.write()
                    with self.assertRaisesRegex(ValueError, field):
                        ExperimentConfig(self.path)

    def test_prompt_provenance_names_actual_project_templates_and_hashes(self):
        self.data["selected_variants"] = ["llm_only", "guidelines"]
        self.write()
        config = ExperimentConfig(self.path)
        metadata = config.variant_metadata(ROOT)
        prompt_path = ROOT / "med_copilot/prompts/diagnosis_plan.py"
        prompts = runpy.run_path(str(prompt_path))
        self.assertEqual(metadata["prompt_provenance"], {
            "file": "med_copilot/prompts/diagnosis_plan.py",
            "sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest()})
        self.assertEqual(list(metadata["variant_definitions"]), self.data["selected_variants"])
        for name, entry in metadata["variant_definitions"].items():
            self.assertEqual(entry["template"], f"VARIANT_TEMPLATES[{name!r}]")
            self.assertEqual(entry["sha256"], hashlib.sha256(prompts["VARIANT_TEMPLATES"][name].encode()).hexdigest())
            self.assertFalse(entry["patient_retrieval"])
            self.assertEqual(entry["guidelines"], name == "guidelines")

    def test_confirmation_counts_align_and_all_effective_settings_are_shown(self):
        experiment = ExperimentConfig(self.path)
        metadata = {"models": {"mini": {"api_key_env": "KEY", "config": {"model_provider": "openai", "model": "mini"}}}}
        cohort = dict(input_patient_count=10, after_presenting_evidence_count=7,
                      eligible_patient_count=5, selected_patient_count=1)
        with redirect_stdout(StringIO()) as output, patch("builtins.input", return_value="y"):
            self.assertTrue(confirm_experiment(experiment, metadata, ROOT, cohort))
        text = output.getvalue()
        rows = [line for line in text.splitlines() if any(label in line for label in (
            "Input patients:", "After presenting-evidence filter:", "After recorded-plan filter:", "Selected after patient limit:"))]
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({len(row) for row in rows}), 1)
        self.assertEqual([int(row.split(":")[-1]) for row in rows], [10, 7, 5, 1])
        for key in ("temperature", "max_retries", "max_patients", "patient_input", "patient_labels",
                    "presenting_evidence_only", "exclude_none_recorded_plans"):
            self.assertIn(key + ":", text)
        self.assertIn("Planned final requests: 1", text)


class ExtensibleComponentConfigTests(unittest.TestCase):
    write = ExperimentConfigTests.write

    def test_component_string_parameter_can_inherit_a_resolved_shared_path(self):
        component = ComponentSpec("Corpus context", ("text",), {
            "corpus_path": Parameter("str", config_field="patient_corpus"),
        })
        self.register(COMPONENTS, {"custom_source": component})
        config = self.load(selected_variants=["custom_a"], variant_settings={})
        settings = config.component_instances("custom_source")[0][1]
        self.assertEqual(settings["corpus_path"], str(ROOT / PIPELINE_DEFAULTS["patient_corpus"]))
        self.assertEqual(config.metadata(ROOT)["variant_settings"]["custom_a"]["custom_source"], settings)
        json.dumps(config.variant_metadata(ROOT), allow_nan=False)

    def setUp(self):
        ExperimentConfigTests.setUp(self)
        component = ComponentSpec(
            "Custom evidence", ("text",), {
                "limit": Parameter("int", 3, minimum=1, maximum=8),
                "strategy": Parameter("str", "focused"),
                "include_scores": Parameter("bool", False),
                "weight": Parameter("number", 0.5, minimum=0, maximum=1),
            }, resources=("guidelines",), credentials=("CUSTOM_API_KEY",),
            implementation="components/custom.py",
        )
        self.register(COMPONENTS, {"custom_source": component})
        self.register(VARIANTS, {
            name: VariantSpec(f"Description {name}", ("custom_source",), "guidelines",
                              {"Key_info": ("custom_source", "text")})
            for name in ("custom_a", "custom_b")
        })

    def register(self, mapping, additions):
        patcher = patch.dict(mapping, additions)
        patcher.start()
        self.addCleanup(patcher.stop)

    def load(self, **updates):
        self.data.update(updates)
        self.write()
        return ExperimentConfig(self.path)

    def test_legacy_defaults_publish_effective_settings_and_shared_component_plan(self):
        config = self.load()
        self.assertEqual(config.variant_settings,
                         {"patient_retrieval_guidelines": DEFAULT_COMPONENT_SETTINGS})
        self.assertEqual(config.variant_components, {"patient_retrieval_guidelines": {
            "patient_retrieval": "patient_retrieval", "guidelines": "guidelines"}})
        self.assertEqual(config.component_plan, {
            name: {"component": name, "settings": values} for name, values in DEFAULT_COMPONENT_SETTINGS.items()})
        self.assertEqual(config.required_resources, {"patient_retrieval", "guidelines"})
        self.assertEqual(config.required_credentials, ["GRAPHRAG_API_KEY", "OPENAI_API_KEY"])
        self.assertFalse(config.needs("absent"))
        self.assertEqual(config.component_instances("absent"), [])

    def test_existing_four_variants_share_exactly_two_component_executions(self):
        config = self.load(selected_variants=["llm_only", "patient_retrieval", "guidelines",
                                              "patient_retrieval_guidelines"])
        self.assertEqual(list(config.component_plan), ["patient_retrieval", "guidelines"])
        self.assertEqual(config.variant_components["llm_only"], {})
        self.assertEqual(config.variant_components["patient_retrieval"], {"patient_retrieval": "patient_retrieval"})
        self.assertEqual(config.variant_components["guidelines"], {"guidelines": "guidelines"})
        self.assertEqual(config.variant_components["patient_retrieval_guidelines"], {
            "patient_retrieval": "patient_retrieval", "guidelines": "guidelines"})
        self.assertEqual(len(config.component_instances("patient_retrieval")), 1)
        self.assertEqual(len(config.component_instances("guidelines")), 1)

    def test_identical_overrides_are_deduplicated_and_snapshots_do_not_alias(self):
        settings = {name: {"custom_source": {"limit": 6}} for name in ("custom_a", "custom_b")}
        config = self.load(selected_variants=["custom_b", "custom_a"], variant_settings=settings)
        self.assertEqual(len(config.component_plan), 1)
        first, = config.component_plan
        self.assertEqual(config.variant_components["custom_a"], {"custom_source": first})
        self.assertEqual(config.variant_components["custom_b"], {"custom_source": first})
        expected = {"limit": 6, "strategy": "focused", "include_scores": False, "weight": 0.5}
        self.assertEqual(config.component_instances("custom_source"), [(first, expected)])
        instances = config.component_instances("custom_source")
        instances[0][1]["limit"] = 1
        metadata = config.metadata(ROOT)
        metadata["variant_settings"]["custom_a"]["custom_source"]["limit"] = 2
        self.assertEqual(config.component_instances("custom_source"), [(first, expected)])
        self.assertEqual(config.variant_settings["custom_a"]["custom_source"]["limit"], 6)

    def test_distinct_overrides_get_distinct_executions_and_stable_order_independent_ids(self):
        values = {"custom_a": {"custom_source": {"limit": 5, "strategy": "broad"}},
                  "custom_b": {"custom_source": {"strategy": "focused", "limit": 6}}}
        first = self.load(selected_variants=["custom_a", "custom_b"], variant_settings=values)
        self.assertEqual(len(first.component_instances("custom_source")), 2)
        refs = deepcopy(first.variant_components)
        self.assertNotEqual(refs["custom_a"], refs["custom_b"])
        values["custom_a"]["custom_source"] = {"strategy": "broad", "limit": 5}
        second = self.load(selected_variants=["custom_b", "custom_a"], variant_settings=values)
        self.assertEqual(second.variant_components, refs)
        self.assertEqual(list(second.component_plan),
                         [refs[name]["custom_source"] for name in ("custom_b", "custom_a")])

    def test_equivalent_integer_and_float_number_overrides_share_one_execution(self):
        config = self.load(selected_variants=["custom_a", "custom_b"], variant_settings={
            "custom_a": {"custom_source": {"weight": 1}},
            "custom_b": {"custom_source": {"weight": 1.0}},
        })
        self.assertEqual(len(config.component_instances("custom_source")), 1)
        self.assertEqual(config.variant_components["custom_a"], config.variant_components["custom_b"])
        self.assertIs(type(config.variant_settings["custom_a"]["custom_source"]["weight"]), float)

    def test_component_instance_identity_collision_is_rejected_instead_of_sharing_wrong_outputs(self):
        first = self.load(selected_variants=["custom_a"],
                          variant_settings={"custom_a": {"custom_source": {"limit": 5}}})
        conflicting_name, = first.component_plan
        self.register(COMPONENTS, {conflicting_name: ComponentSpec("Different implementation", ("text",))})
        self.register(VARIANTS, {"conflicting": VariantSpec("Conflicting", (conflicting_name,), "guidelines",
                                                         {"Key_info": (conflicting_name, "text")})})
        with self.assertRaisesRegex(ValueError, "identity collision"):
            self.load(selected_variants=["custom_a", "conflicting"])

    def test_global_baseline_settings_are_component_defaults_and_explicit_defaults_share(self):
        config = self.load(selected_variants=["patient_retrieval", "patient_retrieval_guidelines"],
                           patient_retrieval_alpha=0.8, patient_retrieval_top_k=7,
                           variant_settings={"patient_retrieval": {"patient_retrieval": {"alpha": 0.8}}})
        self.assertEqual(config.component_instances("patient_retrieval"), [
            ("patient_retrieval", {"alpha": 0.8, "top_k": 7, "rerank_top_k": 5})])
        self.assertEqual(config.variant_settings["patient_retrieval"]["patient_retrieval"]["top_k"], 7)

    def test_inactive_settings_are_validated_and_recorded_without_requesting_resources(self):
        config = self.load(selected_variants=["llm_only"],
                           variant_settings={"custom_a": {"custom_source": {"limit": 7}}})
        self.assertEqual(config.component_plan, {})
        self.assertEqual(config.required_resources, set())
        self.assertEqual(config.required_credentials, [])
        self.assertFalse(config.needs_patient_retrieval)
        self.assertFalse(config.needs_guidelines)
        self.assertEqual(config.variant_settings["custom_a"]["custom_source"], {
            "limit": 7, "strategy": "focused", "include_scores": False, "weight": 0.5})
        self.assertEqual(config.metadata(ROOT)["variant_settings"], config.variant_settings)
        self.assertEqual(config.variant_metadata(ROOT)["variant_definitions"].keys(), {"llm_only"})

    def test_shared_resource_and_credential_requirements_follow_component_not_variant_name(self):
        both = ComponentSpec("Different patient search", ("text",), resources=("patient_retrieval", "guidelines"),
                             credentials=("CUSTOM_API_KEY", "OPENAI_API_KEY", "CUSTOM_API_KEY"))
        self.register(COMPONENTS, {"custom_source": both})
        config = self.load(selected_variants=["custom_a", "custom_b"])
        self.assertTrue(config.needs("custom_source"))
        self.assertFalse(config.needs("patient_retrieval"))
        self.assertTrue(config.needs_patient_retrieval)
        self.assertTrue(config.needs_guidelines)
        self.assertEqual(config.required_resources, {"patient_retrieval", "guidelines"})
        self.assertEqual(config.required_credentials, ["CUSTOM_API_KEY", "OPENAI_API_KEY"])

    def test_generic_component_requirements_integrate_with_existing_credential_preflight(self):
        config = self.load(selected_variants=["custom_a"])
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "CUSTOM_API_KEY"):
            configured_model_credentials({"local": {"api_base": "http://localhost:8000/v1"}}, ["local"],
                                         required_shared_keys=config.required_credentials)
        with patch.dict(os.environ, {"CUSTOM_API_KEY": "PRIVATE_CUSTOM_KEY"}, clear=True):
            self.assertEqual(configured_model_credentials(
                {"local": {"api_base": "http://localhost:8000/v1"}}, ["local"],
                required_shared_keys=config.required_credentials), {"local": (None, "local-no-key")})
        self.assertNotIn("PRIVATE_CUSTOM_KEY", json.dumps(config.metadata(ROOT)))

    def test_metadata_names_actual_template_and_complete_component_registry_provenance(self):
        implementation = self.root / "components/custom.py"
        implementation.parent.mkdir()
        implementation.write_text('"""Placeholder implementation identity."""\n')
        config = self.load(selected_variants=["custom_b", "llm_only"],
                           variant_settings={"custom_b": {"custom_source": {"limit": 5}}})
        metadata = config.variant_metadata(self.root)
        definition = metadata["variant_definitions"]["custom_b"]
        self.assertEqual(definition["description"], "Description custom_b")
        self.assertEqual(definition["template"], "VARIANT_TEMPLATES['guidelines']")
        self.assertEqual(definition["sha256"], hashlib.sha256(VARIANT_TEMPLATES["guidelines"].encode()).hexdigest())
        self.assertEqual(definition["bindings"], {"Key_info": ["custom_source", "text"]})
        instance = definition["components"]["custom_source"]
        component = metadata["component_definitions"][instance]
        self.assertEqual(component["component"], "custom_source")
        self.assertEqual(component["settings"]["limit"], 5)
        self.assertEqual(component["outputs"], ["text"])
        self.assertEqual(component["resources"], ["guidelines"])
        self.assertEqual(component["credentials"], ["CUSTOM_API_KEY"])
        self.assertEqual(component["implementation"], "components/custom.py")
        self.assertEqual(component["implementation_sha256"], hashlib.sha256(implementation.read_bytes()).hexdigest())
        registry_path = ROOT / "scripts/med_copilot_variants.py"
        self.assertEqual(metadata["registry_provenance"], {
            "file": str(registry_path), "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest()})
        component["settings"]["limit"] = 1
        definition["bindings"]["Key_info"][0] = "changed"
        again = config.variant_metadata(self.root)
        self.assertEqual(again["component_definitions"][instance]["settings"]["limit"], 5)
        self.assertEqual(again["variant_definitions"]["custom_b"]["bindings"], {"Key_info": ["custom_source", "text"]})

    def test_missing_implementation_fingerprint_is_explicit_without_loading_component_code(self):
        config = self.load(selected_variants=["custom_a"])
        metadata = config.variant_metadata(self.root)
        self.assertIsNone(metadata["component_definitions"]["custom_source"]["implementation_sha256"])
        self.assertFalse((self.root / "components/custom.py").exists())

    def test_variant_settings_reject_wrong_shapes_unknown_variants_components_and_parameters(self):
        invalid = (None, [], "custom_a", {"unknown": {}}, {"custom_a": []},
                   {"custom_a": {"unknown_component": {}}},
                   {"custom_a": {"custom_source": None}},
                   {"custom_a": {"custom_source": {"api_key": "PRIVATE_KEY"}}},
                   {"llm_only": {"custom_source": {}}})
        for configured in invalid:
            with self.subTest(configured=configured), self.assertRaises(ValueError):
                self.load(selected_variants=["custom_a"], variant_settings=configured)

    def test_variant_settings_validate_types_ranges_and_inactive_overrides(self):
        bad = {"limit": [0, 9, True, 3.0, "3", None],
               "strategy": ["", " \n", 1, False, [], None],
               "include_scores": [0, 1, "false", None],
               "weight": [-0.1, 1.1, True, float("inf"), float("nan"), "0.5", None]}
        for name, values in bad.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "custom_a.custom_source"):
                    self.load(selected_variants=["llm_only"],
                              variant_settings={"custom_a": {"custom_source": {name: value}}})

    def test_duplicate_nested_variant_parameter_keys_are_rejected(self):
        self.path.write_text("selected_models: [mini]\ntemperature: 1\nmax_retries: 3\nmax_patients: null\n"
                             "selected_variants: [custom_a]\nvariant_settings:\n  custom_a:\n    custom_source:\n"
                             "      limit: 3\n      limit: 4\n")
        with self.assertRaisesRegex(ValueError, "Duplicate YAML"):
            ExperimentConfig(self.path)

    def test_invalid_variant_bindings_and_component_declarations_fail_before_execution(self):
        invalid = (
            VariantSpec("", (), "llm_only"),
            VariantSpec("Duplicate", ("custom_source", "custom_source"), "guidelines", {"Key_info": ("custom_source", "text")}),
            VariantSpec("Missing", ("absent",), "guidelines", {"Key_info": ("absent", "text")}),
            VariantSpec("Unknown prompt", (), "absent"),
            VariantSpec("Missing binding", ("custom_source",), "guidelines", {}),
            VariantSpec("Extra binding", (), "llm_only", {"extra": ("custom_source", "text")}),
            VariantSpec("Reserved condition", ("custom_source",), "guidelines", {
                "conditions": ("custom_source", "text"), "Key_info": ("custom_source", "text")}),
            VariantSpec("Missing output", ("custom_source",), "guidelines", {"Key_info": ("custom_source", "missing")}),
            VariantSpec("Wrong source", ("custom_source",), "guidelines", {"Key_info": ("absent", "text")}),
            VariantSpec("Wrong tuple", ("custom_source",), "guidelines", {"Key_info": ["custom_source", "text"]}),
            VariantSpec("No dependencies", (), "guidelines", {"Key_info": ("absent", "text")}),
        )
        for definition in invalid:
            with self.subTest(definition=definition), patch.dict(VARIANTS, {"invalid": definition}), self.assertRaises(ValueError):
                self.load(selected_variants=["invalid"], variant_settings={})
        for outputs in ((), ("text", "text"), ("",), (1,)):
            with self.subTest(outputs=outputs), patch.dict(COMPONENTS, {"custom_source": ComponentSpec("Bad", outputs)}), self.assertRaises(ValueError):
                self.load(selected_variants=["custom_a"], variant_settings={})

    def test_prompt_fields_cannot_traverse_patient_data_or_override_target_conditions(self):
        self.register(VARIANT_TEMPLATES, {"unsafe": "Patient: {conditions.subjective}; {Key_info}"})
        self.register(VARIANTS, {"unsafe": VariantSpec("Unsafe", ("custom_source",), "unsafe", {
            "Key_info": ("custom_source", "text")})})
        with self.assertRaisesRegex(ValueError, "Unsupported prompt field"):
            self.load(selected_variants=["unsafe"], variant_settings={})

    def test_malformed_registry_structures_raise_configuration_errors_not_runtime_type_errors(self):
        invalid_components = (
            None, object(),
            ComponentSpec("", ("text",)), ComponentSpec(None, ("text",)),
            ComponentSpec("Bad", "text"), ComponentSpec("Bad", ["text"]),
            ComponentSpec("Bad", ([],)),
            ComponentSpec("Bad", ("text",), parameters=None),
            ComponentSpec("Bad", ("text",), parameters=["limit"]),
            ComponentSpec("Bad", ("text",), parameters={"limit": 1}),
            ComponentSpec("Bad", ("text",), resources="guidelines"),
            ComponentSpec("Bad", ("text",), resources=(None,)),
            ComponentSpec("Bad", ("text",), resources=(" ",)),
            ComponentSpec("Bad", ("text",), credentials="CUSTOM_API_KEY"),
            ComponentSpec("Bad", ("text",), credentials=(1,)),
            ComponentSpec("Bad", ("text",), credentials=("BAD KEY",)),
            ComponentSpec("Bad", ("text",), implementation=None),
            ComponentSpec("Bad", ("text",), implementation=" "),
        )
        for component in invalid_components:
            with self.subTest(component=component), patch.dict(COMPONENTS, {"custom_source": component}), self.assertRaises(ValueError):
                self.load(selected_variants=["custom_a"], variant_settings={})
        invalid_variants = (
            None, object(), VariantSpec("Bad", None, "llm_only"),
            VariantSpec("Bad", ["custom_source"], "guidelines", {"Key_info": ("custom_source", "text")}),
            VariantSpec("Bad", ([],), "llm_only"),
            VariantSpec("Bad", (), [], {}),
            VariantSpec("Bad", (), "llm_only", None),
        )
        for variant in invalid_variants:
            with self.subTest(variant=variant), patch.dict(VARIANTS, {"invalid": variant}), self.assertRaises(ValueError):
                self.load(selected_variants=["invalid"], variant_settings={})

    def test_blank_and_reserved_variant_ids_are_rejected_even_when_registered_or_inactive(self):
        for variant_id in ("", " \t\n", "not_recorded"):
            definition = VariantSpec("Registered but invalid identity", (), "llm_only")
            with self.subTest(variant_id=variant_id), patch.dict(VARIANTS, {variant_id: definition}):
                with self.assertRaisesRegex(ValueError, "selected_variants"):
                    self.load(selected_variants=[variant_id], variant_settings={})
                with self.assertRaisesRegex(ValueError, "variant_settings"):
                    self.load(selected_variants=["llm_only"], variant_settings={variant_id: {}})

    def test_malformed_parameter_kinds_shared_fields_and_bounds_raise_clean_configuration_errors(self):
        parameters = (
            Parameter("unknown", 3), Parameter(None, 3), Parameter([], 3), Parameter({}, 3),
            Parameter("int", 3, config_field="missing_field"),
            Parameter("int", 3, config_field=""), Parameter("int", 3, config_field=1),
            Parameter("int", 3, config_field=[]), Parameter("int", 3, config_field="component_instances"),
            Parameter("int", 3, minimum="1"), Parameter("int", 3, minimum=True),
            Parameter("int", 3, maximum="8"), Parameter("int", 3, maximum=False),
            Parameter("number", 0.5, minimum=float("nan")),
            Parameter("number", 0.5, maximum=float("inf")),
            Parameter("int", 3, minimum=4, maximum=2),
            Parameter("int", 3, maximum=10 ** 400),
        )
        for parameter in parameters:
            component = ComponentSpec("Malformed parameter", ("text",), {"value": parameter})
            with self.subTest(parameter=parameter), patch.dict(COMPONENTS, {"custom_source": component}), self.assertRaises(ValueError):
                self.load(selected_variants=["custom_a"], variant_settings={})

    def test_component_parameter_can_reuse_shared_temperature_without_config_loader_changes(self):
        component = ComponentSpec("Intermediate model", ("text",), {
            "temperature": Parameter("number", config_field="temperature", minimum=0, maximum=2)})
        with patch.dict(COMPONENTS, {"custom_source": component}):
            config = self.load(selected_variants=["custom_a", "custom_b"], temperature=0.35,
                               variant_settings={"custom_b": {"custom_source": {"temperature": 0.7}}})
        self.assertEqual(config.component_instances("custom_source")[0],
                         ("custom_source", {"temperature": 0.35}))
        self.assertEqual(len(config.component_instances("custom_source")), 2)
        self.assertEqual(config.metadata(ROOT)["temperature"], 0.35)
        self.assertEqual(config.metadata(ROOT)["variant_settings"], {
            "custom_a": {"custom_source": {"temperature": 0.35}},
            "custom_b": {"custom_source": {"temperature": 0.7}},
        })


class PipelineSettingsTests(unittest.TestCase):
    setUp = ExperimentConfigTests.setUp
    write = ExperimentConfigTests.write

    def test_legacy_config_gets_all_effective_pipeline_defaults_and_metadata(self):
        config = ExperimentConfig(self.path)
        expected = {
            "guidelines": "med_copilot/data/guidelines",
            "experiment_output": "experiments/output",
            "patient_corpus": "med_copilot/upstream/soap_with_metadata.json",
            "patient_retrieval_artifacts_dir": "med_copilot/data/patient_retrieval_so",
            "guideline_question_model": "gpt-4o-mini",
            "patient_embedding_model": "emilyalsentzer/Bio_ClinicalBERT",
            "patient_keyword_model": "gpt-4o-mini",
            "patient_reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "patient_retrieval_alpha": 0.5, "patient_retrieval_top_k": 20,
            "patient_rerank_top_k": 5, "guideline_community_level": 2,
            "guideline_response_type": "Multiple Paragraphs", "guideline_verbose": True,
        }
        self.assertEqual(PIPELINE_DEFAULTS, expected)
        for field, value in expected.items():
            self.assertEqual(config.metadata(ROOT)[field], value)
            self.assertEqual(getattr(config, field), ROOT / value if field in PATH_SETTINGS else value)
        self.assertEqual(len(config.metadata(ROOT)), 27)  # 24 effective settings + three provenance fields

    def test_paths_resolve_from_repo_root_and_absolute_paths_are_preserved_from_other_cwd(self):
        for field in PATH_SETTINGS:
            self.data[field] = "alternate resources café/" + field
        self.data["guidelines"] = str(self.root / "external guidelines")
        self.write()
        with chdir(self.root):
            config = ExperimentConfig(self.path)
        for field in PATH_SETTINGS:
            self.assertEqual(getattr(config, field), (ROOT / self.data[field]).resolve())
            self.assertEqual(config.metadata(ROOT)[field], self.data[field])
        self.assertFalse((ROOT / "alternate resources café").exists())

    def test_every_new_setting_validates_types_and_ranges_before_runtime_initialization(self):
        bad_values = {field: [None, "", "  ", [], 3, False] for field in (
            "guidelines", "experiment_output", "patient_corpus", "patient_retrieval_artifacts_dir",
            "guideline_question_model", "patient_embedding_model", "patient_keyword_model", "patient_reranker_model",
            "guideline_response_type")}
        bad_values.update({
            "patient_retrieval_alpha": [None, "0.5", True, -0.1, 1.1, float("nan"), float("inf"), []],
            "patient_retrieval_top_k": [None, 0, -1, True, 1.0, "20"],
            "patient_rerank_top_k": [None, 0, -1, False, 1.0, "5"],
            "guideline_community_level": [None, -1, True, 1.0, "2"],
            "guideline_verbose": [None, 0, 1, "false", []],
        })
        original = dict(self.data)
        for field, values in bad_values.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.data = {**original, field: value}
                    self.write()
                    with self.assertRaisesRegex(ValueError, field):
                        ExperimentConfig(self.path)

    def test_valid_boundaries_and_free_form_guideline_response_types_are_recorded_exactly(self):
        for alpha in (0, 0.0, 0.25, 1, 1.0):
            for response_type in ("Single Paragraph", "Single Sentence", "List of 3-7 Points", "Custom format café"):
                self.data.update(patient_retrieval_alpha=alpha, patient_retrieval_top_k=1,
                                 patient_rerank_top_k=50, guideline_community_level=0,
                                 guideline_verbose=False, guideline_response_type=response_type)
                self.write()
                metadata = ExperimentConfig(self.path).metadata(ROOT)
                for field, value in self.data.items():
                    self.assertEqual(metadata[field], value)

    def test_new_pipeline_fields_do_not_expand_the_confirmation_display(self):
        self.data.update(PIPELINE_DEFAULTS)
        self.write()
        config = ExperimentConfig(self.path)
        models = {"models": {"mini": {"api_key_env": "KEY", "config": {"model_provider": "openai", "model": "mini"}}}}
        with redirect_stdout(StringIO()) as out, patch("builtins.input", return_value="y"):
            self.assertTrue(confirm_experiment(config, models, ROOT))
        for field in PIPELINE_DEFAULTS:
            self.assertNotIn(field + ":", out.getvalue())
        self.assertIn("patient_input:", out.getvalue())
        self.assertIn("temperature: 1", out.getvalue())


if __name__ == "__main__":
    unittest.main()

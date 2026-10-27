"""Offline tests for conditional, ordered patient-encoder initialization."""

import ast
import builtins
from dataclasses import replace
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import yaml

# Direct execution needs project imports before loading the startup module.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.med_copilot_config import ExperimentConfig
from scripts.med_copilot_variants import COMPONENTS, VARIANTS


STARTUP = ROOT / "scripts/med_copilot_startup.py"


class PatientRetrievalStartupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "experiment.yaml"
        self.data = dict(selected_models=["mini"], temperature=1, max_retries=3, max_patients=1,
                         selected_variants=["patient_retrieval"], patient_embedding_model="custom/encoder")
        self.write()
        self.encoder = object()
        self.factory = Mock(return_value=self.encoder)
        sentence = ModuleType("sentence_transformers")
        sentence.SentenceTransformer = self.factory
        original_import = builtins.__import__
        def tracked(name, *args, **kwargs):
            self.assertFalse(name == "faiss" or name == "create_embeddings" or name == "scripts.med_copilot_retrieval")
            return original_import(name, *args, **kwargs)
        with patch.dict(sys.modules, {"sentence_transformers": sentence}), patch("builtins.__import__", side_effect=tracked):
            self.startup_class = runpy.run_path(str(STARTUP))["PatientRetrievalStartup"]
        self.factory.assert_not_called()

    def write(self):
        self.path.write_text(yaml.safe_dump(self.data))

    def test_custom_model_is_loaded_once_and_reused_for_matching_settings(self):
        before = self.path.read_bytes()
        startup = self.startup_class(self.path)
        self.factory.assert_called_once_with("custom/encoder")
        self.assertIs(startup.encoder, self.encoder)
        self.assertIsNone(startup.encoder_error)
        self.assertEqual(startup.experiment.patient_embedding_model, "custom/encoder")
        config = ExperimentConfig(self.path)
        for _ in range(2):
            self.assertIs(startup.encoder_for(config), self.encoder)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_llm_only_and_guideline_only_skip_model_loading(self):
        for variant in ("llm_only", "guidelines"):
            self.data["selected_variants"] = [variant]
            self.write()
            startup = self.startup_class(self.path)
            self.assertIsNotNone(startup.experiment)
            self.assertIsNone(startup.encoder)
            self.assertIsNone(startup.encoder_error)
        self.factory.assert_not_called()

    def test_missing_and_invalid_configuration_defer_reporting_without_loading_models(self):
        for source in (None, "invalid: [", "selected_models: []"):
            with self.subTest(source=source):
                if source is None:
                    self.path.unlink()
                else:
                    self.path.write_text(source)
                startup = self.startup_class(self.path)
                self.assertIsNone(startup.experiment)
                self.assertIsNone(startup.encoder)
                self.assertIsNone(startup.encoder_error)
        self.factory.assert_not_called()

    def test_encoder_failure_is_retained_and_only_raised_when_retrieval_requests_it(self):
        error = RuntimeError("model unavailable")
        self.factory.side_effect = error
        startup = self.startup_class(self.path)
        self.assertIs(startup.encoder_error, error)
        self.assertIsNone(startup.encoder)
        with self.assertRaises(RuntimeError) as caught:
            startup.encoder_for(ExperimentConfig(self.path))
        self.assertIs(caught.exception, error)

    def test_interruptions_during_preload_are_not_swallowed(self):
        for error in (KeyboardInterrupt(), SystemExit(2)):
            self.factory.side_effect = error
            with self.assertRaises(type(error)) as caught:
                self.startup_class(self.path)
            self.assertIs(caught.exception, error)

    def test_changed_encoder_settings_require_restart_before_any_cache_can_be_mislabelled(self):
        startup = self.startup_class(self.path)
        self.data["patient_embedding_model"] = "another/encoder"
        self.write()
        with self.assertRaisesRegex(ValueError, "restart the script"):
            startup.encoder_for(ExperimentConfig(self.path))
        self.factory.assert_called_once_with("custom/encoder")
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_enabling_retrieval_after_baseline_startup_requires_restart(self):
        self.data["selected_variants"] = ["llm_only"]
        self.write()
        startup = self.startup_class(self.path)
        self.data["selected_variants"] = ["patient_retrieval"]
        self.write()
        with self.assertRaisesRegex(ValueError, "restart the script"):
            startup.encoder_for(ExperimentConfig(self.path))
        self.factory.assert_not_called()

    def test_non_encoder_config_changes_can_reuse_the_preloaded_model(self):
        startup = self.startup_class(self.path)
        self.data.update(patient_retrieval_alpha=0.2, selected_models=["other"], guideline_question_model="other")
        self.write()
        self.assertIs(startup.encoder_for(ExperimentConfig(self.path)), self.encoder)
        self.factory.assert_called_once()

    def test_new_variant_ids_reuse_the_same_encoder_for_shared_patient_component(self):
        variants = {
            "custom_patient": replace(VARIANTS["patient_retrieval"], description="Custom patient variant"),
            "custom_combined": replace(VARIANTS["patient_retrieval_guidelines"], description="Custom combined variant"),
        }
        self.data["selected_variants"] = ["patient_retrieval", *variants]
        self.write()
        with patch.dict(VARIANTS, variants):
            startup = self.startup_class(self.path)
            config = ExperimentConfig(self.path)
            self.assertTrue(config.needs_patient_retrieval)
            self.assertIs(startup.encoder_for(config), self.encoder)
            self.assertIs(startup.encoder_for(config), self.encoder)
        self.factory.assert_called_once_with("custom/encoder")

    def test_custom_components_select_encoder_by_resource_not_component_or_variant_name(self):
        patient_component = replace(COMPONENTS["patient_retrieval"], description="Replacement patient component")
        guideline_component = replace(COMPONENTS["guidelines"], description="Replacement guideline component")
        variants = {
            "custom_patient": replace(
                VARIANTS["patient_retrieval"], components=("custom_patient_source",),
                bindings={"example": ("custom_patient_source", "example")},
            ),
            "custom_guidelines": replace(
                VARIANTS["guidelines"], components=("custom_guideline_source",),
                bindings={"Key_info": ("custom_guideline_source", "Key_info")},
            ),
            "custom_prompt_only": replace(VARIANTS["llm_only"], description="Custom S/O prompt"),
        }
        with patch.dict(COMPONENTS, {"custom_patient_source": patient_component,
                                     "custom_guideline_source": guideline_component}), patch.dict(VARIANTS, variants):
            for variant, requires_encoder in (("custom_guidelines", False), ("custom_prompt_only", False),
                                               ("custom_patient", True)):
                with self.subTest(variant=variant):
                    self.data["selected_variants"] = [variant]
                    self.write()
                    startup = self.startup_class(self.path)
                    self.assertIsNotNone(startup.experiment)
                    self.assertEqual(startup.experiment.needs_patient_retrieval, requires_encoder)
                    self.assertIs(startup.encoder, self.encoder if requires_encoder else None)
        self.factory.assert_called_once_with("custom/encoder")

    def test_imports_are_in_header_and_no_faiss_dependency_is_loaded(self):
        tree = ast.parse(STARTUP.read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertTrue(all(node in tree.body for node in imports))
        first_definition = min(node.lineno for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)))
        self.assertTrue(all(node.lineno < first_definition for node in imports))


if __name__ == "__main__":
    unittest.main()

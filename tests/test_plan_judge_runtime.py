"""Offline tests for the independent Plan judge's saved runtime profile."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

# Direct execution needs the repository root before project imports.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_runtime.ollama import ManagedOllama, ollama_for_saved_run
from scripts.experiment_runtime.plan_judging import load_judge_profile, validate_profile


PROFILE = {"concurrency": 4, "context_length": 4096, "flash_attention": True,
           "kv_cache_type": "f16", "host_cache_mib": 2048}


class PlanJudgeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = self.directory / "plan_judging.yaml"
        self.config.write_text(json.dumps(PROFILE))

    def snapshot(self, settings=None):
        return {"settings": deepcopy(PROFILE if settings is None else settings),
                "config_file": str(self.config.resolve()),
                "config_sha256": hashlib.sha256(self.config.read_bytes()).hexdigest()}

    def test_shipped_profile_matches_the_measured_setup(self):
        profile = load_judge_profile(ROOT / "experiments/configs/plan_judging.yaml", {})
        self.assertEqual(profile["settings"], {**PROFILE, "concurrency": 1,
                                              "client_workers": 2, "host_cache_mib": 512})

    def test_separate_workers_are_saved_and_do_not_allocate_extra_gpu_slots(self):
        settings = {**PROFILE, "concurrency": 1, "client_workers": 2, "host_cache_mib": 512}
        self.config.write_text(json.dumps(settings))
        profile = load_judge_profile(self.config, {})
        self.assertEqual(profile["settings"], settings)
        server = ManagedOllama({}, self.directory / "judge.log", parallel=1,
                               context=4096, profile=profile["settings"])
        self.assertEqual(server._environment()["OLLAMA_NUM_PARALLEL"], "1")
        self.config.unlink()
        resumed = load_judge_profile(self.config, {"plan_evaluation": {"runtime_profile": profile}})
        self.assertEqual(resumed, profile)

    def test_productive_trial_workers_migrate_without_reading_config_or_changing_native_profile(self):
        saved = self.snapshot({**PROFILE, "concurrency": 1, "host_cache_mib": 512})
        metadata = {"plan_evaluation": {"runtime_profile": saved, "concurrency": 2,
                                       "client_workers": 2, "gpu_slots": 1}}
        before = deepcopy(metadata)
        self.config.unlink()
        result = load_judge_profile(self.config, metadata)
        self.assertEqual(result["settings"], {**saved["settings"], "client_workers": 2})
        self.assertEqual(metadata, before)

    def test_legacy_stage_without_profile_reuses_separate_native_slots(self):
        metadata = {"plan_evaluation": {"concurrency": 2, "client_workers": 2, "gpu_slots": 1}}
        settings = load_judge_profile(self.config, metadata)["settings"]
        self.assertEqual((settings["concurrency"], settings["client_workers"]), (1, 2))

    def test_worker_validation_and_override_do_not_change_slots_or_hide_invalid_saved_values(self):
        settings = {**PROFILE, "concurrency": 1, "client_workers": 2}
        saved = self.snapshot(settings)
        result = load_judge_profile(self.config, {"plan_evaluation": {"runtime_profile": saved}},
                                    client_workers=4)
        self.assertEqual(result["settings"], {**settings, "client_workers": 4})
        self.assertEqual(saved["settings"], settings)
        for value in (True, False, 0, 65, 2.0, "2", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "client_workers"):
                validate_profile({**settings, "client_workers": value})
        for value in (True, 0, 65, 2.0, "2", None):
            metadata = {"plan_evaluation": {"runtime_profile": self.snapshot(), "client_workers": value}}
            with self.subTest(saved=value), self.assertRaisesRegex(ValueError, "client_workers"):
                load_judge_profile(self.config, metadata, client_workers=2)

    def test_first_stage_uses_file_not_frozen_generation_concurrency(self):
        metadata = {"experiment": {"plan_judge_concurrency": 8, "execution": {
            "ollama_context_length": 16384, "ollama_judge_context_length": 8192}}}
        before = deepcopy(metadata)
        self.assertEqual(load_judge_profile(self.config, metadata), self.snapshot())
        self.assertEqual(metadata, before)

    def test_fingerprint_is_of_exact_file_bytes(self):
        original = self.config.read_bytes()
        self.config.write_bytes(original + b"\n")
        profile = load_judge_profile(self.config, {})
        self.assertEqual(profile["config_sha256"], hashlib.sha256(original + b"\n").hexdigest())
        self.assertEqual(profile["settings"], PROFILE)

    def test_resume_uses_saved_profile_without_reading_current_config(self):
        saved = self.snapshot({**PROFILE, "concurrency": 2, "host_cache_mib": 1024})
        metadata = {"plan_evaluation": {"runtime_profile": saved, "concurrency": 8}}
        before = deepcopy(metadata)
        self.config.unlink()
        with patch("scripts.experiment_runtime.plan_judging.read_yaml_config",
                   side_effect=AssertionError("saved profile must not reread current config")):
            actual = load_judge_profile(self.config, metadata)
        self.assertEqual(actual, saved)
        actual["settings"]["host_cache_mib"] = 0
        self.assertEqual(metadata, before)

    def test_overrides_are_independent_and_do_not_modify_saved_profile(self):
        saved = self.snapshot({**PROFILE, "concurrency": 2, "context_length": 8192})
        metadata = {"plan_evaluation": {"runtime_profile": saved}}
        before = deepcopy(metadata)
        self.assertEqual(load_judge_profile(self.config, metadata, concurrency=8)["settings"],
                         {**saved["settings"], "concurrency": 8})
        self.assertEqual(load_judge_profile(self.config, metadata, context_length=2048)["settings"],
                         {**saved["settings"], "context_length": 2048})
        self.assertEqual(load_judge_profile(self.config, metadata, concurrency=4,
                                           context_length=4096)["settings"], PROFILE)
        self.assertEqual(metadata, before)

    def test_legacy_plan_stage_keeps_its_explicit_runtime_values(self):
        metadata = {"plan_evaluation": {"concurrency": 8, "context_length": 8192}}
        before = deepcopy(metadata)
        expected = {**PROFILE, "concurrency": 8, "context_length": 8192}
        self.assertEqual(load_judge_profile(self.config, metadata)["settings"], expected)
        self.assertEqual(load_judge_profile(self.config, metadata, concurrency=4)["settings"],
                         {**expected, "concurrency": 4})
        self.assertEqual(metadata, before)

    def test_validation_returns_an_independent_copy(self):
        source = deepcopy(PROFILE)
        actual = validate_profile(source)
        actual["concurrency"] = 8
        self.assertEqual(source, PROFILE)

    def test_all_fields_are_required_and_extra_fields_rejected(self):
        for field in PROFILE:
            settings = deepcopy(PROFILE)
            settings.pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                validate_profile(settings)
        for settings in (None, [], "profile", {**PROFILE, "api_key": "private"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_profile(settings)

    def test_numeric_boundaries_and_types(self):
        for field, low, high in (("concurrency", 1, 64), ("context_length", 512, 262144),
                                 ("host_cache_mib", 0, 8192)):
            for valid in (low, high):
                with self.subTest(field=field, valid=valid):
                    self.assertEqual(validate_profile({**PROFILE, field: valid})[field], valid)
            for invalid in (True, False, str(low), float(low), None, low - 1, high + 1):
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    validate_profile({**PROFILE, field: invalid})

    def test_flash_attention_boolean_and_quality_tested_kv_type(self):
        self.assertFalse(validate_profile({**PROFILE, "flash_attention": False})["flash_attention"])
        for invalid in (0, 1, "true", None):
            with self.subTest(flash=invalid), self.assertRaises(ValueError):
                validate_profile({**PROFILE, "flash_attention": invalid})
        for invalid in (None, "q8_0", "q4_0", "F16", 16):
            with self.subTest(kv=invalid), self.assertRaises(ValueError):
                validate_profile({**PROFILE, "kv_cache_type": invalid})

    def test_invalid_config_and_duplicate_yaml_keys_are_rejected(self):
        for text in ("concurrency: 4\nconcurrency: 8\n", "not: [valid", "null", "[]"):
            self.config.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_judge_profile(self.config, {})

    def test_invalid_saved_profile_envelope_is_rejected_without_fallback(self):
        saved = self.snapshot()
        cases = [False, [], "profile", {}, {**saved, "extra": 1},
                 {**saved, "config_file": None}, {**saved, "config_sha256": "bad"},
                 {**saved, "config_sha256": "G" * 64}]
        for malformed in cases:
            with self.subTest(saved=malformed), self.assertRaises(ValueError):
                load_judge_profile(self.config, {"plan_evaluation": {"runtime_profile": malformed}})

    def test_invalid_saved_settings_are_rejected_even_with_an_explicit_override(self):
        for malformed in ([], None, "profile", {**PROFILE, "flash_attention": "true"}):
            saved = {**self.snapshot(), "settings": malformed}
            with self.subTest(settings=malformed), self.assertRaises(ValueError):
                load_judge_profile(self.config, {"plan_evaluation": {"runtime_profile": saved}}, concurrency=4)

    def test_malformed_stage_is_rejected_without_falling_back_to_file(self):
        for malformed in ([], False, 0, "stage"):
            with self.subTest(stage=malformed), self.assertRaises(ValueError):
                load_judge_profile(self.config, {"plan_evaluation": malformed})

    def test_metadata_must_be_an_object(self):
        for malformed in (None, [], False, "metadata"):
            with self.subTest(metadata=malformed), self.assertRaises(ValueError):
                load_judge_profile(self.config, malformed)

    def test_invalid_overrides_rejected_before_files_or_processes_change(self):
        before = self.config.read_bytes()
        with patch("scripts.experiment_runtime.ollama.subprocess.Popen") as start:
            for override in (True, 0, 65, "4", 4.0):
                with self.subTest(concurrency=override), self.assertRaises(ValueError):
                    load_judge_profile(self.config, {}, concurrency=override)
            for override in (True, 0, 511, 262145, "4096", 4096.0):
                with self.subTest(context=override), self.assertRaises(ValueError):
                    load_judge_profile(self.config, {}, context_length=override)
            start.assert_not_called()
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual(list(self.directory.iterdir()), [self.config])

    def test_profile_is_forwarded_explicitly_and_parent_tuning_is_ignored(self):
        settings = {"manage_ollama": True, "ollama_host": "127.0.0.1:11435"}
        parent = {"PATH": "/bin", "HOME": "/preserved/home", "OLLAMA_FLASH_ATTENTION": "0",
                  "OLLAMA_KV_CACHE_TYPE": "q4_0", "LLAMA_ARG_CACHE_RAM": "8192",
                  "OLLAMA_DEBUG": "0", "OLLAMA_NUM_PARALLEL": "64", "OPENAI_API_KEY": "secret"}
        before = deepcopy(settings)
        with patch.dict(os.environ, parent, clear=True):
            server = ManagedOllama(settings, self.directory / "judge.log", parallel=4,
                                   context=4096, profile=PROFILE)
            actual = server._environment()
            self.assertEqual({key: actual[key] for key in (
                "OLLAMA_NUM_PARALLEL", "OLLAMA_CONTEXT_LENGTH", "OLLAMA_FLASH_ATTENTION",
                "OLLAMA_KV_CACHE_TYPE", "LLAMA_ARG_CACHE_RAM", "OLLAMA_DEBUG")}, {
                    "OLLAMA_NUM_PARALLEL": "4", "OLLAMA_CONTEXT_LENGTH": "4096",
                    "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "f16",
                    "LLAMA_ARG_CACHE_RAM": "2048", "OLLAMA_DEBUG": "1"})
            self.assertNotIn("OPENAI_API_KEY", actual)
            self.assertEqual(dict(os.environ), parent)
            ordinary = ManagedOllama(settings, self.directory / "generation.log")
            generated = ordinary._environment()
            for name in ("OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE", "LLAMA_ARG_CACHE_RAM", "OLLAMA_DEBUG"):
                self.assertNotIn(name, generated)
        self.assertEqual(settings, before)
        self.assertEqual(list(self.directory.iterdir()), [self.config])

    def test_disabled_flash_and_zero_host_cache_are_explicit(self):
        profile = {**PROFILE, "flash_attention": False, "host_cache_mib": 0}
        server = ManagedOllama({}, self.directory / "judge.log", parallel=4, context=4096, profile=profile)
        self.assertEqual(server._environment()["OLLAMA_FLASH_ATTENTION"], "0")
        self.assertEqual(server._environment()["LLAMA_ARG_CACHE_RAM"], "0")

    def test_profile_and_server_slots_context_must_match(self):
        for slots, context in ((8, 4096), (4, 8192)):
            with self.subTest(slots=slots, context=context), self.assertRaisesRegex(ValueError, "match"):
                ManagedOllama({}, self.directory / "judge.log", parallel=slots, context=context, profile=PROFILE)

    def test_saved_run_forwards_profile_without_rewriting_generation_metadata(self):
        metadata = {"experiment": {"plan_judge_concurrency": 8, "execution": {
            "manage_ollama": True, "ollama_host": "127.0.0.1:11435", "ollama_context_length": 16384}}}
        path = self.directory / "run.json"
        path.write_text(json.dumps(metadata))
        before = path.read_bytes()
        server = Mock()
        server.__enter__ = Mock(return_value=server)
        server.__exit__ = Mock(return_value=None)
        with patch("scripts.experiment_runtime.ollama.ManagedOllama", return_value=server) as constructor:
            with ollama_for_saved_run(self.directory, concurrency=4, context_length=4096,
                                      runtime_profile=self.snapshot()) as actual:
                self.assertIs(actual, server)
            self.assertEqual(constructor.call_args.kwargs,
                             {"parallel": 4, "context": 4096, "profile": PROFILE})
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

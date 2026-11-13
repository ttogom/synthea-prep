"""Offline native-runtime verification and immutable provenance regressions."""

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

from scripts.experiment_runtime.judge_provenance import native_settings, save_verified_runtime
from scripts.experiment_runtime.ollama import ManagedOllama


PROFILE = {"concurrency": 4, "context_length": 4096, "flash_attention": True,
           "kv_cache_type": "f16", "host_cache_mib": 2048}
NATIVE_LOG = """time=2026-11-12T19:15:46 msg="unrelated server configuration"
cmn  common_param: common_params_print_info: build 1 (6c01da627) with AppleClang 21 for Darwin arm64
load_tensors: offloaded 49/49 layers to GPU
llama_context: n_seq_max             = 4
llama_context: n_ctx                 = 16384
llama_context: n_ctx_seq             = 4096
llama_context: n_batch               = 1024
llama_context: n_ubatch              = 1024
llama_context: flash_attn            = enabled
llama_kv_cache: size = 3072.00 MiB (4096 cells, 48 layers, 4/4 seqs), K (f16): 1536.00 MiB, V (f16): 1536.00 MiB
srv    load_model: prompt cache is enabled, size limit: 2048 MiB
"""


class NativeSettingsTests(unittest.TestCase):
    def test_native_values_are_parsed_and_matched(self):
        actual = native_settings(NATIVE_LOG, PROFILE)
        self.assertEqual(actual["settings"], {
            "parallel_slots": 4, "context_length": 4096, "total_context": 16384,
            "batch": 1024, "microbatch": 1024, "flash_attention": "enabled",
            "kv_cache_type": ("f16", "f16"), "gpu_layers": ("49", "49"),
            "native_build": "common_params_print_info: build 1 (6c01da627) with AppleClang 21 for Darwin arm64",
            "host_cache_mib": 2048})
        self.assertTrue(actual["native_evidence"])
        self.assertFalse(any("unrelated server" in line for line in actual["native_evidence"]))

    def test_every_native_field_requires_evidence(self):
        markers = ("build 1", "offloaded", "n_seq_max", "n_ctx ", "n_ctx_seq", "n_batch",
                   "n_ubatch", "flash_attn", "K (f16)", "prompt cache")
        for marker in markers:
            changed = "\n".join(line for line in NATIVE_LOG.splitlines() if marker not in line)
            with self.subTest(missing=marker), self.assertRaises(ValueError):
                native_settings(changed, PROFILE)

    def test_conflicting_native_values_are_rejected(self):
        lines = [line for line in NATIVE_LOG.splitlines() if "unrelated" not in line]
        for line in lines:
            changed = line.replace("49/49", "48/49").replace("16384", "32768")
            changed = changed.replace("4096", "8192").replace("1024", "512")
            changed = changed.replace("= 4", "= 8").replace("enabled", "disabled")
            changed = changed.replace("(f16)", "(q8_0)").replace("6c01da627", "different")
            with self.subTest(conflict=line), self.assertRaises(ValueError):
                native_settings(NATIVE_LOG + changed + "\n", PROFILE)

    def test_repeated_identical_evidence_is_allowed(self):
        actual = native_settings(NATIVE_LOG + NATIVE_LOG, PROFILE)
        self.assertEqual(actual["settings"], native_settings(NATIVE_LOG, PROFILE)["settings"])

    def test_mismatched_requested_profile_values_are_rejected(self):
        for field, value in (("concurrency", 2), ("context_length", 8192),
                             ("flash_attention", False), ("host_cache_mib", 1024)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                native_settings(NATIVE_LOG, {**PROFILE, field: value})

    def test_wrong_batch_and_microbatch_are_rejected(self):
        for field in ("n_batch", "n_ubatch"):
            changed = "\n".join(line.replace("1024", "512") if field in line else line
                                for line in NATIVE_LOG.splitlines())
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "batch"):
                native_settings(changed, PROFILE)

    def test_partial_or_zero_gpu_offload_is_rejected(self):
        for layers in ("48/49", "0/49", "0/0"):
            with self.subTest(layers=layers), self.assertRaisesRegex(ValueError, "GPU"):
                native_settings(NATIVE_LOG.replace("49/49", layers), PROFILE)

    def test_mixed_kv_cache_types_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "kv_cache_type"):
            native_settings(NATIVE_LOG.replace("V (f16)", "V (q8_0)"), PROFILE)

    def test_zero_host_cache_requires_explicit_disabled_evidence(self):
        profile = {**PROFILE, "host_cache_mib": 0}
        disabled = NATIVE_LOG.replace("prompt cache is enabled, size limit: 2048 MiB",
                                     "prompt cache is disabled - use `--cache-ram N` to enable it")
        self.assertEqual(native_settings(disabled, profile)["settings"]["host_cache_mib"], 0)
        with self.assertRaisesRegex(ValueError, "host prompt-cache"):
            native_settings(NATIVE_LOG, profile)
        with self.assertRaisesRegex(ValueError, "host prompt-cache"):
            native_settings(disabled, PROFILE)

    def test_disabled_flash_attention_can_be_verified(self):
        profile = {**PROFILE, "flash_attention": False}
        log = NATIVE_LOG.replace("flash_attn            = enabled", "flash_attn            = disabled")
        self.assertEqual(native_settings(log, profile)["settings"]["flash_attention"], "disabled")


class RuntimeProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.alias = "local_judge"
        self.model = "qwen2.5:14b"
        self.model_config = {"model": self.model, "api_base": "http://127.0.0.1:11435/v1",
                             "api_key": "private-provider-key"}
        self.models = Mock()
        self.models.metadata.return_value = {"models": {self.alias: {"config": self.model_config}}}
        self.server = Mock()
        self.server.enabled = True
        self.server.base_url = "http://127.0.0.1:11435"
        self.server.profile = deepcopy(PROFILE)
        self.server._log_start = 123
        self.server.log_path = self.run / "ollama-judge.log"
        self.server.invocation_log.return_value = NATIVE_LOG + "private-provider-key\n"
        self.placement = {"name": self.model, "digest": "a" * 64, "size": 9_000_000_000,
                          "size_vram": 12_000_000_000, "context_length": 16384,
                          "details": {"family": "qwen2", "quantization_level": "Q4_K_M"},
                          "unexpected": "private-provider-key"}
        self.server.ps_snapshot.return_value = {"models": [self.placement]}
        self.server.metadata.return_value = {"version": "0.40.16", "base_url": self.server.base_url,
                                              "max_loaded_models": 1, "keep_alive": -1,
                                              "private": "private-provider-key"}

    def save(self):
        return save_verified_runtime(self.server, self.models, self.alias, self.run, self.root)

    def test_unique_durable_files_contain_verified_whitelisted_evidence(self):
        with patch("scripts.experiment_runtime.judge_provenance.os.fsync", wraps=os.fsync) as sync:
            first = self.save()
            first_path = self.run / first["file"]
            original = first_path.read_bytes()
            second = self.save()
            self.assertGreaterEqual(sync.call_count, 6)  # file, provenance directory, run directory each time
        self.assertNotEqual(first["file"], second["file"])
        self.assertEqual(first_path.parent, self.run / "provenance")
        self.assertEqual(first_path.read_bytes(), original)
        self.assertEqual(first["sha256"], hashlib.sha256(original).hexdigest())
        self.assertNotIn(b"private-provider-key", original)
        document = json.loads(original)
        self.assertEqual(document["requested"], PROFILE)
        self.assertEqual(document["effective"]["parallel_slots"], 4)
        self.assertEqual(document["software"]["ollama_version"], "0.40.16")
        self.assertEqual(document["model"]["digest"], "a" * 64)
        self.assertEqual(document["log_start_byte"], 123)
        self.assertEqual(document["verified_log_sha256"],
                         hashlib.sha256(self.server.invocation_log.return_value.encode()).hexdigest())
        self.server.preload.assert_called_with(self.model)

    def test_native_mismatch_creates_no_provenance(self):
        self.server.invocation_log.return_value = NATIVE_LOG.replace("16384", "32768")
        with self.assertRaisesRegex(ValueError, "total_context"):
            self.save()
        self.assertFalse((self.run / "provenance").exists())

    def test_unmanaged_server_is_rejected_before_preload(self):
        self.server.enabled = False
        with self.assertRaisesRegex(ValueError, "manage_ollama"):
            self.save()
        self.server.preload.assert_not_called()
        self.assertFalse((self.run / "provenance").exists())

    def test_different_client_endpoint_is_rejected_before_preload(self):
        for endpoint in (None, "http://127.0.0.1:11434/v1", "https://external.example/v1", "http://127.0.0.1:11435"):
            self.model_config["api_base"] = endpoint
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(ValueError, "endpoint"):
                self.save()
        self.server.preload.assert_not_called()
        self.assertFalse((self.run / "provenance").exists())

    def test_trailing_slash_on_compatible_endpoint_is_allowed(self):
        self.model_config["api_base"] += "/"
        self.assertTrue((self.run / self.save()["file"]).exists())

    def test_wrong_missing_multiple_or_cpu_loaded_models_are_rejected(self):
        cases = [[{**self.placement, "name": "different:14b"}], [], [self.placement, self.placement],
                 [{**self.placement, "digest": ""}], [{**self.placement, "size_vram": 0}]]
        for placement in cases:
            self.server.ps_snapshot.return_value = {"models": placement}
            with self.subTest(placement=placement), self.assertRaisesRegex(ValueError, "identity/GPU"):
                self.save()
        self.assertFalse((self.run / "provenance").exists())

    def test_failed_load_does_not_write_a_success_artifact(self):
        self.server.preload.side_effect = TimeoutError("load failed")
        with self.assertRaisesRegex(TimeoutError, "load failed"):
            self.save()
        self.assertFalse((self.run / "provenance").exists())

    def test_failed_fsync_never_returns_a_reusable_reference(self):
        with patch("scripts.experiment_runtime.judge_provenance.os.fsync", side_effect=OSError("disk failed")):
            with self.assertRaisesRegex(OSError, "disk failed"):
                self.save()

    def test_uuid_collision_does_not_overwrite_existing_provenance(self):
        with patch("scripts.experiment_runtime.judge_provenance.uuid4", return_value=Mock(hex="fixed")):
            first = self.save()
            original = (self.run / first["file"]).read_bytes()
            with self.assertRaises(FileExistsError):
                self.save()
        self.assertEqual((self.run / first["file"]).read_bytes(), original)


class InvocationLogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.log = self.directory / "server.log"
        self.settings = {"manage_ollama": True, "ollama_host": "127.0.0.1:11435"}

    def test_invocation_log_ignores_successful_previous_invocation(self):
        old = NATIVE_LOG.encode()
        self.log.write_bytes(old)
        process = Mock()
        process.poll.return_value = None
        process.pid = 424242
        with patch("scripts.experiment_runtime.ollama.tempfile.gettempdir", return_value=str(self.directory)), \
                patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch.object(ManagedOllama, "_request", return_value={"version": "0.40.16"}), \
                patch("scripts.experiment_runtime.ollama.subprocess.Popen", return_value=process), \
                patch("scripts.experiment_runtime.ollama.os.killpg"):
            with ManagedOllama(self.settings, self.log, parallel=4, context=4096, profile=PROFILE) as server:
                self.assertEqual(server._log_start, len(old))
                with self.log.open("ab") as handle:
                    handle.write(b"current invocation has no native settings\n")
                current = server.invocation_log()
                self.assertEqual(current, "current invocation has no native settings\n")
                with self.assertRaises(ValueError):
                    native_settings(current, PROFILE)

    def test_preload_has_no_prompt_and_uses_server_context(self):
        server = ManagedOllama(self.settings, self.log, parallel=4, context=4096, profile=PROFILE)
        server._process = Mock()
        server._process.poll.return_value = None
        with patch.object(server, "_request", return_value={}) as request:
            server.preload("qwen2.5:14b")
            self.assertEqual(request.call_args.args, ("POST", "/api/generate", {
                "model": "qwen2.5:14b", "stream": False, "keep_alive": -1, "options": {"num_ctx": 4096}}))
            self.assertEqual(request.call_args.kwargs, {"timeout": 180})

    def test_preload_refuses_closed_or_dead_server(self):
        server = ManagedOllama(self.settings, self.log, parallel=4, context=4096, profile=PROFILE)
        with patch.object(server, "_request") as request:
            with self.assertRaisesRegex(RuntimeError, "running"):
                server.preload("qwen2.5:14b")
            server._process = Mock()
            server._process.poll.return_value = 1
            with self.assertRaisesRegex(RuntimeError, "running"):
                server.preload("qwen2.5:14b")
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()

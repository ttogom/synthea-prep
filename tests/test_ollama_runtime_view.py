"""Offline checks of the non-owning parent-server verification client."""

from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

# Direct execution starts in tests/; resolve project imports before using them.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_runtime.ollama import ManagedOllama, OllamaRuntimeView


PROFILE = {"concurrency": 4, "context_length": 4096, "flash_attention": True,
           "kv_cache_type": "f16", "host_cache_mib": 2048}


class RuntimeViewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.log = Path(self.temporary.name) / "native.log"
        self.log.write_bytes(b"old\ncurrent\n")
        self.server = ManagedOllama({"manage_ollama": True}, self.log,
                                     parallel=4, context=4096, profile=PROFILE)
        self.server._version = "0.40.16"
        self.server._log_start = 4
        self.server._process = Mock(pid=12345)
        self.server._process.poll.return_value = None
        self.descriptor = self.server.runtime_descriptor()
        self.alive = patch("scripts.experiment_runtime.ollama.os.kill", return_value=None)
        self.kill = self.alive.start()
        self.addCleanup(self.alive.stop)

    def test_descriptor_is_serializable_detached_and_contains_no_parent_environment(self):
        actual = json.loads(json.dumps(self.descriptor))
        self.assertEqual(actual["server_pid"], 12345)
        self.assertEqual(actual["log_start_byte"], 4)
        actual["profile"]["concurrency"] = 2
        self.assertEqual(self.server.profile["concurrency"], 4)
        self.assertEqual(set(actual), {"managed", "host", "parallel", "context", "max_loaded_models",
                                      "profile", "version", "log_path", "log_start_byte", "server_pid"})

    def test_two_worker_profile_crosses_process_boundary_with_one_native_slot(self):
        self.server.parallel = 1
        self.server.profile.update(concurrency=1, client_workers=2, host_cache_mib=512)
        descriptor = json.loads(json.dumps(self.server.runtime_descriptor()))
        view = OllamaRuntimeView(descriptor)
        self.assertEqual(view.parallel, 1)
        self.assertEqual(view.profile["client_workers"], 2)
        self.assertEqual(view.profile, self.server.profile)
        view.profile["client_workers"] = 1
        self.assertEqual(self.server.profile["client_workers"], 2)

    def test_runtime_descriptor_requires_own_live_server_and_profile(self):
        self.server._process.poll.return_value = 0
        with self.assertRaisesRegex(ValueError, "running owned"):
            self.server.runtime_descriptor()
        self.server._process.poll.return_value = None
        self.server.profile = None
        with self.assertRaises(ValueError):
            self.server.runtime_descriptor()
        self.server.enabled = False
        with self.assertRaises(ValueError):
            self.server.runtime_descriptor()

    def test_view_can_read_metadata_and_ps_but_cannot_manage_processes(self):
        view = OllamaRuntimeView(self.descriptor)
        tags = {"models": [{"name": "qwen2.5:14b", "digest": "digest", "size": 100}]}
        ps = {"models": [{"name": "qwen2.5:14b", "size_vram": 100}]}
        with patch.object(view, "_request", side_effect=[tags, ps]) as request:
            metadata = view.metadata()
            self.assertEqual(metadata["version"], "0.40.16")
            self.assertEqual(metadata["parallel_slots"], 4)
            self.assertEqual(metadata["models"], tags["models"])
            self.assertEqual(view.ps_snapshot(), ps)
            self.assertEqual([call.args[:2] for call in request.call_args_list],
                             [("GET", "/api/tags"), ("GET", "/api/ps")])
        for name in ("close", "unload", "__enter__", "__exit__", "runtime_descriptor"):
            self.assertFalse(hasattr(view, name))
        self.assertTrue(all(call.args == (12345, 0) for call in self.kill.call_args_list))

    def test_preload_uses_parent_endpoint_and_never_starts_or_signals_process(self):
        view = OllamaRuntimeView(self.descriptor)
        with patch.object(view, "_request", return_value={}) as request, \
                patch("scripts.experiment_runtime.ollama.subprocess.Popen") as spawn, \
                patch("scripts.experiment_runtime.ollama.os.killpg") as group:
            view.preload("qwen2.5:14b")
            self.assertEqual(request.call_args.args, ("POST", "/api/generate", {
                "model": "qwen2.5:14b", "stream": False, "keep_alive": -1,
                "options": {"num_ctx": 4096}}))
            self.assertEqual(request.call_args.kwargs, {"timeout": 180})
            spawn.assert_not_called()
            group.assert_not_called()

    def test_dead_parent_server_cannot_be_preloaded_or_used_as_stale_evidence(self):
        view = OllamaRuntimeView(self.descriptor)
        self.kill.side_effect = ProcessLookupError()
        with patch.object(view, "_request") as request:
            for method in (view.metadata, view.ps_snapshot, view.invocation_log,
                           lambda: view.preload("qwen2.5:14b")):
                with self.assertRaisesRegex(RuntimeError, "not running"):
                    method()
            request.assert_not_called()

    def test_log_reader_never_includes_previous_invocation_evidence(self):
        self.assertEqual(OllamaRuntimeView(self.descriptor).invocation_log(), "current\n")
        self.kill.assert_called_with(12345, 0)

    def test_invalid_descriptors_fail_before_network_or_signals(self):
        for field, value in (("host", "127.0.0.1:11434"), ("host", "example.com:11435"),
                             ("parallel", 2), ("context", 8192), ("server_pid", True),
                             ("server_pid", 0), ("managed", False), ("version", ""),
                             ("log_path", "relative.log"), ("log_path", None),
                             ("log_start_byte", -1), ("log_start_byte", True)):
            descriptor = deepcopy(self.descriptor)
            descriptor[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                OllamaRuntimeView(descriptor)
        for descriptor in (None, [], {}, {**self.descriptor, "secret": "private"}):
            with self.subTest(descriptor=descriptor), self.assertRaises(ValueError):
                OllamaRuntimeView(descriptor)
        self.kill.assert_not_called()

    def test_permission_error_is_a_clear_failure_without_claiming_the_server_alive(self):
        view = OllamaRuntimeView(self.descriptor)
        self.kill.side_effect = PermissionError()
        with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
            view.ps_snapshot()

    def test_forced_close_kills_owned_group_even_if_its_leader_already_exited(self):
        process = self.server._process
        process.poll.return_value = 0
        with patch("scripts.experiment_runtime.ollama.os.killpg") as group:
            self.server.close(force_group=True)
        self.assertEqual([call.args for call in group.call_args_list],
                         [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])
        self.assertIsNone(self.server._process)

    def test_forced_close_is_bounded_and_unowned_empty_server_never_signals(self):
        process = self.server._process
        with patch("scripts.experiment_runtime.ollama.os.killpg") as group, \
                patch("scripts.experiment_runtime.ollama.time.sleep") as pause:
            self.server.close(force_group=True)
            self.assertEqual(process.wait.call_args_list[-1].kwargs, {"timeout": 5})
            pause.assert_called_once_with(0.1)
            self.assertEqual(group.call_count, 2)
            self.server.close(force_group=True)
            self.assertEqual(group.call_count, 2)

    def test_forced_close_does_not_reap_leader_before_final_group_signal(self):
        order = []
        self.server._process.wait.side_effect = lambda **kwargs: order.append("wait")
        with patch("scripts.experiment_runtime.ollama.os.killpg",
                   side_effect=lambda pid, sig: order.append(sig)), \
                patch("scripts.experiment_runtime.ollama.time.sleep"):
            self.server.close(force_group=True)
        self.assertEqual(order, [signal.SIGTERM, signal.SIGKILL, "wait"])


if __name__ == "__main__":
    unittest.main()

"""Offline lifecycle tests for the experiment-owned Ollama daemon."""

from copy import deepcopy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

# Direct test execution needs the repository root before project imports.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.med_copilot_ollama import ManagedOllama, ollama_for_saved_run, plan_judge_settings


class ManagedOllamaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Exercise real flock contention only inside this test's directory;
        # offline tests must coexist with an authorized live pilot server.
        lock_directory = patch("scripts.med_copilot_ollama.tempfile.gettempdir", return_value=str(self.root))
        lock_directory.start()
        self.addCleanup(lock_directory.stop)
        self.settings = {"manage_ollama": True, "ollama_host": "127.0.0.1:11435",
                         "ollama_parallel_slots": 4, "ollama_context_length": 8192}
        self.log = self.root / "server.log"
        self.process = Mock()
        self.process.pid = 424242
        self.process.poll.return_value = None
        self.process.wait.return_value = 0
        signals = patch("scripts.med_copilot_ollama.os.killpg")
        self.signal_group = signals.start()
        self.addCleanup(signals.stop)
        self.responses = {"/api/version": {"version": "0.40.16"},
                          "/api/tags": {"models": [{"name": "gemma3:4b", "digest": "actual-digest", "size": 123,
                                                    "unrecorded": "private"}]}}

    def server(self, **kwargs):
        return ManagedOllama(self.settings, self.log, **kwargs)

    def test_owned_server_has_explicit_settings_and_stops_after_success(self):
        with patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch.object(ManagedOllama, "_request", side_effect=lambda method, path, *args, **kw: self.responses[path]), \
                patch("scripts.med_copilot_ollama.subprocess.Popen", return_value=self.process) as start:
            with self.server() as server:
                metadata = server.metadata()
                self.assertEqual(metadata["version"], "0.40.16")
                self.assertEqual(metadata["parallel_slots"], 4)
                self.assertEqual(metadata["context_length"], 8192)
                self.assertEqual(metadata["models"], [{"name": "gemma3:4b", "digest": "actual-digest", "size": 123}])
                self.assertEqual(start.call_args.args, (["ollama", "serve"],))
                self.assertTrue(start.call_args.kwargs["start_new_session"])
                self.assertEqual({key: value for key, value in start.call_args.kwargs["env"].items()
                                  if key.startswith("OLLAMA_")}, {
                    "OLLAMA_HOST": "127.0.0.1:11435", "OLLAMA_NUM_PARALLEL": "4",
                    "OLLAMA_CONTEXT_LENGTH": "8192", "OLLAMA_MAX_LOADED_MODELS": "1",
                    "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_NO_CLOUD": "1",
                })
            self.signal_group.assert_called_once_with(self.process.pid, signal.SIGTERM)
            self.process.wait.assert_called_once_with(timeout=10)
            self.process.terminate.assert_not_called()
            self.process.kill.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "not running"):
                server.metadata()
        self.assertTrue(self.log.exists())

    def test_parallel_and_context_overrides_are_server_settings(self):
        with patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch.object(ManagedOllama, "_request", return_value={"version": "test"}), \
                patch("scripts.med_copilot_ollama.subprocess.Popen", return_value=self.process) as start:
            with self.server(parallel=2, context=4096):
                environment = start.call_args.kwargs["env"]
                self.assertEqual(environment["OLLAMA_NUM_PARALLEL"], "2")
                self.assertEqual(environment["OLLAMA_CONTEXT_LENGTH"], "4096")

    def test_mixed_model_benchmark_can_raise_resident_limit_and_capture_placement(self):
        responses = {**self.responses, "/api/ps": {"models": [{"name": "gemma3:4b", "size_vram": 123}]}}
        with patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch.object(ManagedOllama, "_request", side_effect=lambda method, path, *args, **kw: responses[path]), \
                patch("scripts.med_copilot_ollama.subprocess.Popen", return_value=self.process) as start:
            with self.server(max_loaded_models=3) as server:
                self.assertEqual(start.call_args.kwargs["env"]["OLLAMA_MAX_LOADED_MODELS"], "3")
                self.assertEqual(server.metadata()["max_loaded_models"], 3)
                self.assertEqual(server.ps_snapshot(), responses["/api/ps"])

    def test_parent_credentials_and_gui_settings_are_not_inherited(self):
        environment = {"HOME": "/unchanged/home", "PATH": "/bin", "OPENAI_API_KEY": "private",
                       "GRAPHRAG_API_KEY": "private", "OLLAMA_API_KEY": "private",
                       "HTTPS_PROXY": "https://private@proxy", "OLLAMA_HOST": "127.0.0.1:11434",
                       "OLLAMA_MODELS": "/cached-models", "OLLAMA_NUM_PARALLEL": "20"}
        with patch.dict(os.environ, environment, clear=True):
            child = self.server()._environment()
            self.assertEqual(child["HOME"], "/unchanged/home")
            self.assertEqual(child["OLLAMA_MODELS"], "/cached-models")
            self.assertEqual(child["OLLAMA_NUM_PARALLEL"], "4")
            for name in ("OPENAI_API_KEY", "GRAPHRAG_API_KEY", "OLLAMA_API_KEY", "HTTPS_PROXY"):
                self.assertNotIn(name, child)
            self.assertEqual(os.environ["OLLAMA_HOST"], "127.0.0.1:11434")

    def test_existing_foreign_endpoint_is_never_started_over_or_stopped(self):
        with patch.object(ManagedOllama, "_port_busy", return_value=True), \
                patch("scripts.med_copilot_ollama.subprocess.Popen") as start:
            with self.assertRaisesRegex(ValueError, "unowned"):
                with self.server():
                    self.fail("Foreign server must not be adopted")
            start.assert_not_called()
        self.assertFalse(self.log.exists())

    def test_experiment_port_lock_prevents_another_owner(self):
        with patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch.object(ManagedOllama, "_request", return_value={"version": "test"}), \
                patch("scripts.med_copilot_ollama.subprocess.Popen", return_value=self.process) as start:
            with self.server():
                with self.assertRaisesRegex(ValueError, "another experiment"):
                    with self.server():
                        self.fail("The port already has an owner")
            self.assertEqual(start.call_count, 1)
        # The owner releases the flock on exit, including exceptions.
        server = self.server()
        server._acquire_port()
        server.close()

    def test_disabled_mode_makes_no_network_process_or_log_changes(self):
        self.settings["manage_ollama"] = False
        with patch("scripts.med_copilot_ollama.subprocess.Popen") as start, \
                patch("scripts.med_copilot_ollama.urlopen") as request:
            with self.server() as server:
                self.assertEqual(server.metadata(), {"managed": False})
                self.assertEqual(server.ps_snapshot(), {"models": []})
                server.unload("model")
            start.assert_not_called()
            request.assert_not_called()
        self.assertFalse(self.log.exists())

    def test_body_error_still_stops_owned_server(self):
        with patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch.object(ManagedOllama, "_request", return_value={"version": "test"}), \
                patch("scripts.med_copilot_ollama.subprocess.Popen", return_value=self.process):
            with self.assertRaisesRegex(RuntimeError, "pipeline failed"):
                with self.server():
                    raise RuntimeError("pipeline failed")
        self.signal_group.assert_called_once_with(self.process.pid, signal.SIGTERM)

    def test_startup_exit_is_reported_and_does_not_signal_exited_process(self):
        self.process.poll.return_value = 1
        with patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch.object(ManagedOllama, "_request") as request, \
                patch("scripts.med_copilot_ollama.subprocess.Popen", return_value=self.process):
            with self.assertRaisesRegex(RuntimeError, "before becoming ready"):
                with self.server():
                    pass
            request.assert_not_called()
        self.signal_group.assert_not_called()
        self.process.wait.assert_called_once_with()

    def test_startup_waits_for_api_and_timeout_cleans_up(self):
        with patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch.object(ManagedOllama, "_request", side_effect=URLError("not ready")), \
                patch("scripts.med_copilot_ollama.subprocess.Popen", return_value=self.process), \
                patch("scripts.med_copilot_ollama.time.monotonic", side_effect=[0, 31]), \
                patch("scripts.med_copilot_ollama.time.sleep"):
            with self.assertRaises(TimeoutError):
                with self.server():
                    pass
        self.signal_group.assert_called_once_with(self.process.pid, signal.SIGTERM)

    def test_initial_connection_failure_then_ready(self):
        with patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch.object(ManagedOllama, "_request", side_effect=[URLError("waiting"), {"version": "test"}]), \
                patch("scripts.med_copilot_ollama.subprocess.Popen", return_value=self.process), \
                patch("scripts.med_copilot_ollama.time.sleep") as sleep:
            with self.server():
                pass
            sleep.assert_called_once_with(0.1)

    def test_unresponsive_owned_process_is_killed_after_grace_period(self):
        self.process.wait.side_effect = [subprocess.TimeoutExpired("ollama", 10), 0]
        with patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch.object(ManagedOllama, "_request", return_value={"version": "test"}), \
                patch("scripts.med_copilot_ollama.subprocess.Popen", return_value=self.process):
            with self.server():
                pass
        self.assertEqual([call.args for call in self.signal_group.call_args_list], [
            (self.process.pid, signal.SIGTERM), (self.process.pid, signal.SIGKILL)])
        self.process.terminate.assert_not_called()
        self.process.kill.assert_not_called()
        self.assertEqual(self.process.wait.call_args_list[-1].kwargs, {"timeout": 5})

    def test_invalid_version_response_is_fatal_and_releases_port(self):
        with patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch.object(ManagedOllama, "_request", return_value={}), \
                patch("scripts.med_copilot_ollama.subprocess.Popen", return_value=self.process):
            with self.assertRaisesRegex(ValueError, "version"):
                with self.server():
                    pass
        server = self.server()
        server._acquire_port()
        server.close()

    def test_failed_spawn_releases_owned_log_and_port_without_signaling_anything(self):
        server = self.server()
        with patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch("scripts.med_copilot_ollama.subprocess.Popen", side_effect=FileNotFoundError("ollama")):
            with self.assertRaises(FileNotFoundError):
                with server:
                    pass
        self.signal_group.assert_not_called()
        self.assertIsNone(server._log)
        self.assertIsNone(server._port_lock)
        replacement = self.server()
        replacement._acquire_port()
        replacement.close()

    def test_process_group_that_exits_before_signal_is_still_reaped(self):
        self.signal_group.side_effect = ProcessLookupError()
        with patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch.object(ManagedOllama, "_request", return_value={"version": "test"}), \
                patch("scripts.med_copilot_ollama.subprocess.Popen", return_value=self.process):
            with self.server():
                pass
        self.process.wait.assert_called_once_with(timeout=10)

    def test_unload_is_empty_native_request_and_does_not_generate(self):
        with patch.object(ManagedOllama, "_port_busy", return_value=False), \
                patch.object(ManagedOllama, "_request", return_value={"version": "test"}) as request, \
                patch("scripts.med_copilot_ollama.subprocess.Popen", return_value=self.process):
            with self.server() as server:
                server.unload("gemma3:4b")
                self.assertEqual(request.call_args.args, ("POST", "/api/generate", {
                    "model": "gemma3:4b", "keep_alive": 0, "stream": False}))
                self.assertEqual(request.call_args.kwargs, {"timeout": 30})
                for tag in (None, "", 1):
                    with self.assertRaises(ValueError):
                        server.unload(tag)

    def test_native_request_encodes_json_and_validates_response(self):
        with patch("scripts.med_copilot_ollama.urlopen", return_value=io.BytesIO(b'{"done":true}')) as request:
            value = self.server()._request("POST", "/api/generate", {"model": "test"})
            self.assertEqual(value, {"done": True})
            sent = request.call_args.args[0]
            self.assertEqual(sent.full_url, "http://127.0.0.1:11435/api/generate")
            self.assertEqual(json.loads(sent.data), {"model": "test"})
        with patch("scripts.med_copilot_ollama.urlopen", return_value=io.BytesIO(b'[]')):
            with self.assertRaisesRegex(ValueError, "non-object"):
                self.server()._request("GET", "/api/version")

    def test_port_probe_refuses_uncertain_network_failures(self):
        with patch("scripts.med_copilot_ollama.socket.create_connection", side_effect=ConnectionRefusedError()):
            self.assertFalse(self.server()._port_busy())
        with patch("scripts.med_copilot_ollama.socket.create_connection", side_effect=TimeoutError()):
            with self.assertRaisesRegex(ValueError, "unused"):
                self.server()._port_busy()

    def test_invalid_configuration_is_rejected_before_starting_anything(self):
        for host in ("", "http://example.com:11435", "http://user:secret@localhost:11435", "127.0.0.1:11434",
                     "127.0.0.1", "127.0.0.1:80", "http://localhost:11435/path", "http://localhost:11435?secret=x"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                ManagedOllama({**self.settings, "ollama_host": host}, self.log)
        for value in (True, "4", 0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.server(parallel=value)
        with self.assertRaises(ValueError):
            self.server(context=0)
        with self.assertRaises(ValueError):
            ManagedOllama({**self.settings, "manage_ollama": "yes"}, self.log)


class SavedRunOllamaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, experiment):
        (self.root / "run.json").write_text(json.dumps({"experiment": experiment}))

    def test_recorded_judge_parallelism_and_context_are_reused(self):
        self.write({"execution": {"manage_ollama": True, "ollama_host": "127.0.0.1:11435",
                                  "ollama_parallel_slots": 4, "ollama_context_length": 8192},
                    "plan_judge_concurrency": 8})
        server = Mock()
        server.__enter__ = Mock(return_value=server)
        server.__exit__ = Mock(return_value=None)
        with patch("scripts.med_copilot_ollama.ManagedOllama", return_value=server) as constructor:
            with ollama_for_saved_run(self.root) as actual:
                self.assertIs(actual, server)
            settings, path = constructor.call_args.args
            self.assertEqual(settings["manage_ollama"], True)
            self.assertEqual(path, self.root.resolve() / "ollama-judge.log")
            self.assertEqual(constructor.call_args.kwargs, {"parallel": 8, "context": 8192})
        server.__exit__.assert_called_once()

    def test_old_runs_do_not_start_or_mutate_a_gui_server(self):
        self.write({"plan_judge_concurrency": 2})
        with patch("scripts.med_copilot_ollama.subprocess.Popen") as process:
            with ollama_for_saved_run(self.root) as server:
                self.assertEqual(server.metadata(), {"managed": False})
            process.assert_not_called()
        self.assertFalse((self.root / "ollama-judge.log").exists())

    def test_bad_recorded_settings_are_rejected(self):
        self.write({"execution": ["bad"]})
        with self.assertRaisesRegex(ValueError, "execution"):
            with ollama_for_saved_run(self.root):
                pass

    def test_resume_reuses_previous_stage_and_explicit_override_reaches_server(self):
        metadata = {"experiment": {"execution": {"manage_ollama": True, "ollama_judge_context_length": 8192},
                                   "plan_judge_concurrency": 4},
                    "plan_evaluation": {"status": "interrupted", "concurrency": 8, "context_length": 4096}}
        path = self.root / "run.json"
        path.write_text(json.dumps(metadata))
        before = path.read_bytes()
        server = Mock()
        server.__enter__ = Mock(return_value=server)
        server.__exit__ = Mock(return_value=None)
        with patch("scripts.med_copilot_ollama.ManagedOllama", return_value=server) as constructor:
            with ollama_for_saved_run(self.root):
                pass
            self.assertEqual(constructor.call_args.kwargs, {"parallel": 8, "context": 4096})
            with ollama_for_saved_run(self.root, concurrency=16, context_length=2048):
                pass
            self.assertEqual(constructor.call_args.kwargs, {"parallel": 16, "context": 2048})
        self.assertEqual(path.read_bytes(), before)


class PlanJudgeSettingsTests(unittest.TestCase):
    def test_precedence_is_independent_per_field_and_metadata_is_unchanged(self):
        metadata = {"experiment": {"plan_judge_concurrency": 4, "execution": {
                        "ollama_judge_context_length": 8192, "ollama_context_length": 16384}},
                    "plan_evaluation": {"status": "interrupted", "concurrency": 8, "context_length": 4096}}
        before = deepcopy(metadata)
        self.assertEqual(plan_judge_settings(metadata), {"concurrency": 8, "context_length": 4096})
        self.assertEqual(plan_judge_settings(metadata, concurrency=16), {"concurrency": 16, "context_length": 4096})
        self.assertEqual(plan_judge_settings(metadata, context_length=2048), {"concurrency": 8, "context_length": 2048})
        self.assertEqual(metadata, before)
        for context in (512, 262144):
            self.assertEqual(plan_judge_settings({}, concurrency=64, context_length=context),
                             {"concurrency": 64, "context_length": context})
        metadata["plan_evaluation"].update(concurrency=0, context_length=0)
        self.assertEqual(plan_judge_settings(metadata, concurrency=8, context_length=4096),
                         {"concurrency": 8, "context_length": 4096})

    def test_defaults_and_legacy_stage_null_values_fall_back(self):
        self.assertEqual(plan_judge_settings({}), {"concurrency": 2, "context_length": 8192})
        metadata = {"experiment": {"plan_judge_concurrency": 4, "execution": {"ollama_context_length": 16384}},
                    "plan_evaluation": {"concurrency": None, "context_length": None}}
        self.assertEqual(plan_judge_settings(metadata), {"concurrency": 4, "context_length": 16384})
        metadata["experiment"]["execution"]["ollama_judge_context_length"] = 4096
        self.assertEqual(plan_judge_settings(metadata), {"concurrency": 4, "context_length": 4096})

    def test_invalid_effective_values_fail_without_any_server_or_api_calls(self):
        with patch("scripts.med_copilot_ollama.subprocess.Popen") as process, \
                patch("scripts.med_copilot_ollama.urlopen") as request:
            for value in (True, 0, -1, 65, "8", 1.5):
                with self.subTest(concurrency=value), self.assertRaisesRegex(ValueError, "concurrency"):
                    plan_judge_settings({}, concurrency=value)
            for value in (True, 0, -1, 1, 511, 262145, "4096", 1.5):
                with self.subTest(context=value), self.assertRaisesRegex(ValueError, "context"):
                    plan_judge_settings({}, context_length=value)
            for metadata in ([], {"experiment": []}, {"plan_evaluation": []}, {"experiment": {"execution": []}},
                             {"plan_evaluation": {"concurrency": 0}},
                             {"experiment": {"execution": {"ollama_judge_context_length": 0}}}):
                with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                    plan_judge_settings(metadata)
            process.assert_not_called()
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()

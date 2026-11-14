"""Offline watchdog bounds, ownership, retry and inherited-lock regressions."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

# Direct execution starts in tests/; project helpers require the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_runtime import memory_supervisor as guard
from scripts.med_copilot_run_lock import RunLockError, exclusive_run_lock


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.stop = self.directory / "stop"
        self.events = self.directory / "events.jsonl"
        self.clock = Clock()
        self.worker = Mock(pid=987654, returncode=75)
        self.worker.poll.return_value = None
        self.worker.wait.return_value = 75
        self.server = Mock()

    def test_critical_pressure_stops_without_grace_and_closes_owned_group(self):
        with patch.object(guard, "_sample", return_value={"pressure": 4, "swapout_bytes": 0}), \
                patch.object(guard.time, "monotonic", side_effect=self.clock.monotonic), \
                patch.object(guard.time, "sleep", side_effect=self.clock.sleep):
            self.assertEqual(guard._watch(self.worker, self.server, self.stop, self.events),
                             "critical_memory_pressure")
        self.assertEqual(self.clock.now, 0)
        self.assertEqual(self.stop.read_text(), "critical_memory_pressure")
        self.server.close.assert_called_once_with(force_group=True)
        self.worker.wait.assert_called_once_with(timeout=guard.WORKER_EXIT_SECONDS)

    def test_warning_and_growing_swap_stops_then_bounds_a_hung_worker(self):
        def sample():
            return {"pressure": 2, "swapout_bytes": int(self.clock.now) * 2 * 1024 * 1024}
        self.worker.wait.side_effect = subprocess.TimeoutExpired("worker", 5)
        with patch.object(guard, "_sample", side_effect=sample), \
                patch.object(guard.time, "monotonic", side_effect=self.clock.monotonic), \
                patch.object(guard.time, "sleep", side_effect=self.clock.sleep), \
                patch.object(guard, "_terminate_worker") as terminate:
            self.assertEqual(guard._watch(self.worker, self.server, self.stop, self.events),
                             "sustained_warning_with_swap_growth")
        self.assertLessEqual(self.clock.now, guard.WARNING_SECONDS + guard.GRACE_SECONDS)
        self.server.close.assert_called_once()
        terminate.assert_called_once_with(self.worker)
        self.assertTrue(self.stop.exists())

    def test_monitor_failure_stops_instead_of_assuming_normal(self):
        with patch.object(guard, "_sample", return_value=None), \
                patch.object(guard.time, "monotonic", side_effect=self.clock.monotonic), \
                patch.object(guard.time, "sleep", side_effect=self.clock.sleep):
            reason = guard._watch(self.worker, self.server, self.stop, self.events)
        self.assertEqual(reason, "memory_monitor_failed")
        self.assertLessEqual(self.clock.now, 2 * guard.SAMPLE_INTERVAL_SECONDS + guard.GRACE_SECONDS)

    def test_normal_completed_worker_has_no_shutdown_or_stop_file(self):
        self.worker.poll.return_value = 0
        with patch.object(guard, "_sample") as sample:
            self.assertIsNone(guard._watch(self.worker, self.server, self.stop, self.events))
        sample.assert_not_called()
        self.server.close.assert_not_called()
        self.assertFalse(self.stop.exists())

    def test_isolated_paging_burst_and_compression_traffic_do_not_stop_work(self):
        self.worker.poll.side_effect = lambda: 0 if self.clock.now >= 240 else None
        def sample():
            return {"pressure": 2, "swapout_bytes": 500 * 1024 ** 2 if self.clock.now >= 60 else 0,
                    "swapin_bytes": 0, "compressor_bytes": 6 * 1024 ** 3,
                    "compression_bytes": int(self.clock.now) * 128 * 1024 ** 2,
                    "reclaimable_bytes": 5 * 1024 ** 3}
        with patch.object(guard, "_sample", side_effect=sample), \
                patch.object(guard.time, "monotonic", side_effect=self.clock.monotonic), \
                patch.object(guard.time, "sleep", side_effect=self.clock.sleep):
            self.assertIsNone(guard._watch(self.worker, self.server, self.stop, self.events))
        self.server.close.assert_not_called()
        self.assertFalse(self.stop.exists())
        rows = [json.loads(line) for line in self.events.read_text().splitlines()]
        self.assertFalse(any(row["event"] == "stopping" for row in rows))
        self.assertTrue(any(row.get("phase") == "stable_warning" for row in rows))

    def test_every_sample_is_recorded_quietly_and_critical_skips_optional_diagnostics(self):
        with patch.object(guard, "_sample", return_value={"pressure": 4, "swapout_bytes": 0}), \
                patch.object(guard, "_owned_rss") as rss, \
                patch.object(guard.time, "monotonic", side_effect=self.clock.monotonic), \
                patch.object(guard.time, "sleep", side_effect=self.clock.sleep):
            guard._watch(self.worker, self.server, self.stop, self.events)
        rss.assert_not_called()
        rows = [json.loads(line) for line in self.events.read_text().splitlines()]
        samples = [row for row in rows if row["event"] == "sample"]
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["phase"], "stopping")
        self.assertEqual(samples[0]["sample"]["pressure"], 4)

    def test_rss_diagnostics_include_only_explicit_worker_and_owned_server_group(self):
        self.server._process.pid = 765432
        with patch.object(guard.subprocess, "run", return_value=Mock(stdout=(
                "987654 987654 100\n765432 765432 200\n765433 765432 300\n42 42 999\n"))) as run:
            self.assertEqual(guard._owned_rss(self.worker, self.server), {
                "worker_rss_bytes": 100 * 1024, "server_group_rss_bytes": 500 * 1024})
        self.assertEqual(run.call_args.kwargs["timeout"], 2)
        with patch.object(guard.subprocess, "run", side_effect=subprocess.TimeoutExpired("ps", 2)):
            self.assertEqual(guard._owned_rss(self.worker, self.server), {"error": "TimeoutExpired"})

    def test_quiet_samples_do_not_print_or_fsync_each_sample(self):
        with patch.object(guard.os, "fsync") as sync, patch("builtins.print") as console:
            guard._event(self.events, "sample", quiet=True, sample={"pressure": 1})
        sync.assert_not_called()
        console.assert_not_called()
        self.assertEqual(json.loads(self.events.read_text())["event"], "sample")

    def test_headroom_requires_stable_normal_then_returns(self):
        with patch.object(guard, "_sample", return_value={"pressure": 1, "swapout_bytes": 123}), \
                patch.object(guard.time, "monotonic", side_effect=self.clock.monotonic), \
                patch.object(guard.time, "sleep", side_effect=self.clock.sleep):
            self.assertTrue(guard._headroom())
        self.assertEqual(self.clock.now, guard.RECOVERY_NORMAL_SECONDS)

    def test_headroom_timeout_and_failed_monitor_are_bounded(self):
        with patch.object(guard, "_sample", return_value={"pressure": 2, "swapout_bytes": 123}), \
                patch.object(guard.time, "monotonic", side_effect=self.clock.monotonic), \
                patch.object(guard.time, "sleep", side_effect=self.clock.sleep):
            self.assertFalse(guard._headroom())
        self.assertEqual(self.clock.now, guard.RECOVERY_TIMEOUT_SECONDS)
        self.clock.now = 0
        with patch.object(guard, "_sample", return_value=None), \
                patch.object(guard.time, "monotonic", side_effect=self.clock.monotonic), \
                patch.object(guard.time, "sleep", side_effect=self.clock.sleep):
            self.assertFalse(guard._headroom())
        self.assertLessEqual(self.clock.now, 2 * guard.SAMPLE_INTERVAL_SECONDS)

    def test_sample_failures_are_handled_as_missing_readings(self):
        for error in (OSError("blocked"), ValueError("invalid"), subprocess.TimeoutExpired("sysctl", 2),
                      subprocess.CalledProcessError(1, "sysctl")):
            with self.subTest(error=error), patch.object(guard, "sample_memory", side_effect=error):
                self.assertIsNone(guard._sample())

    def test_terminate_only_signals_owned_session_with_bounded_waits(self):
        self.worker.wait.side_effect = [subprocess.TimeoutExpired("worker", 2), -9]
        with patch.object(guard.os, "killpg") as kill:
            guard._terminate_worker(self.worker)
        self.assertEqual([call.args for call in kill.call_args_list],
                         [(self.worker.pid, signal.SIGTERM), (self.worker.pid, signal.SIGKILL)])
        self.assertTrue(all(call.kwargs == {"timeout": 2} for call in self.worker.wait.call_args_list))

    def test_real_ignoring_child_is_killed_within_the_bound(self):
        child = subprocess.Popen([sys.executable, "-u", "-c", (
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready',flush=True); time.sleep(60)")], stdout=subprocess.PIPE,
            text=True, start_new_session=True)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        self.assertEqual(child.stdout.readline().strip(), "ready")
        guard._terminate_worker(child)
        self.assertEqual(child.returncode, -signal.SIGKILL)
        child.stdout.close()


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        config = self.root / "experiments/configs/plan_judging.yaml"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({"concurrency": 4, "context_length": 4096, "flash_attention": True,
                                      "kv_cache_type": "f16", "host_cache_mib": 2048}))
        (self.run / "run.json").write_text(json.dumps({"experiment": {"execution": {"manage_ollama": True}}}))
        self.original = (self.run / "run.json").read_bytes()
        self.bootstraps = []
        self.servers = []
        self.workers = []

    def server(self, *args, **kwargs):
        self.servers.append(deepcopy(kwargs))
        server = Mock()
        server.runtime_descriptor.return_value = {"owned_test_server": True}
        return nullcontext(server)

    def popen(self, argv, **kwargs):
        self.bootstraps.append(json.loads(Path(argv[-1]).read_text()))
        self.assertEqual(argv[1:3], ["-m", "scripts.experiment_runtime.judge_worker"])
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["pass_fds"], (self.bootstraps[-1]["lock_fd"],))
        worker = Mock(returncode=0 if len(self.bootstraps) > 1 else 75)
        worker.wait.return_value = worker.returncode
        worker.poll.return_value = worker.returncode
        self.workers.append(worker)
        return worker

    def headroom(self, events=None):
        with self.assertRaises(RunLockError):
            with exclusive_run_lock(self.run):
                self.fail("The original lock must cover waiting and retry")
        return True

    def test_one_retry_changes_both_client_profile_and_server_slots_with_lock_held(self):
        with patch.object(guard, "_headroom", side_effect=self.headroom), \
                patch.object(guard, "ollama_for_saved_run", side_effect=self.server), \
                patch.object(guard.subprocess, "Popen", side_effect=self.popen), \
                patch.object(guard, "_watch", side_effect=["critical_memory_pressure", None]):
            self.assertEqual(guard.supervise_cli([str(self.run)], root=self.root), 0)
        self.assertEqual([entry["profile"]["settings"]["concurrency"] for entry in self.bootstraps], [4, 2])
        self.assertEqual([entry["concurrency"] for entry in self.servers], [4, 2])
        self.assertEqual([entry["runtime_profile"]["settings"]["concurrency"] for entry in self.servers], [4, 2])
        self.assertEqual((self.run / "run.json").read_bytes(), self.original)
        with exclusive_run_lock(self.run):
            pass

    def test_normal_exit_has_no_retry(self):
        def popen(argv, **kwargs):
            worker = self.popen(argv, **kwargs)
            worker.returncode = worker.wait.return_value = worker.poll.return_value = 0
            return worker
        with patch.object(guard, "_headroom", return_value=True), \
                patch.object(guard, "ollama_for_saved_run", side_effect=self.server), \
                patch.object(guard.subprocess, "Popen", side_effect=popen), \
                patch.object(guard, "_watch", return_value=None):
            self.assertEqual(guard.supervise_cli([str(self.run)], root=self.root), 0)
        self.assertEqual(len(self.bootstraps), 1)

    def test_shipped_two_clients_one_slot_retries_with_one_client_and_one_slot(self):
        config = self.root / "experiments/configs/plan_judging.yaml"
        config.write_bytes((ROOT / "experiments/configs/plan_judging.yaml").read_bytes())
        with patch.object(guard, "_headroom", side_effect=self.headroom), \
                patch.object(guard, "ollama_for_saved_run", side_effect=self.server), \
                patch.object(guard.subprocess, "Popen", side_effect=self.popen), \
                patch.object(guard, "_watch", side_effect=["sustained_warning_with_swap_growth", None]):
            self.assertEqual(guard.supervise_cli([str(self.run)], root=self.root), 0)
        self.assertEqual([b["profile"]["settings"]["client_workers"] for b in self.bootstraps], [2, 1])
        self.assertEqual([b["profile"]["settings"]["concurrency"] for b in self.bootstraps], [1, 1])
        self.assertEqual([s["concurrency"] for s in self.servers], [1, 1])
        self.assertEqual((self.run / "run.json").read_bytes(), self.original)
        with exclusive_run_lock(self.run):
            pass

    def test_resume_of_productive_trial_preserves_one_native_slot_and_two_clients(self):
        settings = {"concurrency": 1, "context_length": 4096, "flash_attention": True,
                    "kv_cache_type": "f16", "host_cache_mib": 512}
        metadata = json.loads(self.original)
        metadata["plan_evaluation"] = {"status": "interrupted", "concurrency": 2,
            "client_workers": 2, "gpu_slots": 1, "runtime_profile": {
                "settings": settings, "config_file": "/external/trial/profile.yaml", "config_sha256": "a" * 64}}
        (self.run / "run.json").write_text(json.dumps(metadata))
        original = (self.run / "run.json").read_bytes()
        with patch.object(guard, "_headroom", return_value=True), \
                patch.object(guard, "ollama_for_saved_run", side_effect=self.server), \
                patch.object(guard.subprocess, "Popen", side_effect=self.popen), \
                patch.object(guard, "_watch", return_value=None):
            guard.supervise_cli([str(self.run)], root=self.root)
        self.assertEqual(self.servers[0]["concurrency"], 1)
        self.assertEqual(self.bootstraps[0]["profile"]["settings"], {**settings, "client_workers": 2})
        self.assertEqual((self.run / "run.json").read_bytes(), original)

    def test_one_slot_recovers_once_and_two_slots_fall_back_to_one(self):
        for concurrency, expected in ((1, [1, 1]), (2, [2, 1])):
            with self.subTest(concurrency=concurrency):
                self.bootstraps.clear()
                self.servers.clear()
                with patch.object(guard, "_headroom", side_effect=self.headroom), \
                        patch.object(guard, "ollama_for_saved_run", side_effect=self.server), \
                        patch.object(guard.subprocess, "Popen", side_effect=self.popen), \
                        patch.object(guard, "_watch", side_effect=["sustained_memory_warning", None]):
                    self.assertEqual(guard.supervise_cli([str(self.run), "--concurrency", str(concurrency)],
                                                        root=self.root), 0)
                self.assertEqual([entry["profile"]["settings"]["concurrency"] for entry in self.bootstraps], expected)
                self.assertEqual([entry["concurrency"] for entry in self.servers], expected)
                recorded = self.bootstraps[-1]["guard_metadata"]["policy"]
                self.assertEqual(recorded["version"], 2)
                self.assertTrue(recorded["isolated_swap_bursts_continue"])
                self.assertTrue(recorded["compression_traffic_diagnostic_only"])
                self.assertTrue(recorded["critical_pressure_immediate"])
                self.assertEqual({key: recorded[key] for key in guard.policy_metadata()}, guard.policy_metadata())

    def test_user_interrupt_never_launches_a_recovery_attempt(self):
        with patch.object(guard, "_headroom", return_value=True), \
                patch.object(guard, "ollama_for_saved_run", side_effect=self.server), \
                patch.object(guard.subprocess, "Popen", side_effect=self.popen), \
                patch.object(guard, "_watch", side_effect=KeyboardInterrupt):
            self.assertEqual(guard.supervise_cli([str(self.run)], root=self.root), 130)
        self.assertEqual(len(self.bootstraps), 1)

    def test_second_pressure_stop_exits_75_instead_of_retrying_forever(self):
        with patch.object(guard, "_headroom", return_value=True), \
                patch.object(guard, "ollama_for_saved_run", side_effect=self.server), \
                patch.object(guard.subprocess, "Popen", side_effect=self.popen), \
                patch.object(guard, "_watch", return_value="sustained_memory_warning"):
            self.assertEqual(guard.supervise_cli([str(self.run)], root=self.root), guard.MEMORY_STOP_EXIT)
        self.assertEqual(len(self.bootstraps), 2)

    def test_critical_stop_is_not_reported_as_success_when_worker_exits_zero(self):
        def popen(argv, **kwargs):
            worker = self.popen(argv, **kwargs)
            worker.returncode = worker.wait.return_value = worker.poll.return_value = 0
            return worker
        with patch.object(guard, "_headroom", return_value=True), \
                patch.object(guard, "ollama_for_saved_run", side_effect=self.server), \
                patch.object(guard.subprocess, "Popen", side_effect=popen), \
                patch.object(guard, "_watch", return_value="critical_memory_pressure"):
            self.assertEqual(guard.supervise_cli([str(self.run)], root=self.root), guard.MEMORY_STOP_EXIT)
        self.assertEqual(len(self.bootstraps), 2)
        events = [json.loads(line) for path in (self.run / "provenance").glob("*.jsonl")
                  for line in path.read_text().splitlines()]
        self.assertNotIn("finished", [event["event"] for event in events])

    def test_missing_monitor_or_no_headroom_does_not_retry(self):
        with patch.object(guard, "_headroom", return_value=False), \
                patch.object(guard.subprocess, "Popen") as start:
            self.assertEqual(guard.supervise_cli([str(self.run)], root=self.root), guard.MEMORY_STOP_EXIT)
            start.assert_not_called()
        with patch.object(guard, "_headroom", return_value=True), \
                patch.object(guard, "ollama_for_saved_run", side_effect=self.server), \
                patch.object(guard.subprocess, "Popen", side_effect=self.popen), \
                patch.object(guard, "_watch", return_value="memory_monitor_failed"):
            self.assertEqual(guard.supervise_cli([str(self.run)], root=self.root), guard.MEMORY_STOP_EXIT)
        self.assertEqual(len(self.bootstraps), 1)

    def test_inherited_descriptor_must_name_this_runs_locked_inode(self):
        other = self.root / "other"
        other.mkdir()
        with exclusive_run_lock(self.run) as parent:
            with guard.inherited_run_lock(self.run, parent.fileno()):
                with self.assertRaises(RunLockError):
                    with exclusive_run_lock(self.run):
                        pass
            with exclusive_run_lock(other) as stranger:
                with self.assertRaisesRegex(ValueError, "does not match"):
                    with guard.inherited_run_lock(self.run, stranger.fileno()):
                        pass


if __name__ == "__main__":
    unittest.main()

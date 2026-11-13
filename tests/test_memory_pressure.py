"""Offline policy/sampler regressions; no real host readings or model calls."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# Direct execution needs the project root before importing its runtime helper.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_runtime.memory_pressure import (
    MemoryPolicy, WARNING_SWAP_BYTES, sample_memory,
)


def sample(pressure=1, swapped=0):
    return {"pressure": pressure, "swapout_bytes": swapped}


class MemorySamplerTests(unittest.TestCase):
    def test_reads_pressure_and_converts_cumulative_pages_to_bytes(self):
        replies = [SimpleNamespace(stdout="2\n"), SimpleNamespace(stdout=(
            'Mach Virtual Memory Statistics: (page size of 16384 bytes)\nSwapouts: 12345.\n'))]
        with patch("scripts.experiment_runtime.memory_pressure.subprocess.run", side_effect=replies) as run:
            self.assertEqual(sample_memory(), sample(2, 12345 * 16384))
        self.assertEqual(run.call_args_list[0].args[0],
                         ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
        self.assertEqual(run.call_args_list[1].args[0], ["/usr/bin/vm_stat"])
        self.assertTrue(all(call.kwargs["timeout"] == 2 and call.kwargs["check"] for call in run.call_args_list))

    def test_missing_or_invalid_readings_are_errors(self):
        for pressure in ("", "0", "3", "critical", "-1"):
            with self.subTest(pressure=pressure), patch(
                    "scripts.experiment_runtime.memory_pressure.subprocess.run",
                    return_value=SimpleNamespace(stdout=pressure)), self.assertRaises(ValueError):
                sample_memory()

    def test_critical_pressure_does_not_wait_for_secondary_swap_telemetry(self):
        with patch("scripts.experiment_runtime.memory_pressure.subprocess.run",
                   side_effect=[SimpleNamespace(stdout="4\n"), subprocess.TimeoutExpired("vm_stat", 2)]) as run:
            self.assertEqual(sample_memory(), sample(4, 0))
        self.assertEqual(run.call_count, 1)

    def test_missing_or_invalid_swap_telemetry_is_an_error(self):
        for vm in ("Swapouts: 0.\n", "page size of 0 bytes\nSwapouts: 0.\n",
                   "page size of 16384 bytes\nSwapouts: x\n",
                   "page size of 16384 bytes\nSwapouts: -1.\n"):
            with self.subTest(vm=vm), patch(
                    "scripts.experiment_runtime.memory_pressure.subprocess.run",
                    side_effect=[SimpleNamespace(stdout="1"), SimpleNamespace(stdout=vm)]), \
                    self.assertRaises(ValueError):
                sample_memory()

    def test_command_timeout_or_exit_failure_propagates(self):
        for error in (subprocess.TimeoutExpired("sysctl", 2), subprocess.CalledProcessError(1, "sysctl"),
                      FileNotFoundError("sysctl")):
            with self.subTest(error=error), patch(
                    "scripts.experiment_runtime.memory_pressure.subprocess.run", side_effect=error), \
                    self.assertRaises(type(error)):
                sample_memory()


class MemoryPolicyTests(unittest.TestCase):
    def test_brief_warning_recovers_even_with_existing_swap(self):
        policy = MemoryPolicy()
        baseline = 50 * 1024 ** 3
        for now in range(0, 25, 5):
            self.assertIsNone(policy.observe(sample(2, baseline), now))
        self.assertIsNone(policy.observe(sample(1, baseline), 25))
        self.assertIsNone(policy.observe(sample(2, baseline + WARNING_SWAP_BYTES), 30))
        self.assertIsNone(policy.observe(sample(1, baseline + WARNING_SWAP_BYTES), 35))
        self.assertEqual(policy.last_sample, sample(1, baseline + WARNING_SWAP_BYTES))

    def test_warning_needs_duration_and_recent_swap_growth(self):
        policy = MemoryPolicy()
        for now in range(0, 60, 5):
            self.assertIsNone(policy.observe(sample(2, now * WARNING_SWAP_BYTES // 60), now))
        self.assertEqual(policy.observe(sample(2, WARNING_SWAP_BYTES), 60),
                         "sustained_warning_with_swap_growth")
        self.assertEqual(policy.last_event["swapout_growth_bytes"], WARNING_SWAP_BYTES)

    def test_old_swap_growth_outside_window_does_not_trigger(self):
        policy = MemoryPolicy()
        self.assertIsNone(policy.observe(sample(1), 0))
        self.assertIsNone(policy.observe(sample(1, WARNING_SWAP_BYTES * 2), 5))
        self.assertIsNone(policy.observe(sample(1, WARNING_SWAP_BYTES * 2), 10))
        self.assertIsNone(policy.observe(sample(1, WARNING_SWAP_BYTES * 2), 15))
        for now in range(20, 131, 5):
            self.assertIsNone(policy.observe(sample(2, WARNING_SWAP_BYTES * 2), now))

    def test_brief_normal_blip_cannot_reset_mostly_warning_with_growing_swap(self):
        policy = MemoryPolicy()
        for now in range(0, 60, 5):
            pressure = 1 if now == 55 else 2
            self.assertIsNone(policy.observe(sample(pressure, now * WARNING_SWAP_BYTES // 60), now))
        self.assertEqual(policy.observe(sample(2, WARNING_SWAP_BYTES), 60),
                         "sustained_warning_with_swap_growth")

    def test_intermittent_minority_warning_does_not_trigger_rolling_rule(self):
        policy = MemoryPolicy()
        for now in range(0, 65, 5):
            pressure = 2 if now % 10 == 0 else 1
            self.assertIsNone(policy.observe(sample(pressure, now * WARNING_SWAP_BYTES // 60), now))

    def test_sustained_warning_stops_even_without_swap_growth(self):
        policy = MemoryPolicy()
        for now in range(0, 180, 5):
            self.assertIsNone(policy.observe(sample(2, 123456789), now))
        self.assertEqual(policy.observe(sample(2, 123456789), 180), "sustained_memory_warning")

    def test_critical_pressure_stops_on_first_sample(self):
        policy = MemoryPolicy()
        self.assertEqual(policy.observe(sample(4), 0), "critical_memory_pressure")
        self.assertEqual(policy.last_event["sample"], sample(4))

    def test_three_consecutive_failed_samples_stop_and_success_resets_count(self):
        policy = MemoryPolicy()
        self.assertIsNone(policy.observe(None, 0))
        self.assertIsNone(policy.observe(None, 5))
        self.assertIsNone(policy.observe(sample(), 10))
        self.assertEqual(policy.sample_failures, 0)
        self.assertIsNone(policy.observe({"pressure": True, "swapout_bytes": 0}, 15))
        self.assertIsNone(policy.observe({"pressure": 1, "swapout_bytes": -1}, 20))
        self.assertEqual(policy.observe(None, 25), "memory_monitor_failed")

    def test_swap_counter_reset_is_not_negative_growth(self):
        policy = MemoryPolicy()
        policy.observe(sample(2, WARNING_SWAP_BYTES * 3), 0)
        self.assertIsNone(policy.observe(sample(2, 0), 5))
        self.assertIsNone(policy.observe(sample(2, 0), 60))

    def test_headroom_requires_continuous_normal_without_new_swap(self):
        policy = MemoryPolicy()
        self.assertFalse(policy.recovered(sample(1, 1000), 0))
        self.assertFalse(policy.recovered(sample(1, 1000), 25))
        self.assertTrue(policy.recovered(sample(1, 1000), 30))
        self.assertFalse(policy.recovered(sample(1, 1001), 35))
        self.assertTrue(policy.recovered(sample(1, 1001), 65))
        self.assertFalse(policy.recovered(sample(2, 1001), 70))
        self.assertFalse(policy.recovered(sample(1, 1001), 75))
        self.assertTrue(policy.recovered(sample(1, 1001), 105))
        self.assertFalse(policy.recovered(None, 110))

    def test_invalid_time_is_rejected(self):
        for now in (-1, float("nan"), float("inf"), True, "1"):
            for method in (MemoryPolicy().observe, MemoryPolicy().recovered):
                with self.subTest(now=now), self.assertRaises(ValueError):
                    method(sample(), now)


if __name__ == "__main__":
    unittest.main()

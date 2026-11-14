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
    MemoryPolicy, WARNING_SWAP_BYTES, WARNING_CHURN_BYTES, WARNING_COMPRESSION_BYTES,
    MIN_RECLAIMABLE_BYTES, LOW_RESERVE_SECONDS, policy_metadata, sample_memory,
)


def sample(pressure=1, swapped=0):
    return {"pressure": pressure, "swapout_bytes": swapped}


def detailed_sample(pressure=2, **changes):
    return {"pressure": pressure, "swapout_bytes": 0, "swapin_bytes": 0,
            "compressor_bytes": 2 * 1024 ** 3, "compression_bytes": 10 * 1024 ** 3,
            "reclaimable_bytes": 4 * 1024 ** 3, **changes}


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

    def test_reads_bounded_auxiliary_telemetry_without_adding_memory_categories(self):
        vm = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
              'Swapouts: 9.\nSwapins: 7.\nPages occupied by compressor: 11.\n'
              'Compressions: 123.\nPages free: 5.\nPages inactive: 17.\nPages speculative: 3.\n')
        with patch("scripts.experiment_runtime.memory_pressure.subprocess.run", side_effect=[
                SimpleNamespace(stdout="2"), SimpleNamespace(stdout=vm)]):
            self.assertEqual(sample_memory(), {"pressure": 2, "swapout_bytes": 9 * 16384,
                "swapin_bytes": 7 * 16384, "compressor_bytes": 11 * 16384,
                "compression_bytes": 123 * 16384, "reclaimable_bytes": 25 * 16384})


class MemoryPolicyTests(unittest.TestCase):
    def test_stable_warning_with_complete_telemetry_does_not_force_pointless_restart(self):
        policy = MemoryPolicy()
        for now in range(0, 3605, 5):
            self.assertIsNone(policy.observe(detailed_sample(), now))
        self.assertEqual(policy.phase, "stable_warning")
        self.assertLessEqual(len(policy._history), 14)
        self.assertEqual(policy.last_observation["swapout_growth_bytes"], 0)

    def test_startup_never_exempts_critical_or_growing_swap_protection(self):
        critical = MemoryPolicy()
        self.assertEqual(critical.observe(detailed_sample(4), 0), "critical_memory_pressure")
        policy = MemoryPolicy()
        for now in range(0, 60, 5):
            self.assertIsNone(policy.observe(detailed_sample(swapout_bytes=now * WARNING_SWAP_BYTES // 60), now))
        self.assertEqual(policy.observe(detailed_sample(swapout_bytes=WARNING_SWAP_BYTES), 60),
                         "sustained_warning_with_swap_growth")

    def test_persistent_low_reserve_stops_even_without_swap_or_compression_growth(self):
        policy = MemoryPolicy()
        reading = detailed_sample(reclaimable_bytes=MIN_RECLAIMABLE_BYTES - 1)
        for now in range(0, LOW_RESERVE_SECONDS, 5):
            self.assertIsNone(policy.observe(reading, now))
        self.assertEqual(policy.observe(reading, LOW_RESERVE_SECONDS), "sustained_warning_with_memory_churn")
        self.assertEqual(policy.last_event["swapout_growth_bytes"], 0)

    def test_high_reserve_compression_traffic_alone_does_not_stop_judging(self):
        policy = MemoryPolicy()
        base = detailed_sample()
        for now in range(0, 180, 5):
            reading = {**base, "compression_bytes": base["compression_bytes"] + now * WARNING_COMPRESSION_BYTES // 60}
            self.assertIsNone(policy.observe(reading, now))
        self.assertIsNone(policy.observe({**base, "compression_bytes": base["compression_bytes"]
                                         + 3 * WARNING_COMPRESSION_BYTES}, 180))

    def test_low_volume_paging_with_ample_capacity_does_not_trigger_volume_guard(self):
        policy = MemoryPolicy()
        for now in range(0, 180, 5):
            self.assertIsNone(policy.observe(detailed_sample(swapout_bytes=now * WARNING_SWAP_BYTES // 120), now))
        self.assertIsNone(policy.observe(detailed_sample(swapout_bytes=180 * WARNING_SWAP_BYTES // 120), 180))

    def test_incomplete_telemetry_or_low_reserve_cannot_exempt_warning_timeout(self):
        for reading in (sample(2), {**detailed_sample(), "swapin_bytes": None}):
            with self.subTest(reading=reading):
                policy = MemoryPolicy()
                if reading.get("swapin_bytes", 0) is None:
                    for now in (0, 5):
                        self.assertIsNone(policy.observe(reading, now))
                    self.assertEqual(policy.observe(reading, 10), "memory_monitor_failed")
                else:
                    for now in range(0, 180, 5):
                        self.assertIsNone(policy.observe(reading, now))
                    self.assertEqual(policy.observe(reading, 180), "sustained_memory_warning")

    def test_counter_reset_cannot_be_mistaken_for_a_stable_full_window(self):
        policy = MemoryPolicy()
        for now in range(0, 180, 5):
            self.assertIsNone(policy.observe(detailed_sample(), now))
        self.assertEqual(policy.observe(detailed_sample(compression_bytes=0), 180),
                         "sustained_memory_warning")

    def test_headroom_does_not_admit_with_a_known_small_reclaimable_reserve(self):
        policy = MemoryPolicy()
        for now in range(0, 65, 5):
            self.assertFalse(policy.recovered(detailed_sample(1, reclaimable_bytes=MIN_RECLAIMABLE_BYTES - 1), now))
        self.assertFalse(policy.recovered(detailed_sample(1), 65))
        self.assertTrue(policy.recovered(detailed_sample(1), 95))

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

    def test_critical_pressure_does_not_depend_on_valid_auxiliary_counters(self):
        for reading in ({"pressure": 4}, {"pressure": 4, "swapout_bytes": -1},
                        {"pressure": 4, "swapout_bytes": None, "compressor_bytes": float("nan")}):
            with self.subTest(reading=reading):
                policy = MemoryPolicy()
                self.assertEqual(policy.observe(reading, 0), "critical_memory_pressure")
                self.assertEqual(policy.last_event["sample"]["swapout_bytes"], 0)

    def test_isolated_observed_swap_bursts_and_compression_traffic_continue(self):
        for increments in ({25: 1128 * 1024 ** 2, 35: 1128 * 1024 ** 2},
                           {110: 436 * 1024 ** 2}, {5: 200 * 1024 ** 2, 15: 200 * 1024 ** 2,
                                                  105: 436 * 1024 ** 2}):
            with self.subTest(increments=increments):
                policy, swapped = MemoryPolicy(), 0
                for now in range(0, 601, 5):
                    swapped += increments.get(now, 0)
                    self.assertIsNone(policy.observe(detailed_sample(
                        swapout_bytes=swapped, compression_bytes=now * 5 * 1024 ** 3 // 60,
                        reclaimable_bytes=5 * 1024 ** 3), now))

    def test_one_large_burst_plus_tiny_counter_dribble_is_not_sustained_paging(self):
        for dribble in (16384, 1024 ** 2):
            with self.subTest(dribble=dribble):
                policy = MemoryPolicy()
                for now in range(0, 301, 5):
                    swapped = (436 * 1024 ** 2 if now >= 5 else 0) + now * dribble // 5
                    self.assertIsNone(policy.observe(detailed_sample(swapout_bytes=swapped), now))

    def test_repeated_separated_bulk_paging_is_detected_despite_low_active_fraction(self):
        for spacing in (20, 30, 40):
            with self.subTest(spacing=spacing):
                policy, swapped, stopped = MemoryPolicy(), 0, None
                for now in range(0, 181, 5):
                    if now >= 5 and (now - 5) % spacing == 0:
                        swapped += 128 * 1024 ** 2
                    stopped = policy.observe(detailed_sample(swapout_bytes=swapped), now)
                    if stopped:
                        break
                self.assertEqual(stopped, "sustained_warning_with_swap_growth")
                self.assertGreaterEqual(policy.last_observation["swap_growth_episodes"], 3)

    def test_a_cold_burst_split_across_updates_plus_ambient_paging_is_not_sustained(self):
        policy, swapped = MemoryPolicy(), 0
        for now in range(0, 301, 5):
            swapped += {5: 1400, 10: 400, 15: 400}.get(now, 1 if now >= 20 else 0) * 1024 ** 2
            self.assertIsNone(policy.observe(detailed_sample(swapout_bytes=swapped), now))

    def test_brief_low_reserve_recovers_and_normal_pressure_resets_the_low_streak(self):
        policy = MemoryPolicy()
        low = detailed_sample(reclaimable_bytes=MIN_RECLAIMABLE_BYTES - 1)
        for now in (0, 5, 10, 15, 20):
            self.assertIsNone(policy.observe(low, now))
        self.assertIsNone(policy.observe(detailed_sample(), 25))
        self.assertIsNone(policy.observe(low, 30))
        self.assertIsNone(policy.observe(detailed_sample(1, reclaimable_bytes=0), 55))
        for now in range(60, 90, 5):
            self.assertIsNone(policy.observe(low, now))
        self.assertEqual(policy.observe(low, 90), "sustained_warning_with_memory_churn")

    def test_missing_samples_and_long_gaps_reset_sustained_activity_proof(self):
        for interrupted in (None, "gap"):
            with self.subTest(interrupted=interrupted):
                policy = MemoryPolicy()
                for now in range(0, 45, 5):
                    self.assertIsNone(policy.observe(detailed_sample(swapout_bytes=now * 2 * 1024 ** 2), now))
                if interrupted is None:
                    self.assertIsNone(policy.observe(None, 45))
                self.assertIsNone(policy.observe(detailed_sample(swapout_bytes=64 * 1024 ** 2), 65))
                self.assertEqual(policy.last_observation["swap_activity_intervals"], 0)

    def test_policy_metadata_describes_the_actual_burst_detection_and_protections(self):
        policy = policy_metadata()
        self.assertEqual(policy["version"], 2)
        self.assertEqual(policy["low_reserve_seconds"], LOW_RESERVE_SECONDS)
        self.assertEqual(policy["swapout_growth_bytes"], WARNING_SWAP_BYTES)
        self.assertEqual(policy["swap_activity_minimum_fraction"], .5)
        self.assertEqual(policy["swap_maximum_burst_fraction"], .8)
        self.assertEqual(policy["swap_burst_window_seconds"], 15)
        self.assertEqual(policy["swap_episode_count"], 3)
        self.assertEqual(policy["swap_episode_window_seconds"], 120)
        self.assertTrue(policy["critical_pressure_immediate"])
        self.assertTrue(policy["compression_traffic_diagnostic_only"])
        self.assertTrue(policy["isolated_swap_bursts_continue"])

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

"""Small macOS memory-pressure policy for the owned Plan-judge process.

Swap-out is a cumulative OS counter. Decisions use distributed activity over
one minute or repeated episodes over two minutes, never pre-existing swap or
an isolated burst. Sampling and waiting belong to the supervisor; this module
owns no processes, threads, or patient artifacts.
"""

from collections import deque
import math
import re
import subprocess


SAMPLE_INTERVAL_SECONDS = 5
WARNING_SECONDS = 60
WARNING_SWAP_BYTES = 64 * 1024 * 1024
WARNING_SAMPLE_FRACTION = 0.8
MAX_WARNING_SECONDS = 180
STARTUP_SECONDS = 120
# Retained telemetry reference levels; traffic alone is not capacity exhaustion.
WARNING_CHURN_BYTES = 64 * 1024 * 1024
WARNING_COMPRESSION_BYTES = 256 * 1024 * 1024
MIN_RECLAIMABLE_BYTES = 1024 * 1024 * 1024
LOW_RESERVE_SECONDS = 30
MAX_SAMPLE_GAP_SECONDS = 15
SWAP_ACTIVITY_MIN_BYTES = 1024 * 1024
SWAP_ACTIVITY_SPAN_SECONDS = 40
SWAP_ACTIVITY_FRACTION = 0.5
SWAP_MAX_BURST_FRACTION = 0.8
SWAP_BURST_WINDOW_SECONDS = 15
SWAP_ACTIVITY_EPISODES = 3
SWAP_EPISODE_QUIET_SECONDS = 10
SWAP_EPISODE_WINDOW_SECONDS = 120
SWAP_EPISODE_SPAN_SECONDS = 60
RECOVERY_NORMAL_SECONDS = 30
RECOVERY_TIMEOUT_SECONDS = 300
MAX_SAMPLE_FAILURES = 3


def policy_metadata():
    """Describe actual production decisions, shared by CLI and external harnesses."""
    return {"version": 2, "sample_seconds": SAMPLE_INTERVAL_SECONDS,
            "warning_seconds": WARNING_SECONDS, "warning_sample_fraction": WARNING_SAMPLE_FRACTION,
            "swapout_growth_bytes": WARNING_SWAP_BYTES, "warning_review_seconds": MAX_WARNING_SECONDS,
            "startup_seconds": STARTUP_SECONDS, "minimum_reclaimable_bytes": MIN_RECLAIMABLE_BYTES,
            "low_reserve_seconds": LOW_RESERVE_SECONDS, "maximum_sample_gap_seconds": MAX_SAMPLE_GAP_SECONDS,
            "swap_activity_minimum_interval_bytes": SWAP_ACTIVITY_MIN_BYTES,
            "swap_activity_minimum_span_seconds": SWAP_ACTIVITY_SPAN_SECONDS,
            "swap_activity_minimum_fraction": SWAP_ACTIVITY_FRACTION,
            "swap_maximum_burst_fraction": SWAP_MAX_BURST_FRACTION,
            "swap_burst_window_seconds": SWAP_BURST_WINDOW_SECONDS,
            "swap_episode_count": SWAP_ACTIVITY_EPISODES, "swap_episode_quiet_seconds": SWAP_EPISODE_QUIET_SECONDS,
            "swap_episode_window_seconds": SWAP_EPISODE_WINDOW_SECONDS,
            "swap_episode_minimum_span_seconds": SWAP_EPISODE_SPAN_SECONDS,
            "swap_episode_growth_bytes": WARNING_SWAP_BYTES,
            "compression_traffic_diagnostic_only": True, "isolated_swap_bursts_continue": True,
            "full_auxiliary_window_required_after_warning_review": True,
            "critical_pressure_immediate": True, "normal_recovery_seconds": RECOVERY_NORMAL_SECONDS,
            "recovery_timeout_seconds": RECOVERY_TIMEOUT_SECONDS, "max_sample_failures": MAX_SAMPLE_FAILURES}


def _swap_intervals(history, cutoff):
    return [(max(left[0], cutoff), right[0])
            for left, right in zip(history, list(history)[1:])
            if right[0] > cutoff and right[1]["swapout_bytes"] - left[1]["swapout_bytes"]
            >= SWAP_ACTIVITY_MIN_BYTES]


def _largest_swap_burst(history):
    # Group a burst across multiple sampler updates. Otherwise the same cold
    # allocation would be classified differently solely by polling alignment.
    increments = [(right[0], right[1]["swapout_bytes"] - left[1]["swapout_bytes"])
                  for left, right in zip(history, list(history)[1:])]
    return max((sum(value for at, value in increments
                    if end - SWAP_BURST_WINDOW_SECONDS <= at <= end)
                for end, _ in increments), default=0)


def sample_memory():
    """Read pressure and cumulative swap-out bytes with bounded local commands.

    OSError, TimeoutExpired and ValueError intentionally propagate so the
    supervisor can count failures; missing monitoring must never imply safety.
    """
    pressure_result = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        capture_output=True, text=True, check=True, timeout=2,
    )
    pressure = pressure_result.stdout.strip()
    if pressure not in ("1", "2", "4"):
        raise ValueError("Unknown macOS memory-pressure level")
    if pressure == "4":
        # Emergency pressure must not wait for auxiliary swap telemetry, which
        # may itself fail or stall while the machine is critically constrained.
        return {"pressure": 4, "swapout_bytes": 0}
    vm_result = subprocess.run(
        ["/usr/bin/vm_stat"], capture_output=True, text=True, check=True, timeout=2,
    )
    page_size = re.search(r"page size of\s+(\d+)\s+bytes", vm_result.stdout)
    swapouts = re.search(r"^Swapouts:\s+(\d+)\.?\s*$", vm_result.stdout, re.MULTILINE)
    if page_size is None or swapouts is None or int(page_size.group(1)) <= 0:
        raise ValueError("Cannot read macOS swap-out counter/page size")
    unit = int(page_size.group(1))
    sample = {"pressure": int(pressure), "swapout_bytes": unit * int(swapouts.group(1))}
    # Additional telemetry is optional for older OS formats. Missing fields keep
    # the conservative warning timeout; they can never authorize an exemption.
    fields = {"Swapins": "swapin_bytes", "Pages occupied by compressor": "compressor_bytes",
              "Compressions": "compression_bytes"}
    for label, key in fields.items():
        count = re.search(r"^" + re.escape(label) + r":\s+(\d+)\.?\s*$", vm_result.stdout, re.MULTILINE)
        if count is not None:
            sample[key] = unit * int(count.group(1))
    reclaimable = [re.search(r"^" + re.escape(label) + r":\s+(\d+)\.?\s*$",
                             vm_result.stdout, re.MULTILINE)
                   for label in ("Pages free", "Pages inactive", "Pages speculative")]
    if all(count is not None for count in reclaimable):
        # An approximation of reclaimable pages, not a promise that all are free
        # or independent of file-backed memory. Do not add this to process RSS.
        sample["reclaimable_bytes"] = unit * sum(int(count.group(1)) for count in reclaimable)
    return sample


def _valid_sample(sample):
    return (isinstance(sample, dict) and type(sample.get("pressure")) is int
            and sample["pressure"] in (1, 2, 4)
            and type(sample.get("swapout_bytes")) is int and sample["swapout_bytes"] >= 0
            and all(type(sample[key]) is int and sample[key] >= 0 for key in (
                "swapin_bytes", "compressor_bytes", "compression_bytes", "reclaimable_bytes") if key in sample))


class MemoryPolicy:
    """Evaluate samples without waiting, signaling, or rewriting run metadata.

    ``observe(None, now)`` records a failed sample. ``recovered`` is separate so
    a caller can require stable headroom after releasing the model. Create a
    fresh policy for each owned-server attempt.
    """

    def __init__(self):
        self.last_sample = None
        self.last_event = None
        self.sample_failures = 0
        self._warning_since = None
        self._history = deque()
        self._paging_history = deque()
        self._normal_since = None
        self._normal_swapout = None
        self._low_reserve_since = None
        self._started_at = None
        self.phase = "startup"
        self.last_observation = {}

    def observe(self, sample, now):
        """Return a stop reason, or None while conditions remain acceptable."""
        self._check_time(now)
        if isinstance(sample, dict) and type(sample.get("pressure")) is int and sample["pressure"] == 4:
            # Emergency protection must not depend on optional counters being
            # present/valid, nor put nonfinite auxiliary values into the trace.
            safe = {"pressure": 4, "swapout_bytes": 0}
            safe.update({key: value for key, value in sample.items()
                         if key != "pressure" and type(value) is int and value >= 0})
            self.last_sample = safe
            return self._stop("critical_memory_pressure", now, safe, 0)
        if not _valid_sample(sample):
            self._history.clear()
            self._paging_history.clear()
            self._low_reserve_since = None
            self.sample_failures += 1
            if self.sample_failures >= MAX_SAMPLE_FAILURES:
                return self._stop("memory_monitor_failed", now, None, 0)
            return None
        self.sample_failures = 0
        self.last_sample = dict(sample)
        if self._started_at is None:
            self._started_at = now
        if self._history:
            previous_time, previous = self._history[-1]
            if now <= previous_time:
                raise ValueError("Memory policy time must increase between valid observations")
            if (now - previous_time > MAX_SAMPLE_GAP_SECONDS
                    or any(sample.get(key, 0) < previous.get(key, 0)
                           for key in ("swapout_bytes", "swapin_bytes", "compression_bytes"))):
                self._history.clear()  # A gap/reset cannot prove sustained deterioration.
                self._paging_history.clear()
                self._low_reserve_since = None
        self._history.append((now, dict(sample)))
        self._paging_history.append((now, dict(sample)))
        while len(self._paging_history) > 1 and self._paging_history[1][0] <= now - SWAP_EPISODE_WINDOW_SECONDS:
            self._paging_history.popleft()
        cutoff = now - WARNING_SECONDS
        # Keep one baseline at or before the window boundary.
        while len(self._history) > 1 and self._history[1][0] <= cutoff:
            self._history.popleft()
        baseline = self._history[0][1]
        growth = sample["swapout_bytes"] - baseline["swapout_bytes"]
        complete = all(all(key in item for key in (
            "swapin_bytes", "compressor_bytes", "compression_bytes", "reclaimable_bytes"))
            for _, item in self._history)
        deltas = {key: max(0, sample[key] - baseline[key]) for key in (
            "swapin_bytes", "compressor_bytes", "compression_bytes") if key in sample and key in baseline}
        if "reclaimable_bytes" in sample and "reclaimable_bytes" in baseline:
            deltas["reclaimable_drop_bytes"] = max(0, baseline["reclaimable_bytes"] - sample["reclaimable_bytes"])
        full_window = now - self._history[0][0] >= WARNING_SECONDS
        intervals = _swap_intervals(self._history, cutoff)
        active_seconds = sum(end - start for start, end in intervals)
        span = intervals[-1][1] - intervals[0][1] if intervals else 0
        episode_intervals = _swap_intervals(self._paging_history, now - SWAP_EPISODE_WINDOW_SECONDS)
        episodes = int(bool(episode_intervals)) + sum(
            right[0] - left[1] >= SWAP_EPISODE_QUIET_SECONDS
            for left, right in zip(episode_intervals, episode_intervals[1:]))
        episode_span = (episode_intervals[-1][1] - episode_intervals[0][1]
                        if episode_intervals else 0)
        episode_growth = sample["swapout_bytes"] - self._paging_history[0][1]["swapout_bytes"]
        largest = _largest_swap_burst(self._history)
        episode_largest = _largest_swap_burst(self._paging_history)
        sustained_paging = (full_window and growth >= WARNING_SWAP_BYTES
                            and span >= SWAP_ACTIVITY_SPAN_SECONDS
                            and largest <= growth * SWAP_MAX_BURST_FRACTION
                            and active_seconds >= WARNING_SECONDS * SWAP_ACTIVITY_FRACTION)
        repeated_paging = (episode_growth >= WARNING_SWAP_BYTES and episodes >= SWAP_ACTIVITY_EPISODES
                           and episode_span >= SWAP_EPISODE_SPAN_SECONDS
                           and episode_largest <= episode_growth * SWAP_MAX_BURST_FRACTION)
        self.phase = "startup" if now - self._started_at < STARTUP_SECONDS else "normal"
        self.last_observation = {"phase": self.phase, "window_seconds": now - self._history[0][0],
                                 "swapout_growth_bytes": growth,
                                 "swap_activity_seconds": active_seconds,
                                 "swap_activity_span_seconds": span,
                                 "swap_activity_intervals": len(intervals),
                                 "swap_largest_burst_bytes": largest,
                                 "swap_growth_episodes": episodes, "swap_episode_span_seconds": episode_span,
                                 "swap_episode_growth_bytes": episode_growth, **deltas}
        if sample["pressure"] == 1:
            self._warning_since = None
            self._low_reserve_since = None
            return None
        if self._warning_since is None:
            self._warning_since = now
        duration = now - self._warning_since
        window = [entry for entry in self._history if entry[0] >= cutoff]
        mostly_warning = (now - self._history[0][0] >= WARNING_SECONDS
                          and sum(entry[1]["pressure"] == 2 for entry in window) / len(window) >= WARNING_SAMPLE_FRACTION)
        if (duration >= WARNING_SECONDS or mostly_warning) and (sustained_paging or repeated_paging):
            return self._stop("sustained_warning_with_swap_growth", now, sample, growth)
        if sample.get("reclaimable_bytes", MIN_RECLAIMABLE_BYTES) < MIN_RECLAIMABLE_BYTES:
            if self._low_reserve_since is None:
                self._low_reserve_since = now
            if now - self._low_reserve_since >= LOW_RESERVE_SECONDS:
                return self._stop("sustained_warning_with_memory_churn", now, sample, growth)
        else:
            self._low_reserve_since = None
        if duration >= MAX_WARNING_SECONDS:
            # Warning pressure, compression traffic and one paging burst do not
            # establish exhaustion. Capacity and sustained paging are checked
            # above; missing/gapped telemetry still cannot authorize continuation.
            if not complete or not full_window:
                return self._stop("sustained_memory_warning", now, sample, growth)
            self.phase = "warning_paging" if growth else "stable_warning"
        elif self.phase != "startup":
            self.phase = "warning"
        self.last_observation["phase"] = self.phase
        return None

    def recovered(self, sample, now):
        """Require 30 continuous normal seconds without any additional swap-out."""
        self._check_time(now)
        if not _valid_sample(sample) or sample["pressure"] != 1:
            self._normal_since = self._normal_swapout = None
            return False
        if sample.get("reclaimable_bytes", MIN_RECLAIMABLE_BYTES) < MIN_RECLAIMABLE_BYTES:
            self._normal_since = self._normal_swapout = None
            return False
        if self._normal_since is None or sample["swapout_bytes"] != self._normal_swapout:
            self._normal_since = now
            self._normal_swapout = sample["swapout_bytes"]
        return now - self._normal_since >= RECOVERY_NORMAL_SECONDS

    @staticmethod
    def _check_time(now):
        if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
            raise ValueError("Memory policy time must be finite monotonic seconds")

    def _stop(self, reason, now, sample, growth):
        self.phase = "stopping"
        self.last_observation["phase"] = self.phase
        self.last_event = {"reason": reason, "monotonic_seconds": now,
                           "sample": None if sample is None else dict(sample),
                           "swapout_growth_bytes": growth}
        return reason

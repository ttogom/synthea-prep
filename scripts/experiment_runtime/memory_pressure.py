"""Small macOS memory-pressure policy for the owned Plan-judge process.

Swap-out is a cumulative OS counter. Decisions use growth in the latest minute,
not swap already consumed before judging. Sampling and waiting belong to the
supervisor; this module owns no processes, threads, or patient artifacts.
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
# Conservative new churn guards, not benchmark-calibrated throughput settings.
# A stable warning needs a full minute below these growth limits and a reserve.
WARNING_CHURN_BYTES = 64 * 1024 * 1024
WARNING_COMPRESSION_BYTES = 256 * 1024 * 1024
MIN_RECLAIMABLE_BYTES = 1024 * 1024 * 1024
RECOVERY_NORMAL_SECONDS = 30
RECOVERY_TIMEOUT_SECONDS = 300
MAX_SAMPLE_FAILURES = 3


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
        self._normal_since = None
        self._normal_swapout = None
        self._started_at = None
        self.phase = "startup"
        self.last_observation = {}

    def observe(self, sample, now):
        """Return a stop reason, or None while conditions remain acceptable."""
        self._check_time(now)
        if not _valid_sample(sample):
            self.sample_failures += 1
            if self.sample_failures >= MAX_SAMPLE_FAILURES:
                return self._stop("memory_monitor_failed", now, None, 0)
            return None
        self.sample_failures = 0
        self.last_sample = dict(sample)
        if self._started_at is None:
            self._started_at = now
        if self._history and any(sample.get(key, 0) < self._history[-1][1].get(key, 0)
                                 for key in ("swapout_bytes", "swapin_bytes", "compression_bytes")):
            self._history.clear()  # Counter reset: do not invent negative growth.
        self._history.append((now, dict(sample)))
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
        churn = (complete and (deltas["swapin_bytes"] >= WARNING_CHURN_BYTES
                 or deltas["compressor_bytes"] >= WARNING_CHURN_BYTES
                 or deltas["compression_bytes"] >= WARNING_COMPRESSION_BYTES
                 or deltas["reclaimable_drop_bytes"] >= WARNING_CHURN_BYTES))
        self.phase = "startup" if now - self._started_at < STARTUP_SECONDS else "normal"
        self.last_observation = {"phase": self.phase, "window_seconds": now - self._history[0][0],
                                 "swapout_growth_bytes": growth, **deltas}
        if sample["pressure"] == 4:
            return self._stop("critical_memory_pressure", now, sample, growth)
        if sample["pressure"] == 1:
            self._warning_since = None
            return None
        if self._warning_since is None:
            self._warning_since = now
        duration = now - self._warning_since
        window = [entry for entry in self._history if entry[0] >= cutoff]
        mostly_warning = (now - self._history[0][0] >= WARNING_SECONDS
                          and sum(entry[1]["pressure"] == 2 for entry in window) / len(window) >= WARNING_SAMPLE_FRACTION)
        if (duration >= WARNING_SECONDS or mostly_warning) and growth >= WARNING_SWAP_BYTES:
            return self._stop("sustained_warning_with_swap_growth", now, sample, growth)
        # Loading model/cache pages can move these counters even with useful
        # headroom. Churn alone must not shorten the original 180-second window.
        if ((duration >= WARNING_SECONDS or mostly_warning) and full_window and churn
                and sample["reclaimable_bytes"] < MIN_RECLAIMABLE_BYTES):
            return self._stop("sustained_warning_with_memory_churn", now, sample, growth)
        if duration >= MAX_WARNING_SECONDS:
            # A static warning is not itself evidence that the run is worsening.
            # Permit it only with a complete, stable minute of auxiliary readings
            # and a modest reclaimable reserve. Old/missing telemetry stays strict.
            if (not complete or not full_window or churn or growth != 0
                    or sample["reclaimable_bytes"] < MIN_RECLAIMABLE_BYTES):
                return self._stop("sustained_memory_warning", now, sample, growth)
            self.phase = "stable_warning"
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

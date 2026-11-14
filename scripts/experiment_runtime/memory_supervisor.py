"""Bounded, lightweight watchdog for an isolated Plan-judging worker and server."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

from scripts.experiment_runtime.memory_pressure import (
    MemoryPolicy, sample_memory, SAMPLE_INTERVAL_SECONDS, RECOVERY_TIMEOUT_SECONDS,
    WARNING_SECONDS, WARNING_SWAP_BYTES, MAX_WARNING_SECONDS, RECOVERY_NORMAL_SECONDS,
    MAX_SAMPLE_FAILURES, WARNING_SAMPLE_FRACTION,
    STARTUP_SECONDS, WARNING_CHURN_BYTES, WARNING_COMPRESSION_BYTES, MIN_RECLAIMABLE_BYTES,
    policy_metadata,
)
from scripts.experiment_runtime.ollama import ollama_for_saved_run
from scripts.experiment_runtime.plan_judging import load_judge_profile
from scripts.med_copilot_run_lock import exclusive_run_lock


MEMORY_STOP_EXIT = 75
GRACE_SECONDS = 30
WORKER_EXIT_SECONDS = 5


class MemoryPressureStop(KeyboardInterrupt):
    """Stop new scalar calls while completed replies remain durably reusable."""


@contextmanager
def inherited_run_lock(run_dir, descriptor):
    """Use the parent's inherited open-file lock, never an environment-only bypass."""
    actual = os.fstat(descriptor)
    expected = (Path(run_dir) / ".run.lock").stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("Inherited judge lock does not match the run directory")
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    yield  # Parent releases the shared lock only after worker/server cleanup.


def _sample():
    try:
        return sample_memory()
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _event(path, kind, *, quiet=False, **detail):
    record = {"event": kind, "unix_seconds": time.time(), **detail}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        if not quiet:
            os.fsync(handle.fileno())
    if not quiet:
        print(f"Plan memory guard: {kind}" + (f" ({detail['reason']})" if 'reason' in detail else ""),
              file=sys.stderr, flush=True)


def _headroom(events=None):
    policy = MemoryPolicy()
    deadline = time.monotonic() + RECOVERY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        sample = _sample()
        now = time.monotonic()
        if events is not None:
            _event(events, "sample", quiet=True, phase="recovering", sample=sample, monotonic_seconds=now)
        if policy.observe(sample, now) == "memory_monitor_failed":
            return False
        if policy.recovered(sample, now):
            return True
        time.sleep(SAMPLE_INTERVAL_SECONDS)
    return False


def _owned_rss(worker, server):
    """Bounded RSS diagnostics only; shared/GPU pages are not unique footprint."""
    process = getattr(server, "_process", None)
    group = getattr(process, "pid", None)
    if type(group) is not int or group <= 0:
        return None
    try:
        result = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,rss="],
                                capture_output=True, text=True, check=True, timeout=2)
        rows = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
        return {"worker_rss_bytes": sum(rss * 1024 for pid, _, rss in rows if pid == worker.pid),
                "server_group_rss_bytes": sum(rss * 1024 for _, pgid, rss in rows if pgid == group)}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"error": type(exc).__name__}


def _terminate_worker(worker):
    """Signal only the private session we created, with bounded TERM/KILL waits."""
    for sig, timeout in ((signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        if worker.poll() is not None:
            return
        try:
            os.killpg(worker.pid, sig)
        except ProcessLookupError:
            pass
        try:
            worker.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    raise RuntimeError("Owned Plan worker did not terminate within the shutdown bound")


def _watch(worker, server, stop_file, events):
    policy = MemoryPolicy()
    previous_pressure = None
    while worker.poll() is None:
        sample = _sample()
        now = time.monotonic()
        reason = policy.observe(sample, now)
        # Every sample goes to disk; policy retains only the latest minute.
        # Emergency shutdown never waits for optional per-process diagnostics.
        _event(events, "sample", quiet=True, sample=sample, monotonic_seconds=now,
               **policy.last_observation,
               owned_rss=None if reason == "critical_memory_pressure" else _owned_rss(worker, server))
        pressure = None if sample is None else sample["pressure"]
        if pressure != previous_pressure:
            _event(events, "pressure_changed", sample=sample)
            previous_pressure = pressure
        if reason:
            _event(events, "stopping", **policy.last_event)
            stop_file.write_text(reason, encoding="utf-8")
            deadline = time.monotonic() + (0 if reason == "critical_memory_pressure" else GRACE_SECONDS)
            while worker.poll() is None and time.monotonic() < deadline:
                sample = _sample()
                if sample is not None and sample["pressure"] == 4:
                    break
                time.sleep(1)
            # This always owns a separate native-server process group. Closing
            # it also unblocks network calls before the final worker deadline.
            server.close(force_group=True)
            try:
                worker.wait(timeout=WORKER_EXIT_SECONDS)
            except subprocess.TimeoutExpired:
                _terminate_worker(worker)
            _event(events, "stopped", reason=reason, worker_exit=worker.returncode)
            return reason
        time.sleep(SAMPLE_INTERVAL_SECONDS)
    return None


def supervise_cli(argv, *, root):
    """Hold one run lock across at most two attempts, including C1 recovery."""
    parser = argparse.ArgumentParser(description="Judge saved Plans with automatic memory protection")
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--context-length", type=int)
    args = parser.parse_args(argv)
    run_dir = args.run_directory.resolve()
    try:
        with exclusive_run_lock(run_dir) as lock, TemporaryDirectory(prefix="plan-judge-guard-") as temporary:
            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            profile = load_judge_profile(root / "experiments/configs/plan_judging.yaml", metadata,
                                         concurrency=args.concurrency, context_length=args.context_length)
            events = run_dir / "provenance" / f"plan-judge-memory-{uuid4().hex}.jsonl"
            events.parent.mkdir(parents=True, exist_ok=True)
            policy = {**policy_metadata(), "grace_seconds": GRACE_SECONDS,
                      "worker_exit_seconds": WORKER_EXIT_SECONDS, "max_attempts": 2}
            _event(events, "started", policy=policy)
            for attempt in range(2):
                if not _headroom(events):
                    _event(events, "paused", reason="headroom_not_restored")
                    return MEMORY_STOP_EXIT
                effective = profile["settings"]
                settings = {key: effective[key] for key in ("concurrency", "context_length")}
                stop_file = Path(temporary) / "stop"
                stop_file.unlink(missing_ok=True)
                with ollama_for_saved_run(run_dir, **settings, runtime_profile=profile) as server:
                    bootstrap = Path(temporary) / "worker.json"
                    bootstrap.write_text(json.dumps({"run_dir": str(run_dir), "root": str(root),
                        "profile": profile, "server": server.runtime_descriptor(), "lock_fd": lock.fileno(),
                        "stop_file": str(stop_file), "guard_metadata": {"events_file": str(events.relative_to(run_dir)),
                        "attempt": attempt + 1, "policy": policy}}), encoding="utf-8")
                    worker = subprocess.Popen([sys.executable, "-m", "scripts.experiment_runtime.judge_worker", str(bootstrap)],
                        cwd=root, pass_fds=(lock.fileno(),), start_new_session=True)
                    try:
                        reason = _watch(worker, server, stop_file, events)
                        code = worker.wait(timeout=WORKER_EXIT_SECONDS)
                    finally:
                        try:
                            server.close(force_group=True)
                        finally:
                            if worker.poll() is None:
                                _terminate_worker(worker)
                if reason is None:
                    _event(events, "finished", worker_exit=code)
                    return code if code >= 0 else 1
                if attempt or reason == "memory_monitor_failed":
                    _event(events, "paused", reason=reason)
                    return MEMORY_STOP_EXIT
                reduced = 2 if effective["concurrency"] > 2 else 1
                reduced_workers = min(effective.get("client_workers", effective["concurrency"]), reduced)
                profile["settings"]["concurrency"] = reduced
                if "client_workers" in effective:
                    profile["settings"]["client_workers"] = reduced_workers
                _event(events, "retrying", concurrency=reduced_workers, server_slots=reduced)
            return MEMORY_STOP_EXIT
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Plan memory guard failed: {exc}", file=sys.stderr)
        return 1

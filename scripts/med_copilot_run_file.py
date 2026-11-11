"""Own run.json and record run errors using only the standard library.

The runner imports this before its heavy dependencies, so a run directory and
run.json exist (and can record the failure) even when those imports fail.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import traceback
from zoneinfo import ZoneInfo


PACIFIC = ZoneInfo("America/Los_Angeles")
RUN_FILE = "run.json"
SHARED_CREDENTIALS = ("OPENAI_API_KEY", "GRAPHRAG_API_KEY")
# Stages named by run.json error entries.
RUN_STAGES = (
    "startup", "input_loading", "corpus_loading", "retrieval_initialization",
    "keyword_extraction", "patient_retrieval", "reranking", "guideline_question_generation",
    "guideline_retrieval", "agentic_retrieval", "final_generation", "scoring", "output_saving",
)
# The run/patient a failure belongs to and the active stage, local to each execution.
_ERROR_SCOPE = ContextVar("med_copilot_error_scope", default=None)
_STAGE = ContextVar("med_copilot_stage", default=None)
# Where an escaping exception was raised; consumed when the failure is recorded.
_ERROR_ORIGIN = "_med_copilot_error_origin"


# %% ------------------------------------------------------------------
# Run naming and directories
# ---------------------------------------------------------------------

def pacific_time(instant):
    """Convert an aware datetime to whole-second America/Los_Angeles time."""
    return instant.astimezone(PACIFIC).replace(microsecond=0)


def pacific_now():
    return pacific_time(datetime.now(timezone.utc))


def path_safe_branch(branch):
    """Make a branch name usable as one directory-name component."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", branch).strip(".-")
    if not safe:
        raise ValueError(f"Branch name {branch!r} has no path-safe characters")
    return safe


def _timestamp(started_at):
    return started_at.strftime("%Y-%m-%d_%H%M%S") + started_at.tzname()


def make_run_name(started_at, input_path, branch):
    """<Pacific timestamp>__<input filename stem>__<branch name>."""
    return f"{_timestamp(started_at)}__{Path(input_path).stem}__{path_safe_branch(branch)}"


def _make_dirs(path):
    """Create path and any missing parents; return those created, deepest first."""
    path = Path(path).resolve()
    missing = [directory for directory in (path, *path.parents) if not directory.exists()]
    path.mkdir(parents=True, exist_ok=True)
    return missing


def create_run_dir(output_root, run_name):
    """Create a new run directory, never reusing an existing one."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_name
    try:
        run_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(errno.EEXIST, "Run directory already exists; refusing to overwrite",
                              str(run_dir)) from exc
    return run_dir


def write_run_json(run_dir, metadata):
    """Atomically replace run.json so readers never see a partial file."""
    run_dir = Path(run_dir)
    temporary = run_dir / f".{RUN_FILE}.tmp"
    try:
        temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
                             + "\n", encoding="utf-8")
        os.replace(temporary, run_dir / RUN_FILE)
    finally:
        temporary.unlink(missing_ok=True)


# %% ------------------------------------------------------------------
# Error recording
# ---------------------------------------------------------------------

class RunMetadataError(RuntimeError):
    """run.json could not be updated, so the run cannot continue safely."""


def redact_secrets(text, secrets):
    """Replace every non-empty secret in text, longest first."""
    for secret in sorted({value for value in secrets if value}, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return text


def redact_shared_credentials(text):
    """Redact the shared API keys; used until the selected model keys are known."""
    return redact_secrets(text, [os.environ.get(name) for name in SHARED_CREDENTIALS])


def _mark_origin(exc, field, value):
    """Keep the innermost value as an exception escapes nested scopes."""
    try:
        origin = exc.__dict__.setdefault(_ERROR_ORIGIN, {})
    except AttributeError:
        return
    origin.setdefault(field, value)


@contextmanager
def error_scope(run, *, patient_id=None):
    """Attribute failures recorded inside this block to one run and patient."""
    scope_token = _ERROR_SCOPE.set({"run": run, "patient_id": patient_id})
    stage_token = _STAGE.set(None)
    try:
        yield
    except BaseException as exc:
        _mark_origin(exc, "patient_id", patient_id)
        raise
    finally:
        _STAGE.reset(stage_token)
        _ERROR_SCOPE.reset(scope_token)


@contextmanager
def pipeline_stage(stage):
    """Mark the active pipeline stage; an escaping exception keeps the innermost one."""
    if stage not in RUN_STAGES:
        raise ValueError(f"Unknown pipeline stage {stage!r}")
    token = _STAGE.set(stage)
    try:
        yield
    except BaseException as exc:
        _mark_origin(exc, "stage", stage)
        raise
    finally:
        _STAGE.reset(token)


def record_error(exc, *, run=None, stage=None, unique=False):
    """Save one failure to run.json immediately and return its entry.

    Call this once, where the failure is handled. The run, patient, and stage
    default to where the exception was raised, then to the active error_scope.
    The message comes from the exception, and the traceback keeps chained
    causes. Without a run, the entry is returned unsaved. With unique=True, an
    entry already saved with the same patient, stage, type, and message is kept
    instead (for failures that a resumed evaluation can meet again). If run.json
    cannot be written, both failures go to stderr and RunMetadataError is raised.
    """
    scope = _ERROR_SCOPE.get() or {}
    origin = exc.__dict__.pop(_ERROR_ORIGIN, {}) if hasattr(exc, "__dict__") else {}
    run = run or scope.get("run")
    redact = getattr(run, "redact", None) or (lambda text: text)
    entry = {
        "timestamp": pacific_now().isoformat(),
        "patient_id": origin.get("patient_id", scope.get("patient_id")),
        "stage": stage or origin.get("stage") or _STAGE.get(),
        "exception_type": type(exc).__name__,
        "message": redact(str(exc)),
        "traceback": redact("".join(traceback.format_exception(exc))),
    }
    identity = ("patient_id", "stage", "exception_type", "message")
    if unique and run is not None:
        for saved in getattr(run, "metadata", {}).get("errors", []):
            if all(saved.get(field) == entry[field] for field in identity):
                return saved
    if run is not None:
        try:
            run.add_error(entry)
        except Exception as write_exc:
            print(f"Could not save error details to {RUN_FILE}. Original failure:\n"
                  f"{entry['traceback']}\n{RUN_FILE} write failure:", file=sys.stderr)
            traceback.print_exception(write_exc)
            raise RunMetadataError(f"Could not save error details to {RUN_FILE}") from write_exc
    return entry


# %% ------------------------------------------------------------------
# run.json owner
# ---------------------------------------------------------------------

class RunFile:
    """Own one run directory's run.json; updates are serialized across tasks."""

    def __init__(self, run_dir, metadata, created=(), started_at=None):
        # Absolute, so stages that change the working directory still save here.
        self.run_dir = Path(run_dir).resolve()
        self.metadata = metadata
        self.started_at = started_at
        self.redact = redact_shared_credentials
        # Output directories this run created, removed again if it is cancelled.
        self._created = list(created)
        self._lock = threading.Lock()

    @classmethod
    def begin(cls, output_root, *, started_at=None, run_name=None):
        """Create the run directory with a running run.json.

        Without a run name the directory is provisional,
        `<timestamp>__starting-<suffix>`, until move() names it.
        """
        started_at = started_at or pacific_now()
        created = _make_dirs(output_root)
        if run_name is None:
            run_dir = Path(tempfile.mkdtemp(prefix=f"{_timestamp(started_at)}__starting-",
                                            dir=output_root))
        else:
            run_dir = create_run_dir(output_root, run_name)
        run = cls(run_dir, {
            "run_name": run_dir.name,
            "started_at": started_at.isoformat(),
            "finished_at": None,
            "status": "running",
            "branch": None,
            "input": None,
            "git_commit": None,
            "reference_data": None,
            "errors": [],
        }, created, started_at)
        run.update()
        return run

    def update(self, **fields):
        with self._lock:
            self.metadata.update(fields)
            write_run_json(self.run_dir, self.metadata)

    def add_error(self, entry):
        """Append one error entry and persist it before execution continues."""
        with self._lock:
            self.metadata["errors"].append(entry)
            write_run_json(self.run_dir, self.metadata)

    def move(self, output_root, run_name):
        """Rename the run directory, never replacing an existing run."""
        with self._lock:
            self._created += [path for path in _make_dirs(output_root) if path not in self._created]
            target = create_run_dir(output_root, run_name).resolve()
            for path in self.run_dir.iterdir():
                os.replace(path, target / path.name)
            self.run_dir.rmdir()
            self.run_dir = target
            self.metadata["run_name"] = target.name
            write_run_json(self.run_dir, self.metadata)
            # Drop output directories created only for the provisional location.
            for directory in sorted(self._created, key=lambda path: len(path.parts), reverse=True):
                if not target.is_relative_to(directory):
                    try:
                        directory.rmdir()
                        self._created.remove(directory)
                    except OSError:
                        pass

    def discard(self):
        """Remove a cancelled run's run.json, directory, and newly created parents."""
        with self._lock:
            (self.run_dir / RUN_FILE).unlink(missing_ok=True)
            self.run_dir.rmdir()
            for directory in sorted(self._created, key=lambda path: len(path.parts), reverse=True):
                try:
                    directory.rmdir()
                except OSError:
                    pass  # Not empty: other runs live there.

    def record_fatal(self, exc, *, stage=None):
        """Record the failure that stops this run, unless saving run.json is what failed."""
        if isinstance(exc, RunMetadataError):
            return
        try:
            record_error(exc, run=self, stage=stage)
        except RunMetadataError:
            pass  # record_error already reported both failures to stderr.

    def fail(self, exc, *, stage=None):
        """Record a fatal failure, then mark the run failed_incomplete."""
        self.record_fatal(exc, stage=stage)
        self.stop(pacific_now())

    def stop(self, finished_at):
        """Best-effort failed_incomplete finalization; reports a failed write."""
        try:
            self.update(status="failed_incomplete", finished_at=finished_at.isoformat())
        except Exception:
            # The caller re-raises the original failure after this report.
            print(f"Could not mark {RUN_FILE} as stopped:", file=sys.stderr)
            traceback.print_exc()

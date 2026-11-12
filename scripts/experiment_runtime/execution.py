"""Durable job checkpoints and bounded execution for the experiment coordinator.

Workers return independent payloads; one JobStore connection serializes every
transaction. Canonical results remain the public artifact, not this internal DB.
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from threading import RLock
from time import perf_counter

from scripts.med_copilot_run_file import RunMetadataError, _fsync_directory


SCHEMA_VERSION = 1


def _reject_constant(value):
    raise ValueError(f"Nonfinite checkpoint value: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint key: {key}")
        result[key] = value
    return result


def _decode(text):
    value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    # The re-encoding also rejects finite-looking numbers that overflow to infinity.
    json.dumps(value, allow_nan=False)
    return value


def _key(key):
    if (not isinstance(key, tuple) or len(key) != 4
            or any(not isinstance(value, str) or not value.strip() for value in key[:2])
            or any(value is not None and (not isinstance(value, str) or not value.strip())
                   for value in key[2:])):
        raise ValueError("Job key must be (stage, patient_id, variant_or_none, model_or_none)")
    return (*key[:2], *(value or "" for value in key[2:]))


def _public_key(key):
    return (*key[:2], *(value or None for value in key[2:]))


class JobStore:
    """One serialized SQLite writer with durable, independently recoverable jobs.

    Identity is an arbitrary non-secret JSON object describing the run's inputs
    and settings. Existing identities are checked read-only before any mutation.
    save() returns only after a FULL-synchronous WAL transaction commits. Callers
    must hold the experiment directory lock while using this connection.
    """

    def __init__(self, path, *, identity=None, redact=None):
        if identity is not None and not isinstance(identity, dict):
            raise TypeError("Job-store identity must be an object or None")
        if redact is not None and not callable(redact):
            raise TypeError("redact must be callable or None")
        self.path = Path(path).resolve()
        self._lock = RLock()
        self._connection = None
        self._redact = redact or (lambda text: text)
        encoded_identity = json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False)
        existing = self.path.exists()
        if existing:
            self._validate_existing(encoded_identity if identity is not None else None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA busy_timeout=30000")
            if not existing:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
                connection.execute(
                    "CREATE TABLE jobs (stage TEXT NOT NULL, patient TEXT NOT NULL, "
                    "variant TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL, "
                    "payload TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision > 0), "
                    "PRIMARY KEY(stage, patient, variant, model))")
                connection.executemany("INSERT INTO metadata VALUES (?, ?)", (
                    ("schema_version", str(SCHEMA_VERSION)), ("identity", encoded_identity)))
                connection.execute("COMMIT")
                _fsync_directory(self.path.parent)
        except BaseException:
            connection.close()
            raise
        self._connection = connection

    def _validate_existing(self, expected_identity):
        try:
            connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
            try:
                metadata = dict(connection.execute("SELECT name, value FROM metadata"))
                if metadata.get("schema_version") != str(SCHEMA_VERSION):
                    raise ValueError("Unsupported job-store schema")
                _decode(metadata["identity"])
                if expected_identity is not None and metadata["identity"] != expected_identity:
                    raise ValueError("Job-store identity differs from this experiment")
                for stage, patient, variant, model, status, payload, revision in connection.execute(
                        "SELECT stage, patient, variant, model, status, payload, revision FROM jobs"):
                    _key(_public_key((stage, patient, variant, model)))
                    if (not isinstance(status, str) or not status.strip()
                            or type(revision) is not int or revision < 1
                            or not isinstance(_decode(payload), dict)):
                        raise ValueError("Invalid saved job checkpoint")
            finally:
                connection.close()
        except (sqlite3.Error, KeyError, TypeError) as exc:
            raise ValueError("Invalid job store") from exc

    def _open(self):
        if self._connection is None:
            raise RuntimeError("Job store is closed")
        return self._connection

    def save(self, key, payload, status="checkpoint"):
        """Durably replace one job and return its new monotonically increasing revision."""
        key = _key(key)
        if not isinstance(payload, dict):
            raise TypeError("Job checkpoint payload must be an object")
        if not isinstance(status, str) or not status.strip():
            raise ValueError("Job status must be a nonempty string")
        encoded = self._redact(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        if not isinstance(encoded, str) or not isinstance(_decode(encoded), dict):
            raise ValueError("Redaction must preserve a JSON-object checkpoint")
        with self._lock:
            connection = self._open()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT revision FROM jobs WHERE stage=? AND patient=? AND variant=? AND model=?", key).fetchone()
                revision = 1 if row is None else row[0] + 1
                connection.execute(
                    "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(stage,patient,variant,model) DO UPDATE SET "
                    "status=excluded.status,payload=excluded.payload,revision=excluded.revision",
                    (*key, status, encoded, revision))
                connection.execute("COMMIT")
            except sqlite3.Error as exc:
                if connection.in_transaction:
                    try:
                        connection.execute("ROLLBACK")
                    except sqlite3.Error as rollback_error:
                        raise RunMetadataError("Could not roll back durable job checkpoint") from rollback_error
                raise RunMetadataError("Could not save durable job checkpoint") from exc
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return revision

    def state(self, key):
        """Read a detached payload, status, and revision for one job, or None."""
        key = _key(key)
        with self._lock:
            row = self._open().execute(
                "SELECT status,payload,revision FROM jobs WHERE stage=? AND patient=? AND variant=? AND model=?", key).fetchone()
        return None if row is None else {"status": row[0], "payload": _decode(row[1]), "revision": row[2]}

    def load(self, key):
        """Read a detached job payload, or None when the job has never been saved."""
        state = self.state(key)
        return None if state is None else state["payload"]

    def entries(self, stage=None):
        """Return deterministic (key, state) pairs, optionally filtered by stage."""
        if stage is not None and (not isinstance(stage, str) or not stage.strip()):
            raise ValueError("stage must be a nonempty string or None")
        query = "SELECT stage,patient,variant,model,status,payload,revision FROM jobs"
        arguments = () if stage is None else (stage,)
        if stage is not None:
            query += " WHERE stage=?"
        query += " ORDER BY stage,patient,variant,model"
        with self._lock:
            rows = self._open().execute(query, arguments).fetchall()
        return [(_public_key(row[:4]), {"status": row[4], "payload": _decode(row[5]), "revision": row[6]})
                for row in rows]

    def close(self):
        """Release the connection; committed checkpoints remain durable."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, kind, value, traceback):
        self.close()


def _timed_call(function, item):
    start = perf_counter()
    result = function(item)
    return result, perf_counter() - start


def bounded_map(function, iterable, *, workers, max_pending=None):
    """Yield (item, result, execution_seconds) with a bounded submission window.

    Results are delivered as they finish. Every item gets its own ContextVar
    context; callers must still give each task independent mutable payloads.
    On failure or generator close, queued work is cancelled and running calls
    finish their durable checkpointing before this generator exits.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    limit = workers * 2 if max_pending is None else max_pending
    if type(limit) is not int or limit < workers:
        raise ValueError("max_pending must be an integer at least workers")
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="experiment-job")
    iterator = iter(iterable)
    pending = {}
    exhausted = False
    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < limit:
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = executor.submit(copy_context().run, _timed_call, function, item)
                pending[future] = item
            if pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    item = pending.pop(future)
                    result, seconds = future.result()
                    yield item, result, seconds
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def write_results(records, path):
    """Atomically export canonical JSONL; incomplete output never replaces the old file."""
    path = Path(path).resolve()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".results-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

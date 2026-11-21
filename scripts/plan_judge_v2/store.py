"""Per-variant output directory: manifest.json, judgments.jsonl checkpoint, and a writer lease.

Layout: <out>/<run-name>/<variant>/
  manifest.json    shared metadata written once per run, plus one entry per session
  judgments.jsonl  append-only rows that reference manifest.run_id; fsynced per batch
  judgments.csv    optional export, one row per (patient, generator model)
  .lease           present while one writer owns the variant

Rows are the checkpoint: a resumed run skips keys already scored or recorded as
unavailable and retries invalid/error rows up to max_attempts. A manifest made
with a different protocol, input hash, selection, or judge is refused rather
than mixed.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import uuid


MANIFEST = "manifest.json"
JUDGMENTS = "judgments.jsonl"
CSV_FILE = "judgments.csv"
LEASE = ".lease"
MANIFEST_VERSION = 1
ROW_VERSION = 1
FINAL_STATUSES = ("scored", "unavailable")
RETRY_STATUSES = ("invalid_response", "error")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_writer_id():
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _fsync_directory(directory):
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return  # Some filesystems (e.g. Drive FUSE, Windows) cannot open directories.
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_text(path, text):
    """Write a sibling temporary file, fsync it, then rename it over path."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class OutputConflict(RuntimeError):
    """The output directory belongs to an incompatible run, or is corrupt."""


class LeaseHeld(RuntimeError):
    """Another writer owns this variant."""


class LeaseLost(RuntimeError):
    """This writer's lease was taken over or removed while judging."""


def compatibility_key(manifest):
    """Fields that must match for rows to be comparable; paths may differ between users."""
    return {
        "manifest_version": manifest["manifest_version"],
        "protocol": manifest["protocol"],
        "variant": manifest["variant"],
        "inputs": {name: entry["sha256"] for name, entry in manifest["inputs"].items()},
        "selection": manifest["selection"],
        "judge": manifest["judge"],
    }


def _differences(saved, current, prefix=""):
    if isinstance(saved, dict) and isinstance(current, dict):
        for key in sorted(set(saved) | set(current)):
            yield from _differences(saved.get(key), current.get(key), f"{prefix}{key}.")
    elif saved != current:
        yield f"{prefix[:-1]}: saved {saved!r}, current {current!r}"


def open_manifest(directory, proposed):
    """Create the manifest, or return the saved one when it is compatible."""
    path = Path(directory) / MANIFEST
    if not path.exists():
        manifest = {**proposed, "run_id": uuid.uuid4().hex, "created_at": utc_now(), "sessions": []}
        atomic_write_json(path, manifest)
        return manifest
    saved = read_json(path)
    if saved.get("manifest_version") != MANIFEST_VERSION or saved.get("protocol", {}).get("version") \
            != proposed["protocol"]["version"]:
        raise OutputConflict(f"{path} was written by another manifest/protocol version; "
                             "use a new output directory")
    differences = list(_differences(compatibility_key(saved), compatibility_key(proposed)))
    if differences:
        raise OutputConflict(f"{path} belongs to a different judging run:\n  " + "\n  ".join(differences)
                             + "\nUse a new output directory instead of mixing results.")
    return saved


class Lease:
    """Exclusive writer ownership of one variant directory.

    Created with O_EXCL, then re-read to confirm ownership (network filesystems
    such as Google Drive do not guarantee O_EXCL across machines, so the re-read
    and per-batch ownership checks are the second line of defence). A background
    thread refreshes the heartbeat; a lease whose heartbeat is older than
    stale_after seconds can be taken over only when takeover_stale is set.
    """

    def __init__(self, directory, writer_id, owner=None, *, stale_after=900, heartbeat=60,
                 settle=0.5, takeover_stale=False, clock=time.time):
        self.path = Path(directory) / LEASE
        self.writer_id, self.owner = writer_id, owner
        self.stale_after, self.heartbeat, self.settle = stale_after, heartbeat, settle
        self.takeover_stale, self.clock = takeover_stale, clock
        self._stop = threading.Event()
        self._thread = None
        self.lost = None

    def _content(self, started_at):
        return {"writer_id": self.writer_id, "owner": self.owner, "host": socket.gethostname(),
                "pid": os.getpid(), "started_at": started_at, "heartbeat_unix": self.clock()}

    def read(self):
        try:
            return read_json(self.path)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return {"writer_id": None, "heartbeat_unix": 0, "unreadable": True}

    def acquire(self):
        started = utc_now()
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = self.read() or {}
            age = self.clock() - float(holder.get("heartbeat_unix") or 0)
            if age < self.stale_after:
                raise LeaseHeld(f"Variant is being judged by {holder.get('owner') or holder.get('writer_id')} "
                                f"(heartbeat {age:.0f}s ago). Wait for it to finish.") from None
            if not self.takeover_stale:
                raise LeaseHeld(f"Stale lease from {holder.get('owner') or holder.get('writer_id')} "
                                f"(heartbeat {age:.0f}s ago). Confirm nobody is judging this variant, "
                                "then rerun with takeover_stale enabled.") from None
            atomic_write_json(self.path, self._content(started))
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self._content(started), handle)
                handle.flush()
                os.fsync(handle.fileno())
        time.sleep(self.settle)
        self.verify()
        self._started = started
        self._thread = threading.Thread(target=self._beat, name="plan-judge-lease", daemon=True)
        self._thread.start()

    def verify(self):
        if self.lost is not None:
            raise LeaseLost(self.lost)
        holder = self.read()
        if not holder or holder.get("writer_id") != self.writer_id:
            self.lost = (f"Lease now held by {holder.get('owner') or holder.get('writer_id')}" if holder
                         else "Lease file was removed")
            raise LeaseLost(self.lost)

    def _beat(self):
        while not self._stop.wait(self.heartbeat):
            try:
                self.verify()
                atomic_write_json(self.path, self._content(self._started))
            except LeaseLost:
                return
            except OSError as exc:
                self.lost = f"Could not refresh lease: {exc}"
                return

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        holder = self.read()
        if holder and holder.get("writer_id") == self.writer_id:
            self.path.unlink(missing_ok=True)


@contextmanager
def held_lease(lease):
    lease.acquire()
    try:
        yield lease
    finally:
        lease.release()


class JudgmentStore:
    """The append-only judgments.jsonl for one variant directory."""

    def __init__(self, directory, run_id):
        self.path = Path(directory) / JUDGMENTS
        self.run_id = run_id
        self.repaired_bytes = 0

    def load(self):
        """Return saved rows, repairing only an incomplete final line (an interrupted append)."""
        if not self.path.exists():
            return []
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            tail = data[cut:]
            try:
                json.loads(tail)
            except ValueError:
                with open(self.path, "r+b") as handle:
                    handle.truncate(cut)
                    handle.flush()
                    os.fsync(handle.fileno())
                self.repaired_bytes = len(tail)
                data = data[:cut]
            else:
                with open(self.path, "ab") as handle:
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                data += b"\n"
        rows = []
        for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise OutputConflict(f"{self.path}:{number} is corrupt; inspect it before resuming") from exc
            if row.get("run_id") != self.run_id:
                raise OutputConflict(f"{self.path}:{number} belongs to run {row.get('run_id')}, "
                                     f"not manifest run {self.run_id}")
            rows.append(row)
        return rows

    def append(self, rows):
        if not rows:
            return
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        with open(self.path, "ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def summarize(rows, max_attempts):
    """Per-key state: the first final row wins; otherwise count failed attempts.

    Returns (final, attempts, duplicates): final maps key -> row, attempts maps
    key -> failed attempt count, duplicates counts keys finalized more than once
    (evidence of two writers; the earliest row is kept).
    """
    final, attempts, duplicates = {}, {}, 0
    for row in rows:
        key = row["patient_id"], row["generator_model"]
        if row["status"] in FINAL_STATUSES:
            if key in final:
                duplicates += 1
            else:
                final[key] = row
        elif row["status"] in RETRY_STATUSES:
            attempts[key] = attempts.get(key, 0) + 1
    return final, attempts, duplicates


def latest_rows(rows):
    """One row per key for exports: the first final row, else the latest failed attempt."""
    chosen = {}
    for row in rows:
        key = row["patient_id"], row["generator_model"]
        current = chosen.get(key)
        if current is None or (current["status"] not in FINAL_STATUSES):
            chosen[key] = row
    return chosen

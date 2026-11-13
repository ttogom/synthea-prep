"""One filesystem writer per experiment directory, shared by runner and evaluator."""

from contextlib import contextmanager
import fcntl
from pathlib import Path


class RunLockError(ValueError):
    """Another runner or evaluator owns the experiment directory."""


@contextmanager
def exclusive_run_lock(run_dir):
    """Hold a nonblocking process lock; retain its inode after release.

    Never unlink this file: another process may already have opened the inode
    while waiting to acquire it. The OS releases the lock after a crash.
    """
    path = Path(run_dir).resolve() / ".run.lock"
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunLockError(f"Run directory is already in use by a runner or evaluator: {path.parent}") from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

"""File locks and disk reservations shared between survey processes.

Locks: ``fcntl.flock`` on POSIX, ``msvcrt.locking`` (one byte at offset 0)
on Windows. Both are released by the operating system when the process
ends, so a lock held by a killed process never blocks the next run.

Disk reservations: survey processes that write to the same disk (for
example one per partition, each with its own scratch dir) register the
bytes they are about to download in ``--shared-state-dir``, and each
process subtracts the other processes' reservations on the same file
system from the free space before it checks the disk floor. A process
holds a lock on its own reservation for its whole life, so a reservation
left by a killed process is recognised (its lock is free) and removed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:                    # pragma: no cover - Windows
    fcntl = None
try:
    import msvcrt
except ImportError:                    # POSIX
    msvcrt = None

logger = logging.getLogger(__name__)

LOCKING_AVAILABLE = fcntl is not None or msvcrt is not None
RESERVE_PREFIX = "disk_reserve."


def try_lock(fp) -> bool:
    """Exclusive, non-blocking lock on the open file ``fp``. True when taken,
    False when another process (or another open handle) holds it. Raises
    OSError when the platform has no file locking."""
    if fcntl is not None:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False
    if msvcrt is not None:
        try:
            fp.seek(0)
            msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    raise OSError("no file locking on this platform")


def unlock(fp):
    try:
        if fcntl is not None:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        elif msvcrt is not None:
            fp.seek(0)
            msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


@contextmanager
def locked(path: Path, timeout: float = 30.0, poll: float = 0.02):
    """Hold an exclusive lock on ``path`` (created when missing) for the
    ``with`` block, waiting up to ``timeout`` seconds for it. Without any
    locking mechanism the block runs unlocked."""
    fp = open(path, "a+")
    try:
        if LOCKING_AVAILABLE:
            deadline = time.monotonic() + timeout
            while not try_lock(fp):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"lock {path} not free after {timeout:.0f} s")
                time.sleep(poll)
        try:
            yield
        finally:
            if LOCKING_AVAILABLE:
                unlock(fp)
    finally:
        fp.close()


class DiskReservations:
    """Bytes this process is about to write, published in ``state_dir`` for
    the other survey processes, and the sum of theirs on the same file
    system. With ``state_dir`` None it does nothing (others() is 0)."""

    def __init__(self, state_dir: Path | None, disk_path: Path):
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self.disk_path = Path(disk_path)
        self.bytes = 0
        self._lock_fp = None
        # one writer at a time: the producer publishes from every fetch
        # thread, and interleaved writes could publish a torn or stale file
        self._write_lock = threading.RLock()
        if self.state_dir is None:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.dev = os.stat(self.disk_path).st_dev
        self.pid = os.getpid()
        self._file = self.state_dir / f"{RESERVE_PREFIX}{self.pid}.json"
        if LOCKING_AVAILABLE:
            # Held for the life of the process: others read a free lock as
            # "this reservation's owner is gone".
            self._lock_fp = open(self.state_dir / f"{RESERVE_PREFIX}{self.pid}.lock", "a+")
            if not try_lock(self._lock_fp):
                logger.warning("could not lock the disk reservation of pid %d", self.pid)
        self._write(0)

    def _write(self, nbytes: int):
        if self.state_dir is None:
            return
        with self._write_lock:
            self.bytes = int(nbytes)
            tmp = self._file.with_name(f"{self._file.name}.{threading.get_ident()}.tmp")
            try:
                tmp.write_text(json.dumps({"pid": self.pid, "dev": self.dev, "bytes": self.bytes}),
                               encoding="utf-8")
                tmp.replace(self._file)
            except OSError as e:
                logger.warning("could not write the disk reservation %s: %s", self._file, e)

    def reserve(self, nbytes):
        """Publish ``nbytes`` as this process's in-flight reservation.
        ``nbytes`` may be a callable: it is then evaluated under the write
        lock, so concurrent callers always publish the current total and an
        older total never overwrites a newer one."""
        if self.state_dir is None:
            return
        with self._write_lock:
            self._write(nbytes() if callable(nbytes) else nbytes)

    def release(self):
        self._write(0)

    def others(self) -> int:
        """Bytes reserved by the other live processes on this file system.
        Reservations whose owner is gone (lock free) are removed."""
        if self.state_dir is None:
            return 0
        total = 0
        for p in self.state_dir.glob(f"{RESERVE_PREFIX}*.json"):
            if p == self._file:
                continue
            lock_path = p.with_suffix(".lock")
            if LOCKING_AVAILABLE and lock_path.exists():
                try:
                    with open(lock_path, "a+") as fp:
                        if try_lock(fp):            # owner gone
                            unlock(fp)
                            p.unlink(missing_ok=True)
                            lock_path.unlink(missing_ok=True)
                            continue
                except OSError:
                    pass
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if doc.get("dev") == self.dev:
                total += max(0, int(doc.get("bytes") or 0))
        return total

    def free(self) -> int:
        """Free bytes on the disk minus the other processes' reservations."""
        return shutil.disk_usage(self.disk_path).free - self.others()

    def close(self):
        if self.state_dir is None:
            return
        try:
            self._file.unlink(missing_ok=True)
        except OSError:
            pass
        if self._lock_fp is not None:
            unlock(self._lock_fp)
            self._lock_fp.close()
            self._lock_fp = None
            try:
                (self.state_dir / f"{RESERVE_PREFIX}{self.pid}.lock").unlink(missing_ok=True)
            except OSError:
                pass

"""Per-book claim lock so several re-OCR workers can share one worklist.

Why: ``run_ocr.py`` does one book at a time; to throw more compute at a running
campaign you start a *second* worker (a separate invocation on its own port
range). But ``--skip-done`` only skips books whose final PDF already exists, so
two workers would both pick the first not-yet-done book and collide on its output
dir + page cache. A claim lock fixes that: each worker atomically claims a book
before OCR'ing it and releases it when done, so an added worker only picks up
books nobody is working on.

Stealing a dead worker's claim (so its half-OCR'd book gets finished rather than
stranded) is the other half: combined with the per-page OCR checkpoint, a killed
worker's partial book resumes from cache under whoever steals it.

  - Same host: the owner pid is authoritative. A claim is stealable iff that pid
    is no longer alive (``os.kill(pid, 0)``) — no waiting on a timeout, so a
    restart right after an OOM kill reclaims immediately.
  - Other host: pid liveness isn't checkable remotely, so fall back to a
    heartbeat TTL — the live owner ``touch``es the file periodically, and the
    claim is stealable only once its mtime is older than ``ttl``.

Single-worker runs use it too (claim → release per book); it's a cheap no-op that
also makes an interrupted single worker resumable by the next run.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    """True if a process with this pid exists (signal 0 = existence probe)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True            # exists, owned by someone else
    return True


class ClaimLock:
    """An advisory, stealable lock backed by a single JSON file.

    Use as a context manager or call ``acquire()`` / ``release()`` directly.
    ``acquire()`` returns True if this worker now owns the book, False if a live
    worker already holds it (caller should skip the book).
    """

    def __init__(self, path, *, ttl: float = 900.0, heartbeat: float = None,
                 now=time.time, host: str = None, pid: int = None):
        self.path = Path(path)
        self.ttl = float(ttl)
        # Refresh well within ttl so a live cross-host owner never looks stale.
        self.heartbeat = float(heartbeat) if heartbeat else self.ttl / 3.0
        self._now = now
        self.host = host or socket.gethostname()
        self.pid = pid if pid is not None else os.getpid()
        self._stop = threading.Event()
        self._beat = None

    # -- introspection ----------------------------------------------------- #
    def _read(self):
        return self.read(self.path)

    @staticmethod
    def read(path):
        """Parse a claim file -> dict, or None if missing/corrupt."""
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return None

    def _stealable(self, info, mtime: float) -> bool:
        """Can the current claim (info/mtime) be taken over?"""
        if not info:
            return True                                   # unreadable/corrupt
        if info.get("host") == self.host:
            return not _pid_alive(int(info.get("pid", -1)))   # pid is truth
        return self._now() - mtime > self.ttl                 # cross-host: TTL

    def _payload(self) -> str:
        return json.dumps({"host": self.host, "pid": self.pid,
                           "claimed_at": self._now()})

    def _write_new(self) -> bool:
        """Atomically create the claim (O_EXCL). True if we created it."""
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        try:
            os.write(fd, self._payload().encode())
        finally:
            os.close(fd)
        return True

    def _steal(self) -> None:
        """Overwrite an existing (dead/stale) claim with our own, atomically."""
        tmp = self.path.with_name(f".{self.path.name}.tmp{self.pid}")
        tmp.write_text(self._payload())
        os.replace(tmp, self.path)

    # -- public API -------------------------------------------------------- #
    def acquire(self, *, start_heartbeat: bool = True) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._write_new():
            owned = True
        else:
            try:
                mtime = self.path.stat().st_mtime
            except FileNotFoundError:
                # Raced with a release between O_EXCL and stat — retry once.
                owned = self._write_new()
                if start_heartbeat and owned:
                    self._start_heartbeat()
                return owned
            if self._stealable(self._read(), mtime):
                self._steal()
                owned = True
            else:
                owned = False
        if owned and start_heartbeat:
            self._start_heartbeat()
        return owned

    def _start_heartbeat(self) -> None:
        if self._beat is not None:
            return

        def loop():
            while not self._stop.wait(self.heartbeat):
                try:
                    os.utime(self.path, None)      # bump mtime: "still alive"
                except OSError:
                    return
        self._beat = threading.Thread(target=loop, daemon=True)
        self._beat.start()

    def update(self, **fields) -> bool:
        """Merge extra fields into the claim (atomically), but only while we
        still own it. Used to record the per-book OCR subprocess pid so a watcher
        can list / kill the book it belongs to. Returns True if written."""
        info = self._read()
        if not (info and info.get("host") == self.host
                and info.get("pid") == self.pid):
            return False
        info.update(fields)
        tmp = self.path.with_name(f".{self.path.name}.tmp{self.pid}")
        tmp.write_text(json.dumps(info))
        os.replace(tmp, self.path)
        return True

    def release(self) -> None:
        self._stop.set()
        if self._beat is not None:
            self._beat.join(timeout=1.0)
            self._beat = None
        # Only unlink a claim we still own (don't clobber a worker that stole it
        # after we stalled past the TTL).
        info = self._read()
        if info and info.get("host") == self.host and info.get("pid") == self.pid:
            try:
                self.path.unlink()
            except OSError:
                pass

    def __enter__(self):
        self.owned = self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False

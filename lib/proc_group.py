"""Track spawned child processes and tear the whole subtree down on
signal / interpreter exit — so killing a parent never leaves orphan
``surya_ocr`` / ``llama-server`` processes running against the GPU and ports
(which then corrupt later runs).

Shared by ``ocr_one.py`` (which spawns surya_ocr + llama-server via
``parallel_ocr``) and the ``run_ocr.py`` batch driver (which spawns
``ocr_one`` itself). Each level installs the handler and registers what it
launches, so a clean teardown cascades no matter where the kill lands.

Two launch flavours:

* ``popen(cmd, group=True)`` — start the child in its OWN session (it becomes a
  process-group leader). Use for a *subtree* you want to kill as a unit: the
  cleanup ``killpg``s the whole group (the child and everything it spawns).
  ``run_ocr`` launches ``ocr_one`` this way.
* ``popen(cmd)`` (``group=False``, default) — leave the child in the current
  process group. Use for leaf workers (surya_ocr, llama-server) so that a
  ``killpg`` of an ancestor group also reaps them, and the local handler can
  terminate them directly too. ``parallel_ocr`` launches its workers this way.

Caveat (macOS): there is no ``PR_SET_PDEATHSIG``, so a *SIGKILL of the direct
parent* can't be caught and leaf children would orphan — for a hard kill, kill
the process group (``kill -- -<pgid>``). Catchable signals (SIGINT/SIGTERM/
SIGHUP) and normal/exception exit are all handled here.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import threading
import time

_children: dict = {}          # Popen -> group_leader? (bool)
_lock = threading.Lock()
_installed = False


def popen(cmd, *, group: bool = False, **kwargs) -> subprocess.Popen:
    """``subprocess.Popen`` that is tracked for teardown. ``group=True`` starts
    it in a new session (process-group leader). Returns the Popen."""
    if group:
        kwargs.setdefault("start_new_session", True)
    proc = subprocess.Popen(cmd, **kwargs)
    with _lock:
        _children[proc] = group
    return proc


def reap(proc) -> None:
    """Stop tracking a child that has finished (call after ``proc.wait()``)."""
    with _lock:
        _children.pop(proc, None)


def _signal(proc, group, sig) -> None:
    try:
        if group:
            os.killpg(os.getpgid(proc.pid), sig)
        else:
            proc.send_signal(sig)
    except (ProcessLookupError, PermissionError):
        pass


def kill_all(*_) -> None:
    """SIGTERM every tracked child (group-kill the leaders), wait briefly, then
    SIGKILL whatever is still alive."""
    with _lock:
        items = [(p, g) for p, g in _children.items() if p.poll() is None]
    for p, g in items:
        _signal(p, g, signal.SIGTERM)
    deadline = time.time() + 5
    for p, _g in items:
        try:
            p.wait(timeout=max(0, deadline - time.time()))
        except Exception:
            pass
    for p, g in items:
        if p.poll() is None:
            _signal(p, g, signal.SIGKILL)


def install() -> None:
    """Register atexit + signal handlers (idempotent). Call once from the main
    thread (signal handlers can only be set there)."""
    global _installed
    if _installed:
        return
    _installed = True
    atexit.register(kill_all)

    def _handler(signum, _frame):
        kill_all()
        sys.exit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass          # not main thread / unsupported — atexit still covers it

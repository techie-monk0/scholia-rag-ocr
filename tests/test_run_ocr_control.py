"""run_ocr's status/control surface: listing the books being processed and
killing a specific one (built on the claim files)."""
import json
import os
import subprocess
import sys
import time

import pytest

import run_ocr
from claimlock import _pid_alive


def _claim(out_root, name, **fields):
    d = out_root / name
    d.mkdir(parents=True)
    base = {"host": __import__("socket").gethostname(), "pid": os.getpid(),
            "claimed_at": time.time()}
    base.update(fields)
    (d / ".claim").write_text(json.dumps(base))
    return d / ".claim"


def test_active_claims_lists_live_skips_dead(tmp_path):
    _claim(tmp_path, "e1")                       # live worker (this pid)
    _claim(tmp_path, "e2", pid=2)                # dead worker pid -> reclaimable
    names = {n for n, _ in run_ocr.active_claims(tmp_path)}
    assert names == {"e1"}


def test_list_active_empty(tmp_path, capsys):
    assert run_ocr.list_active(tmp_path) == 0
    assert "No books are being processed" in capsys.readouterr().out


def test_list_active_shows_book(tmp_path, capsys):
    _claim(tmp_path, "e49", ocr_pid=os.getpid())
    run_ocr.list_active(tmp_path)
    out = capsys.readouterr().out
    assert "e49" in out and "1 book(s) in progress" in out


def test_port_ranges_contiguous_when_free():
    """In an empty range, blocks are non-overlapping and spaced by width."""
    bases, skipped = run_ocr.find_port_ranges(8401, 3, 2)
    assert bases == [8401, 8403, 8405]
    assert skipped is False


def test_port_ranges_skips_busy_port():
    """A bound port is stepped over, and `skipped` flags it (another worker)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 8411))
    try:
        bases, skipped = run_ocr.find_port_ranges(8411, 1, 2)
        assert 8411 not in range(bases[0], bases[0] + 2)   # didn't reuse the busy one
        assert skipped is True
    finally:
        s.close()


def test_kill_no_claim(tmp_path, capsys):
    assert run_ocr.kill_books(tmp_path, ["e99"]) == 1
    assert "no active claim" in capsys.readouterr().out


def test_kill_claim_without_live_ocr(tmp_path, capsys):
    _claim(tmp_path, "e49", ocr_pid=2)           # ocr pid dead
    assert run_ocr.kill_books(tmp_path, ["e49"]) == 1
    assert "no live OCR process" in capsys.readouterr().out


def test_kill_stops_real_process_group(tmp_path, capsys):
    """End-to-end: a real child in its own process group is SIGTERM'd by name."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                            start_new_session=True)          # own process group
    try:
        _claim(tmp_path, "e49", ocr_pid=proc.pid)
        assert _pid_alive(proc.pid)
        rc = run_ocr.kill_books(tmp_path, ["e49"])
        assert rc == 0
        assert proc.wait(timeout=5) != 0                     # died from SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
    assert "SIGTERM" in capsys.readouterr().out

"""ClaimLock: the per-book cooperative lock that lets a second re-OCR worker
join a running campaign without colliding (lib/claimlock.py)."""
import json
import os

import pytest
from claimlock import ClaimLock, _pid_alive


def test_acquire_creates_claim(tmp_path):
    c = ClaimLock(tmp_path / "e1" / ".claim")
    assert c.acquire(start_heartbeat=False) is True
    assert (tmp_path / "e1" / ".claim").exists()
    info = json.loads((tmp_path / "e1" / ".claim").read_text())
    assert info["pid"] == os.getpid()


def test_live_same_host_owner_blocks(tmp_path):
    """A claim held by a live pid on this host is NOT stealable."""
    path = tmp_path / ".claim"
    a = ClaimLock(path)
    assert a.acquire(start_heartbeat=False) is True
    # A different worker (distinct pid) on the same host sees a live owner.
    b = ClaimLock(path, pid=os.getpid() + 0)   # same live pid
    other = ClaimLock(path, pid=999999, host=a.host)
    assert other.acquire(start_heartbeat=False) is False


def test_dead_same_host_owner_is_stolen(tmp_path):
    """A claim whose owner pid is dead (same host) is reclaimed immediately,
    regardless of ttl — the OOM-restart case."""
    path = tmp_path / ".claim"
    dead_pid = 2  # pid 2 is kernel-reserved; _pid_alive(2) is False for us
    assert not _pid_alive(dead_pid)
    ClaimLock(path, pid=dead_pid).acquire(start_heartbeat=False)
    fresh = ClaimLock(path, ttl=10_000)        # big ttl: only pid-death lets it in
    assert fresh.acquire(start_heartbeat=False) is True
    assert json.loads(path.read_text())["pid"] == os.getpid()


def test_cross_host_respects_ttl(tmp_path):
    """A claim from another host is stealable only once mtime passes ttl."""
    path = tmp_path / ".claim"
    clock = {"t": 1000.0}
    owner = ClaimLock(path, ttl=100, host="other-host", pid=4242,
                      now=lambda: clock["t"])
    assert owner.acquire(start_heartbeat=False) is True
    # Fresh claim from THIS host, just after: still within ttl -> blocked.
    me = ClaimLock(path, ttl=100, now=lambda: clock["t"])
    clock["t"] = 1050.0
    os.utime(path, (clock["t"], clock["t"]))   # owner's last heartbeat
    assert me.acquire(start_heartbeat=False) is False
    # Past ttl with no heartbeat -> stale -> stealable.
    clock["t"] = 1200.0
    assert me.acquire(start_heartbeat=False) is True


def test_release_unlinks_only_own_claim(tmp_path):
    path = tmp_path / ".claim"
    mine = ClaimLock(path)
    mine.acquire(start_heartbeat=False)
    # Someone stole it (different pid recorded). Our release must NOT delete it.
    path.write_text(json.dumps({"host": mine.host, "pid": 999999,
                                "claimed_at": 0}))
    mine.release()
    assert path.exists()


def test_release_removes_own_claim(tmp_path):
    path = tmp_path / ".claim"
    c = ClaimLock(path)
    c.acquire(start_heartbeat=False)
    c.release()
    assert not path.exists()


def test_corrupt_claim_is_stealable(tmp_path):
    path = tmp_path / ".claim"
    path.write_text("{ not json")
    assert ClaimLock(path).acquire(start_heartbeat=False) is True


def test_context_manager(tmp_path):
    path = tmp_path / ".claim"
    with ClaimLock(path) as c:
        assert c.owned is True
        assert path.exists()
    assert not path.exists()


def test_two_workers_one_book_only_one_wins(tmp_path):
    """The core guarantee: concurrent acquire on the same book -> exactly one
    owner (the other must skip)."""
    path = tmp_path / ".claim"
    a = ClaimLock(path, pid=os.getpid())
    b = ClaimLock(path, pid=999998, host=a.host)
    got_a = a.acquire(start_heartbeat=False)
    got_b = b.acquire(start_heartbeat=False)
    assert got_a is True and got_b is False

"""build_rag_manifest — pub_id/content_hash stamping from the catalogue's v_holding_files.

The RAG consumer keys citations on the catalogue's stable `pub_id`; the manifest is where it's
transported. See the catalogue's external_tool_dependency_contract.md.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))          # repo root, for build_rag_manifest
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))  # for the shared identity layer

import build_rag_manifest as B
import identity


def test_stamp_identity_matches_by_edition_then_source():
    rows = [
        {"edition": "e1", "source": "/a.pdf"},   # source matches a holding hash
        {"edition": "e2", "source": "/b.pdf"},   # no source match, but a sole holding hash
        {"edition": "e9", "source": "/z.pdf"},   # not catalogued
    ]
    identity = {
        1: {"pub_id": "u1", "hashes": {"/a.pdf": "h1"}},
        2: {"pub_id": "u2", "hashes": {"/x.pdf": "h2"}},
    }
    missing = B._stamp_identity(rows, identity)
    assert rows[0]["pub_id"] == "u1" and rows[0]["content_hash"] == "h1"
    assert rows[1]["pub_id"] == "u2" and rows[1]["content_hash"] == "h2"
    assert rows[2]["pub_id"] is None and rows[2]["content_hash"] is None
    assert missing == ["e9"]


def test_stamp_leaves_hash_none_when_ambiguous():
    rows = [{"edition": "e1", "source": "/nomatch.pdf"}]
    identity = {1: {"pub_id": "u1", "hashes": {"/a.pdf": "h1", "/b.pdf": "h2"}}}
    B._stamp_identity(rows, identity)
    assert rows[0]["pub_id"] == "u1" and rows[0]["content_hash"] is None   # 2 hashes, no source match


def test_catalogue_identity_missing_db_is_empty(tmp_path):
    assert identity.catalogue_identity(tmp_path / "nope.db") == {}


def test_catalogue_identity_reads_v_holding_files(tmp_path):
    p = tmp_path / "cat.db"
    c = sqlite3.connect(p)
    c.executescript(
        "CREATE TABLE edition (id INTEGER, pub_id TEXT);"
        "CREATE TABLE holding (edition_id INTEGER, file_path TEXT, content_hash TEXT);"
        "CREATE VIEW v_holding_files AS "
        " SELECT h.edition_id, e.pub_id, h.file_path, h.content_hash "
        " FROM holding h JOIN edition e ON e.id = h.edition_id;")
    c.execute("INSERT INTO edition VALUES (1, 'u1')")
    c.execute("INSERT INTO holding VALUES (1, '/a.pdf', 'h1')")
    c.commit(); c.close()
    assert identity.catalogue_identity(p) == {1: {"pub_id": "u1", "hashes": {"/a.pdf": "h1"}}}


def _cat_db(tmp_path, *, with_pub_id=True, version=None):
    """A throwaway catalogue-shaped DB. `version`, if given, is stamped into schema_meta as the
    external-read-contract version."""
    p = tmp_path / "cat.db"
    c = sqlite3.connect(p)
    cols = "h.edition_id, e.pub_id, h.file_path, h.content_hash" if with_pub_id else \
           "h.edition_id, h.file_path, h.content_hash"
    c.executescript(
        "CREATE TABLE edition (id INTEGER, pub_id TEXT);"
        "CREATE TABLE holding (edition_id INTEGER, file_path TEXT, content_hash TEXT);"
        f"CREATE VIEW v_holding_files AS SELECT {cols} "
        " FROM holding h JOIN edition e ON e.id = h.edition_id;")
    if version is not None:
        c.executescript("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT);")
        c.execute("INSERT INTO schema_meta VALUES ('external_read_contract_version', ?)",
                  (str(version),))
    c.commit(); c.close()
    return p


def test_handshake_passes_on_current_version(tmp_path):
    p = _cat_db(tmp_path, version=identity.REQUIRED_CONTRACT_VERSION)
    assert identity.catalogue_identity(p) == {}   # no rows, but no ContractError


def test_handshake_rejects_missing_pub_id_column(tmp_path):
    import pytest
    p = _cat_db(tmp_path, with_pub_id=False)
    with pytest.raises(identity.ContractError):
        identity.catalogue_identity(p)


def test_handshake_rejects_older_stamped_version(tmp_path):
    import pytest
    p = _cat_db(tmp_path, version=identity.REQUIRED_CONTRACT_VERSION - 1)
    with pytest.raises(identity.ContractError):
        identity.catalogue_identity(p)

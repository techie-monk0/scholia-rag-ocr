"""Single source of truth for a book's stable *identity*.

The catalogue is the system-of-record for edition identity. The contract token the
downstream RAG/LLM consumer keys on is **``pub_id``** — an opaque, stable UUID — with
**``content_hash``** as the version (for re-embed staleness). Both are surfaced two ways:

* the live catalogue's ``v_holding_files`` external read-contract view, and
* an ``edition.json`` sidecar snapshot written into each book's output dir at OCR time.

This module is the ONE place that reads either source, so everything that needs to name
or match a book — the corpus export (``lib/corpus.export_book``), the RAG handoff manifest
(``build_rag_manifest``), anything future — keys on the SAME identifier instead of each
re-deriving it. In particular ``corpus_key()`` is the abstraction the corpus export goes
through: change the keying policy here, not in the callers.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import namedtuple
from pathlib import Path

# pub_id: the catalogue<->LLM contract UUID. content_hash: the version. edition: the
# catalogue integer (internal). source: original holding path. doc_id: the doc.json id.
Identity = namedtuple("Identity", "pub_id content_hash edition source doc_id")

_ENAME = re.compile(r"e(\d+)")


def sidecar_identity(book_dir) -> Identity:
    """Resolve a finished book dir's identity from its ``edition.json`` snapshot —
    offline, no catalogue DB needed (this is what OCR stamped at the time). The
    ``edition`` integer falls back to a legacy ``eNNN`` dir name; every field may be
    ``None`` if the sidecar is absent."""
    book = Path(book_dir)
    meta = {}
    p = book / "edition.json"
    if p.exists():
        try:
            meta = json.loads(p.read_text())
        except (OSError, ValueError):
            meta = {}
    edition = meta.get("edition")
    if edition is None:
        m = _ENAME.fullmatch(book.name)                 # legacy eNNN dir
        edition = int(m.group(1)) if m else None
    return Identity(
        pub_id=meta.get("pub_id"),
        content_hash=meta.get("content_hash"),
        edition=int(edition) if edition is not None else None,
        source=meta.get("source"),
        doc_id=meta.get("doc_id"),
    )


def corpus_key(book_dir) -> str:
    """The stable name under which a book is exported into the corpus — its
    ``pub_id`` UUID, the catalogue<->LLM contract identifier the RAG ingester keys on.

    Fallbacks keep an export from ever colliding or crashing on a not-yet-stamped
    book: ``e<edition>`` when pub_id is unresolved, then the source dir name."""
    ident = sidecar_identity(book_dir)
    if ident.pub_id:
        return ident.pub_id
    if ident.edition is not None:
        return f"e{ident.edition}"
    return Path(book_dir).name


# --- external read-contract handshake ----------------------------------------
# The version of the catalogue's external read-contract this pipeline was built against, and the
# `v_holding_files` columns it depends on. The catalogue publishes the contract as a versioned,
# language-neutral descriptor + a `schema_meta.external_read_contract_version` stamped into every
# DB — we verify against that instead of importing catalogue code. See library_cataloging
# docs/access/external_tool_dependency_contract.md ("Versioned read-contract descriptor").
REQUIRED_CONTRACT_VERSION = 1
_REQUIRED_HOLDING_COLUMNS = ("edition_id", "pub_id", "file_path", "content_hash")


class ContractError(RuntimeError):
    """The catalogue's external read-contract is incompatible with what this pipeline needs."""


def verify_catalogue_contract(conn):
    """Handshake against the catalogue's published read-contract, over a connection we already
    hold. Raises ``ContractError`` on a hard incompatibility: a required ``v_holding_files``
    column is missing, or the DB advertises a contract version older than
    ``REQUIRED_CONTRACT_VERSION``. An UNSTAMPED DB (version ``None`` — a catalogue not yet
    re-inited since the descriptor landed) passes on the strength of the column check, which is
    the property correctness actually depends on. Tolerant reader: extra columns are ignored.
    Returns the advertised version (or ``None``)."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(v_holding_files)")}
    missing = [c for c in _REQUIRED_HOLDING_COLUMNS if c not in have]
    if missing:
        raise ContractError(
            f"catalogue v_holding_files is missing required columns {missing} "
            f"(external-read-contract v{REQUIRED_CONTRACT_VERSION}); the identity surface "
            "changed incompatibly — see the catalogue's external read-contract descriptor.")
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='external_read_contract_version'").fetchone()
        ver = int(row[0]) if row and row[0] is not None else None
    except sqlite3.Error:
        ver = None            # schema_meta absent (older/foreign DB) — columns already vouch for shape
    if ver is not None and ver < REQUIRED_CONTRACT_VERSION:
        raise ContractError(
            f"catalogue external-read-contract v{ver} < required v{REQUIRED_CONTRACT_VERSION}; "
            "a breaking change bumped it — upgrade this pipeline.")
    return ver


def catalogue_identity(db_path) -> dict:
    """Map catalogue ``edition_id -> {pub_id, hashes: {file_path: content_hash}}`` from
    the stable ``v_holding_files`` external read-contract. Opened IMMUTABLE read-only,
    so reading identity never locks or mutates the live catalogue. ``{}`` if DB absent.
    Verifies the read-contract handshake first (raises ``ContractError`` on drift)."""
    if not db_path or not Path(db_path).exists():
        return {}
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?immutable=1", uri=True)
    try:
        verify_catalogue_contract(conn)
        out: dict = {}
        for eid, pub_id, file_path, content_hash in conn.execute(
                "SELECT edition_id, pub_id, file_path, content_hash FROM v_holding_files"):
            rec = out.setdefault(eid, {"pub_id": pub_id, "hashes": {}})
            rec["pub_id"] = rec["pub_id"] or pub_id
            if file_path and content_hash:
                rec["hashes"][file_path] = content_hash
        return out
    finally:
        conn.close()

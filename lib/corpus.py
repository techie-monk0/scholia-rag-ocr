"""Corpus export + post-OCR health verification.

Two jobs, one module:

1. **health(book_dir)** — verify a finished book actually produced usable output.
   An interrupted OCR/merge leaves an empty ``{}`` results.json stub, which then
   yields a `doc.json` with ``blocks: []`` that *looks* finished (a searchable PDF
   even exists). This is the e3 failure mode: the book was treated as "done" and
   skipped forever. ``health`` gates on CONTENT (results cache non-empty, doc has
   blocks), not file-existence, so such a book fails loudly and gets re-OCR'd.

2. **export_book / export_all** — mirror the small, durable artifacts (doc.json,
   body/full text, pages.jsonl) of a *verified* book into a self-contained corpus
   folder (``$OCR_CORPUS_DIR`` or ``<repo>/_ocr_corpus``), indexed by manifest.jsonl
   with paths relative to that folder. The 48 GB of regenerable page images stay in
   ``out/``; only the LLM-relevant text/structure is exported.

The pipeline calls health()+export_book() automatically after each book
(``ocr_one.run_one``); ``is_done()`` is the health-gated "done" test the driver's
--skip-done uses so a broken book is never skipped.
"""
from __future__ import annotations

import json
import os
import shutil
from collections import namedtuple
from pathlib import Path

import identity
import surya_backend as sb

REPO_ROOT = Path(__file__).resolve().parent.parent


def corpus_dir(override=None) -> Path | None:
    """Resolve the corpus destination. ``override`` (CLI) wins; else $OCR_CORPUS_DIR;
    else ``<repo>/_ocr_corpus``. The literal ``none`` (any case) disables export."""
    val = override or os.environ.get("OCR_CORPUS_DIR") or str(REPO_ROOT / "_ocr_corpus")
    if str(val).strip().lower() == "none":
        return None
    return Path(val)


Health = namedtuple("Health", "ok reasons n_blocks n_pages")


def _doc_path(book: Path):
    docs = [p for p in book.glob("*.doc.json")]
    return docs[0] if docs else None


def health(book_dir) -> Health:
    """Is this finished book's output actually usable? Checks CONTENT, not mere
    file-existence — the guard against the e3 empty-stub failure mode.

    Fails when: the OCR results cache is an empty ``{}`` stub (interrupted run), the
    doc.json is missing/unparseable/schema-less, or it carries zero blocks. Native
    (digital-first) books have no results.json, so that check is skipped for them;
    the doc-blocks check still applies."""
    book = Path(book_dir)
    reasons: list[str] = []

    # OCR results cache — scanned books only (native has embedded text, no cache).
    # An empty `{}` stub is POSITIVE evidence of breakage (the e3 signature); a
    # cache that is simply absent is NOT — it may have been cleaned to reclaim disk
    # after a successful run (the searchable PDF is the surviving output). So only a
    # present-but-empty cache counts against health.
    res = book / "_pages" / ".surya_ocr" / "results.json"
    n_pages = 0
    res_usable = False
    if res.exists():
        if sb.usable_cache(res):
            res_usable = True
            try:
                n_pages = len(json.loads(res.read_text()))
            except (OSError, ValueError):
                pass
        else:
            reasons.append("results.json is an empty stub (interrupted OCR — re-OCR)")

    # doc.json — the RAG contract. A present doc must parse, be schema-tagged, and
    # hold blocks. A MISSING doc is only a problem when an OCR cache is present (OCR
    # ran but assembly never finished); with no cache either, the book is a
    # successfully-finished run whose regenerable doc/cache were cleaned — not broken.
    docp = _doc_path(book)
    n_blocks = 0
    if docp is None:
        if res_usable:
            reasons.append("OCR cache present but no doc.json (assembly incomplete)")
    else:
        try:
            doc = json.loads(docp.read_text())
        except (OSError, ValueError) as e:
            reasons.append(f"doc.json unreadable ({e})")
            doc = None
        if doc is not None:
            n_blocks = len(doc.get("blocks") or [])
            if not doc.get("schema_version"):
                reasons.append("doc.json missing schema_version")
            if n_blocks == 0:
                reasons.append("doc.json has 0 blocks")

    return Health(ok=not reasons, reasons=reasons, n_blocks=n_blocks, n_pages=n_pages)


def write_marker(book_dir, h: Health) -> None:
    """Record the health verdict as a sidecar (``.ocr_ok`` / ``.ocr_incomplete``),
    clearing the opposite one. Best-effort — never raises."""
    book = Path(book_dir)
    try:
        for m in (".ocr_ok", ".ocr_incomplete"):
            p = book / m
            if p.exists():
                p.unlink()
        name = ".ocr_ok" if h.ok else ".ocr_incomplete"
        (book / name).write_text(json.dumps(
            {"ok": h.ok, "reasons": h.reasons,
             "n_blocks": h.n_blocks, "n_pages": h.n_pages}))
    except OSError:
        pass


def is_done(book_dir) -> bool:
    """A book is *done* only if it has a final searchable PDF AND passes health.
    This is the health-gated replacement for the old "PDF exists == done" test that
    let the broken e3 book be skipped indefinitely."""
    book = Path(book_dir)
    pdfs = [p for p in book.glob("*.pdf") if not p.name.endswith("_frontmatter.pdf")]
    return bool(pdfs) and health(book).ok


# --- export -------------------------------------------------------------------

_ARTIFACTS = ("body/all_pages.txt", "full/all_pages.txt", "body/pages.jsonl")


def manifest_row(book: Path, doc: dict, dst_files: dict) -> dict:
    """One corpus manifest row with paths RELATIVE to the corpus root, mirroring the
    rag_handoff schema. ``dst_files`` maps logical keys to the copied relative paths
    (absent ones are null). Keyed on ``pub_id`` — the catalogue<->LLM contract UUID
    the RAG ingester acknowledges (see [[lib/identity.py]]); ``edition`` is retained
    as an internal cross-reference."""
    ident = identity.sidecar_identity(book)
    scanned = (book / "_pages" / ".surya_ocr" / "results.json").exists()
    return {
        "pub_id": ident.pub_id,
        "content_hash": ident.content_hash,
        "edition": f"e{ident.edition}" if ident.edition is not None else book.name,
        "doc_id": doc.get("doc_id"),
        "title": doc.get("title"),
        "kind": "scanned" if scanned else "native",
        "source": doc.get("source"),
        "schema_version": doc.get("schema_version"),
        "doc_json": dst_files.get("doc_json"),
        "body_text": dst_files.get("body_text"),
        "full_text": dst_files.get("full_text"),
        "pages_jsonl": dst_files.get("pages_jsonl"),
        "n_blocks": len(doc.get("blocks") or []),
        "nfc": True,
    }


def export_book(book_dir, corpus_root) -> dict:
    """Copy a verified book's durable artifacts into ``corpus_root/<pub_id>/`` and
    return its manifest row (relative paths). The dir is keyed by the catalogue<->LLM
    contract UUID via ``identity.corpus_key`` — the corpus is a per-``pub_id`` handoff
    tray the RAG ingester consumes. Call only after ``health().ok``."""
    book = Path(book_dir)
    corpus_root = Path(corpus_root)
    key_name = identity.corpus_key(book)
    dst = corpus_root / key_name
    docp = _doc_path(book)
    if docp is None:
        raise ValueError(f"{key_name}: no doc.json to export")
    doc = json.loads(docp.read_text())

    (dst / "body").mkdir(parents=True, exist_ok=True)
    (dst / "full").mkdir(parents=True, exist_ok=True)

    dst_files: dict = {}
    shutil.copy2(docp, dst / docp.name)
    dst_files["doc_json"] = f"{key_name}/{docp.name}"
    key = {"body/all_pages.txt": "body_text", "full/all_pages.txt": "full_text",
           "body/pages.jsonl": "pages_jsonl"}
    for rel in _ARTIFACTS:
        src = book / rel
        if src.is_file():
            shutil.copy2(src, dst / rel)
            dst_files[key[rel]] = f"{key_name}/{rel}"

    return manifest_row(book, doc, dst_files)


def write_manifest(corpus_root, rows: list[dict]) -> Path:
    """Write the whole corpus manifest.jsonl (sorted by edition number)."""
    corpus_root = Path(corpus_root)
    corpus_root.mkdir(parents=True, exist_ok=True)
    mp = corpus_root / "manifest.jsonl"

    def _key(r):
        e = r.get("edition", "")
        return (0, int(e[1:])) if e[1:].isdigit() else (1, e)

    with open(mp, "w") as f:
        for r in sorted(rows, key=_key):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return mp


def _row_key(r: dict):
    """Stable identity of a corpus manifest row: ``pub_id`` (the contract UUID) when
    present, else the internal ``edition`` — the same precedence the export dir uses."""
    return r.get("pub_id") or r.get("edition")


def upsert_manifest_row(corpus_root, row: dict) -> None:
    """Insert/replace a single book's row in the corpus manifest (used by the live
    pipeline as each book finishes), matched on its stable identity (``pub_id``).
    Best-effort; keeps the file sorted."""
    corpus_root = Path(corpus_root)
    mp = corpus_root / "manifest.jsonl"
    rows = []
    if mp.is_file():
        for line in mp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if _row_key(r) != _row_key(row):
                rows.append(r)
    rows.append(row)
    write_manifest(corpus_root, rows)

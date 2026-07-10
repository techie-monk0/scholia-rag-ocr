#!/usr/bin/env python3
"""docjson — read-only access layer for the doc.json contract.

The producer (build_doc/structure) emits a forward-compatible doc.json:
``schema_version`` + a ``features`` list + additive optional keys, no sidecar
(see ARCHITECTURE.md §"Consuming doc.json"). This module encapsulates the
defensive-reading rules ONCE so clients (RAG ingestion/embedding) don't
re-implement them and drift:

  - read optional keys via ``.get`` with sensible defaults (e.g. region → "body"
    when absent, so pre-region docs still work),
  - treat ``type`` as an OPEN vocabulary — an unknown type is kept as content,
    never silently dropped,
  - expose ``features`` so callers gate behaviour on what the doc actually has.

It is READ-ONLY, stdlib-only (safe to vendor into a consumer repo), and never
hides data — the raw dicts stay reachable via ``Block.raw`` / ``Doc.raw``.

    doc = Doc.from_file("e10.doc.json")
    for b in doc.content_blocks(region="body"):   # drops furniture/nav
        embed(b.text)
    for v in doc.verses():
        index_verse(v.text, v.ref_label, v.line_bbox)
    for row in load_manifest("out/rag_handoff_manifest.jsonl"):
        doc = Doc.from_file(row["doc_json"]); ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

# Block types that are navigation/furniture — droppable from an embedding corpus.
# (An UNKNOWN type is NOT in here, so it is treated as content by default.)
FURNITURE_TYPES = frozenset({
    "page_header", "page_footer", "page_number", "running_head",
    "title_page", "copyright", "colophon", "dedication", "toc",
})
# Verse-like content the consumer usually wants kept as a unit.
VERSE_TYPES = frozenset({"verse", "quote", "epigraph"})
NOTE_TYPES = frozenset({"footnote", "endnote"})


class Block:
    """Thin, forward-safe wrapper over one doc.json block dict."""

    __slots__ = ("raw",)

    def __init__(self, raw: dict):
        self.raw = raw

    def get(self, key, default=None):
        return self.raw.get(key, default)

    # --- core fields (always present) --------------------------------------- #
    @property
    def id(self) -> Optional[str]:
        return self.raw.get("id")

    @property
    def type(self) -> str:
        return self.raw.get("type", "body")

    @property
    def text(self) -> str:
        return self.raw.get("text", "") or ""

    @property
    def page(self):
        return self.raw.get("page")

    @property
    def bbox(self):
        return self.raw.get("bbox")

    @property
    def confidence(self):
        return self.raw.get("confidence")

    @property
    def needs_review(self) -> bool:
        return bool(self.raw.get("needs_review"))

    @property
    def chapter(self):
        return self.raw.get("chapter")

    @property
    def marker(self):
        return self.raw.get("marker")

    @property
    def refs(self) -> list:
        return self.raw.get("refs") or []

    # --- optional / feature-gated keys (safe defaults) ---------------------- #
    @property
    def region(self) -> str:
        # Absent region (pre-region docs / non-structure blocks) → "body": the
        # safe default, so nothing is wrongly excluded as front/back matter.
        return self.raw.get("region", "body")

    @property
    def indent(self):
        return self.raw.get("indent")

    @property
    def centered(self) -> bool:
        return bool(self.raw.get("centered"))

    @property
    def ref_label(self):
        return self.raw.get("ref_label")

    @property
    def line_bbox(self):
        return self.raw.get("line_bbox")

    @property
    def continuation(self) -> bool:
        return bool(self.raw.get("continuation"))

    # --- classification helpers --------------------------------------------- #
    def is_furniture(self) -> bool:
        return self.type in FURNITURE_TYPES

    def is_note(self) -> bool:
        return self.type in NOTE_TYPES

    def is_verse_like(self) -> bool:
        return self.type in VERSE_TYPES

    def is_content(self) -> bool:
        """Real content to embed: not furniture/nav and not a note. UNKNOWN types
        count as content (forward-safe — never drop what we don't recognize)."""
        return not self.is_furniture() and not self.is_note()

    def __repr__(self):
        return f"Block({self.id!r}, {self.type!r}, p{self.page})"


class Doc:
    """Read-only view over a doc.json document."""

    def __init__(self, raw: dict):
        self.raw = raw
        self._blocks = [Block(b) for b in raw.get("blocks", [])]

    @classmethod
    def from_file(cls, path) -> "Doc":
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data: dict) -> "Doc":
        return cls(data)

    # --- metadata ----------------------------------------------------------- #
    @property
    def schema_version(self) -> str:
        return self.raw.get("schema_version", "0")

    @property
    def features(self) -> frozenset:
        return frozenset(self.raw.get("features", ()))

    def has_feature(self, name: str) -> bool:
        return name in self.features

    @property
    def doc_id(self):
        return self.raw.get("doc_id")

    @property
    def title(self):
        return self.raw.get("title")

    @property
    def source(self):
        return self.raw.get("source")

    @property
    def footnotes(self) -> dict:
        return self.raw.get("footnotes", {}) or {}

    @property
    def blocks(self) -> list:
        return self._blocks

    def __len__(self):
        return len(self._blocks)

    def __iter__(self):
        return iter(self._blocks)

    # --- queries (blocks are already in reading order) ---------------------- #
    def iter_blocks(self, *, region: Optional[str] = None,
                    types: Optional[Iterable[str]] = None,
                    exclude_types: Optional[Iterable[str]] = None,
                    chapter: Optional[str] = None) -> Iterator[Block]:
        types = frozenset(types) if types is not None else None
        exclude = frozenset(exclude_types) if exclude_types is not None else None
        for b in self._blocks:
            if region is not None and b.region != region:
                continue
            if types is not None and b.type not in types:
                continue
            if exclude is not None and b.type in exclude:
                continue
            if chapter is not None and b.chapter != chapter:
                continue
            yield b

    def content_blocks(self, *, region: Optional[str] = "body") -> list:
        """Blocks worth embedding: content only (furniture + notes dropped),
        in reading order. ``region`` defaults to body; pass None for all regions.
        Region filtering is only applied when the doc declares the feature (so an
        old doc without regions yields everything rather than nothing)."""
        use_region = region if (region and self.has_feature("region")) else None
        return [b for b in self.iter_blocks(region=use_region) if b.is_content()]

    def text(self, *, region: Optional[str] = "body", sep: str = "\n\n") -> str:
        """Clean concatenated text for embedding (content blocks only)."""
        return sep.join(b.text for b in self.content_blocks(region=region) if b.text)

    def by_type(self, *types: str) -> list:
        return list(self.iter_blocks(types=types))

    def verses(self) -> list:
        return list(self.iter_blocks(types=VERSE_TYPES))

    def headings(self) -> list:
        return self.by_type("heading")

    # --- footnote backlinks ------------------------------------------------- #
    def note(self, key: str) -> Optional[dict]:
        return self.footnotes.get(key)

    def note_source_block(self, key: str) -> Optional[Block]:
        """The block a footnotes-map entry was attached to (via its block_id)."""
        bid = (self.footnotes.get(key) or {}).get("block_id")
        if not bid:
            return None
        return next((b for b in self._blocks if b.id == bid), None)


# --------------------------------------------------------------------------- #
# Ingestion manifest (ingest_manifest.jsonl / rag_handoff_manifest.jsonl)
# --------------------------------------------------------------------------- #
def load_manifest(path) -> Iterator[dict]:
    """Yield each manifest row (one finished book). Tolerant of blank/partial
    trailing lines so it is safe to tail a file being appended to."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue            # a half-written trailing line; skip


def open_doc(row: dict) -> Optional[Doc]:
    """Load the Doc referenced by a manifest row, or None if its file is gone."""
    p = row.get("doc_json")
    if p and Path(p).is_file():
        return Doc.from_file(p)
    return None

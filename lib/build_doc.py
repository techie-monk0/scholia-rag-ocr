#!/usr/bin/env python3
"""Phase 1 of the structure-preserving contract (docs/ARCHITECTURE.md): turn the
normalized ``Page``/``Block`` model into a single ``<book>.doc.json`` per book.

This phase delivers: the type taxonomy, heading levels, furniture / navigation
labeling, table-HTML preservation, per-block ``refs`` (the [^N] markers a body
block references), per-block ``marker`` (a note's own number), unicode
normalization, confidence carry-through, and a basic ``needs_review``.

Out of scope here (later phases): the resolved ``footnotes`` map + scope keys
(Phase 2) and cross-column reading-order reconstruction (Phase 3). Blocks are
emitted in backend order for now; the ``footnotes`` map is empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import surya_backend as sb
import notes
import reading_order
from notes import extract_marker, find_refs, page_no as _page_no
from textnorm import normalize_block_text

# Surya layout label -> contract type. Unknown labels fall back to "body" (keep
# the text as content rather than dropping it).
LABEL_TO_TYPE = {
    "Text": "body", "TextInlineMath": "body", "Formula": "body", "Form": "body",
    "Title": "heading", "SectionHeader": "heading", "Section-header": "heading",
    "PageHeader": "page_header", "Page-header": "page_header",
    "PageFooter": "page_footer", "Page-footer": "page_footer",
    "PageNumber": "page_number", "Page-number": "page_number",
    "Footnote": "footnote",
    "Caption": "caption",
    "ListItem": "list_item", "List-item": "list_item", "ListGroup": "list_item",
    "Table": "table",
    "Figure": "figure", "Picture": "figure",
    "TableOfContents": "toc",
}
DEFAULT_TYPE = "body"


def _schema_version():
    """doc.json format version — single source of truth is structure.SCHEMA_VERSION
    (same whether or not the structure layer runs)."""
    try:
        import structure
        return structure.SCHEMA_VERSION
    except Exception:
        return "1.2"

_HEADING_TAG = re.compile(r"<h([1-6])\b", re.I)
_CHAPTER = re.compile(r"^\s*(?:chapter|part)\s+([0-9]+|[ivxlcdm]+|[a-z]+)\b", re.I)
_PAGE_NUMBER = re.compile(r"^\s*[\[(]?\s*(?:\d{1,4}|[ivxlcdm]{1,7})\s*[\])]?\s*$", re.I)
# Detailed multi-level contents lines the layout model mislabels as footnotes —
# "Abandoning Belief in Permanence ......... 65" / "Enthusiasm for Emptiness 33".
# Dot-leaders running to a page number are unambiguously a TOC entry; a short line
# merely ending in a bare page number is only treated as TOC inside a detected
# contents section (so real notes ending in a year/number aren't touched).
_TOC_LEADERS = re.compile(r"\.{4,}\s*\d{1,4}\b")
_TOC_DOTS = re.compile(r"\.{4,}")
_TOC_TAIL = re.compile(r"\S\s+\d{1,4}\s*$")


def _footnote_is_toc(text, in_toc_section) -> bool:
    t = (text or "").replace("\n", " ").strip()
    if _TOC_LEADERS.search(t):                       # "Title ..... 65" — unambiguous
        return True
    # inside a detected contents section, bare leaders or a trailing page number
    # are TOC too (real notes don't live in a contents section)
    return bool(in_toc_section and (_TOC_DOTS.search(t)
                                    or (len(t) <= 80 and _TOC_TAIL.search(t))))
_NAV_SECTIONS = [
    (re.compile(r"^\s*(table of\s+)?contents\s*$", re.I), "toc"),
    (re.compile(r"^\s*index(\s+of\b.*)?\s*$", re.I), "index"),
    (re.compile(r"^\s*(bibliography|references|works cited|sources)\s*$", re.I),
     "bibliography"),
]
# Back-matter often has no SectionHeader — only a running head ("INDEX OF
# ENGLISH WORDS", "Bibliography") repeated each page. Detect nav per page from
# the header text (contains-match, looser than the heading anchors above).
_NAV_HEADER = [
    (re.compile(r"\bindex\b", re.I), "index"),
    (re.compile(r"\b(bibliography|works cited)\b", re.I), "bibliography"),
    (re.compile(r"\btable of contents\b", re.I), "toc"),
]


def _page_nav(pg) -> str | None:
    """Nav section implied by a page's running header (back-matter pages)."""
    for blk in pg.blocks:
        if (blk.label or "") in ("PageHeader", "Page-header"):
            for rx, t in _NAV_HEADER:
                if rx.search(blk.text or ""):
                    return t
    return None


def _doc_id_for(source) -> str | None:
    if source is None:
        return None
    p = Path(source)
    try:
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            h = hashlib.sha256(str(source).encode()).hexdigest()
    except OSError:
        h = hashlib.sha256(str(source).encode()).hexdigest()
    return h[:16]


def _heading_level(block) -> int:
    m = _HEADING_TAG.search(block.html or "")
    if m:
        return int(m.group(1))
    return 1 if (block.label or "") == "Title" else 2


def build_doc_dict(pages, *, source=None, title=None, doc_id=None,
                   mode="auto", low_conf: float = 0.6, detect_verses=True,
                   det_by_stem=None) -> dict:
    """Pure transform: ``Page`` list -> contract dict. No file IO (testable).

    Structure analysis (verse/quote + geometry + later region/types) is merged
    ADDITIVELY into each block and declared via doc-level ``schema_version`` +
    ``features``; CORE fields never change. ``detect_verses`` (default True) gates
    the structure layer — with it off, blocks carry only the core fields and
    ``features`` is empty."""
    # Phase 2: resolve notes first — the engine is authoritative for note
    # classification (it re-types blocks Surya mislabeled) and gives per-block
    # annotations keyed by (page_index, block_index).
    fmap, ann = notes.resolve(pages, mode=mode, return_annotations=True) \
        if pages else ({}, {})

    # Structure layer (verse/quote/region/types). Lazy import + try/except so a
    # failure here can NEVER break the build — blocks just emit core fields only.
    struct, features, schema_ver = {}, [], _schema_version()
    if detect_verses:
        try:
            import structure
            struct, features = structure.analyze(pages)
            schema_ver = structure.SCHEMA_VERSION
        except Exception as e:                 # pragma: no cover - defensive
            print(f"  (structure analysis disabled: {e})", flush=True)
            struct, features = {}, []
    if det_by_stem:                              # per-line boxes available
        features = features + ["line_bbox"]
    cmod = None                                  # citation-cue signals (shared module)
    if detect_verses:
        try:
            import citation_cues as cmod
            features = features + ["citation_cues"]
        except Exception:                        # pragma: no cover - defensive
            cmod = None

    blocks_out = []
    bid = 0
    chapter = None
    prev_text = ""           # previous block IN READING ORDER (for citation lead-ins)
    nav_mode = None          # nav section type currently in effect (toc/index/...)
    for pi, pg in enumerate(pages):                 # 0-based to match ann keys
        pnum = _page_no(pg, pi + 1)
        det_boxes = (det_by_stem or {}).get(Path(pg.name).stem) if det_by_stem else None
        page_nav = _page_nav(pg)          # nav implied by this page's running head
        # Phase 3: emit blocks in reading order; flag ambiguous-layout blocks.
        idx_of = {id(b): bi for bi, b in enumerate(pg.blocks)}
        ordered, _uncertain = reading_order.order_page(pg.blocks)
        flagged = (reading_order.uncertain_indices(pg.blocks)
                   | reading_order.multicolumn_indices(pg))   # incl. merged columns
        uncertain_ids = {id(pg.blocks[i]) for i in flagged}
        for blk in ordered:
            bi = idx_of[id(blk)]
            label = blk.label or ""
            btype = LABEL_TO_TYPE.get(label, DEFAULT_TYPE)
            text = normalize_block_text(blk.text or "")
            note_ann = ann.get((pi, bi))

            level = None
            if btype == "heading":
                level = _heading_level(blk)
                nav_mode = next((t for rx, t in _NAV_SECTIONS if rx.search(text)),
                                None)
                cm = _CHAPTER.match(text)
                if cm:
                    chapter = cm.group(1).lower()   # match notes' case-normalized scope
            elif (page_nav or nav_mode) and btype in ("body", "list_item") \
                    and not note_ann:
                btype = page_nav or nav_mode   # content in a nav section inherits it
            if btype in ("page_header", "page_footer") and _PAGE_NUMBER.match(text):
                btype = "page_number"

            marker, refs, continuation = None, [], False
            # A dense contents line ("Abandoning Belief in Permanence … 65") that the
            # layout model OR the notes engine grabbed as a note is really TOC — wins
            # over the notes engine, and must skip the note marker-stripping below.
            _claimed_note = (note_ann["type"] if note_ann
                             else btype if btype in ("footnote", "endnote") else None)
            if _claimed_note and _footnote_is_toc(
                    text, page_nav == "toc" or nav_mode == "toc"):
                btype = "toc"                  # dense contents lines mis-tagged as notes
            elif note_ann:                     # notes engine is authoritative
                btype = note_ann["type"]       # footnote / endnote (fixes mislabels)
                _m, text = extract_marker(text, bare=True)
                continuation = note_ann.get("continuation", False)
                marker = None if continuation else (note_ann.get("marker") or _m)
            elif btype in ("footnote", "endnote"):
                marker, text = extract_marker(text)
            else:
                refs = find_refs(text)

            table = None
            if btype == "table" and blk.html:
                table = {"format": "html", "content": blk.html}

            conf = blk.confidence
            review = bool((conf is not None and conf < low_conf)
                          or id(blk) in uncertain_ids)

            # --- Structure layer (additive): override `type` for body blocks the
            #     analysis re-typed (verse/quote/region furniture), attach the
            #     optional geometry/region keys, and (verse only) keep line breaks.
            attrs = struct.get(id(blk))
            extra = {}
            if attrs:
                stype = attrs.get("type")
                if stype in ("verse", "quote", "list_item", "epigraph"):
                    if btype == "body":                # geometry/list refine body only
                        btype = stype
                        if stype in ("verse", "epigraph"):    # keep line breaks
                            text = normalize_block_text(blk.text or "",
                                                        dehyphenate=False)
                elif stype in ("page_header", "page_footer", "page_number",
                               "title_page", "dedication", "colophon"):
                    if btype != "table":               # drop-furniture: override
                        btype = stype                  # even a mis-typed "heading"
                elif stype:                            # copyright / toc:
                    if btype not in ("heading", "table"):   # unambiguous → override
                        btype = stype                       # incl. a mis-typed footnote
                for k in ("region", "indent", "centered", "ref_label"):
                    if attrs.get(k) is not None:
                        extra[k] = attrs[k]
                review = review or bool(attrs.get("needs_review"))

            # Optional per-line boxes — only on verse/quote/epigraph blocks (where
            # per-line alignment matters), and only from REAL detection boxes.
            if det_boxes and btype in ("verse", "quote", "epigraph"):
                lb = sb.block_line_boxes(blk, det_boxes)
                if lb:
                    extra["line_bbox"] = lb

            # Citation cues (additive signals; the CONSUMER adjudicates verse/quote/
            # body). Emit on geometry candidates OR any block with a lead-in/source.
            if cmod is not None:
                try:
                    cue = cmod.cues(prev_text, text)
                except Exception:
                    cue = {}
                strong = cue.get("lead_in_verb") or cue.get("trailing_source")
                if cue and (btype in ("verse", "quote", "epigraph")
                            or "indent" in extra or strong):
                    extra.update(cue)
                    # A body block the geometry MISSED but the cues bracket → flag a
                    # candidate (e.g. a verse whose line breaks OCR flattened).
                    if btype == "body" and strong:
                        extra["quote_candidate"] = True
                        review = True
                    if cue.get("trailing_source") and "ref_label" not in extra:
                        extra["ref_label"] = cue["trailing_source"]
            prev_text = text

            bid += 1
            block = {
                "id": f"b{bid:05d}",
                "type": btype,
                "text": text,
                "page": pnum,
                "level": level,
                "chapter": chapter,
                "refs": refs,
                "marker": marker,
                "table": table,
                "bbox": [round(float(v), 1) for v in blk.bbox] if blk.bbox else None,
                "confidence": conf,
                "needs_review": review,
            }
            block.update(extra)                        # additive optional keys
            if continuation:
                block["continuation"] = True
            blocks_out.append(block)

            # Backlink: point the footnotes-map entry at the block that carries it
            # (the primary, non-continuation note block).
            if note_ann and not continuation:
                sk = note_ann.get("scope_key")
                if sk and sk in fmap and fmap[sk].get("block_id") is None:
                    fmap[sk]["block_id"] = block["id"]

    return {
        "schema_version": schema_ver,
        "features": features,
        "doc_id": doc_id or _doc_id_for(source),
        "source": str(source) if source is not None else None,
        "title": title,
        "blocks": blocks_out,
        "footnotes": fmap,
    }


def build_doc(pages, out_path, *, detect_verses=True, **kwargs) -> dict:
    """Build the contract dict and write it to ``out_path`` as JSON. Structure
    attributes (verse/quote/region/…) are merged into the doc additively under
    ``schema_version``/``features`` — no sidecar."""
    doc = build_doc_dict(pages, detect_verses=detect_verses, **kwargs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    return doc


def main():
    p = sb.common_parser(__doc__.splitlines()[0])
    p.add_argument("-o", "--output", type=Path, default=Path("doc.json"))
    p.add_argument("--source", default=None, help="Original file path (for doc_id "
                   "content hash + provenance).")
    p.add_argument("--title", default=None)
    p.add_argument("--detect-verses", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Run the structure layer (verse/quote/region/… merged into "
                        "doc.json under schema_version/features). On by default; "
                        "--no-detect-verses emits core fields only.")
    args = p.parse_args()
    pages = sb.get_pages(args)
    doc = build_doc(pages, args.output, source=args.source, title=args.title,
                    detect_verses=args.detect_verses)
    types = {}
    for b in doc["blocks"]:
        types[b["type"]] = types.get(b["type"], 0) + 1
    print(f"Wrote {args.output}  ({len(doc['blocks'])} blocks: "
          + ", ".join(f"{k}={v}" for k, v in sorted(types.items())) + ")",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())

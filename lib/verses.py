#!/usr/bin/env python3
"""Verse / block-quote detection from block geometry (ARCHITECTURE.md
§"Verses & block quotations").

Verses (root verses, kārikās, dohās) and block quotations are distinct content
the chunker treats specially — a verse is kept as an atomic citable unit with its
line breaks preserved; a quote is attributed to its source, not the book's author.
Both sit at a larger left offset than body text; a verse keeps short line breaks,
a prose quote flows. We detect them purely from each block's bbox **relative to
its column's body margin** (never absolute pixels — pages have different widths
and skew), so this needs no raster and no OCR change.

This module is consumed only to build the ``<name>.verses.json`` SIDECAR (behind
``--detect-verses``); it never alters ``doc.json``. Block-level only: per-line
boxes (``line_bbox``) and deskew are out of scope (see the spec / plan).

    detect(blocks) -> { id(block): {type, indent, centered, ref_label, needs_review} }

Only blocks classified ``verse``/``quote`` appear in the result. ``type`` is one
of ``"verse"`` / ``"quote"``.
"""

from __future__ import annotations

import re
from statistics import median

import reading_order

# Tunable thresholds (fractions of column width). Re-tuned (Phase 4): the
# precision job is now done by region gating + the type taxonomy (front/back
# furniture, list_item, copyright, toc are excluded BEFORE verse detection), so
# the indent threshold can be low enough to recover real verses (e.g. the e10 p42
# tercet, ~3.8% indented). verse vs quote = FLOW (a quote is a long uniform
# justified block; anything else indented/centered is verse).
INDENT_MIN = 0.035    # left offset past the body margin to count as "indented"
CENTER_TOL = 0.07     # block center within this of the column center -> centered
CENTER_GAP = 0.06     # AND both side gaps at least this (symmetric) -> centered
MIN_LINES = 2         # a verse/quote spans >=2 lines; 1-line blocks are furniture
MARGIN_MIN = 2        # need this many text blocks in a column for a reliable margin

# Candidate labels (only these get reclassified verse/quote); the margin baseline
# is computed from a broader "text-ish" set so verse-heavy pages still get one.
_CAND_LABELS = (None, "Text")
_NON_TEXT = ("PageHeader", "PageFooter", "Table", "Figure", "Picture")

# ref_label: a leading verse number / dotted canonical ref ("24", "24.18").
_VERSE_NUM = re.compile(r"^\s*(\d+(?:\.\d+)+|\d+)\s*[.)—\-\s]")


def _box(b):
    bb = getattr(b, "bbox", None)
    if not bb or len(bb) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bb[:4])
    except (TypeError, ValueError):
        return None
    return (x0, y0, x1, y1) if x1 > x0 else None


def _is_candidate(b) -> bool:
    return getattr(b, "label", None) in _CAND_LABELS and _box(b) is not None


def _is_text_ish(b) -> bool:                # used for the margin baseline only
    return getattr(b, "label", None) not in _NON_TEXT and _box(b) is not None


def ref_label(text: str):
    """Best-effort leading verse number / canonical ref, else None."""
    if not text:
        return None
    m = _VERSE_NUM.match(text)
    return m.group(1) if m else None


def _flowed(text: str) -> bool:
    """A flowed (justified) paragraph: >=4 lines where all-but-the-last reach
    ~full width — a block QUOTE, not a verse (whose lines stay ragged/short)."""
    lines = [l for l in (text or "").split("\n") if l.strip()]
    if len(lines) < 4:
        return False
    mx = max(len(l) for l in lines)
    full = sum(1 for l in lines if len(l) >= 0.8 * mx)
    return full >= len(lines) - 1


def _classify(indent, centered, nlines, text):
    """'verse' | 'quote' | None. Requires >=MIN_LINES (single short lines are
    furniture). Indented/centered + flowed(many uniform lines) -> quote; otherwise
    (centered, or left-indented with ragged/short lines) -> verse."""
    if nlines < MIN_LINES:
        return None
    if not (indent >= INDENT_MIN or centered):
        return None
    return "quote" if _flowed(text) else "verse"


def detect(blocks) -> dict:
    """Geometry-based verse/quote detection. Returns ``{id(block): info}`` for
    blocks classified verse/quote (see module docstring)."""
    out: dict = {}
    for col in reading_order.columns(list(blocks)):
        margin_blocks = [b for b in col if _is_text_ish(b)]
        if len(margin_blocks) < MARGIN_MIN:     # need a reliable margin baseline
            continue
        margin_left = median(_box(b)[0] for b in margin_blocks)
        margin_right = median(_box(b)[2] for b in margin_blocks)
        col_width = margin_right - margin_left
        if col_width <= 0:
            continue
        col_center = (margin_left + margin_right) / 2.0
        for b in col:
            if not _is_candidate(b):
                continue
            x0, _y0, x1, _y1 = _box(b)
            indent = max(0.0, (x0 - margin_left) / col_width)
            left_gap = (x0 - margin_left) / col_width
            right_gap = (margin_right - x1) / col_width
            centered = (left_gap >= CENTER_GAP and right_gap >= CENTER_GAP
                        and abs((x0 + x1) / 2.0 - col_center) / col_width <= CENTER_TOL)
            nlines = sum(1 for ln in (b.text or "").split("\n") if ln.strip())
            btype = _classify(indent, centered, nlines, b.text or "")
            if btype is None:
                continue
            out[id(b)] = {
                "type": btype,
                "indent": round(indent, 3),
                "centered": bool(centered),
                "ref_label": ref_label(b.text),
                "needs_review": False,
            }
    return out

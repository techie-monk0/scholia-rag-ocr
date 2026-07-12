#!/usr/bin/env python3
"""Phase 3 — cross-column reading-order reconstruction (docs/ARCHITECTURE.md #1).

Scrambled text is fatal for chunking: a multi-column page read in the wrong
order yields incoherent chunks. We reorder a page's blocks into reading order
(columns left→right, top→bottom within each), keep the footnote band last, and
flag layouts whose order is *uncertain* (overlapping bboxes) so the builder can
set ``needs_review`` and the chunker can quarantine them.

Multi-column safety net (intra-block scramble): when Surya merges several
columns into ONE block (e.g. a glossary/index/dictionary), block-level
reordering can't unscramble the text *inside* it. We detect such a block by a
vertical whitespace gutter inside its own bbox and flag it ``needs_review`` so
the chunker quarantines it rather than trusting scrambled order.

FOLLOW-ON ENHANCEMENT (not implemented — see README "Known limitations"):
column-strip re-OCR — crop each detected column and OCR it separately, then
concatenate in column order, which removes the scramble at the source.

Public API:
    order_page(blocks) -> (ordered_blocks, uncertain: bool)
    uncertain_indices(blocks) -> set[int]      # indices in an ambiguous overlap
    multicolumn_indices(page) -> set[int]      # wide blocks with an inner gutter
"""

from __future__ import annotations

_FOOTNOTE_LABELS = {"Footnote", "Page-footnote"}


def _bbox(b):
    return b.bbox if b.bbox else [0, 0, 0, 0]


def _overlaps(a, b) -> bool:
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def uncertain_indices(blocks) -> set:
    """Indices of blocks whose bbox overlaps another's — the layout there is
    ambiguous (can't be cleanly separated into columns/rows)."""
    blocks = list(blocks)
    bad = set()
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if _overlaps(blocks[i], blocks[j]):
                bad.add(i)
                bad.add(j)
    return bad


def _split_columns(blocks):
    """Group blocks into columns by merging their x-intervals; a gap between
    merged intervals is a gutter. Returns columns ordered left→right."""
    if len(blocks) <= 1:
        return [list(blocks)]
    intervals = sorted(([_bbox(b)[0], _bbox(b)[2]] for b in blocks))
    merged = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    if len(merged) == 1:                      # single column
        return [list(blocks)]
    cols = [[] for _ in merged]
    for b in blocks:
        cx = (_bbox(b)[0] + _bbox(b)[2]) / 2
        idx = next((i for i, (mx0, mx1) in enumerate(merged) if mx0 <= cx <= mx1),
                   None)
        if idx is None:                        # nearest column by center
            idx = min(range(len(merged)),
                      key=lambda i: min(abs(cx - merged[i][0]), abs(cx - merged[i][1])))
        cols[idx].append(b)
    return cols                                # merged is x-sorted -> left→right


def columns(blocks):
    """Public wrapper around ``_split_columns``: group blocks into columns by
    x-overlap, returned left→right as ``list[list[block]]``. Used by verse/quote
    detection to compute a per-column body margin."""
    return _split_columns(blocks)


def _has_inner_gutter(im, bbox, *, central=(0.30, 0.70), gutter_max=0.012,
                      side_min=0.08):
    """True if the block's image region has a near-empty vertical channel in its
    central x-range with ink on both sides — i.e. ≥2 columns merged into one
    block."""
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    crop = im[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0 or crop.shape[1] < 20:
        return False
    col_ink = (crop < 128).mean(axis=0)        # per-column ink fraction
    w = col_ink.shape[0]
    lo, hi = int(w * central[0]), int(w * central[1])
    if hi - lo < 3:
        return False
    return (col_ink[lo:hi].min() <= gutter_max          # an empty central channel
            and col_ink[:lo].max() >= side_min          # text to the left
            and col_ink[hi:].max() >= side_min)         # and to the right


def multicolumn_indices(page) -> set:
    """Indices of wide blocks that contain an internal vertical gutter (multiple
    columns merged into one block). Needs the page raster; empty without it."""
    if not getattr(page, "image_path", None):
        return set()
    try:
        import numpy as np
        from PIL import Image
        im = np.asarray(Image.open(page.image_path).convert("L"))
    except Exception:
        return set()
    W = im.shape[1]
    out = set()
    for i, b in enumerate(page.blocks):
        if not b.bbox:
            continue
        if (b.bbox[2] - b.bbox[0]) >= 0.5 * W and _has_inner_gutter(im, b.bbox):
            out.add(i)
    return out


def order_page(blocks):
    """Return (blocks in reading order, uncertain). Footnote-band blocks sort
    after the body; remaining blocks order by column then top-to-bottom."""
    blocks = list(blocks)
    if not blocks:
        return [], False
    uncertain = bool(uncertain_indices(blocks))
    foot = [b for b in blocks if (b.label or "") in _FOOTNOTE_LABELS]
    body = [b for b in blocks if (b.label or "") not in _FOOTNOTE_LABELS]
    ordered = []
    for col in _split_columns(body):
        ordered += sorted(col, key=lambda b: _bbox(b)[1])
    ordered += sorted(foot, key=lambda b: _bbox(b)[1])
    return ordered, uncertain

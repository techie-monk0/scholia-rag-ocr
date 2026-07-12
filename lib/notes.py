#!/usr/bin/env python3
"""Phase 2 — footnote / endnote resolution (docs/ARCHITECTURE.md §Footnotes).

The unified note engine: detect note bodies (page-bottom footnotes via a
raster *rule* + geometry, since Surya's `Footnote` label is unreliable — it was
absent on facing pages of the same book; or endnotes via a back-of-book "Notes"
section), parse each note's own marker, stitch page-bottom notes that continue
across a page break (no repeated marker), key by scope (``p<page>`` for
footnotes, ``ch<chapter>`` / ``book`` for endnotes), join to the body's `[^N]`
references, and fall back to sequential position when an OCR'd number breaks the
match. Failure is benign: an unmatched note is kept raw and flagged
``needs_review``.

Entry point: ``resolve(pages) -> {scope:marker -> Footnote}``. ``build_doc``
calls ``resolve(pages, return_annotations=True)`` to also get per-block roles.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---- shared note-parsing helpers (also imported by build_doc) ------------- #
# A note's own leading marker, kept deliberately broad so we're not tied to one
# book's convention: [^N] (from <sup>); a number that is wrapped and/or
# terminated — "13.", "13)", "13:", "(13)", "[13]"; or a symbol run "* † ‡ § ¶
# # ‖", optionally doubled ("††"). The strict form requires punctuation/wrapping
# so body prose isn't mis-parsed; `bare=True` (used only inside a confirmed note
# context) additionally accepts a plain "13 " with no punctuation.
_MARK = re.compile(
    r"^\s*(?:\[\^([^\]]+)\]"
    r"|[\[(]?(\d{1,4})[\])]?[.):]"
    r"|[\[(](\d{1,4})[\])]"
    r"|([*†‡§¶#‖]+))\s*")
_BARE_MARK = re.compile(r"^\s*(\d{1,4})\s+")
# Body citation markers in OCR text: [^N] (superscript markers Surya renders this
# way) OR [N] (plain bracketed digits). The bare-bracket form is DIGITS-ONLY so an
# editorial insertion like "[Valid cognitions…]" is never mistaken for a marker.
_REF_ANY = re.compile(r"\[\^([^\]]+)\]|\[(\d{1,3})\]")


def find_refs(text: str) -> list:
    """All note markers referenced in a body block's text ([^N] and [N] forms)."""
    return [a or b for a, b in _REF_ANY.findall(text or "")]
# Note-section heading, matched on space-flattened text so OCR letter-spacing
# garble ("N otes to C hapter One") still resolves. Titles: Notes, Endnotes,
# Reference Notes, Chapter Notes; optional "to (the) (chapter) <scope>".
_NOTES_PREFIX = re.compile(r"(endnotes|referencenotes|chapternotes|notes)")
_NOTES_SCOPE_CH = re.compile(r"chapter([a-z0-9]+)$")          # "...chapter three"
_NOTES_SCOPE_TO = re.compile(r"to(?:the)?([a-z0-9]+)$")       # "...to the introduction"
_FOOTNOTE_LABELS = {"Footnote", "Page-footnote"}
_HEADING_LABELS = {"SectionHeader", "Section-header", "Title"}
_FURNITURE_LABELS = {"PageHeader", "Page-header", "PageFooter", "Page-footer"}

# geometry knobs
_BAND_FRAC = 0.70        # no-rule fallback: blocks below this page fraction
_RULE_TOL = 18           # px slack below a detected rule


def extract_marker(text: str, *, bare: bool = False):
    """Pull a leading note marker off the front of ``text``; returns
    (marker, remaining_body) or (None, text). ``bare=True`` (only inside a
    confirmed note context) also accepts a plain "13 " with no punctuation."""
    m = _MARK.match(text)
    if m:
        marker = m.group(1) or m.group(2) or m.group(3) or m.group(4)
        return marker, text[m.end():].lstrip()
    if bare:
        m = _BARE_MARK.match(text)
        if m:
            return m.group(1), text[m.end():].lstrip()
    return None, text


def parse_notes_heading(text):
    """(is_notes_heading, chapter_scope) from a heading, OCR-space-tolerant.
    Chapter is the token after 'to (the) (chapter)'; None -> book scope."""
    flat = re.sub(r"\s+", "", text or "").lower()
    if not _NOTES_PREFIX.match(flat):
        return False, None
    m = _NOTES_SCOPE_CH.search(flat) or _NOTES_SCOPE_TO.search(flat)
    return True, (_norm_chapter(m.group(1)) if m else None)


def page_no(pg, idx: int) -> int:
    """1-based source page: trailing digits in the image name, else backend
    page field, else the running index."""
    m = re.search(r"(\d+)\D*$", Path(pg.name).stem) if pg.name else None
    if m:
        return int(m.group(1))
    if getattr(pg, "page_no", None):
        return int(pg.page_no)
    return idx


def _terminated(text: str) -> bool:
    """Does this note text look complete (vs cut off mid-phrase)?"""
    t = text.rstrip()
    return (not t) or t[-1] in ".!?\"')]”’…"


# --------------------------------------------------------------------------- #
# Raster rule detection (the footnote separator line)
# --------------------------------------------------------------------------- #
def _max_run_per_row(ink):
    """Longest run of consecutive True per row, vectorized over columns."""
    import numpy as np
    H, W = ink.shape
    runs = np.zeros(H, dtype=int)
    cur = np.zeros(H, dtype=int)
    for x in range(W):
        col = ink[:, x]
        cur = np.where(col, cur + 1, 0)
        runs = np.maximum(runs, cur)
    return runs


# POSSIBLE ENHANCEMENT (not implemented — current behavior verified adequate on
# real annotated pages, e.g. edition-383 p.57): hand annotations (pencil/pen
# underlining, highlighting) could in principle (a) be mistaken for the footnote
# separator rule, or (b) corrupt OCR of the marked text. In practice the
# detector already rejects underlines via their waviness (no long *unbroken*
# run) and non-isolation (text immediately above), and Surya read underlined
# text correctly at 300 DPI. If heavier/straighter marks ever cause trouble,
# harden here with: straightness (constant run-y across width), left-margin
# anchoring (the rule starts at the text margin; underlines start mid-line), and
# a pre-OCR de-annotation pass (drop grey graphite / saturated colored pen;
# dark-ink underlines fall back to confidence-based needs_review).
def detect_footnote_rule(image_path, *, lower_frac=0.5, min_width_frac=0.10,
                         max_thick=8):
    """Return the y of the topmost horizontal *rule* in the lower part of the
    page (the footnote separator), or None. A rule is a thin, isolated band with
    a long unbroken dark run — unlike text (short letter-runs) or solid blocks."""
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None
    try:
        im = np.asarray(Image.open(image_path).convert("L"))
    except Exception:
        return None
    H, W = im.shape
    ink = im < 128
    runs = _max_run_per_row(ink)
    row_mean = ink.mean(axis=1)
    y0 = int(H * lower_frac)
    is_rule_row = (runs >= W * min_width_frac) & (row_mean < 0.5)
    is_rule_row[:y0] = False
    # group consecutive rule rows into bands; keep thin + isolated ones
    y = y0
    while y < H:
        if not is_rule_row[y]:
            y += 1
            continue
        start = y
        while y < H and is_rule_row[y]:
            y += 1
        thick = y - start
        if thick <= max_thick:
            above = row_mean[max(0, start - 4):start].mean() if start else 0.0
            if above < 0.05:                  # whitespace gap above the rule
                return start
    return None


# --------------------------------------------------------------------------- #
# Footnote band (page-bottom)
# --------------------------------------------------------------------------- #
def _band_blocks(page, rule_y):
    """(block_index, block) for blocks in the page's footnote band, sorted by y.
    Uses the rule when found, else a bottom-fraction geometry fallback, plus the
    `Footnote` label as a corroborating signal."""
    H = page.height or 1
    has_image = bool(page.image_path)
    out = []
    for bi, b in enumerate(page.blocks):
        y0 = (b.bbox[1] if b.bbox else 0)
        # Require strong evidence: a detected rule or Surya's Footnote label.
        # The bottom-fraction geometry fallback fires ONLY when there's no image
        # to find a rule in (synthetic tests / born-digital) — using it on real
        # pages turns ordinary bottom-of-page body into spurious footnotes.
        in_band = ((b.label in _FOOTNOTE_LABELS)
                   or (rule_y is not None and y0 >= rule_y - _RULE_TOL)
                   or (not has_image and y0 >= _BAND_FRAC * H))
        if in_band:
            out.append((bi, b))
    out.sort(key=lambda t: (t[1].bbox[1] if t[1].bbox else 0))
    return out


def _body_refs(page, band_idx):
    """Markers referenced (in [^N]) by NON-band blocks, in reading order."""
    refs = []
    for bi, b in enumerate(page.blocks):
        if bi in band_idx:
            continue
        refs += find_refs(b.text or "")
    return refs


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def resolve(pages, *, mode="auto", return_annotations=False):
    """Build the scope-keyed footnote/endnote map from the Page model.

    ``mode``: "auto" (detect per book), "page", "end", or "both". With
    ``return_annotations`` also returns ``{(page_idx, block_idx): role}`` so the
    doc builder can re-type the blocks Surya mislabeled and merge continuations.
    """
    fmap: dict = {}
    ann: dict = {}
    endnote_chapter = None
    open_note = None            # last page-bottom note still "open" (unterminated)
    open_endnote = None         # last endnote still "open" across a page break

    want_page = mode in ("auto", "page", "both")
    want_end = mode in ("auto", "end", "both")

    for pi, page in enumerate(pages):
        pno = page_no(page, pi + 1)

        # section-mode transitions from this page's headings
        page_is_notes = False
        for b in page.blocks:
            if b.label in _HEADING_LABELS:
                txt = (b.text or "").strip()
                is_notes, ch = parse_notes_heading(txt)
                if want_end and is_notes:
                    endnote_chapter = ch
                    page_is_notes = True

        # A page is an endnote page if a Notes heading is on it, or its content
        # is a full-page run of numbered entries (so the back-of-book Notes
        # section is caught even when its heading isn't in view / is OCR-garbled).
        if want_end and (page_is_notes or _looks_like_notes_page(page)):
            # A Notes *heading* opens a fresh section, so any note left open on a
            # prior page is closed; a heading-less continuation page (caught by
            # _looks_like_notes_page) may instead carry the previous note's tail.
            if page_is_notes:
                open_endnote = None
            open_endnote = _resolve_endnotes(page, pi, pno, endnote_chapter,
                                             fmap, ann, open_endnote)
            open_note = None
            continue

        # leaving the endnote stream closes any open endnote
        open_endnote = None
        if want_page:
            open_note = _resolve_page_footnotes(page, pi, pno, fmap, ann, open_note)

    if return_annotations:
        return fmap, ann
    return fmap


def _norm_chapter(ch):
    if not ch:
        return None
    ch = ch.strip()
    m = re.match(r"^([0-9]+|[ivxlcdm]+)$", ch, re.I)
    return m.group(1) if m else ch.lower()


def _plausible_next_marker(marker, prev_marker) -> bool:
    """Could ``marker`` be the note right after ``prev_marker`` in a numbered
    sequence? Used to tell a genuine next note (e.g. "2" after "1") from a bare
    number that is really continuation text (e.g. a year "906" tailing note 1).
    Non-numeric markers (symbols) are always treated as plausible new notes."""
    if not (str(marker).isdigit() and str(prev_marker).isdigit()):
        return True
    return 0 < int(marker) - int(prev_marker) <= 2     # next (allow one skipped)


def _looks_like_notes_page(page):
    """An endnote-section page: a full-page run of numbered entries. Requires a
    numbered entry in the UPPER half (notes fill the page) so a normal page with
    body on top and a few footnotes at the bottom is NOT misread as endnotes."""
    H = page.height or 1
    skip = _HEADING_LABELS | _FURNITURE_LABELS       # ignore headings + furniture
    items = [(b.bbox[1] if b.bbox else 0, b.text) for b in page.blocks
             if b.label not in skip and (b.text or "").strip()]
    if not items:
        return False
    numbered = sum(1 for _, t in items if extract_marker(t, bare=True)[0] is not None)
    upper_numbered = any(extract_marker(t, bare=True)[0] is not None and y < 0.5 * H
                         for y, t in items)
    # ≥3 entries: a dense run of notes, not one stray numbered block (which a
    # single-footnote page would otherwise look like).
    return upper_numbered and numbered >= 3 and numbered >= len(items) // 2


def _resolve_page_footnotes(page, pi, pno, fmap, ann, open_note):
    rule_y = detect_footnote_rule(page.image_path) if page.image_path else None
    band = _band_blocks(page, rule_y)
    band_idx = {bi for bi, _ in band}
    refs = _body_refs(page, band_idx)

    # Pass 1 — segment the band into notes; stitch continuations at text level.
    # Markers/keys are assigned in pass 2 (a same-page continuation can precede
    # its owning note's key being known).
    notes = []          # {bi, marker, text, term, cont_bis}
    for bi, b in band:
        marker, body = extract_marker((b.text or "").strip(), bare=True)
        if marker is None:
            if not notes and open_note is not None and not open_note["term"]:
                # cross-page continuation: append to the previous page's open
                # note, mutating its map entry directly.
                entry = fmap[open_note["key"]]
                entry["text"] += " " + body
                if pno not in entry["span_pages"]:
                    entry["span_pages"].append(pno)
                open_note["term"] = _terminated(body)
                ann[(pi, bi)] = {"type": "footnote", "marker": open_note["marker"],
                                 "scope_key": open_note["key"], "continuation": True}
                continue
            if notes:                       # continuation of this page's last note
                notes[-1]["text"] += " " + body
                notes[-1]["term"] = _terminated(body)
                notes[-1]["cont_bis"].append(bi)
                continue
            # orphan unmarked note with no open note to attach to
        notes.append({"bi": bi, "marker": marker, "text": body,
                      "term": _terminated(body), "cont_bis": []})

    # Pass 2 — join markers to body refs (sequential fallback) + assign keys.
    used = set()
    last_keyed = None
    for note in notes:
        review = False
        if note["marker"] is None or note["marker"] not in refs:
            nxt = next((r for r in refs if r not in used), None)
            if note["marker"] is None and nxt is not None:
                note["marker"], review = nxt, True      # misread number recovered
            elif note["marker"] is not None and note["marker"] not in refs:
                review = True                           # orphan note (kept, flagged)
        if note["marker"] is None:
            continue                                    # unkeyable -> drop silently
        used.add(note["marker"])
        key = f"p{pno}:{note['marker']}"
        fmap[key] = {"marker": note["marker"], "kind": "footnote",
                     "text": note["text"], "page": pno, "chapter": None,
                     "block_id": None, "needs_review": review, "span_pages": [pno]}
        ann[(pi, note["bi"])] = {"type": "footnote", "marker": note["marker"],
                                 "scope_key": key, "continuation": False}
        for cb in note["cont_bis"]:
            ann[(pi, cb)] = {"type": "footnote", "marker": note["marker"],
                             "scope_key": key, "continuation": True}
        note["key"] = key
        last_keyed = note

    if last_keyed and not last_keyed["term"]:
        return {"marker": last_keyed["marker"], "key": last_keyed["key"],
                "term": False}
    return None


def _resolve_endnotes(page, pi, pno, ch, fmap, ann, open_endnote=None):
    """Resolve one endnote-section page; returns the note left open at the page's
    end (unterminated, so it may continue onto the next page) or None.

    ``open_endnote`` is the note carried in from the previous page: when this
    page opens with an UNMARKED block before any marked note, that block is its
    tail (end-of-chapter notes wrap a long citation across the page break with no
    repeated marker, exactly like page-bottom footnotes), so it is appended to
    the carried note rather than emitted as a spurious note."""
    scope = f"ch{ch}" if ch else "book"
    last_key = None
    seen_marked = False
    for bi, b in enumerate(page.blocks):
        if b.label in _HEADING_LABELS or b.label in _FURNITURE_LABELS:
            continue
        text = (b.text or "").strip()
        if not text:
            continue
        marker, body = extract_marker(text, bare=True)
        # A continued citation can resume with a bare number (a year/page like
        # "906 (Cambridge…" tailing "…589–"). Before the first marked note on a
        # heading-less page that carries an open endnote, such a leading number
        # is NOT a new note marker unless it's the plausible next one in sequence.
        if (marker is not None and not seen_marked and last_key is None
                and open_endnote is not None and not open_endnote["term"]
                and not _plausible_next_marker(marker, open_endnote["marker"])):
            marker, body = None, text     # keep the digits — they're citation text
        if marker is None:
            if last_key:                    # continuation of previous endnote
                fmap[last_key]["text"] += " " + body
                ann[(pi, bi)] = {"type": "endnote", "marker": fmap[last_key]["marker"],
                                 "scope_key": last_key, "continuation": True}
            elif (not seen_marked and open_endnote is not None
                  and not open_endnote["term"]):
                # cross-page continuation of the prior page's open endnote
                entry = fmap[open_endnote["key"]]
                entry["text"] += " " + body
                if pno not in entry["span_pages"]:
                    entry["span_pages"].append(pno)
                open_endnote["term"] = _terminated(body)
                ann[(pi, bi)] = {"type": "endnote", "marker": open_endnote["marker"],
                                 "scope_key": open_endnote["key"], "continuation": True}
                last_key = open_endnote["key"]
            continue
        seen_marked = True
        key = f"{scope}:{marker}"
        fmap[key] = {"marker": marker, "kind": "endnote", "text": body,
                     "page": pno, "chapter": ch, "block_id": None,
                     "needs_review": False, "span_pages": [pno]}
        ann[(pi, bi)] = {"type": "endnote", "marker": marker, "scope_key": key,
                         "continuation": False}
        last_key = key

    if last_key and not _terminated(fmap[last_key]["text"]):
        return {"marker": fmap[last_key]["marker"], "key": last_key, "term": False}
    return None

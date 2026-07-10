#!/usr/bin/env python3
"""Structure-analysis layer for the doc.json contract (ARCHITECTURE.md).

Computes per-block structure attributes and merges them ADDITIVELY into doc.json,
with a doc-level ``schema_version`` + ``features`` list so consumers are forward-
compatible (check ``features`` to know what's populated; read keys via ``.get()``).

Layers (grown phase by phase; each adds to ``FEATURES``):
  Phase 0  verse / quote + geometry (indent, centered, ref_label)
  Phase 1  printer-key + copyright labeling
  Phase 2  list_item vs numbered-verse
  Phase 3  TOC detection
  Phase 4  region (front | body | back)
  Phase 5  running heads / folios by repetition
  Phase 6  glossary per-entry split          (DEFERRED)

This NEVER touches OCR: it runs at assembly on already-recognized blocks, and
``build_doc`` calls ``analyze`` guarded, so a failure here cannot break the build
(blocks simply emit with their core fields only).

    analyze(pages) -> (per_block, features)
      per_block : {id(block): {attr: value, ...}}   # only blocks with extras
      features  : list[str]                          # what THIS pipeline computed
"""

import re

SCHEMA_VERSION = "1.2"

# What this pipeline version computes. Declared whenever analysis runs (even if a
# given doc has zero verses), so a consumer can tell "computed, none present" from
# "not computed by this pipeline". Grows as later phases land.
FEATURES = ["indent", "centered", "verse", "quote", "ref_label",
            "printer_key", "copyright", "list_item", "toc", "region",
            "running_head", "title_page", "dedication", "colophon", "epigraph"]

# --- Phase 1: printer's key + copyright ------------------------------------- #
# Copyright/edition-page signals (any region).
_COPYRIGHT_RE = re.compile(
    r"\b(ISBN|ISSN|LCCN|Library of Congress|Cataloging-in-Publication|"
    r"All rights reserved|©|Copyright ©|Printed in)\b", re.I)
_INT_TOKEN = re.compile(r"^\d{1,2}$")


def is_printer_key(text: str) -> bool:
    """A printer's key: a line whose tokens are ALL small integers (<=~24) in a
    strictly monotonic run, e.g. '10 9 8 7 6 5 4 3 2 1', '1 3 5 7 9',
    '04 03 02 01 00'. No prose. Appears in copyright OR back-matter colophon."""
    toks = (text or "").split()
    if len(toks) < 4 or not all(_INT_TOKEN.match(t) for t in toks):
        return False
    nums = [int(t) for t in toks]
    if any(n > 24 for n in nums):
        return False
    d = [b - a for a, b in zip(nums, nums[1:])]
    return all(x > 0 for x in d) or all(x < 0 for x in d)   # strictly monotonic


def is_copyright(text: str) -> bool:
    return bool(_COPYRIGHT_RE.search(text or ""))


# --- Phase 2: list_item vs numbered-verse ----------------------------------- #
# Leading enumerator: 1. 1) 1* (a) a) a. a* iv. • – . Requires trailing
# punctuation (or a bullet), so a bare verse stanza number ("3 It is a practice…")
# does NOT match — those are verses, distinguished by their preserved line breaks.
_ENUM = re.compile(
    r"^\s*(\([0-9a-z]{1,4}\)|[0-9]{1,3}[.)*]|[a-z][.)*]|[ivxlcdm]{1,4}[.)])\s+\S",
    re.I)
_BULLET = re.compile(r"^\s*[•▪◦‣·–—-]\s+\S")


def has_enumerator(text: str) -> bool:
    return bool(_ENUM.match(text or "") or _BULLET.match(text or ""))


def is_ragged(text: str) -> bool:
    """True if the block has >=2 lines markedly shorter than its widest line — the
    line-break-preserved signature of verse (vs. a flowed list item / paragraph,
    whose interior lines are ~uniform full width)."""
    lines = [l for l in (text or "").split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    mx = max(len(l) for l in lines)
    return sum(1 for l in lines if len(l) < 0.6 * mx) >= 2


def is_list_item(text: str) -> bool:
    """A flowed enumerated/bulleted item — NOT a verse. Enumerated + not ragged
    (single-line items, or wrapped prose); ragged enumerated blocks are verse."""
    return has_enumerator(text) and not is_ragged(text)


# --- Phase 3: table of contents --------------------------------------------- #
# Leader dots ("Preface . . . . 7" / "Preface....7") OR >=2 lines that end in a
# page number (a TOC entry: "title  <pagenum>"). The Contents-heading nav path in
# build_doc already inherits-tag a dotless TOC; this catches TOC blocks directly.
_LEADER_DOTS = re.compile(r"\.\s?\.\s?\.|\.{4,}|…{2,}|…\s*\d")
_TOC_ENTRY = re.compile(r"\S.*?\s\d{1,4}\s*$")


def is_toc_block(text: str) -> bool:
    t = text or ""
    if _LEADER_DOTS.search(t):
        return True
    lines = [l for l in t.split("\n") if l.strip()]
    return sum(1 for l in lines if _TOC_ENTRY.match(l)) >= 2


# --- Phase 4: region (front | body | back) ---------------------------------- #
_BACKMATTER_RE = re.compile(
    r"^\s*(notes|endnotes|appendix|appendices|glossary|bibliography|references|"
    r"works cited|index|colophon)\b", re.I)


def _is_heading(blk) -> bool:
    return (getattr(blk, "label", None) or "") in ("SectionHeader", "Title")


def _prose_block(blk) -> bool:
    """A substantial body-prose block (front matter has little of this)."""
    return (getattr(blk, "label", None) in (None, "Text")
            and len((blk.text or "")) > 80)


_DEDICATION_RE = re.compile(
    r"\b(dedicat|in memory of|in loving memory|for my |to my |to the memory)", re.I)
_COLOPHON_RE = re.compile(r"\b(colophon|printed (and bound )?in|typeset|"
                          r"set in \w+ type|first (printing|edition))\b", re.I)


def is_dedication(text: str) -> bool:
    t = text or ""
    return bool(_DEDICATION_RE.search(t)) and len(t) < 200


def _nlines(blk) -> int:
    return sum(1 for l in (getattr(blk, "text", "") or "").split("\n") if l.strip())


def assign_regions(pages, sig):
    """front | body | back per page index, from per-page signals (sig[i])."""
    n = len(pages)
    reg = ["body"] * n
    if n <= 2:
        return reg
    # FRONT: the leading run of pages with a front signal (copyright/TOC, or no
    # body prose), ending at the first prose page that isn't copyright/TOC. Capped
    # so a prose-light book can't be all-front.
    cap = max(8, int(0.25 * n))
    front_end = 0
    for i in range(n):
        s = sig[i]
        if not (s["copyright"] or s["toc"] or s["prose"] == 0):
            break
        front_end = i + 1
        if front_end >= cap:
            break
    for i in range(front_end):
        reg[i] = "front"
    # BACK: from the first back-matter section heading in the latter half to end.
    for i in range(max(front_end, n // 2), n):
        if sig[i]["backmatter"]:
            for j in range(i, n):
                reg[j] = "back"
            break
    return reg


# --- Phase 5: running heads / folios by repetition -------------------------- #
_DIGITS = re.compile(r"\d+")


def running_heads(pages, min_repeat=3):
    """{id(block): 'page_header'|'page_footer'|'page_number'} for blocks in the
    top/bottom margin band whose DIGIT-MASKED text repeats across >= min_repeat
    pages (a running head/folio, even if Surya labeled it Text/SectionHeader)."""
    from collections import defaultdict
    seen = defaultdict(set)
    info = {}
    for pi, pg in enumerate(pages):
        H = getattr(pg, "height", 0) or 0
        if H <= 0:
            continue
        for blk in pg.blocks:
            bb = getattr(blk, "bbox", None)
            if not bb or len(bb) < 4:
                continue
            yc = (bb[1] + bb[3]) / 2.0
            band = "top" if yc < 0.10 * H else ("bot" if yc > 0.90 * H else None)
            if band is None:
                continue
            masked = _DIGITS.sub("#", (blk.text or "").strip())[:80]
            seen[(band, masked)].add(pi)
            info[id(blk)] = (band, masked)
    repeated = {k for k, ps in seen.items() if len(ps) >= min_repeat}
    out = {}
    for bid, (band, masked) in info.items():
        if (band, masked) not in repeated:
            continue
        if masked.strip("# ") == "":               # pure folio (page number)
            out[bid] = "page_number"
        else:
            out[bid] = "page_header" if band == "top" else "page_footer"
    return out


def analyze(pages):
    """Return ({id(block): attrs}, features). Passes: (1) running-head repetition;
    (2) per-block candidate types + per-page signals; (3) region pre-pass, gate
    verse/quote out of front/back, attach `region` to every block."""
    import verses

    running = running_heads(pages)
    per_block: dict = {}
    page_of: dict = {}
    sig = []
    for pi, pg in enumerate(pages):
        try:
            vinfo = verses.detect(pg.blocks)
        except Exception:
            vinfo = {}
        copyright_pg = toc_pg = backmatter_pg = False
        prose = 0
        for blk in pg.blocks:
            key = id(blk)
            page_of[key] = pi
            attrs: dict = {}
            vi = vinfo.get(key)
            if vi:
                attrs.update({"type": vi.get("type"), "indent": vi.get("indent"),
                              "centered": vi.get("centered"),
                              "ref_label": vi.get("ref_label"),
                              "needs_review": vi.get("needs_review")})
            text = blk.text or ""
            if id(blk) in running:               # repeated band block = furniture
                attrs["type"] = running[id(blk)]
            elif is_printer_key(text) or is_copyright(text):
                attrs["type"] = "copyright"
                copyright_pg = True
            elif is_list_item(text) and attrs.get("type") not in ("verse", "quote"):
                attrs["type"] = "list_item"
            elif is_toc_block(text) and attrs.get("type") not in ("verse", "quote"):
                attrs["type"] = "toc"
                toc_pg = True
            if attrs:
                per_block.setdefault(key, {}).update(attrs)
            if _prose_block(blk):
                prose += 1
            if _is_heading(blk) and _BACKMATTER_RE.match(text):
                backmatter_pg = True
            if _is_heading(blk) and re.match(r"^\s*(contents|table of contents)\b",
                                             text, re.I):
                toc_pg = True
        sig.append({"copyright": copyright_pg, "toc": toc_pg, "prose": prose,
                    "backmatter": backmatter_pg, "nblocks": len(pg.blocks)})

    region = assign_regions(pages, sig)
    for pi, pg in enumerate(pages):
        r = region[pi]
        # A title page: front, sparse, with no copyright/TOC/prose signal.
        title_pg = (r == "front" and not sig[pi]["copyright"] and not sig[pi]["toc"]
                    and sig[pi]["prose"] == 0 and sig[pi]["nblocks"] <= 8)
        for blk in pg.blocks:
            a = per_block.setdefault(id(blk), {})
            t = a.get("type")
            text = blk.text or ""
            if r != "body":
                # Region gate. A verse/quote-shaped block in front/back is either a
                # genuine opening epigraph (>=3 lines of quoted text, not on a title
                # page -> KEEP as `epigraph`) or furniture (title/author/dedication
                # -> drop the override).
                if t in ("verse", "quote"):
                    if not title_pg and _nlines(blk) >= 3:
                        a["type"] = "epigraph"
                    else:
                        a.pop("type")
                        t = None
                if r == "back" and t == "copyright":
                    a["type"] = "colophon"           # printer's note in back matter
                elif r == "back" and _is_heading(blk) and _COLOPHON_RE.search(text):
                    a["type"] = "colophon"
                elif r == "front" and is_dedication(text) \
                        and t not in ("copyright", "toc"):
                    a["type"] = "dedication"
                elif r == "front" and title_pg and t not in (
                        "copyright", "toc", "page_header", "page_footer",
                        "page_number", "epigraph", "dedication"):
                    a["type"] = "title_page"
            a["region"] = r
    return per_block, list(FEATURES)

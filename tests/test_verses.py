"""Verse / quote detection (lib/verses.py) + the doc.json-unchanged sidecar
invariant in build_doc. Geometry fixtures use EXPLICIT bboxes (conftest auto-stacks
only when bbox is None)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import verses
from ocr_backend import Block
from build_doc import build_doc_dict
from textnorm import normalize_block_text


def _body(text, x0, x1, y):
    """A full-width body block (establishes the column margin)."""
    return Block(text, [x0, y, x1, y + 50], label="Text")


# Two full-width body blocks define margin_left=100, margin_right=700 (col_width 600).
def _col(*extra):
    base = [_body("Body paragraph one across the full width.", 100, 700, 100),
            _body("Body paragraph two across the full width.", 100, 700, 1000)]
    return base + list(extra)


# --------------------------------------------------------------------------- #
# detect()
# --------------------------------------------------------------------------- #
def test_centered_short_multiline_is_verse():
    v = Block("24. a centered\nshort second line", [300, 300, 520, 380], label="Text")
    info = verses.detect(_col(v))[id(v)]
    assert info["type"] == "verse"
    assert info["centered"] is True
    assert info["ref_label"] == "24"


def test_left_indented_flowed_is_quote():
    # A flowed block quote: indented + >=4 uniform (justified) lines.
    q = Block("Quoted prose line one that flows wide\n"
              "Quoted prose line two that flows wide\n"
              "Quoted prose line three flows just as wide\n"
              "short tail", [180, 300, 690, 460], label="Text")
    info = verses.detect(_col(q))[id(q)]
    assert info["type"] == "quote"


def test_full_width_body_not_flagged():
    b = _body("A perfectly ordinary body paragraph.", 100, 700, 300)
    assert id(b) not in verses.detect(_col(b))


def test_centered_heading_not_reclassified():
    h = Block("A Centered Heading", [250, 300, 550, 360], label="SectionHeader")
    assert id(h) not in verses.detect(_col(h))         # non-body label excluded


def test_single_line_indented_not_flagged():
    # single short indented/centered lines are title-page furniture (author, ISBN),
    # the dominant real-corpus false positive — require >=2 lines, so not flagged.
    s = Block("An indented single line", [180, 300, 680, 350], label="Text")
    assert id(s) not in verses.detect(_col(s))


def test_two_column_right_block_not_false_verse():
    # left column body (margin 50..350), right column body (margin 450..750)
    left = [Block("L body one", [50, 100, 350, 150], label="Text"),
            Block("L body two", [50, 200, 350, 250], label="Text")]
    right = [Block("R body one", [450, 100, 750, 150], label="Text"),
             Block("R body two", [450, 200, 750, 250], label="Text")]
    # a right-column block at its own margin: big absolute x0 but ~0 indent
    rblk = Block("R body three at margin", [450, 300, 720, 350], label="Text")
    assert id(rblk) not in verses.detect(left + right + [rblk])


def test_bboxless_block_no_crash():
    nob = Block("no bbox here", None, label="Text")
    out = verses.detect(_col(nob))                     # must not raise
    assert id(nob) not in out


def test_ref_label_heuristic():
    assert verses.ref_label("24. The aggregates...") == "24"
    assert verses.ref_label("24.18 emptiness") == "24.18"
    assert verses.ref_label("Plain prose, no number") is None


def test_empty_and_degenerate_inputs():
    assert verses.detect([]) == {}
    assert verses.detect([Block("x", [0, 0, 0, 0], label="Text")]) == {}   # zero width


# --------------------------------------------------------------------------- #
# build_doc sidecar invariant + verse de-hyphenation
# --------------------------------------------------------------------------- #
def _pages():
    from ocr_backend import Page
    blocks = [_body("Plain body para.", 100, 700, 100),
              _body("Second body para.", 100, 700, 1000),
              Block("12. broken across\nthe verse li-\nnes here",
                    [300, 400, 520, 520], label="Text")]
    return [Page(None, "page-0001", 800, 1400, blocks, 1)]


CORE_KEYS = {"id", "type", "text", "page", "level", "chapter", "refs",
             "marker", "table", "bbox", "confidence", "needs_review"}
# Optional, feature-gated additive keys the structure layer may attach.
OPT_KEYS = {"continuation", "region", "indent", "centered", "ref_label",
            "line_bbox", "preceded_by_colon", "lead_in_verb", "trailing_source",
            "quote_candidate"}


def test_structure_is_additive_core_unchanged():
    # The structure layer must be ADDITIVE: with it off, blocks carry only core
    # fields and features is empty; with it on, the doc gains schema_version +
    # features and per-block optional keys, but CORE fields of non-structure
    # blocks are byte-identical.
    pages = _pages()
    core = build_doc_dict(pages, detect_verses=False)
    full = build_doc_dict(pages, detect_verses=True)
    assert core["features"] == [] and "verse" in full["features"]
    assert core["schema_version"] == full["schema_version"]
    cb = {b["id"]: b for b in core["blocks"]}
    fb = {b["id"]: b for b in full["blocks"]}
    assert cb.keys() == fb.keys()
    for bid, c in cb.items():
        f = fb[bid]
        if f["type"] in ("verse", "quote"):
            continue                            # structure intentionally refined this
        assert f["type"] == c["type"]           # non-structure blocks unchanged
        assert f["text"] == c["text"]
        assert set(f) <= CORE_KEYS | OPT_KEYS    # only additive optional keys


def test_line_bbox_emitted_on_verse_with_detection():
    pages = _pages()
    pg = pages[0]
    # 3 detection line boxes inside the 3-line verse block ([300,400,520,520])
    det = {Path(pg.name).stem: [[300, 400, 520, 440],
                                [300, 440, 520, 480],
                                [300, 480, 520, 520]]}
    doc = build_doc_dict(pages, det_by_stem=det)
    assert "line_bbox" in doc["features"]
    verse = next(b for b in doc["blocks"] if b["type"] == "verse")
    assert verse.get("line_bbox") and len(verse["line_bbox"]) == 3
    # without detection boxes, no line_bbox (a band-slice is useless for alignment)
    doc2 = build_doc_dict(pages)
    assert "line_bbox" not in doc2["features"]
    assert all("line_bbox" not in b for b in doc2["blocks"])


def test_verse_block_merged_into_doc_keeps_linebreaks():
    pages = _pages()
    doc = build_doc_dict(pages, detect_verses=True)
    verse = next(b for b in doc["blocks"] if b["type"] == "verse")
    assert verse["ref_label"] == "12" and "indent" in verse
    assert "\n" in verse["text"] and "li-\nnes" in verse["text"]   # not de-hyphenated
    # body de-hyphenation still works via textnorm
    assert normalize_block_text("li-\nnes") == "lines"
    assert normalize_block_text("li-\nnes", dehyphenate=False) == "li-\nnes"

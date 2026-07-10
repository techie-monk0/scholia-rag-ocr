"""Tests for the detect-only path's native-text extraction + layout merge.

These cover the pure logic (block-text join, bbox coverage, label merge) without
PyMuPDF on a real PDF or any GPU. The surya_layout pass itself is exercised
end-to-end on the GPU separately.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

import native_text as nt
from ocr_backend import Page, Block


def _pg(blocks):
    return Page(None, "page-0001.png", 200, 300,
                [Block(t, bb, label=l) for t, bb, l in blocks])


def test_coverage_math():
    assert nt._coverage([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert nt._coverage([0, 0, 10, 10], [5, 0, 15, 10]) == 0.5
    assert nt._coverage([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_block_text_joins_lines_drops_blanks():
    b = {"lines": [{"spans": [{"text": "foo "}, {"text": "bar"}]},
                   {"spans": [{"text": "   "}]},          # blank line dropped
                   {"spans": [{"text": "baz"}]}]}
    assert nt._block_text(b) == "foo bar\nbaz"


def test_merge_assigns_region_label():
    pg = _pg([("130 Title", [10, 10, 180, 55], None),       # in the header band
              ("body text here", [10, 80, 180, 280], None)])  # in the text band
    layout = {"page-0001": [([0, 0, 200, 60], "PageHeader"),
                            ([0, 70, 200, 290], "Text")]}
    nt.merge_layout([pg], layout)
    assert pg.blocks[0].label == "PageHeader"
    assert pg.blocks[1].label == "Text"


def test_merge_picks_max_coverage_region():
    # block straddles two regions; the one covering more of it wins
    pg = _pg([("x", [0, 0, 100, 100], None)])
    layout = {"page-0001": [([0, 0, 100, 30], "PageHeader"),   # covers 30%
                            ([0, 20, 100, 100], "Table")]}     # covers 80%
    nt.merge_layout([pg], layout)
    assert pg.blocks[0].label == "Table"


def test_merge_preserves_native_picture():
    pg = _pg([("", [0, 0, 200, 300], "Picture")])             # native image block
    nt.merge_layout([pg], {"page-0001": [([0, 0, 200, 300], "Text")]})
    assert pg.blocks[0].label == "Picture"


def test_merge_defaults_to_text_when_no_region():
    pg = _pg([("orphan", [10, 10, 50, 50], None)])
    _, n = nt.merge_layout([pg], {"page-0001": []})
    assert pg.blocks[0].label == "Text" and n == 0

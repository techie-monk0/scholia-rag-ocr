"""Tests for per-book layout overrides + column-strip re-OCR (lib/layout_overrides,
lib/column_strips)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import layout_overrides as lo
import column_strips as cs
from ocr_backend import Block, Page


# --------------------------------------------------------------------------- #
# layout_overrides
# --------------------------------------------------------------------------- #
def test_manifest_parse_and_lookup(tmp_path):
    m = tmp_path / "ov.txt"
    m.write_text(
        "# comment\n"
        "edition:129 :: two-up\n"
        "edition:655 :: columns=2\n"
        "Mapping the Tibetan World :: columns=3\n"
        "\n"
        "edition:7 :: two-up=auto, columns=2\n"
    )
    e = lo.load(m)
    assert lo.lookup(e, edition=129) == {"two_up": "on"}
    assert lo.lookup(e, edition=655) == {"columns": 2}
    assert lo.lookup(e, edition=7) == {"two_up": "auto", "columns": 2}
    # bare key matches by stem substring (case-insensitive), not by edition
    assert lo.lookup(e, stem="atlas_mapping the tibetan world_v2") == {"columns": 3}
    assert lo.lookup(e, edition=999, stem="something else") == {}


def test_manifest_two_column_alias(tmp_path):
    m = tmp_path / "ov.txt"
    m.write_text("edition:5 :: two-column\n")
    assert lo.lookup(lo.load(m), edition=5) == {"columns": 2}


def test_manifest_drop_pages(tmp_path):
    m = tmp_path / "ov.txt"
    m.write_text("edition:3 :: drop-pages=even\nedition:406 :: drop-pages=odd\n")
    e = lo.load(m)
    assert lo.lookup(e, edition=3) == {"drop_pages": "even"}
    assert lo.lookup(e, edition=406) == {"drop_pages": "odd"}
    bad = tmp_path / "bad.txt"
    bad.write_text("edition:1 :: drop-pages=verso\n")   # only even/odd allowed
    with pytest.raises(SystemExit):
        lo.load(bad)


def test_manifest_bad_directive_raises(tmp_path):
    m = tmp_path / "ov.txt"
    m.write_text("edition:1 :: wobble\n")
    with pytest.raises(SystemExit):
        lo.load(m)


# --------------------------------------------------------------------------- #
# column_strips: gutter detection
# --------------------------------------------------------------------------- #
def _two_col_page(w=600, h=800, gutter=(280, 320)):
    """White page (255) with ink (0) in two text columns and a blank gutter."""
    g = np.full((h, w), 255, np.uint8)
    g[100:700, 40:gutter[0]] = 0          # left column ink
    g[100:700, gutter[1]:w - 40] = 0      # right column ink
    return g


def test_split_at_central_gutter():
    g = _two_col_page()
    cuts = cs.column_split_xs(g, 2)
    assert len(cuts) == 1
    assert 280 <= cuts[0] <= 320          # inside the blank gutter


def test_no_split_without_gutter():
    g = np.full((800, 600), 255, np.uint8)
    g[100:700, 40:560] = 0                # single full-width text block
    assert cs.column_split_xs(g, 2) == []     # no near-empty central channel
    assert cs.column_split_xs(g, 1) == []


def test_no_split_blank_page():
    assert cs.column_split_xs(np.full((800, 600), 255, np.uint8), 2) == []


# --------------------------------------------------------------------------- #
# column_strips: recombine
# --------------------------------------------------------------------------- #
def test_recombine_orders_columns_and_offsets_bbox():
    # two strips of one page: c0 (left, x_off 0), c1 (right, x_off 300)
    left = Page(None, "p1__c0", 300, 800,
                [Block("LEFT-A", [10, 20, 200, 60]),
                 Block("LEFT-B", [10, 100, 200, 140])])
    right = Page(None, "p1__c1", 300, 800,
                 [Block("RIGHT-A", [5, 25, 180, 65])])
    mapping = {"p1__c0": ("p1", 0), "p1__c1": ("p1", 300)}
    pages = cs.recombine([right, left], mapping, [])   # order-insensitive input
    assert len(pages) == 1
    pg = pages[0]
    assert pg.name == "p1"
    # left column fully, then right column
    assert [b.text for b in pg.blocks] == ["LEFT-A", "LEFT-B", "RIGHT-A"]
    # right strip's bbox is shifted by its x_offset (300)
    assert pg.blocks[-1].bbox == [305, 25, 480, 65]

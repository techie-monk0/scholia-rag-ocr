"""Phase 3 — cross-column reading-order reconstruction (lib/reading_order.py).

Executable specs; SKIP until ``lib/reading_order.py`` exists. This is contract
item #1 (scrambled text is fatal) — the canonical failure is a multi-column
glossary mis-ordered by column interleaving (one such page scored 54%).

Expected API (provisional):
    reading_order.order_page(blocks) -> (ordered_blocks, uncertain: bool)
        returns blocks in reading order (columns top→bottom, left→right) and a
        flag set when the layout is ambiguous (-> needs_review on those blocks).
"""
import pytest

ro = pytest.importorskip("reading_order",
                         reason="Phase 3: lib/reading_order.py not built yet")

from conftest import mk_block


def _b(text, x0, y0, x1, y1):
    return mk_block(text, bbox=[x0, y0, x1, y1])


def test_single_column_orders_top_to_bottom():
    blocks = [_b("second", 0, 200, 500, 260), _b("first", 0, 0, 500, 60)]
    ordered, _ = ro.order_page(blocks)
    assert [b.text for b in ordered] == ["first", "second"]


def test_two_columns_not_interleaved():
    # Left column (x 0–480) reads fully before right column (x 520–1000),
    # NOT zig-zagged by y. This is the glossary-interleaving failure.
    blocks = [
        _b("L1", 0, 0, 480, 40), _b("R1", 520, 0, 1000, 40),
        _b("L2", 0, 60, 480, 100), _b("R2", 520, 60, 1000, 100),
    ]
    ordered, _ = ro.order_page(blocks)
    assert [b.text for b in ordered] == ["L1", "L2", "R1", "R2"]


def test_footnote_band_sorts_after_body():
    body = _b("body", 0, 100, 1000, 200)
    foot = mk_block("[^1]note", label="Footnote", bbox=[0, 1300, 1000, 1360])
    ordered, _ = ro.order_page([foot, body])
    assert [b.text for b in ordered] == ["body", "[^1]note"]


def test_ambiguous_overlap_flags_uncertain():
    # Overlapping/indeterminate column boundaries -> uncertain True so the
    # builder can set needs_review on these blocks.
    blocks = [
        _b("A", 0, 0, 600, 40), _b("B", 400, 10, 1000, 50),   # x-overlap
    ]
    _, uncertain = ro.order_page(blocks)
    assert uncertain is True


def test_clean_layout_not_flagged():
    blocks = [_b("a", 0, 0, 1000, 40), _b("b", 0, 60, 1000, 100)]
    _, uncertain = ro.order_page(blocks)
    assert uncertain is False


def _write_img(path, two_col):
    """White page with ink either in two column bands (gutter in the middle) or
    spread across the full width (single column)."""
    import numpy as np
    from PIL import Image
    a = np.full((400, 600), 255, dtype="uint8")
    if two_col:
        a[50:350, 40:260] = 0          # left column
        a[50:350, 340:560] = 0         # right column  (gutter 260-340)
    else:
        a[50:350, 40:560] = 0          # full-width single column
    Image.fromarray(a).save(path)


def test_multicolumn_block_flagged(tmp_path):
    from conftest import mk_block, mk_page
    img = tmp_path / "p.png"
    _write_img(img, two_col=True)
    page = mk_page([mk_block("merged glossary cols", bbox=[0, 0, 600, 400])],
                   width=600, height=400)
    page.image_path = str(img)
    assert ro.multicolumn_indices(page) == {0}


def test_single_column_block_not_flagged(tmp_path):
    from conftest import mk_block, mk_page
    img = tmp_path / "p.png"
    _write_img(img, two_col=False)
    page = mk_page([mk_block("normal paragraph", bbox=[0, 0, 600, 400])],
                   width=600, height=400)
    page.image_path = str(img)
    assert ro.multicolumn_indices(page) == set()

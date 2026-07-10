"""descramble — parse running-head page numbers and reorder into reading order.

Locks the roman/arabic parsing (strict, edge-anchored so title words never look
like numbers), the reorder+dedup plan (roman before arabic, keep first of a
duplicate, anchor unnumbered pages), the monotonic verify, and the end-to-end
unscramble() with a fake OCR (so no engine is needed).
"""
from pathlib import Path

import pytest
from PIL import Image

import descramble as ds


# --- roman_to_int ---------------------------------------------------------- #
@pytest.mark.parametrize("s,n", [("i", 1), ("iv", 4), ("viii", 8), ("x", 10),
                                 ("xvii", 17), ("XL", 40), ("ix", 9)])
def test_roman_valid(s, n):
    assert ds.roman_to_int(s) == n


@pytest.mark.parametrize("s", ["iiii", "vx", "the", "hum", "", "ip", "12"])
def test_roman_invalid(s):
    assert ds.roman_to_int(s) is None


# --- parse_running_head ---------------------------------------------------- #
def test_parse_running_head():
    assert ds.parse_running_head("8 / The Great Hūṃ") == ("arabic", 8)      # verso
    assert ds.parse_running_head("Translator's Introduction / 3") == ("arabic", 3)
    assert ds.parse_running_head("viii / The Great Hūṃ") == ("roman", 8)    # front
    assert ds.parse_running_head("The Great Hūṃ") is None                   # no number
    assert ds.parse_running_head("") is None


def test_parse_running_head_with_trailing_body_text():
    # REAL crops grab the running head AND the first body line, pushing the number
    # out of the string's edges — the number must be found next to the '/'.
    assert ds.parse_running_head(
        "Translator's Introduction / 3 draws in particular from those of") == ("arabic", 3)
    assert ds.parse_running_head(
        "8 / The Great Hūṃ Mañjughoṣa, the bodhisattva of wisdom") == ("arabic", 8)
    assert ds.parse_running_head(
        "viii / The Great Hūṃ CHAPTERS THAT DEVELOP") == ("roman", 8)


# --- plan_descramble ------------------------------------------------------- #
def _lab(kind, v):
    return (kind, v)


def test_plan_orders_roman_then_arabic_and_dedupes():
    # file order scrambled: pairs swapped + a duplicate of page 8
    records = [
        ("f01", _lab("roman", 8)),      # viii
        ("f02", _lab("roman", 10)),     # x
        ("f03", _lab("arabic", 3)),
        ("f04", _lab("arabic", 2)),
        ("f05", _lab("arabic", 5)),
        ("f06", _lab("arabic", 4)),
        ("f07", _lab("arabic", 8)),
        ("f08", _lab("arabic", 8)),     # duplicate of page 8 -> dropped
        ("f09", _lab("arabic", 7)),
    ]
    order, drops = ds.plan_descramble(records)
    # roman (8,10) first, then arabic sorted (2,3,4,5,7,8); dup dropped
    assert order == ["f01", "f02", "f04", "f03", "f06", "f05", "f09", "f07"]
    assert [n for n, _ in drops] == ["f08"]


def test_plan_anchors_unnumbered_pages():
    records = [
        ("cover", None),                # leading unnumbered -> front
        ("f1", _lab("arabic", 1)),
        ("plate", None),                # sticks after its file-order predecessor f1
        ("f2", _lab("arabic", 2)),
    ]
    order, drops = ds.plan_descramble(records)
    assert order == ["cover", "f1", "plate", "f2"]
    assert drops == []


# --- infer_missing --------------------------------------------------------- #
def test_infer_missing_fills_gap_for_openers():
    # reading order: page 16, [opener, no number], page 18 -> opener inferred 17
    order = ["a", "b", "c"]
    label_by = {"a": ("arabic", 16), "b": None, "c": ("arabic", 18)}
    inf = ds.infer_missing(order, label_by)
    assert inf == {"b": ("arabic", 17)}


def test_infer_missing_two_page_gap():
    order = ["a", "b", "c", "d"]
    label_by = {"a": ("arabic", 16), "b": None, "c": None, "d": ("arabic", 19)}
    inf = ds.infer_missing(order, label_by)
    assert inf == {"b": ("arabic", 17), "c": ("arabic", 18)}


def test_infer_missing_no_gap_is_a_plate():
    # contiguous numbers (16,17) with an unnumbered leaf between -> NOT inferred
    order = ["a", "plate", "b"]
    label_by = {"a": ("arabic", 16), "plate": None, "b": ("arabic", 17)}
    assert ds.infer_missing(order, label_by) == {}      # stays unnumbered


def test_infer_missing_leading_unnumbered_not_inferred():
    order = ["cover", "a"]
    label_by = {"cover": None, "a": ("arabic", 1)}
    assert ds.infer_missing(order, label_by) == {}      # front matter -> no guess


# --- flag_suspects --------------------------------------------------------- #
def _recs(*vals):
    return [(f"f{i}", ("arabic", v) if v is not None else None)
            for i, v in enumerate(vals)]


def test_flag_suspects_catches_isolated_misread():
    # a '7' wedged among pages in the 40s -> misread (isolated from all neighbors)
    recs = _recs(44, 45, 46, 7, 48, 49)
    assert ds.flag_suspects(recs) == {"f3"}


def test_flag_suspects_allows_facing_swaps_and_clean_runs():
    assert ds.flag_suspects(_recs(44, 45, 46, 47, 48)) == set()
    assert ds.flag_suspects(_recs(40, 42, 41, 43, 45, 44)) == set()   # swaps ok


def test_flag_suspects_allows_genuine_gap():
    # a real gap (missing scans) 45 -> 56: page 56 is supported by its far-side
    # neighbors (57, 58), so it is NOT flagged
    assert ds.flag_suspects(_recs(44, 45, 56, 57, 58)) == set()


# --- verify_increasing ----------------------------------------------------- #
def test_verify_increasing():
    assert ds.verify_increasing([("arabic", 1), ("arabic", 2), ("arabic", 5)]) == []
    bad = ds.verify_increasing([("arabic", 3), ("arabic", 2)])
    assert len(bad) == 1


# --- unscramble (end-to-end with a fake OCR) ------------------------------- #
def test_unscramble_reorders_and_drops(tmp_path):
    src = tmp_path / "pages"
    src.mkdir()
    # 6 files whose running heads (faked below) are scrambled with one duplicate
    heads = {
        "p1.jpg": "Translator's Introduction / 3",
        "p2.jpg": "2 / The Great Hūṃ",
        "p3.jpg": "Translator's Introduction / 5",
        "p4.jpg": "4 / The Great Hūṃ",
        "p5.jpg": "4 / The Great Hūṃ",           # duplicate of page 4
        "p6.jpg": "6 / The Great Hūṃ",
    }
    for n in heads:
        Image.new("RGB", (40, 60), "white").save(src / n)
    import surya_backend as sb
    images = sb.list_images(src)

    def fake_ocr(strip_imgs, strips_dir):        # inject: map cropped strip -> head
        return {Path(s).name: heads[Path(s).name] for s in strip_imgs}

    final_dir, final_imgs = ds.unscramble(src, images, ocr_strips=fake_ocr)
    assert final_dir == src / "_reordered"
    # manifest lists pages 2,3,4,5,6 in order (dup dropped)
    import json
    man = json.loads((final_dir / "manifest.json").read_text())
    assert [p["page"] for p in man["pages"]] == ["2", "3", "4", "5", "6"]
    assert man["dropped"] == 1
    assert (final_dir / "_dropped" / "p5.jpg").exists()
    assert len(final_imgs) == 5


def test_unscramble_noop_when_no_numbers(tmp_path):
    src = tmp_path / "pages"
    src.mkdir()
    for n in ("a.jpg", "b.jpg"):
        Image.new("RGB", (40, 60), "white").save(src / n)
    import surya_backend as sb
    images = sb.list_images(src)
    final_dir, final_imgs = ds.unscramble(
        src, images, ocr_strips=lambda si, sd: {Path(s).name: "" for s in si})
    assert final_dir == src                       # unchanged — no numbers read
    assert len(final_imgs) == 2

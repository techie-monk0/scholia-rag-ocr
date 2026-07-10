"""preprocess — the shared rotate -> swap -> pad chain, the --pages window, and
the arg plumbing.

Both run_ocr.py and ocr_one.py drive image preprocessing through this module
(they forward the tokens after --preprocess to preprocess.parse), so these lock
the page-window grammar, chaining/order, the no-op case, window scoping
(rotate/swap only touch pages in --pages), and the helpers (parse, validate,
has_any, label, step_order).
"""
import argparse

import pytest
from PIL import Image

import preprocess
import surya_backend as sb


def _ns(**kw):
    base = {"pad_page_numbers": False, "pages": None, "rotate": None,
            "swap": None, "auto_unscramble_pages": False}
    base.update(kw)
    return argparse.Namespace(**base)


# --- parse_pages ----------------------------------------------------------- #
def test_parse_pages_valid_and_open_ended():
    assert preprocess.parse_pages("9:20") == (9, 20)
    assert preprocess.parse_pages("  9 : 200 ") == (9, 200)
    assert preprocess.parse_pages("9:9") == (9, 9)
    assert preprocess.parse_pages("9:end") == (9, None)    # to the last page
    assert preprocess.parse_pages("9:END") == (9, None)    # case-insensitive
    assert preprocess.parse_pages(None) is None
    assert preprocess.parse_pages((3, 8)) == (3, 8)        # pre-parsed tuple


@pytest.mark.parametrize("bad", ["9", "9:", ":20", "a:b", "9-20", "9:20:30",
                                 "0:5", "20:9"])
def test_parse_pages_rejected(bad):
    # a bare number / trailing colon is NOT a range now — the colon+end is required
    with pytest.raises(SystemExit):
        preprocess.parse_pages(bad)


def test_normalize_argv_swap_only_eats_a_range():
    n = preprocess.normalize_argv
    # a colon-range following --swap is absorbed; anything else is left positional
    assert n(["--swap", "9:20"]) == ["--swap=9:20"]
    assert n(["--swap", "9:end"]) == ["--swap=9:end"]
    assert n(["--swap", "9"]) == ["--swap=", "9"]          # bare number = folder now
    assert n(["--swap", "D"]) == ["--swap=", "D"]          # folder path, not a range
    assert n(["--swap", "D", "--rotate", "x"]) == ["--swap=", "D", "--rotate", "x"]
    assert n(["--swap"]) == ["--swap"]                     # bare at end
    assert n(["--swap", "--rotate", "x"]) == ["--swap", "--rotate", "x"]  # next is opt
    assert n(["D", "--swap"]) == ["D", "--swap"]           # other tokens untouched


def test_normalize_argv_then_parse_resolves_ambiguity():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    preprocess.add_preprocess_args(ap)
    a = ap.parse_args(preprocess.normalize_argv(["--swap", "scans"]))
    assert a.folders == ["scans"] and a.swap is True       # bare, folder preserved
    # a folder literally named like a number is safe now (no colon = not a range)
    b = ap.parse_args(preprocess.normalize_argv(["--swap", "9"]))
    assert b.folders == ["9"] and b.swap is True
    c = ap.parse_args(preprocess.normalize_argv(["--swap", "9:20", "scans"]))
    assert c.folders == ["scans"] and c.swap == "9:20"


def test_split_rotate():
    assert preprocess.split_rotate("180:CW") == ("180:CW", None)
    assert preprocess.split_rotate("180:CW@9:end") == ("180:CW", "9:end")
    assert preprocess.split_rotate("90:CCW@9:20") == ("90:CCW", "9:20")
    assert preprocess.split_rotate(None) == (None, None)


# --- arg plumbing ---------------------------------------------------------- #
def test_add_preprocess_args_registers_all():
    ap = argparse.ArgumentParser()
    preprocess.add_preprocess_args(ap)
    args = ap.parse_args(["--pages", "9:20", "--rotate", "180:CW", "--swap"])
    assert args.pad_page_numbers is True             # ON by default
    assert args.pages == "9:20"
    assert args.rotate == "180:CW"
    assert args.swap is True


def test_no_pad_page_numbers_turns_padding_off():
    ap = argparse.ArgumentParser()
    preprocess.add_preprocess_args(ap)
    assert ap.parse_args([]).pad_page_numbers is True                 # default on
    assert ap.parse_args(["--no-pad-page-numbers"]).pad_page_numbers is False


def test_step_order_follows_command_line():
    ap = argparse.ArgumentParser()
    preprocess.add_preprocess_args(ap)
    a = ap.parse_args(["--swap", "--rotate", "180:CW"])
    assert preprocess.step_order(a) == ["swap", "rotate"]
    assert preprocess.label(a) == " swap rotate-180:CW"

    b = ap.parse_args(["--rotate", "180:CW", "--swap"])
    assert preprocess.step_order(b) == ["rotate", "swap"]
    assert preprocess.label(b) == " rotate-180:CW swap"


def test_split_forward():
    s = preprocess.split_forward
    assert s(["a.pdf", "--scanned-no-text"]) == (["a.pdf", "--scanned-no-text"], [])
    assert s(["a.pdf", "--", "--rotate", "180:CW"]) == (["a.pdf"], ["--rotate", "180:CW"])
    assert s(["a.pdf", "--preprocess", "--swap", "9:end"]) == (["a.pdf"], ["--swap", "9:end"])
    # tolerate '-- --preprocess …'
    assert s(["a.pdf", "--", "--preprocess", "--swap"]) == (["a.pdf"], ["--swap"])
    # marker with nothing after -> empty tail
    assert s(["a.pdf", "--"]) == (["a.pdf"], [])


def test_parse_forwards_and_validates():
    # parse() consumes the tokens after --preprocess (no folder), validating them
    assert preprocess.parse([]) is None
    ns = preprocess.parse(["--pages", "9:20", "--swap", "9:end", "--rotate", "180:CW"])
    assert ns.pad_page_numbers and ns.pages == "9:20"    # padding on by default
    assert ns.swap == "9:end" and ns.rotate == "180:CW"
    assert preprocess.step_order(ns) == ["swap", "rotate"]   # CLI order preserved
    with pytest.raises(SystemExit):
        preprocess.parse(["--pages", "9"])                   # bare number not a range


def test_has_any_and_label():
    # padding is real work; --pages alone (padding off) is a no-op
    assert not preprocess.has_any(_ns(pages="9:20"))       # _ns() defaults padding off
    assert preprocess.has_any(_ns(pad_page_numbers=True))  # padding on = has work
    # label: padding on (the default) adds nothing; padding OFF shows 'no-pad'
    assert preprocess.label(_ns(pad_page_numbers=True)) == ""
    assert preprocess.label(_ns()) == " no-pad"

    a = _ns(pad_page_numbers=True, pages="9:20", rotate="180:CW", swap=True)
    assert preprocess.has_any(a)
    assert preprocess.label(a) == " pages-9:20 rotate-180:CW swap"


def test_validate_rejects_bad_specs():
    with pytest.raises(SystemExit):
        preprocess.validate(_ns(rotate="200:CW"))
    with pytest.raises(SystemExit):
        preprocess.validate(_ns(pages="20:9"))
    preprocess.validate(_ns())                             # all-default is fine


# --- the chain ------------------------------------------------------------- #
def test_preprocess_images_noop_returns_same(tmp_path):
    src = tmp_path / "pages"
    src.mkdir()
    Image.new("RGB", (10, 20), "white").save(src / "Great hum - 1.jpg")
    imgs = sb.list_images(src)
    final_dir, final_imgs = preprocess.preprocess_images(src, imgs)
    assert final_dir == src and [p.name for p in final_imgs] == ["Great hum - 1.jpg"]


def test_preprocess_images_chains_pad_rotate_swap(tmp_path):
    src = tmp_path / "pages"
    src.mkdir()
    for n in range(9, 13):                                 # pages 9..12 -> width 2
        Image.new("L", (20, 10), 0).save(src / f"Great hum - {n}.jpg")
    imgs = sb.list_images(src)

    final_dir, final_imgs = preprocess.preprocess_images(
        src, imgs, pad=True, pages="9:12", rotate="90:CW", swap=True)

    # Padding runs FIRST, so the chained dir order is _renumbered -> _rotated -> _swapped
    assert final_dir == src / "_renumbered" / "_rotated" / "_swapped"
    assert [p.name for p in final_imgs] == [
        "Great hum - 09.jpg", "Great hum - 10.jpg",        # width from largest (12)
        "Great hum - 11.jpg", "Great hum - 12.jpg"]
    assert Image.open(final_imgs[0]).size == (10, 20)      # 90° rotate: 20x10 -> 10x20


def test_preprocess_images_runs_steps_in_given_order(tmp_path):
    """The chained dir nesting reflects execution order; --pad (if any) is first."""
    src = tmp_path / "pages"
    src.mkdir()
    for n in (1, 2, 3, 4):
        Image.new("L", (20, 10), 0).save(src / f"p{n}.jpg")
    imgs = sb.list_images(src)

    fd, _ = preprocess.preprocess_images(
        src, imgs, rotate="90:CW", swap=True, order=["swap", "rotate"])
    assert fd == src / "_swapped" / "_rotated"            # swap ran first

    fd2, _ = preprocess.preprocess_images(
        src, imgs, rotate="90:CW", swap=True, order=["rotate", "swap"])
    assert fd2 == src / "_rotated" / "_swapped"           # rotate ran first


def test_marker_records_final_dir_and_noop_clears_it(tmp_path):
    src = tmp_path / "pages"
    src.mkdir()
    for n in (1, 2, 3):
        Image.new("L", (20, 10), 0).save(src / f"p{n}.jpg")
    imgs = sb.list_images(src)

    final, _ = preprocess.preprocess_images(src, imgs, rotate="90:CW@2:3")
    assert final == src / "_rotated"
    assert preprocess.final_dir(src) == src / "_rotated"      # marker points at it
    # a no-op run clears the marker (final == source)
    preprocess.preprocess_images(src, sb.list_images(src))
    assert preprocess.final_dir(src) == src                   # falls back to source


def test_force_rebuilds_after_range_change(tmp_path):
    """Changing a rotate range with --force re-rotates correctly (no stale cache)."""
    src = tmp_path / "pages"
    src.mkdir()
    for n in (1, 2, 3):
        Image.new("L", (20, 10), 0).save(src / f"p{n}.jpg")   # landscape 20x10

    preprocess.preprocess_images(src, sb.list_images(src), rotate="90:CW@2:3")
    # now rotate ONLY page 1, forcing a clean rebuild
    final, imgs = preprocess.preprocess_images(
        src, sb.list_images(src), rotate="90:CW@1:1", force=True)
    size = {p.name: Image.open(p).size for p in imgs}
    assert size["p1.jpg"] == (10, 20)                         # rotated now
    assert size["p2.jpg"] == (20, 10) and size["p3.jpg"] == (20, 10)  # NOT rotated


def test_clean_step_dirs_removes_everything(tmp_path):
    src = tmp_path / "pages"
    src.mkdir()
    Image.new("L", (20, 10), 0).save(src / "p1.jpg")
    preprocess.preprocess_images(src, sb.list_images(src), pad=True, rotate="90:CW@1:1")
    assert (src / "_renumbered").is_dir()
    preprocess.clean_step_dirs(src)
    assert not any(p.is_dir() for p in src.iterdir())        # subdirs gone
    assert not (src / preprocess._MARKER).exists()           # marker gone


def test_preprocess_images_auto_unscramble_step(tmp_path):
    """--auto-unscramble-pages runs as the last preprocess step (fake reader)."""
    src = tmp_path / "pages"
    src.mkdir()
    heads = {"p1.jpg": "3 / T", "p2.jpg": "2 / T", "p3.jpg": "4 / T"}
    for n in heads:
        Image.new("RGB", (40, 60), "white").save(src / n)
    imgs = sb.list_images(src)
    fake = lambda si, sd: {p.name: heads[p.name] for p in si}
    final, fimgs = preprocess.preprocess_images(
        src, imgs, auto_unscramble=True, ocr_strips=fake)
    assert final == src / "_reordered"
    import json
    man = json.loads((final / "manifest.json").read_text())
    assert [p["page"] for p in man["pages"]] == ["2", "3", "4"]   # reordered


def test_preprocess_images_pad_width_spans_all_pages(tmp_path):
    """Auto width comes from the whole folder's largest page, even when only a
    subset of images is processed (a --limit spot-check)."""
    src = tmp_path / "pages"
    src.mkdir()
    for n in range(1, 121):                               # pages 1..120 -> width 3
        (src / f"p{n}.jpg").write_bytes(b"x")
    limited = sb.list_images(src)[:5]                     # p1..p5 only

    final_dir, final_imgs = preprocess.preprocess_images(src, limited, pad=True)
    assert [p.name for p in final_imgs] == [
        "p001.jpg", "p002.jpg", "p003.jpg", "p004.jpg", "p005.jpg"]  # 3-digit


def test_preprocess_images_pages_window_scopes_rotate(tmp_path):
    """--rotate only touches pages inside --pages; others pass through unchanged."""
    src = tmp_path / "pages"
    src.mkdir()
    for n in (1, 2, 3):
        Image.new("L", (20, 10), 0).save(src / f"p{n}.jpg")   # landscape 20x10
    imgs = sb.list_images(src)

    final_dir, final_imgs = preprocess.preprocess_images(
        src, imgs, pages="2:end", rotate="90:CW")             # rotate pages 2,3 only
    by = {p.name: Image.open(p).size for p in final_imgs}
    assert by["p1.jpg"] == (20, 10)                            # outside window: as-is
    assert by["p2.jpg"] == (10, 20) and by["p3.jpg"] == (10, 20)  # rotated


def test_swap_own_range_overrides_pages(tmp_path):
    """--swap 5:end swaps within 5.., ignoring the global --pages window."""
    src = tmp_path / "pages"
    src.mkdir()
    for n in range(1, 9):
        (src / f"p{n}.jpg").write_bytes(f"PAGE-{n}".encode())
    imgs = sb.list_images(src)
    final_dir, _ = preprocess.preprocess_images(src, imgs, pages="2:end", swap="5:end")
    r = lambda n: (final_dir / f"p{n}.jpg").read_bytes()
    assert r(2) == b"PAGE-2" and r(3) == b"PAGE-3"        # below swap's own range
    assert r(5) == b"PAGE-6" and r(6) == b"PAGE-5"        # 5<->6
    assert r(7) == b"PAGE-8" and r(8) == b"PAGE-7"        # 7<->8


def test_rotate_own_range_overrides_pages(tmp_path):
    """--rotate …@2:3 rotates only pages 2-3, ignoring the global --pages window."""
    src = tmp_path / "pages"
    src.mkdir()
    for n in (1, 2, 3, 4):
        Image.new("L", (20, 10), 0).save(src / f"p{n}.jpg")   # landscape 20x10
    imgs = sb.list_images(src)
    final_dir, final_imgs = preprocess.preprocess_images(
        src, imgs, pages="5:end", rotate="90:CW@2:3")
    size = {p.name: Image.open(p).size for p in final_imgs}
    assert size["p1.jpg"] == (20, 10) and size["p4.jpg"] == (20, 10)   # untouched
    assert size["p2.jpg"] == (10, 20) and size["p3.jpg"] == (10, 20)   # rotated


def test_preprocess_images_swap_defaults_to_whole_book(tmp_path):
    """--swap without --pages swaps from page 1 through the last."""
    src = tmp_path / "pages"
    src.mkdir()
    for n in range(1, 5):
        (src / f"p{n}.jpg").write_bytes(f"PAGE-{n}".encode())
    imgs = sb.list_images(src)
    final_dir, _ = preprocess.preprocess_images(src, imgs, swap=True)
    assert (final_dir / "p1.jpg").read_bytes() == b"PAGE-2"   # 1<->2
    assert (final_dir / "p3.jpg").read_bytes() == b"PAGE-4"   # 3<->4

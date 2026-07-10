"""swap_pages — pairwise page reordering (the --swap step, scoped by --pages).

For scans whose facing pairs were captured out of order. These lock the pairing
plan (which file gets which page's content) and that make_swapped preserves
filenames (page numbering keys off the stem) and is incremental. The range
``(start, end)`` is supplied by the caller; its grammar is parsed and tested in
test_preprocess.py (preprocess.parse_pages).
"""
import pytest

import swap_pages as sp


# --- swap_plan ------------------------------------------------------------- #
def _names(n_lo, n_hi, prefix="Great hum - "):
    from pathlib import Path
    return [Path(f"{prefix}{n}.jpg") for n in range(n_lo, n_hi + 1)]


def test_swap_plan_pairs_within_range():
    imgs = _names(1, 14)
    plan = sp.swap_plan(imgs, 9, 14)
    by = {p.name: plan[p].name for p in imgs}
    # outside the range: untouched
    for n in list(range(1, 9)):
        assert by[f"Great hum - {n}.jpg"] == f"Great hum - {n}.jpg"
    # inside: consecutive pairs trade content
    assert by["Great hum - 9.jpg"] == "Great hum - 10.jpg"
    assert by["Great hum - 10.jpg"] == "Great hum - 9.jpg"
    assert by["Great hum - 13.jpg"] == "Great hum - 14.jpg"
    assert by["Great hum - 14.jpg"] == "Great hum - 13.jpg"


def test_swap_plan_end_none_runs_to_last_page():
    imgs = _names(1, 12)
    plan = sp.swap_plan(imgs, 9, None)             # 9<->10, 11<->12; 1..8 untouched
    by = {p.name: plan[p].name for p in imgs}
    assert by["Great hum - 8.jpg"] == "Great hum - 8.jpg"
    assert by["Great hum - 9.jpg"] == "Great hum - 10.jpg"
    assert by["Great hum - 11.jpg"] == "Great hum - 12.jpg"


def test_swap_plan_odd_tail_left_alone():
    imgs = _names(9, 13)
    plan = sp.swap_plan(imgs, 9, 13)               # 9<->10, 11<->12, 13 unpaired
    by = {p.name: plan[p].name for p in imgs}
    assert by["Great hum - 13.jpg"] == "Great hum - 13.jpg"


def test_swap_plan_missing_partner_skips_pair():
    from pathlib import Path
    imgs = [Path("p9.jpg"), Path("p11.jpg"), Path("p12.jpg")]  # no p10
    plan = sp.swap_plan(imgs, 9, 12)
    assert plan[Path("p9.jpg")] == Path("p9.jpg")             # 9's partner absent
    assert plan[Path("p11.jpg")] == Path("p12.jpg")           # 11<->12 still swaps
    assert plan[Path("p12.jpg")] == Path("p11.jpg")


# --- make_swapped ---------------------------------------------------------- #
def test_make_swapped_moves_content_keeps_names(tmp_path):
    src = tmp_path / "pages"
    src.mkdir()
    for n in range(8, 14):
        (src / f"Great hum - {n}.jpg").write_text(f"PAGE-{n}")
    imgs = [src / f"Great hum - {n}.jpg" for n in range(8, 14)]

    out_dir = sp.make_swapped(imgs, src / "_swapped", 9, 12)
    names = sorted(p.name for p in out_dir.iterdir() if p.suffix == ".jpg")
    assert names == sorted(p.name for p in imgs)             # same filenames
    read = lambda n: (out_dir / f"Great hum - {n}.jpg").read_text()
    assert read(8) == "PAGE-8"                               # outside range
    assert read(9) == "PAGE-10" and read(10) == "PAGE-9"     # swapped
    assert read(11) == "PAGE-12" and read(12) == "PAGE-11"   # swapped
    assert read(13) == "PAGE-13"                             # outside range


def test_make_swapped_is_incremental(tmp_path):
    src = tmp_path / "pages"
    src.mkdir()
    for n in (9, 10):
        (src / f"p{n}.jpg").write_text(f"PAGE-{n}")
    imgs = [src / f"p{n}.jpg" for n in (9, 10)]

    out_dir = sp.make_swapped(imgs, src / "_swapped", 9, 10)
    mtime = (out_dir / "p9.jpg").stat().st_mtime_ns
    sp.make_swapped(imgs, out_dir, 9, 10)                    # up-to-date -> untouched
    assert (out_dir / "p9.jpg").stat().st_mtime_ns == mtime
    sp.make_swapped(imgs, out_dir, 9, 10, force=True)        # force rewrites
    assert (out_dir / "p9.jpg").read_text() == "PAGE-10"     # still correct content

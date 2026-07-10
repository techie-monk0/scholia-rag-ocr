"""pad_pages — zero-pad page numbers in image filenames to a uniform width.

Locks auto_width (digits of the largest page number), the name rewriting (no
truncation of wider numbers, files without a trailing number untouched), and that
make_padded copies content, is incremental, and rejects a width that collides two
pages onto one name.
"""
from pathlib import Path

import pytest

import pad_pages as pp


# --- auto_width ------------------------------------------------------------ #
def test_auto_width_from_largest_page():
    imgs = [Path(f"Great hum - {n}.jpg") for n in (1, 9, 10, 976)]
    assert pp.auto_width(imgs) == 3                        # 976 -> 3 digits
    assert pp.auto_width([Path("p1.jpg"), Path("p9.jpg")]) == 1
    assert pp.auto_width([Path("cover.jpg")]) == 1         # no numbers -> 1


# --- padded_name ----------------------------------------------------------- #
def test_padded_name_pads_to_width():
    assert pp.padded_name(Path("Great hum - 1.jpg"), 3) == "Great hum - 001.jpg"
    assert pp.padded_name(Path("p7.png"), 4) == "p0007.png"


def test_padded_name_never_truncates_wider_numbers():
    assert pp.padded_name(Path("Great hum - 1000.jpg"), 3) == "Great hum - 1000.jpg"


def test_padded_name_no_trailing_number_unchanged():
    assert pp.padded_name(Path("cover.jpg"), 3) == "cover.jpg"


# --- make_padded ----------------------------------------------------------- #
def test_make_padded_copies_content_under_padded_name(tmp_path):
    src = tmp_path / "pages"
    src.mkdir()
    for n in (1, 10, 100):
        (src / f"Great hum - {n}.jpg").write_text(f"PAGE-{n}")
    imgs = [src / f"Great hum - {n}.jpg" for n in (1, 10, 100)]

    out = pp.make_padded(imgs, src / "_renumbered", 3)
    names = sorted(p.name for p in out.iterdir())
    assert names == ["Great hum - 001.jpg", "Great hum - 010.jpg",
                     "Great hum - 100.jpg"]
    assert (out / "Great hum - 001.jpg").read_text() == "PAGE-1"   # content preserved


def test_make_padded_is_incremental(tmp_path):
    src = tmp_path / "pages"
    src.mkdir()
    (src / "p1.jpg").write_text("PAGE-1")
    imgs = [src / "p1.jpg"]
    out = pp.make_padded(imgs, src / "_renumbered", 3)
    mtime = (out / "p001.jpg").stat().st_mtime_ns
    pp.make_padded(imgs, out, 3)                          # up-to-date -> untouched
    assert (out / "p001.jpg").stat().st_mtime_ns == mtime


def test_make_padded_rejects_width_below_one(tmp_path):
    src = tmp_path / "pages"
    src.mkdir()
    (src / "p1.jpg").write_text("x")
    with pytest.raises(SystemExit):
        pp.make_padded([src / "p1.jpg"], src / "_renumbered", 0)


def test_make_padded_rejects_collision(tmp_path):
    src = tmp_path / "pages"
    src.mkdir()
    (src / "p1.jpg").write_text("a")
    (src / "p01.jpg").write_text("b")                    # both -> p001.jpg at N=3
    with pytest.raises(SystemExit):
        pp.make_padded([src / "p1.jpg", src / "p01.jpg"], src / "_renumbered", 3)

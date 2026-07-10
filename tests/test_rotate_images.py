"""rotate_images — --rotate D:CW/CCW spec parsing + fixed-angle page rotation.

Surya expects upright text; a rotated scan OCRs to garbage. These lock the spec
grammar (bounds, CW/CCW) and the rotation geometry/direction, and that
make_rotated preserves filenames (page numbering keys off the stem) and is
incremental.
"""
import numpy as np
import pytest
from PIL import Image

import rotate_images as ri


# --- parse_rotate ---------------------------------------------------------- #
def test_parse_rotate_valid():
    assert ri.parse_rotate("180:CW") == (180, True)
    assert ri.parse_rotate("90:CCW") == (90, False)
    assert ri.parse_rotate("  5 : cw ") == (5, True)      # whitespace + case


def test_parse_rotate_none_is_noop():
    assert ri.parse_rotate(None) is None
    assert ri.parse_rotate("") is None
    assert ri.parse_rotate("0:CW") is None                # 0 degrees == no rotation


@pytest.mark.parametrize("bad", ["200:CW", "181:CCW"])
def test_parse_rotate_over_180_rejected(bad):
    with pytest.raises(SystemExit):
        ri.parse_rotate(bad)


@pytest.mark.parametrize("bad", ["90", "90:LEFT", "abc", "cw:90", "-90:CW"])
def test_parse_rotate_malformed_rejected(bad):
    with pytest.raises(SystemExit):
        ri.parse_rotate(bad)


# --- rotate_image ---------------------------------------------------------- #
def test_rotate_image_expands_canvas():
    im = Image.new("RGB", (100, 40), "white")
    assert ri.rotate_image(im, 90, True).size == (40, 100)
    assert ri.rotate_image(im, 180, False).size == (100, 40)   # 180 keeps dims


def test_rotate_image_direction():
    # Top row black: 90 CW sends the top edge to the RIGHT column, CCW to the LEFT.
    a = np.full((4, 6), 255, np.uint8)
    a[0, :] = 0
    im = Image.fromarray(a, "L")
    cw = np.asarray(ri.rotate_image(im, 90, True), "L")
    ccw = np.asarray(ri.rotate_image(im, 90, False), "L")
    assert (cw[:, -1] == 0).all() and not (cw[:, 0] == 0).all()
    assert (ccw[:, 0] == 0).all() and not (ccw[:, -1] == 0).all()


def test_rotate_image_180_is_point_symmetric():
    a = np.arange(24, dtype=np.uint8).reshape(4, 6)
    out = np.asarray(ri.rotate_image(Image.fromarray(a, "L"), 180, True), "L")
    assert np.array_equal(out, a[::-1, ::-1])


# --- make_rotated ---------------------------------------------------------- #
def test_make_rotated_preserves_names_and_is_incremental(tmp_path):
    src = tmp_path / "pages"
    src.mkdir()
    names = ["p001.png", "p002.png"]
    for n in names:
        Image.new("L", (20, 10), 255).save(src / n)
    imgs = [src / n for n in names]

    rot_dir = ri.make_rotated(imgs, src / "_rotated", 90, True)
    out = sorted(p.name for p in rot_dir.iterdir() if p.suffix == ".png")
    assert out == names                                   # same filenames
    assert Image.open(rot_dir / "p001.png").size == (10, 20)   # rotated 90

    # Incremental: an up-to-date copy is left untouched; --force rewrites it.
    mtime = (rot_dir / "p001.png").stat().st_mtime_ns
    ri.make_rotated(imgs, rot_dir, 90, True)
    assert (rot_dir / "p001.png").stat().st_mtime_ns == mtime

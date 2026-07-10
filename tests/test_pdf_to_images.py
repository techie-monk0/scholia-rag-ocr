"""Two-up split detection + rasterization (lib/pdf_to_images.py)."""
import numpy as np
import fitz

from pdf_to_images import two_up_split_x, pdf_to_images


def _blank(h, w):
    return np.full((h, w), 255.0)


def test_split_detects_central_gutter():
    a = _blank(600, 1000)               # landscape sheet
    a[50:550, 50:460] = 0               # left page
    a[50:550, 540:950] = 0              # right page (gutter 460-540)
    x = two_up_split_x(a)
    assert x is not None and 460 <= x <= 540


def test_portrait_not_split():
    a = _blank(1000, 600)               # portrait
    a[50:950, 50:550] = 0
    assert two_up_split_x(a) is None


def test_landscape_fullbleed_not_split():
    # landscape but ink spans the centre (a wide single page / full-bleed figure)
    a = _blank(600, 1000)
    a[50:550, 50:950] = 0
    assert two_up_split_x(a) is None


def _pdf(tmp_path, name, w, h, rects):
    doc = fitz.open()
    pg = doc.new_page(width=w, height=h)
    for r in rects:
        pg.draw_rect(fitz.Rect(*r), fill=(0, 0, 0))
    p = tmp_path / name
    doc.save(p)
    return p


def test_two_up_pdf_splits_into_left_right(tmp_path):
    # landscape page with two solid text columns separated by a gutter
    pdf = _pdf(tmp_path, "twoup.pdf", 1000, 600,
               [(60, 40, 460, 560), (560, 40, 940, 560)])
    out = pdf_to_images(pdf, tmp_path / "a", dpi=150, two_up="auto", quiet=True)
    names = sorted(p.name for p in out.glob("*.png"))
    assert names == ["page-0001a.png", "page-0001b.png"]


def test_portrait_pdf_not_split(tmp_path):
    pdf = _pdf(tmp_path, "port.pdf", 600, 800, [(60, 40, 540, 760)])
    out = pdf_to_images(pdf, tmp_path / "b", dpi=150, two_up="auto", quiet=True)
    names = sorted(p.name for p in out.glob("*.png"))
    assert names == ["page-0001.png"]


def test_two_up_off_keeps_landscape_whole(tmp_path):
    pdf = _pdf(tmp_path, "twoup2.pdf", 1000, 600,
               [(60, 40, 460, 560), (560, 40, 940, 560)])
    out = pdf_to_images(pdf, tmp_path / "c", dpi=150, two_up="off", quiet=True)
    names = sorted(p.name for p in out.glob("*.png"))
    assert names == ["page-0001.png"]

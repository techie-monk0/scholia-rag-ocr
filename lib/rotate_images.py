"""Rotate page images by a fixed angle before OCR (``--rotate D:CW/CCW``).

Surya's detection/recognition models expect upright text; a scan fed in rotated
(commonly 180°, i.e. upside-down) comes back as garbage or near-empty text — and
180° in particular can slip past the health gates as low-quality-looking text
rather than a clean failure. This rotates every page a fixed amount up front, so
the whole downstream pipeline (OCR, column strips, line detection, and the
emitted searchable PDF) works on — and outputs — the corrected orientation.

Pipeline:
    rot_dir = make_rotated(images, image_dir / "_rotated", degrees, cw)
    image_dir, images = rot_dir, sb.list_images(rot_dir)

Filenames are preserved, so page numbering (keyed by filename stem) is unchanged.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_NUM_RE = re.compile(r"(\d+)\s*$")          # trailing integer of the filename stem


def _page_no(path):
    """Trailing page number in a filename stem, or ``None`` if it has none."""
    m = _NUM_RE.search(Path(path).stem)
    return int(m.group(1)) if m else None


def _in_window(n, start, end):
    """Is page number ``n`` inside the ``[start, end]`` window (``None`` bound =
    open on that side)? A page with no number (``n is None``) is never in-window."""
    return (n is not None
            and (start is None or n >= start)
            and (end is None or n <= end))


def parse_rotate(spec):
    """``'D:CW'`` / ``'D:CCW'`` -> ``(degrees, clockwise)`` with ``0 <= D <= 180``.

    Returns ``None`` for a falsy spec or ``D == 0`` (both mean "no rotation").
    """
    if not spec:
        return None
    m = re.fullmatch(r"\s*(\d{1,3})\s*:\s*(CW|CCW)\s*", spec, re.I)
    if not m:
        raise SystemExit(
            f"--rotate must be D:CW or D:CCW (e.g. 180:CW, 90:CCW); got {spec!r}")
    degrees = int(m.group(1))
    if degrees > 180:
        raise SystemExit(f"--rotate degrees must be <= 180; got {degrees}")
    if degrees == 0:
        return None
    return degrees, m.group(2).upper() == "CW"


def _fill(im):
    """White fill for the triangular corners a non-right-angle rotation exposes
    (a scan page is black-on-white; black corners would read as ink)."""
    mode = im.mode
    if mode in ("1", "L", "I", "F"):
        return 255
    if mode == "LA":
        return (255, 255)
    if mode == "RGBA":
        return (255, 255, 255, 255)
    return (255, 255, 255)


def rotate_image(im, degrees: int, cw: bool):
    """Rotate a PIL image ``degrees`` clockwise (``cw``) or counter-clockwise.

    PIL's ``rotate`` is counter-clockwise-positive, so a clockwise turn is a
    negative angle. ``expand=True`` grows the canvas so nothing is cropped.
    Right-angle turns map pixels exactly (NEAREST, no blur); other angles use
    BICUBIC on an RGB copy with white-filled corners.
    """
    from PIL import Image
    angle = -degrees if cw else degrees          # CCW-positive
    if angle % 90 == 0:
        return im.rotate(angle, expand=True, resample=Image.NEAREST)
    if im.mode == "P":                            # palette + interpolation -> RGB
        im = im.convert("RGB")
    return im.rotate(angle, expand=True, resample=Image.BICUBIC,
                     fillcolor=_fill(im))


def make_rotated(images, rot_dir: Path, degrees: int, cw: bool, *, force=False,
                 pages=None):
    """Write each image in ``images`` rotated into ``rot_dir`` under the SAME
    filename. Returns ``rot_dir``. Incremental: skips a page whose rotated copy is
    already current (present and no older than the source).

    ``pages=(start, end)`` limits rotation to page numbers in the ``[start, end]``
    window (``end=None`` = through the last page); pages OUTSIDE the window — and
    any file without a page number — are copied through unchanged, so ``rot_dir``
    always holds the full set. ``pages=None`` rotates every page."""
    from PIL import Image
    rot_dir = Path(rot_dir)
    rot_dir.mkdir(parents=True, exist_ok=True)
    start, end = pages if pages else (None, None)
    for img in images:
        img = Path(img)
        out = rot_dir / img.name
        if (not force and out.exists()
                and out.stat().st_mtime >= img.stat().st_mtime):
            continue
        if pages is not None and not _in_window(_page_no(img), start, end):
            shutil.copy2(img, out)                # outside the window -> passthrough
            continue
        with Image.open(img) as im:
            rotate_image(im, degrees, cw).save(out)
    return rot_dir

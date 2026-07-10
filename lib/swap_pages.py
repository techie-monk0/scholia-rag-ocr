"""Swap adjacent page-image pairs within a range (the ``--swap`` step, scoped by
``--pages START:END``).

Some scans come back with each facing pair captured in the wrong order — every
two consecutive pages transposed (a recto/verso or spread-half flip during
capture). This rewrites a page range so consecutive pages trade places pairwise:
START<->START+1, START+2<->START+3, ... up to and including END. A page with no
partner (an odd tail of the range) is left as-is.

The range ``(start, end)`` is supplied by the caller (from ``--pages``); the
grammar is parsed in ``preprocess.parse_pages``.

Page numbers come from the filename: ``<prefix><page number><ext>`` (the trailing
integer of the stem), e.g. ``Great hum - 9.jpg`` -> 9.

Like the rotation pass, swapped copies go to a sibling ``_swapped/`` dir under the
SAME filenames — only the image *content* moves between the two names of a pair.
Page numbering (keyed by filename stem) is therefore preserved, and the whole
downstream pipeline (OCR, line detection, output PDF) just uses the corrected
reading order.

Pipeline:
    swap_dir = make_swapped(images, image_dir / "_swapped", start, end)
    image_dir, images = swap_dir, sb.list_images(swap_dir)
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_NUM_RE = re.compile(r"(\d+)\s*$")          # trailing integer of the filename stem


def _page_no(path: Path):
    """Trailing page number in a filename stem, or ``None`` if it has no digits."""
    m = _NUM_RE.search(path.stem)
    return int(m.group(1)) if m else None


def swap_plan(images, start: int, end):
    """Map each image Path to the Path whose CONTENT it should receive.

    Within ``[start, end]``, pages pair up (start,start+1), (start+2,start+3), …;
    each paired file takes its partner's content. Every other file (outside the
    range, or an unpaired odd tail) maps to itself. Missing pages in a pair are
    skipped (that pair simply doesn't swap). ``end=None`` extends the range to the
    highest page number present (swap from START through the last page)."""
    by_no = {}
    for img in images:
        n = _page_no(Path(img))
        if n is not None:
            by_no[n] = Path(img)
    if end is None:
        end = max(by_no) if by_no else start
    plan = {Path(img): Path(img) for img in images}
    p = start
    while p + 1 <= end:
        a, b = by_no.get(p), by_no.get(p + 1)
        if a is not None and b is not None:
            plan[a], plan[b] = b, a
        p += 2
    return plan


def make_swapped(images, swap_dir: Path, start: int, end, *, force=False):
    """Write each image into ``swap_dir`` under its OWN filename but carrying the
    content of its swap partner (an unswapped page carries its own content).
    Returns ``swap_dir``. Incremental: skips a page whose swapped copy is already
    current (present and no older than the content source it copies from)."""
    swap_dir = Path(swap_dir)
    swap_dir.mkdir(parents=True, exist_ok=True)
    plan = swap_plan(images, start, end)
    for img in images:
        img = Path(img)
        src = plan[img]                       # file whose content this name gets
        out = swap_dir / img.name
        if (not force and out.exists()
                and out.stat().st_mtime >= src.stat().st_mtime):
            continue
        shutil.copy2(src, out)                # byte copy (no re-encode); keeps mtime
    return swap_dir

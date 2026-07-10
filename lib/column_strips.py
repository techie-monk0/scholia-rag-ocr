"""Column-strip re-OCR — the source fix for the multi-column reading-order
scramble (ARCHITECTURE.md, reading order).

A two-column page OCR'd whole comes back with the columns interleaved (or merged
into one block), which is fatal for chunking. Instead we crop each page into its
columns, OCR each strip on its own (a single column never scrambles), then
recombine the strip blocks onto the original page in column order, shifting each
block's bbox back to page coordinates.

Only the *OCR* (text + reading order) uses the strips. Line *detection* for the
PDF text layer keeps running on the whole page — line positions are correct
regardless of column order — so the searchable PDF is unaffected.

Pipeline:
    strips_dir, mapping = make_strips(orig_images, strips_dir, columns)
    strip_pages = backend.ocr_pages(list_images(strips_dir), strips_dir)
    pages = recombine(strip_pages, mapping, orig_images)
"""

from __future__ import annotations

from pathlib import Path

from ocr_backend import Block, Page
import surya_backend as sb


def column_split_xs(gray, ncols: int, *, search=0.14, gutter_max=0.02,
                    side_min=0.04):
    """Interior x cut positions for splitting ``gray`` (HxW uint8 ndarray) into up
    to ``ncols`` columns, splitting ONLY at a genuine gutter.

    For each of the ``ncols-1`` nominal boundaries (at i*W/ncols) we search a
    window of +-``search``*W for the lowest-ink vertical line and accept it as a
    cut only if that channel is near-empty (ink <= ``gutter_max``) with text on
    both sides (>= ``side_min``) — a real column gutter. A page with no such
    channel (a map, a full-width or single-column page, a chapter opener) yields
    fewer cuts, or none — so it stays whole instead of being sliced through its
    content. Returns the accepted cuts, sorted (length 0..ncols-1)."""
    import numpy as np
    H, W = gray.shape[:2]
    if ncols <= 1 or W < 8:
        return []
    col_ink = (gray < 128).mean(axis=0)          # per-column ink fraction
    half = max(1, int(search * W))
    cuts = []
    for i in range(1, ncols):
        nominal = round(i * W / ncols)
        a, b = max(1, nominal - half), min(W - 1, nominal + half)
        if b - a < 2:
            continue
        x = a + int(np.argmin(col_ink[a:b]))     # deepest valley in the window
        # Accept only a near-empty channel with ink on BOTH sides of the cut —
        # i.e. an actual gutter between two text columns, not a page margin.
        if (col_ink[x] <= gutter_max
                and col_ink[:x].max(initial=0) >= side_min
                and col_ink[x:].max(initial=0) >= side_min):
            cuts.append(int(x))
    return sorted(set(cuts))


def make_strips(images, strips_dir: Path, columns: int, *, force=False):
    """Crop each page image in ``images`` into ``columns`` vertical strips written
    to ``strips_dir`` as ``<stem>__c{i}.png``. Returns
    ``{strip_stem: (orig_stem, x_offset)}`` (x_offset = the strip's left edge in
    page pixels). Incremental: skips a page whose strips are already current."""
    import numpy as np
    from PIL import Image
    strips_dir = Path(strips_dir)
    strips_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict = {}
    for img in images:
        im = Image.open(img)
        W, H = im.size
        gray = np.asarray(im.convert("L"))
        cuts = column_split_xs(gray, columns)
        bounds = [0, *cuts, W]
        for i in range(len(bounds) - 1):
            x0, x1 = bounds[i], bounds[i + 1]
            if x1 - x0 < 2:
                continue
            stem = f"{img.stem}__c{i}"
            out = strips_dir / f"{stem}.png"
            mapping[stem] = (img.stem, x0)
            if (not force and out.exists()
                    and out.stat().st_mtime >= img.stat().st_mtime):
                continue
            im.crop((x0, 0, x1, H)).save(out)
    return mapping


def _col_index(strip_stem: str) -> int:
    try:
        return int(strip_stem.rsplit("__c", 1)[1])
    except (IndexError, ValueError):
        return 0


def recombine(strip_pages, mapping, orig_images) -> list:
    """Merge OCR'd strip ``Page``s back into one ``Page`` per original page.

    ``mapping``: strip_stem -> (orig_stem, x_offset). ``orig_images``: iterable
    of the original page image paths (for image + true dimensions). Blocks are
    concatenated column-by-column (left->right), each shifted right by its strip
    x_offset, so the linearized text reads in true column order."""
    from PIL import Image
    orig_by_stem = {Path(p).stem: Path(p) for p in orig_images}
    # group strip pages by their original page
    groups: dict = {}
    for pg in strip_pages:
        orig_stem, x_off = mapping.get(pg.name) or mapping.get(Path(pg.name).stem) \
            or (None, 0)
        if orig_stem is None:
            continue
        groups.setdefault(orig_stem, []).append((x_off, _col_index(pg.name), pg))

    pages: list = []
    for orig_stem, items in groups.items():
        items.sort(key=lambda t: (t[1], t[0]))      # by column index then x
        img = orig_by_stem.get(orig_stem)
        width = height = None
        if img:
            try:
                width, height = Image.open(img).size
            except Exception:
                pass
        blocks: list = []
        page_no = None
        for x_off, _ci, pg in items:
            if page_no is None:
                page_no = pg.page_no
            for b in pg.blocks:
                bb = b.bbox
                shifted = ([bb[0] + x_off, bb[1], bb[2] + x_off, bb[3]]
                           if bb else bb)
                blocks.append(Block(b.text, shifted, b.confidence, b.label,
                                    b.html))
        if not width:                                # fall back to strip extents
            width = max((b.bbox[2] for b in blocks if b.bbox), default=1000)
            height = max((b.bbox[3] for b in blocks if b.bbox), default=1000)
        pages.append(Page(img, orig_stem, int(width), int(height), blocks, page_no))

    pages.sort(key=lambda p: sb.natural_key(p.name))
    return pages

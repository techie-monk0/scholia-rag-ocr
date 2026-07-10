#!/usr/bin/env python3
"""Rasterize a PDF to a folder of page images — the one seam ocr_one.py is
missing.

ocr_one.py already takes a folder of page images all the way to a searchable
PDF + text (Surya, cached). The only thing it can't do is start from a PDF.
This module fills that gap with a thin PyMuPDF (``fitz``) wrapper:

    pdf_to_images("Book.pdf", "pages/", dpi=300) -> Path("pages/")

It works the same for both target classes:

  * Image-only (scanned) PDFs — the embedded page bitmaps are rasterized.
  * Mojibake born-digital PDFs — the *visually correct glyphs* are rendered to
    pixels and the garbage text layer is discarded. Re-OCR'ing those pixels is
    exactly how rasterize-then-OCR repairs mojibake.

Pages are written zero-padded and sequential (``page-0001.png`` ...) so they
sort in reading order without needing the --pad-page-numbers step.

Render high (~300 DPI / zoom ~4): Surya upscales internally and low-res input
mangles diacritics (per Surya's own perf notes), so we never downscale here —
file-size trimming happens later in the PDF-assembly stage.

Standalone:
    python3 lib/pdf_to_images.py Book.pdf                 # -> Book/
    python3 lib/pdf_to_images.py Book.pdf -o pages/ --dpi 350
    python3 lib/pdf_to_images.py /folder/of/pdfs -o out/  # batch: one dir each
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF — already a pipeline dependency


def two_up_split_x(gray, *, min_aspect=1.15, central=(0.35, 0.65),
                   gutter_max=0.01, side_min=0.05):
    """For a two-up scan (two book pages on one landscape sheet), return the x to
    split at (the central whitespace gutter), or None. Requires landscape shape,
    a near-empty vertical channel in the central x-range, and ink on both sides
    (so a genuinely wide single page / full-bleed figure isn't split)."""
    h, w = gray.shape
    if w < min_aspect * h:
        return None
    col_ink = (gray < 128).mean(axis=0)
    lo, hi = int(w * central[0]), int(w * central[1])
    if hi - lo < 3:
        return None
    gx = lo + int(col_ink[lo:hi].argmin())
    if col_ink[gx] > gutter_max:
        return None
    if col_ink[:lo].max() < side_min or col_ink[hi:].max() < side_min:
        return None
    return gx


def _pix_gray(pix):
    import numpy as np
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr[..., :3].mean(axis=2) if pix.n >= 3 else arr[..., 0]


def _pix_pil(pix):
    from PIL import Image
    mode = "RGBA" if pix.n == 4 else "RGB" if pix.n == 3 else "L"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples)


def _render_page(pix, base, ext, two_up):
    """Save a rendered page, splitting a two-up sheet into left/right ('a'/'b').
    Returns the list of output filenames written."""
    landscape = pix.width >= 1.15 * pix.height
    if two_up == "off" or (two_up == "auto" and not landscape):
        pix.save(f"{base}.{ext}")
        return [f"{base}.{ext}"]
    gx = (two_up_split_x(_pix_gray(pix)) if two_up == "auto"
          else (pix.width // 2 if landscape else None))
    if gx:
        img = _pix_pil(pix)
        img.crop((0, 0, gx, pix.height)).save(f"{base}a.{ext}")
        img.crop((gx, 0, pix.width, pix.height)).save(f"{base}b.{ext}")
        return [f"{base}a.{ext}", f"{base}b.{ext}"]
    pix.save(f"{base}.{ext}")
    return [f"{base}.{ext}"]


def pdf_to_images(pdf_path, out_dir=None, dpi: int = 300, *,
                  fmt: str = "png", force: bool = False, quiet: bool = False,
                  two_up: str = "auto"):
    """Render every page of ``pdf_path`` to ``out_dir`` as zero-padded images.

    Returns the output directory (``Path``). ``out_dir`` defaults to a sibling
    folder named after the PDF stem (``Book.pdf`` -> ``Book/``).

    Incremental by default: a page is re-rendered only if its image is missing
    or older than the PDF; pass ``force=True`` to always re-render. ``dpi`` sets
    the render resolution (zoom = dpi/72); keep it high (300+) — see module doc.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"no such PDF: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"not a .pdf: {pdf_path}")

    out_dir = Path(out_dir) if out_dir is not None else pdf_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    ext = fmt.lower().lstrip(".")
    pdf_mtime = pdf_path.stat().st_mtime

    rendered = skipped = 0
    with fitz.open(pdf_path) as doc:
        n = doc.page_count
        width = max(4, len(str(n)))               # page-0001 for <=9999 pages
        for i, page in enumerate(doc, start=1):
            base = out_dir / f"page-{i:0{width}d}"
            # cache hit: any output for this source page (single, or two-up a/b)
            outs = [Path(f"{base}.{ext}"), Path(f"{base}a.{ext}")]
            if (not force and any(o.exists() and o.stat().st_mtime >= pdf_mtime
                                  for o in outs)):
                skipped += 1
                continue
            pix = page.get_pixmap(matrix=matrix)
            names = _render_page(pix, str(base), ext, two_up)
            rendered += 1
            if not quiet:
                tag = " (split two-up)" if len(names) > 1 else ""
                print(f"  page {i}/{n} -> {Path(names[0]).name}"
                      f"{'..'+Path(names[-1]).name+tag if len(names)>1 else ''}",
                      flush=True)

    if not quiet:
        note = f"{rendered} rendered" + (f", {skipped} cached" if skipped else "")
        print(f"{pdf_path.name}: {note} @ {dpi} DPI -> {out_dir}/", flush=True)
    return out_dir


def _is_pdf(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".pdf"


def list_pdfs(folder) -> list[Path]:
    """PDFs directly inside ``folder``, in natural sort order."""
    folder = Path(folder)
    return sorted((p for p in folder.iterdir() if _is_pdf(p)),
                  key=lambda p: p.name.lower())


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("pdf", type=Path,
                    help="A .pdf file, or a folder of PDFs (batch: one image "
                         "folder per PDF).")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output dir. Single PDF: the image folder (default: "
                         "sibling folder named after the PDF). Folder of PDFs: "
                         "the parent under which one <stem>/ folder is made per "
                         "PDF (default: alongside each PDF).")
    ap.add_argument("--dpi", type=int, default=300,
                    help="Render resolution (zoom = dpi/72). Keep high (300+); "
                         "Surya upscales and low-res mangles diacritics.")
    ap.add_argument("--format", default="png", choices=["png", "jpg", "jpeg"],
                    help="Output image format.")
    ap.add_argument("--force", action="store_true",
                    help="Re-render every page, ignoring the incremental cache.")
    ap.add_argument("--two-up", choices=["auto", "on", "off"], default="off",
                    help="Split two-up scans (two book pages on one landscape "
                         "sheet) into left/right page images (named ...a/...b). "
                         "off (default) = never split — the safe default, since "
                         "two-up books are rare and gutter detection mis-splits "
                         "landscape plates/maps/tables in ordinary books. on = "
                         "split every landscape page at the midpoint (use for a "
                         "known two-up-throughout book); auto = split only "
                         "landscape pages with a detected central whitespace "
                         "gutter (heuristic, mistake-prone — opt in deliberately).")
    args = ap.parse_args(argv)

    if not args.pdf.exists():
        raise SystemExit(f"no such path: {args.pdf}")

    if args.pdf.is_dir():
        pdfs = list_pdfs(args.pdf)
        if not pdfs:
            raise SystemExit(f"no PDFs in {args.pdf}")
        print(f"Batch: {len(pdfs)} PDF(s) in {args.pdf}", flush=True)
        for pdf in pdfs:
            out = (args.out / pdf.stem) if args.out else None
            pdf_to_images(pdf, out, dpi=args.dpi, fmt=args.format,
                          force=args.force, two_up=args.two_up)
    else:
        if not _is_pdf(args.pdf):
            raise SystemExit(f"not a .pdf: {args.pdf}")
        pdf_to_images(args.pdf, args.out, dpi=args.dpi, fmt=args.format,
                      force=args.force, two_up=args.two_up)


if __name__ == "__main__":
    sys.exit(main())

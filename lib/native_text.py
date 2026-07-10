"""Native-text extraction for the detect-only path (digital-first books).

For a digital-first PDF the embedded text is already clean — OCR'ing the
rasterized page only degrades it (and corrupts diacritics). So instead of
recognition we read the text straight out of the PDF with PyMuPDF and hand it to
the SAME assembly the OCR path uses. The region/furniture LABELS still come from
the model (surya_layout, merged in by ocr_one) so a digital-first book ends up
labeled the same way a scanned book does.

`native_pages()` returns a list of ocr_backend.Page with:
  * one Block per PDF text block, text = its lines joined, label = None
    (filled later by the surya_layout merge),
  * one Block per embedded image, text = "", label = "Picture",
  * bboxes in IMAGE PIXELS at the given dpi (PDF points * dpi/72), so they line
    up with the rasterized page images and the layout regions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_backend import Block, Page


def _block_text(b) -> str:
    """Join a PDF text block's lines (spans concatenated, blank lines dropped)."""
    lines = []
    for ln in b.get("lines", []):
        t = "".join(s.get("text", "") for s in ln.get("spans", []))
        if t.strip():
            lines.append(t)
    return "\n".join(lines).strip()


def native_pages(source_pdf, dpi: int = 300, page_names=None):
    """Build Page objects from ``source_pdf``'s embedded text layer. bboxes are
    in image pixels at ``dpi`` (so they align with the rasterized images and
    surya_layout output). ``page_names[i]`` names page i+1's image (defaults to
    page-0001.png …, matching pdf_to_images)."""
    z = dpi / 72.0
    out = []
    with fitz.open(source_pdf) as doc:
        width = max(4, len(str(doc.page_count)))
        for i, page in enumerate(doc, start=1):
            d = page.get_text("dict")
            W, H = int(round(d["width"] * z)), int(round(d["height"] * z))
            blocks = []
            for b in d.get("blocks", []):
                x0, y0, x1, y1 = b["bbox"]
                bbox = [x0 * z, y0 * z, x1 * z, y1 * z]
                if b.get("type") == 1:                 # embedded image -> figure
                    blocks.append(Block("", bbox, label="Picture"))
                    continue
                txt = _block_text(b)
                if txt:
                    blocks.append(Block(txt, bbox, label=None))
            name = page_names[i - 1] if page_names else f"page-{i:0{width}d}.png"
            out.append(Page(None, name, W, H, blocks, i))
    return out


def native_line_boxes(source_pdf, dpi: int = 300, page_names=None) -> dict:
    """{image-stem: [line bbox in image px, ...]} from the PDF's own text lines —
    the native equivalent of surya_detect's per-line boxes, so the detect-only
    path can emit ``line_bbox`` like the OCR path (build_doc attaches it to
    verse/ambiguous blocks via block_line_boxes)."""
    z = dpi / 72.0
    out = {}
    with fitz.open(source_pdf) as doc:
        width = max(4, len(str(doc.page_count)))
        for i, page in enumerate(doc, start=1):
            d = page.get_text("dict")
            lines = []
            for b in d.get("blocks", []):
                if b.get("type") == 1:                 # image block — no text lines
                    continue
                for ln in b.get("lines", []):
                    if any(s.get("text", "").strip() for s in ln.get("spans", [])):
                        x0, y0, x1, y1 = ln["bbox"]
                        lines.append([x0 * z, y0 * z, x1 * z, y1 * z])
            name = page_names[i - 1] if page_names else f"page-{i:0{width}d}.png"
            out[Path(name).stem] = lines
    return out


def _coverage(a, b) -> float:
    """Fraction of block ``a`` covered by region ``b`` (intersection / area(a))."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0) / max(1.0, (ax1 - ax0) * (ay1 - ay0))


def merge_layout(pages, layout_by_stem, min_coverage: float = 0.5):
    """Label each native text block with the surya_layout region it sits inside
    (the one covering the most of it). Native image blocks keep their 'Picture'.
    Blocks with no covering region default to 'Text'. Mutates pages; returns
    (pages, n_labeled)."""
    labeled = 0
    for pg in pages:
        regions = layout_by_stem.get(Path(pg.name).stem, [])
        for blk in pg.blocks:
            if blk.label:                              # native image -> Picture
                continue
            best, best_cov = None, min_coverage
            for bbox, label in regions:
                cov = _coverage(blk.bbox, bbox)
                if cov > best_cov:
                    best, best_cov = label, cov
            blk.label = best or "Text"
            if best:
                labeled += 1
    return pages, labeled

#!/usr/bin/env python3
"""Goal 1 — build a searchable PDF (image + invisible OCR text layer).

Each output page is the original JPEG with an *invisible* (render mode 3)
text layer positioned over it, so the PDF looks identical to the scan but is
fully selectable / searchable.  The text comes from the shared Surya backend.

Line-level boxes (default): each OCR block is split into its visual lines and
positioned with per-line boxes from ``surya_detect`` (falls back to slicing the
block into equal bands when detection and OCR disagree on the line count).
Use ``--lines split`` to skip the detection pass and use band-slicing only.

Default font is "Arial Unicode" (covers IAST diacritics + Tibetan +
Devanagari).  Override with --font for a different glyph repertoire.

Usage:
    python3 ocr_to_pdf.py "/path/to/Steps 4" -o steps4.pdf
    python3 ocr_to_pdf.py "/path/to/Steps 4" --lines split -o steps4.pdf
    python3 ocr_to_pdf.py "/path/to/Steps 4" --limit 3 -o smoke.pdf
"""
from __future__ import annotations

import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance

import surya_backend as sb

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",  # fallback: Latin/IAST only
]


def pick_font(path: str | None) -> tuple[fitz.Font, str]:
    candidates = [path] if path else FONT_CANDIDATES
    for cand in candidates:
        if cand and Path(cand).exists():
            return fitz.Font(fontfile=cand), cand
    raise SystemExit("No usable text-layer font found. Pass one with --font.")


def _supported(font: fitz.Font, text: str) -> tuple[str, int]:
    """Replace chars the font can't render with a space (keeps the rest
    searchable; avoids TextWriter raising on a missing glyph)."""
    out, drops = [], 0
    for ch in text:
        if ch in "\r\n\t" or font.has_glyph(ord(ch)):
            out.append(ch)
        else:
            out.append(" ")
            drops += 1
    return "".join(out), drops


def _otsu_threshold(gray_image):
    """Otsu's method: pick the grey level that best separates ink from paper,
    from the image histogram (so b&w handles uneven scan lighting)."""
    hist = gray_image.histogram()[:256]
    total = sum(hist) or 1
    sum_all = sum(i * h for i, h in enumerate(hist))
    sum_b = w_b = 0
    best_var, thresh = -1.0, 128
    for i, h in enumerate(hist):
        w_b += h
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * h
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, thresh = var, i
    return thresh


def _encode_image(image_path, *, color, quality, max_dim, contrast=1.0):
    """Re-encode a page image for embedding. ``color`` is 'color' | 'grey' |
    'bw'; ``contrast`` >1 boosts contrast (text pops to crisper black-on-white).
    Returns image bytes (JPEG for color/grey, 1-bit PNG for bw), or None to embed
    the original untouched (color, no actual resize, no quality/contrast change)."""
    im = Image.open(image_path)
    resized = False
    if max_dim:
        w, h = im.size
        if max(w, h) > max_dim:                          # only shrink, never upscale
            s = max_dim / max(w, h)
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
            resized = True
    # Nothing would change the pixels -> embed the original bytes losslessly.
    if color == "color" and contrast == 1.0 and not resized and not quality:
        return None
    if contrast != 1.0:
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im = ImageEnhance.Contrast(im).enhance(contrast)
    if color == "bw":
        g = im.convert("L")
        t = _otsu_threshold(g)
        bw = g.point(lambda p, t=t: 255 if p > t else 0).convert(
            "1", dither=Image.Dither.NONE)
        buf = io.BytesIO()
        bw.save(buf, format="PNG", optimize=True)      # 1-bit / Flate
        return buf.getvalue()
    if color == "grey":
        im = im.convert("L")
    elif im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality or 90, optimize=True)
    return buf.getvalue()


def _page_lines(pg, det_boxes, font):
    """(line_text, bbox) for one page.

    Surya returns paragraph-level text with NO line breaks, so we can't trust a
    newline split. Instead we treat the detection line boxes as the real lines
    and reflow each block's words across the boxes that fall inside it (greedy
    width fill), so every line sits on its true box — the key to an accurate,
    selectable text layer. Blocks with no detected lines fall back to one line.
    """
    det = sorted(det_boxes or [], key=lambda b: (b[1], b[0]))
    out = []
    for blk in pg.blocks:
        text = " ".join(blk.text.split())          # collapse newlines/space
        if not text:
            continue
        x0, y0, x1, y1 = blk.bbox
        inside = sorted(
            (b for b in det
             if x0 - 2 <= (b[0] + b[2]) / 2 <= x1 + 2
             and y0 - 2 <= (b[1] + b[3]) / 2 <= y1 + 2),
            key=lambda b: b[1])
        if not inside:
            out.append((text, [x0, y0, x1, y1]))
            continue
        # Distribute words across ALL inside line boxes proportional to each
        # box's width, so every real line is used and the last word lands on the
        # last box — keeps top and bottom anchored, error bounded to <1 line.
        words = text.split()
        sp = font.text_length(" ", fontsize=1.0) or 0.0
        ww = [font.text_length(w, fontsize=1.0) + sp for w in words]
        total_ww = sum(ww) or 1.0
        bw = [b[2] - b[0] for b in inside]
        total_bw = sum(bw) or 1.0
        cum = cumbw = 0.0
        wi = 0
        for j, box in enumerate(inside):
            cumbw += bw[j]
            target = cumbw / total_bw * total_ww
            last = j == len(inside) - 1
            line = []
            while wi < len(words) and (last or cum < target):
                line.append(words[wi])
                cum += ww[wi]
                wi += 1
            if line:
                out.append((" ".join(line), box))
    return out


def build_pdf(pages, output, *, font_path=None, det_by_stem=None, lines="detect",
              image_quality=None, max_dim=None, uniform=False, uniform_except=(),
              color="color", contrast=1.0, quiet=False, encode_workers=None):
    """Build a searchable PDF (page images + invisible text layer) into
    ``output``. ``det_by_stem`` supplies per-line boxes (lines='detect').

    ``image_quality`` (JPEG 1-95) and ``max_dim`` (cap the longest edge, px)
    shrink the embedded page images; ``color`` ('color'|'grey'|'bw') sets their
    color depth (the first page — the cover — always stays color); ``uniform``
    makes every PDF page the same size (the max page box of the uniform pages),
    scaling each scan to fit (centered, no distortion). ``uniform_except`` is a set
    of 1-based page indices left at their ORIGINAL size (e.g. a foldout/plate that
    shouldn't be shrunk into the common box). All reuse the OCR cache (geometry only).
    """
    det_by_stem = det_by_stem or {}
    uniform_except = set(uniform_except)
    font, fpath = pick_font(font_path)
    print(f"Text-layer font: {fpath}  (lines={lines})", flush=True)

    target = None
    if uniform:                              # common box = max over the UNIFORM pages
        uni = [pg for i, pg in enumerate(pages, 1) if i not in uniform_except]
        if uni:
            target = (max(pg.width for pg in uni), max(pg.height for pg in uni))

    # Image (re-)encoding is the heavy part and is independent per page, so do it
    # in parallel up front (PIL releases the GIL); PyMuPDF assembly stays serial.
    # `streams[i]` is JPEG/PNG bytes, or None to embed the original file as-is.
    def _enc(idx_pg):
        i, pg = idx_pg
        if not pg.image_path:
            return None
        cover = i == 1                                   # leave the cover alone
        return _encode_image(pg.image_path, color="color" if cover else color,
                             quality=image_quality, max_dim=max_dim,
                             contrast=1.0 if cover else contrast)
    workers = encode_workers or min(8, (os.cpu_count() or 4))
    if workers > 1 and len(pages) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            streams = list(ex.map(_enc, enumerate(pages, 1)))
    else:
        streams = [_enc(ip) for ip in enumerate(pages, 1)]

    doc = fitz.open()
    dropped_chars = 0
    for i, pg in enumerate(pages, 1):
        # Map the scan into the page: scale-to-fit + center when uniform (unless this
        # page is exempted, in which case it keeps its original size).
        if target and i not in uniform_except:
            W, H = target
            s = min(W / pg.width, H / pg.height)
            ox, oy = (W - pg.width * s) / 2, (H - pg.height * s) / 2
        else:
            W, H, s, ox, oy = pg.width, pg.height, 1.0, 0.0, 0.0

        page = doc.new_page(width=W, height=H)
        if pg.image_path:
            rect = fitz.Rect(ox, oy, ox + pg.width * s, oy + pg.height * s)
            stream = streams[i - 1]
            if stream is not None:
                page.insert_image(rect, stream=stream)
            else:
                page.insert_image(rect, filename=str(pg.image_path))

        stem = Path(pg.name).stem
        det = det_by_stem.get(stem)
        if det is None and pg.image_path:
            det = det_by_stem.get(pg.image_path.stem)
        if lines == "detect" and det:
            items = _page_lines(pg, det, font)       # reflow text onto real lines
        else:
            items = sb.line_items(pg, None)          # no detection: band-slice

        tw = fitz.TextWriter(page.rect)
        for text, (x0, y0, x1, y1) in items:
            # same scale+offset as the image, so the text layer stays aligned
            x0, y0, x1, y1 = ox + x0 * s, oy + y0 * s, ox + x1 * s, oy + y1 * s
            clean, drops = _supported(font, text)
            dropped_chars += drops
            box_w, box_h = x1 - x0, y1 - y0
            if not clean.strip() or box_w <= 0 or box_h <= 0:
                continue
            # Fit the line's text to its box WIDTH so the invisible glyphs track
            # the scanned line end-to-end (the font's natural width differs from
            # the scan's, which is what wrecks selection/copy). Clamp so a short
            # line in a loose box doesn't balloon; floor keeps it selectable.
            natural = font.text_length(clean, fontsize=1.0)
            if natural <= 0:
                continue
            fontsize = max(2.0, min(box_w / natural, box_h * 1.3))
            # vertically center the (possibly shorter) text within the line box
            baseline = (x0, y0 + box_h * 0.5 + fontsize * 0.3)
            try:
                tw.append(baseline, clean, font=font, fontsize=fontsize)
            except Exception:
                pass
        tw.write_text(page, render_mode=3)  # 3 = invisible, still selectable
        if not quiet:
            print(f"[{i}/{len(pages)}] {pg.name}: {len(items)} lines", flush=True)

    try:
        doc.subset_fonts()          # embed only used glyphs (Arial Unicode is ~15 MB)
    except Exception:
        pass
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output), deflate=True, garbage=4)
    doc.close()
    note = (f"  ({dropped_chars} glyphs unsupported by font, omitted)"
            if dropped_chars else "")
    if not quiet:
        print(f"\nWrote {output}{note}", flush=True)


def main():
    p = sb.common_parser(__doc__.splitlines()[0])
    p.add_argument("-o", "--output", type=Path, default=Path("searchable.pdf"))
    p.add_argument("--font", default=None,
                   help="TTF/OTF/TTC for the invisible text layer "
                        "(default: Arial Unicode).")
    p.add_argument("--lines", choices=["detect", "split"], default="detect",
                   help="detect: per-line boxes from surya_detect (default). "
                        "split: slice blocks into bands, no detection pass.")
    args = p.parse_args()

    pages = sb.get_pages(args)

    det_by_stem = {}
    if args.lines == "detect":
        det_path = sb.ensure_detection(args.image_dir, sb.resolve_cache_dir(args),
                                       force=args.force, limit=args.limit)
        det_by_stem = sb.load_line_boxes(det_path, args.image_dir)

    build_pdf(pages, args.output, font_path=args.font,
              det_by_stem=det_by_stem, lines=args.lines)


if __name__ == "__main__":
    sys.exit(main())

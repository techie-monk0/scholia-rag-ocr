#!/usr/bin/env python3
"""Goal 2 — extract bare OCR text for later LLM chunking.

Writes three things into the output directory (shares the same Surya cache as
ocr_to_pdf.py, so OCR is not re-run):

  * one ``NNNN_<name>.txt`` per page (raw, clean text)
  * ``all_pages.txt`` — every page concatenated with page-boundary markers
    (a form-feed plus a ``===== PAGE ... =====`` header) so a chunker can
    split on page boundaries
  * ``pages.jsonl`` — one JSON object per page (index, name, page_no, text,
    and per-block text/bbox/confidence/label) for structured / coordinate-aware
    chunking

Layout-aware body text (``--body-only``): surya-ocr-2 labels every region it
reads, so we simply drop the non-body regions — running headers/footers, which
also carry the page numbers (labels PageHeader/PageFooter) — and keep Text,
SectionHeader, Footnote, Caption, lists, etc. Tune with ``--drop-labels``.

Usage:
    python3 ocr_to_text.py "/path/to/Steps 4" -o steps4_text
    python3 ocr_to_text.py "/path/to/Steps 4" --body-only -o steps4_body
    python3 ocr_to_text.py "/path/to/Steps 4" --limit 3 -o smoke_text
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import surya_backend as sb


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


_EM_DASH = "—"
_FOOTNOTE_REF = re.compile(r"[ \t]*\[\^[^\]]*\]")


def normalize_emit(text, *, ascii_dashes=True, strip_footnote_refs=False):
    """Normalize the *bare text* emitted for downstream chunking/embedding.

    The PDF text layer keeps the faithful Unicode; this only touches the .txt /
    all_pages / jsonl text. ``ascii_dashes`` rewrites em dashes (—) as ``--``
    (en dashes in number ranges are left alone). ``strip_footnote_refs`` removes
    the ``[^N]`` in-text footnote markers (and the space before them)."""
    if strip_footnote_refs:
        text = _FOOTNOTE_REF.sub("", text)
    if ascii_dashes:
        text = text.replace(_EM_DASH, "--")
    return text


def build_text(pages, outdir, *, drop=frozenset(), ascii_dashes=True,
               strip_footnote_refs=False):
    """Write per-page .txt + all_pages.txt + pages.jsonl into ``outdir``.

    ``drop`` is a set of layout labels to omit (body-only). ``ascii_dashes`` /
    ``strip_footnote_refs`` tune the emitted bare text (see ``normalize_emit``).
    Returns (n_pages, total_dropped_regions).
    """
    def emit(s):
        return normalize_emit(s, ascii_dashes=ascii_dashes,
                              strip_footnote_refs=strip_footnote_refs)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    combined_parts: list[str] = []
    total_dropped = 0
    jsonl_path = outdir / "pages.jsonl"
    with jsonl_path.open("w") as jf:
        for i, pg in enumerate(pages, 1):
            kept = [b for b in pg.blocks if b.label not in drop]
            total_dropped += len(pg.blocks) - len(kept)
            text = emit("\n\n".join(b.text for b in kept if b.text.strip()))
            stem = f"{i:04d}_{_safe(Path(pg.name).stem)}"
            (outdir / f"{stem}.txt").write_text(text + "\n")

            header = f"===== PAGE {i:04d} :: {pg.name} ====="
            combined_parts.append("\f\n" + header + "\n\n" + text)

            jf.write(json.dumps({
                "index": i,
                "name": pg.name,
                "page_no": pg.page_no,
                "width": pg.width,
                "height": pg.height,
                "text": text,
                "blocks": [
                    {"text": emit(b.text), "label": b.label,
                     "bbox": b.bbox, "confidence": b.confidence}
                    for b in kept
                ],
            }, ensure_ascii=False) + "\n")

            drp = len(pg.blocks) - len(kept)
            print(f"[{i}/{len(pages)}] {pg.name}: {len(kept)} blocks"
                  + (f" (-{drp} non-body)" if drp else "")
                  + f", {len(text)} chars", flush=True)

    combined = outdir / "all_pages.txt"
    combined.write_text("\n".join(combined_parts) + "\n")
    return len(pages), total_dropped


def main():
    p = sb.common_parser(__doc__.splitlines()[0])
    p.add_argument("-o", "--output-dir", type=Path, default=Path("ocr_text"))
    p.add_argument("--body-only", action="store_true",
                   help="Keep only body regions; drop running headers/footers "
                        "and page numbers (layout-aware, uses Surya's labels).")
    p.add_argument("--drop-labels", default=",".join(sorted(sb.NON_BODY_LABELS)),
                   help="Comma-separated layout labels to drop with --body-only "
                        f"(default: {','.join(sorted(sb.NON_BODY_LABELS))}).")
    p.add_argument("--ascii-dashes", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Emit em dashes (—) as '--' in the bare text (the PDF "
                        "text layer keeps the Unicode). --no-ascii-dashes keeps —.")
    p.add_argument("--strip-footnote-refs", action="store_true",
                   help="Remove in-text footnote markers [^N] from the bare text "
                        "(they're kept by default so they stay recoverable).")
    args = p.parse_args()

    pages = sb.get_pages(args)
    drop = {s.strip() for s in args.drop_labels.split(",") if s.strip()} \
        if args.body_only else set()

    n, total_dropped = build_text(pages, args.output_dir, drop=drop,
                                  ascii_dashes=args.ascii_dashes,
                                  strip_footnote_refs=args.strip_footnote_refs)

    extra = f", dropped {total_dropped} non-body regions" if args.body_only else ""
    print(f"\nWrote {n} page files + all_pages.txt + pages.jsonl "
          f"into {args.output_dir}/{extra}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

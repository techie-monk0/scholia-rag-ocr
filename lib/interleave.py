"""interleave.py — a book-pipelined OCR runner that keeps the GPU continuously fed.

The per-book serial path (ocr_one.run_one) leaves the GPU idle through each book's
rasterize (CPU/disk) and assembly (CPU), and reloads the llama-servers per book.
This runner fixes both for a batch of books:

  * PERSISTENT servers (#3): the llama-servers are launched ONCE
    (backend.persistent_servers) and reused for every book — no per-book GGUF
    reload. One server set, so no extra memory vs the serial path (running whole
    books in parallel instead would multiply servers and swap).

  * STAGE PIPELINE (#2): three stages run concurrently, joined by bounded queues:
        rasterize (1 thread, CPU)  ->  ocr+detect (1 thread, the GPU)  ->  assemble (1 thread, CPU)
    so while the GPU OCRs book N, the CPU rasterizes N+1 and assembles N-1. The
    GPU stage is single-threaded (one book at a time on the shared servers); the
    CPU stages overlap it and never block it. Queue sizes bound how far rasterize
    runs ahead (rasterized pages live on disk, not RAM).

Scope (v1): supports the campaign config — PDF/folder input, two_up + drop-pages
layout, --lines detect, --text both, --doc, a single PDF. Column-strip re-OCR
(columns>=2) is NOT yet pipelined; such books fall back to whole-page OCR here
(run them on the serial path if column order matters). Reuses the same builders
as run_one, so output is identical per book.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import surya_backend as sb
from pdf_to_images import pdf_to_images
from ocr_to_text import build_text
from ocr_to_pdf import build_pdf
from build_doc import build_doc
import corpus

_SENTINEL = object()


def _ts():
    return time.strftime("%H:%M:%S")


def _split_drop(images, drop_pages):
    """Apply a drop-pages=even|odd layout: (ocr_imgs, skip_imgs). skip_imgs are
    kept image-only in the PDF (Surya can't read the native script on them)."""
    if drop_pages not in ("even", "odd"):
        return images, []
    want_even = drop_pages == "even"
    ocr_imgs, skip = [], []
    for idx, img in enumerate(images, 1):              # 1-based page index
        (skip if ((idx % 2 == 0) == want_even) else ocr_imgs).append(img)
    return ocr_imgs, skip


def _rasterize(job, opts):
    """STAGE 1 (CPU): a PDF -> page images under <out>/_pages; a folder is used
    as-is. Returns the image dir."""
    src = job.src
    if src.is_file() and src.suffix.lower() == ".pdf":
        return pdf_to_images(src, out_dir=job.out / "_pages", dpi=opts.dpi,
                             force=opts.force, two_up=job.layout.get("two_up", "off"))
    return src


def _ocr(job, image_dir, backend, opts):
    """STAGE 2 (GPU): OCR (+ optional line-box detection) against the persistent
    servers. Returns (pages, det_by_stem)."""
    images = sb.list_images(image_dir)
    if opts.limit:
        images = images[:opts.limit]
    ocr_imgs, skip_imgs = _split_drop(images, job.layout.get("drop_pages"))
    pages = backend.ocr_pages(ocr_imgs, image_dir, force=opts.force)
    if skip_imgs:                                      # re-insert as image-only
        from PIL import Image as _Image
        for img in skip_imgs:
            try:
                w, h = _Image.open(img).size
            except Exception:
                w, h = 1000, 1000
            pages.append(sb.Page(img, img.name, int(w), int(h), [], None))
        pages.sort(key=lambda p: sb.natural_key(p.name))
    det = {}
    if opts.lines == "detect":
        det = backend.line_boxes(ocr_imgs, image_dir, force=opts.force) or {}
    return pages, det


def _assemble(job, pages, det, opts):
    """STAGE 3 (CPU): full/body text + doc.json + searchable PDF (same builders
    as run_one). Returns the page count."""
    import ocr_one                                     # reuse the non-Latin filter
    # Output FILES are named after the source stem (matching run_one), not the
    # edition slug — the slug is only the output DIRECTORY (job.out).
    out, name = job.out, getattr(job, "stem", None) or job.name
    out.mkdir(parents=True, exist_ok=True)
    if opts.drop_nonlatin:
        ocr_one.drop_nonlatin_pages(pages, threshold=opts.nonlatin_threshold)
    body_labels = {s.strip() for s in opts.drop_labels.split(",") if s.strip()}
    emit = dict(ascii_dashes=opts.ascii_dashes,
                strip_footnote_refs=opts.strip_footnote_refs)
    build_text(pages, out / "full", drop=set(), **emit)
    npages, _dropped = build_text(pages, out / "body", drop=body_labels, **emit)
    if opts.doc:
        build_doc(pages, out / f"{name}.doc.json", source=str(job.src),
                  title=opts.title, det_by_stem=det)
    color = "grey" if opts.color == "gray" else opts.color
    build_pdf(pages, out / f"{name}.pdf", color=color, max_dim=opts.pdf_max_dim,
              image_quality=opts.pdf_image_quality, font_path=opts.font,
              det_by_stem=det, lines=opts.lines, uniform=opts.uniform_pages,
              contrast=opts.contrast)
    return npages


def default_opts(**over) -> SimpleNamespace:
    """The campaign defaults; override per call."""
    o = dict(dpi=300, force=False, limit=None, lines="detect", text="both",
             doc=True, drop_nonlatin=True, nonlatin_threshold=0.5,
             drop_labels=",".join(sorted(sb.NON_BODY_LABELS)), ascii_dashes=True,
             strip_footnote_refs=False, title=None, color="bw", pdf_max_dim=1200,
             pdf_image_quality=None, font=None, uniform_pages=False, contrast=1.25,
             skip_done=True, prefetch=2, profiler=None)
    o.update(over)
    return SimpleNamespace(**o)


def run_books(jobs, backend, opts, log_dir) -> dict:
    """Pipeline ``jobs`` (each: SimpleNamespace name/src/out/layout) through
    rasterize -> ocr -> assemble against ``backend``'s persistent servers.
    Returns {name: 'ok' | 'skip' | 'FAIL: ...'}."""
    todo = []
    for j in jobs:
        stem = getattr(j, "stem", None) or j.name      # output file == source stem
        if opts.skip_done and corpus.is_done(j.out):
            print(f"[{_ts()}] skip {j.name} (done)", flush=True)
            continue
        todo.append(j)
    results = {j.name: "skip" for j in jobs}
    if not todo:
        return results

    ras_q = queue.Queue(maxsize=max(1, opts.prefetch))   # rasterized, awaiting GPU
    asm_q = queue.Queue(maxsize=max(1, opts.prefetch))   # OCR'd, awaiting assembly
    lock = threading.Lock()
    totals = {"pages": 0, "secs": 0.0}                   # running OCR throughput

    def rasterizer():
        for j in todo:
            try:
                image_dir = _rasterize(j, opts)
                ras_q.put((j, image_dir))
            except Exception as e:
                with lock:
                    results[j.name] = f"FAIL rasterize: {e}"
                print(f"[{_ts()}] {j.name}: FAIL rasterize: {e}", flush=True)
        ras_q.put(_SENTINEL)

    def ocrer():
        while True:
            item = ras_q.get()
            if item is _SENTINEL:
                break
            j, image_dir = item
            try:
                if opts.profiler is not None:        # tag [PERF] with the GPU book
                    opts.profiler.enter_stage(f"ocr:{j.name}")
                print(f"[{_ts()}] {j.name}: STAGE ocr ...", flush=True)
                t0 = time.perf_counter()
                pages, det = _ocr(j, image_dir, backend, opts)
                asm_q.put((j, pages, det, time.perf_counter() - t0))
            except Exception as e:
                with lock:
                    results[j.name] = f"FAIL ocr: {e}"
                print(f"[{_ts()}] {j.name}: FAIL ocr: {e}", flush=True)
        asm_q.put(_SENTINEL)

    def assembler():
        while True:
            item = asm_q.get()
            if item is _SENTINEL:
                break
            j, pages, det, ocr_secs = item
            try:
                print(f"[{_ts()}] {j.name}: STAGE assembly ...", flush=True)
                _assemble(j, pages, det, opts)
                # Health-gate before counting this book done: a broken/empty run
                # (e.g. interrupted OCR -> empty results stub -> 0-block doc) fails
                # loudly and is not exported — the e3 failure mode.
                hlth = corpus.health(j.out)
                corpus.write_marker(j.out, hlth)
                if not hlth.ok:
                    with lock:
                        results[j.name] = "FAIL health: " + "; ".join(hlth.reasons)
                    print(f"[{_ts()}] {j.name}: HEALTH FAIL — "
                          + "; ".join(hlth.reasons) + " (needs re-OCR)", flush=True)
                    continue
                dest = corpus.corpus_dir(getattr(opts, "corpus_dir", None))
                if dest is not None:
                    try:
                        corpus.upsert_manifest_row(dest, corpus.export_book(j.out, dest))
                    except Exception as e:
                        print(f"[{_ts()}] {j.name}: corpus export skipped: {e}",
                              flush=True)
                pg = len(pages)
                spp = ocr_secs / pg if pg else 0.0
                with lock:
                    results[j.name] = "ok"
                    totals["pages"] += pg
                    totals["secs"] += ocr_secs
                    overall = totals["secs"] / totals["pages"] if totals["pages"] else 0.0
                    tot_pages = totals["pages"]
                print(f"[{_ts()}] {j.name}: done — {pg} pages, {spp:.2f} s/pg  "
                      f"(overall {overall:.2f} s/pg over {tot_pages} pages)",
                      flush=True)
            except Exception as e:
                with lock:
                    results[j.name] = f"FAIL assemble: {e}"
                print(f"[{_ts()}] {j.name}: FAIL assemble: {e}", flush=True)

    with backend.persistent_servers(log_dir):
        threads = [threading.Thread(target=t, name=t.__name__)
                   for t in (rasterizer, ocrer, assembler)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return results

#!/usr/bin/env python3
"""End-to-end: OCR a folder of page images, then emit both bare text and a
single searchable PDF.

The OCR engine is pluggable (see ``ocr_backend.py``): pick one with
``--backend`` (default ``surya``). Each backend owns its own parallelism /
scaling / caching — you pass options into the generic layer with ``--instances``
(a parallelism hint) and/or ``--backend-opt KEY=VALUE`` (repeatable, engine
specific), and they're forwarded to the chosen backend. Assembly (text + PDF)
is backend-agnostic, so adding Tesseract / a cloud engine touches only a new
backend module.

The input may be a folder of page images OR a ``.pdf`` — a PDF is rasterized to
page images (``lib/pdf_to_images.py``, with two-up scan splitting) and then runs
down the same image path, so it also repairs mojibake born-digital PDFs by
re-OCR'ing the rendered glyphs. Tune render resolution with ``--dpi``.

Pass MORE than one book — multiple inputs, a folder of PDFs, or a ``.txt``
worklist of paths — to run a BATCH (re-OCR many books): each goes to
``<--out>/<name>/``, failures are isolated, ``--skip-done`` resumes. The
single-book pipeline (``run_one``) is shared, so batch gains every capability
(two-up, doc.json, footnote/endnote resolution, ...) for free.

Usage:
    python3 ocr_pipeline/ocr_one.py "/path/to/Steps 4" --out out/steps4
    python3 ocr_pipeline/ocr_one.py "/path/to/Book.pdf" --out out/book   # PDF in
    python3 ocr_pipeline/ocr_one.py /folder/of/pdfs --out out --skip-done # batch
    python3 ocr_pipeline/ocr_one.py worklist.txt --out out               # batch
    python3 ocr_pipeline/ocr_one.py a.pdf b.pdf c.pdf --out out          # batch
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))   # helper modules live in lib/
import surya_backend as sb
import ocr_backend
import ocr_surya  # noqa: F401  — registers the "surya" backend
import proc_group
from ocr_to_text import build_text
from ocr_to_pdf import build_pdf
from pdf_to_images import pdf_to_images, list_pdfs
from build_doc import build_doc
import page_lang
import preprocess               # pad / pages / rotate / swap (shared)
import add_pages                # --add-pages-after: hand-added pages merged post-preprocess


def _ts():
    """Wall-clock HH:MM:SS for per-stage progress timestamps."""
    return time.strftime("%H:%M:%S")


def _nonlatin_fraction(text):
    """(fraction of letters that are NOT Latin/IAST, total letter count).
    Latin/IAST = Basic/Latin-1/Latin-Extended (incl. ā ī ñ ś) + Latin Extended
    Additional (ṃ ṇ ṛ ṣ ṭ ...). Other scripts (Tibetan, Devanagari, Odia,
    Bengali, ...) count as non-Latin — which, from this engine, means
    hallucinated garbage on script it can't read."""
    latin = other = 0
    for ch in text:
        if not ch.isalpha():
            continue
        o = ord(ch)
        if o < 0x250 or 0x1E00 <= o <= 0x1EFF:
            latin += 1
        else:
            other += 1
    tot = latin + other
    return (other / tot if tot else 0.0), tot


def drop_nonlatin_pages(pages, *, threshold=0.5, min_letters=20):
    """Blank pages whose OCR text is mostly non-Latin (kept image-only in the
    PDF, empty in the text). Returns the list of blanked page names."""
    dropped = []
    for pg in pages:
        frac, tot = _nonlatin_fraction(pg.text)
        if tot >= min_letters and frac >= threshold:
            pg.blocks = []
            dropped.append(pg.name)
    return dropped


_NUM_RE = re.compile(r"^(.*?)(\d+)(\.[^.]+)$")


def inconsistent_number_widths(image_dir):
    """Returns {(prefix, ext): sorted([widths])} for filename groups whose
    trailing numbers have more than one digit width (e.g. '9' and '200')."""
    groups = {}
    for p in sb.list_images(image_dir):
        m = _NUM_RE.match(p.name)
        if m:
            groups.setdefault((m.group(1), m.group(3).lower()), set()).add(
                len(m.group(2)))
    return {k: sorted(w) for k, w in groups.items() if len(w) > 1}


def _parse_opts(pairs) -> dict:
    """``KEY=VALUE`` strings -> dict, with int/float/bool coercion."""
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--backend-opt must be KEY=VALUE, got {pair!r}")
        k, v = pair.split("=", 1)
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        elif v.isdigit():
            v = int(v)
        else:
            try:
                v = float(v)
            except ValueError:
                pass
        out[k.strip()] = v
    return out


# --pdf-size presets: bundle the resolution/quality/colour knobs.
PDF_SIZE_PRESETS = {
    "small":  {"pdf_max_dim": 1600, "pdf_image_quality": 90, "color": "grey"},
    "medium": {"pdf_max_dim": 2000, "pdf_image_quality": 90, "color": "grey"},
    "large":  {"pdf_max_dim": 0,    "pdf_image_quality": 90, "color": "grey"},
}


class _HelpFormatter(argparse.RawDescriptionHelpFormatter,
                     argparse.ArgumentDefaultsHelpFormatter):
    """Keep the raw epilog AND append '(default: ...)' to every option (but not
    to required ones, where 'default: None' would be misleading)."""

    def _get_help_string(self, action):
        if getattr(action, "required", False):
            return action.help
        return argparse.ArgumentDefaultsHelpFormatter._get_help_string(self, action)


def resolve_inputs(inputs):
    """Expand the positional inputs into a flat list of book sources (a .pdf
    file, or a folder of page images). Triggers BATCH when there's more than one:
      * a .pdf file            -> that book
      * a folder of PDFs       -> one book per PDF
      * a folder of images     -> that one book
      * a .txt/.list worklist  -> one path per non-comment line
    """
    out = []
    for inp in inputs:
        inp = Path(inp)
        if inp.is_file() and inp.suffix.lower() == ".pdf":
            out.append(inp)
        elif inp.is_file() and inp.suffix.lower() in (".txt", ".list"):
            for line in inp.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    out.append(Path(line))
        elif inp.is_dir():
            pdfs = list_pdfs(inp)
            out.extend(pdfs if pdfs else [inp])     # folder of PDFs -> batch
        else:
            out.append(inp)                          # let run_one report the error
    return out


def manifest_row(out, name, doc, source=None):
    """One ingestion-manifest record for a finished book: the doc.json (RAG
    contract), the verses.json sidecar, the searchable PDF, and the body text /
    pages.jsonl, plus doc_id + block count. Paths are absolute; missing optional
    artifacts are null."""
    out = Path(out)
    body = out / "body" / "all_pages.txt"
    pj = out / "body" / "pages.jsonl"
    return {
        "edition": out.name,
        "title": (doc or {}).get("title"),
        "doc_id": (doc or {}).get("doc_id"),
        "source": (doc or {}).get("source") or (str(source) if source else None),
        "schema_version": (doc or {}).get("schema_version"),
        "features": (doc or {}).get("features", []),
        "n_blocks": len((doc or {}).get("blocks", [])),
        "doc_json": str((out / f"{name}.doc.json").resolve()),
        "pdf": str((out / f"{name}.pdf").resolve()) if (out / f"{name}.pdf").exists() else None,
        "body_text": str(body.resolve()) if body.exists() else None,
        "pages_jsonl": str(pj.resolve()) if pj.exists() else None,
    }


def append_manifest(out, name, doc, source=None):
    """Append this finished book to ``<out-root>/ingest_manifest.jsonl`` (the
    parent of the per-book dir) so the RAG ingestion/embedding task can tail it.
    Best-effort — never fails the run."""
    try:
        manifest = Path(out).parent / "ingest_manifest.jsonl"
        with open(manifest, "a") as f:
            f.write(json.dumps(manifest_row(out, name, doc, source),
                               ensure_ascii=False) + "\n")
        print(f"Manifest += {Path(out).name} -> {manifest}", flush=True)
    except Exception as e:
        print(f"  (manifest append skipped: {e})", flush=True)


# Processes that do the GPU OCR work (recognition + Surya detection), so the
# per-stage profile can attribute the ocr/detection stages to them specifically.
_OCR_PROC_MATCH = ("llama-server", "surya")


def _perf_print(sample, stage, book):
    """Emit one machine sample as a [PERF] JSON line on the SAME stream as the
    task log, so perf and task events (incl. 'OCR checkpoint ...') share one
    wall-clock timeline — to see what the app was doing for any slice you read
    adjacent lines, no cross-file join. Self-describing: time, book, stage,
    machine metrics, and llama/surya/tree proc aggregates."""
    procs = sample.get("procs") or {}
    llama = sum(r["cpu_pct"] for r in procs.values()
                if "llama-server" in r.get("name", "").lower())
    # Match the surya_ocr/surya_detect *executable*, not just "surya" — the
    # llama-server's cmd contains the model path "surya-ocr-2-gguf" and would
    # otherwise be double-counted here.
    surya = sum(r["cpu_pct"] for r in procs.values()
                if "surya_ocr" in r.get("cmd", "").lower()
                or "surya_detect" in r.get("cmd", "").lower())
    rec = {"iso": time.strftime("%H:%M:%S",
                                time.localtime(sample.get("t_wall", 0.0))),
           "book": book, "stage": stage, "nproc": len(procs),
           "llama_cpu": round(llama, 1), "surya_cpu": round(surya, 1),
           "tree_cpu": round(sum(r["cpu_pct"] for r in procs.values()), 1),
           "tree_rss": round(sum(r["rss_gb"] for r in procs.values()), 2)}
    for k, v in sample.items():
        if k.startswith("sys.") or k.startswith("gpu."):
            rec[k] = round(v, 2)
    print("[PERF] " + json.dumps(rec), flush=True)


def _perf_start(args, name):
    """Begin streaming per-stage machine samples for this book if --perf-stats is
    set. Returns a started StageProfiler over this process's subtree (which owns
    the llama-server / surya children); each sample prints as a [PERF] line.
    Never raises — opt-in instrumentation must not affect OCR."""
    if not getattr(args, "perf_stats", False):
        return None
    try:
        import perfprofile
        prof = perfprofile.StageProfiler(
            root_pid=os.getpid(), interval=args.perf_interval,
            on_sample=lambda s, stage: _perf_print(s, stage, name))
        prof.add_detector(perfprofile.DipDetector())   # GPU dips, classified
        return prof.start()
    except Exception as e:
        print(f"[PERF] disabled: {e}", flush=True)
        return None


def _perf_finish(prof, name, npages):
    """Stop sampling and print a [PERF] SUMMARY line per stage (aggregated +
    interpreted, including the GPU duty-cycle / idle-episode count that exposes a
    checkpoint sawtooth) into the same log. Guarded — never breaks the run."""
    if prof is None:
        return
    try:
        prof.stop()
        rep = prof.report()
        for stage, p in rep["stages"].items():
            g = (p["machine"].get("gpu.util_pct") or {})
            mem = (p["machine"].get("sys.mem_avail_gb") or {})
            v = p["interpretation"]
            duty = (p.get("duty", {}) or {}).get("gpu", {})
            rec = {
                "summary": True, "book": name, "stage": stage,
                "duration_s": p["duration_s"], "samples": p["samples"],
                "gpu_mean": g.get("mean"), "gpu_max": g.get("max"),
                # busy_mean = GPU during work (the saturation signal for instance
                # decisions); idle_frac = the warmup/stall overhead.
                "gpu_busy_mean": duty.get("busy_mean"),
                "gpu_idle_frac": duty.get("idle_frac"),
                "gpu_idle_episodes": duty.get("episodes"),
                "mem_avail_min_gb": mem.get("min"),
                "swap_delta_gb": p.get("deltas", {}).get("sys.swap_used_gb"),
                "bottleneck": v["bottleneck"], "swapping": v["swapping"]}
            if stage == "ocr" and npages:          # throughput, the decision metric
                rec["pages"] = npages
                rec["s_per_page"] = round(p["duration_s"] / npages, 2)
            print("[PERF] " + json.dumps(rec), flush=True)
        for f in rep.get("findings", []):          # detector findings (dips, ...)
            print("[PERF] " + json.dumps({"finding": True, "book": name, **f}),
                  flush=True)
    except Exception as e:
        print(f"[PERF] summary failed: {e}", flush=True)


def _parse_page_list(spec):
    """'3,5,100-110' -> frozenset of 1-based page indices (for --no-uniform-pages)."""
    out = set()
    try:
        for tok in str(spec).split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                a, b = tok.split("-", 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(tok))
    except ValueError:
        raise SystemExit(f"--no-uniform-pages: bad page list {spec!r} (use comma-"
                         "separated 1-based indices/ranges, e.g. '3,5,100-110')")
    return frozenset(out)


def run_one(args, source, out):
    """Process one book (a .pdf or an image folder) into ``out``. A PDF is
    rasterized into ``out/_pages`` first (kept local, not next to the source)."""
    source = Path(source)
    doc_source = source
    # Per-book layout: start from the CLI flags, let a matching --layout-overrides
    # entry (by filename stem) override them. Two-up splits at rasterize time;
    # columns>=2 switches the OCR pass to column-strip re-OCR (below).
    two_up, columns, drop_pages = args.two_up, args.columns, args.drop_pages
    if args.layout_overrides:
        import layout_overrides as lo
        d = lo.lookup(lo.load(args.layout_overrides), stem=source.stem)
        two_up = args.two_up if args.two_up != "off" else d.get("two_up", "off")
        columns = args.columns if args.columns != 1 else d.get("columns", 1)
        drop_pages = args.drop_pages or d.get("drop_pages")
    if two_up != "off" or columns >= 2 or drop_pages:
        print(f"Layout: two-up={two_up}, columns={columns}, "
              f"drop-pages={drop_pages} ({source.stem})", flush=True)

    # Name is known up front (PDF stem / folder name) so the perf profiler can
    # start before rasterize and tag every stage, including rasterize.
    is_pdf = source.is_file() and source.suffix.lower() == ".pdf"
    name = source.stem if is_pdf else source.name
    prof = _perf_start(args, name)      # per-stage machine profiler (or None)

    if is_pdf:
        if out is None:
            raise SystemExit("--out is required for PDF input")
        if prof:
            prof.enter_stage("rasterize")
        print(f"[{_ts()}] {name}: STAGE rasterize — {source.name} @ {args.dpi} DPI ...",
              flush=True)
        _ras0 = time.perf_counter()
        image_dir = pdf_to_images(source, out_dir=Path(out) / "_pages",
                                  dpi=args.dpi, force=args.force, two_up=two_up)
        if prof:
            prof.record("rasterize", _ras0, time.perf_counter())
    else:
        image_dir = source

    if not image_dir.is_dir():
        raise SystemExit(f"not a folder: {image_dir}")

    if out is None:
        raise SystemExit("--out is required")
    out = Path(out)

    # --prepare-output: assembly-only. Resolve the dir the OCR stage actually used
    # (via the .preprocess_final.json marker — added pages were folded in there during
    # OCR) and rebuild the deliverables from its cached OUTPUT (results.json +
    # detection.json). NO preprocess (which would clobber the marker), NO OCR — so you
    # can regenerate the PDF/text/doc with different appearance options without re-OCR.
    if getattr(args, "prepare_output", False):
        base = (out / "_pages") if is_pdf else Path(image_dir)
        final = preprocess.final_dir(base)
        cache = final / ".surya_ocr"
        if not sb.usable_cache(cache / "results.json"):
            raise SystemExit(
                f"--prepare-output needs an existing OCR cache at "
                f"{cache/'results.json'} — run OCR on this book first.")
        pages = sb.load_pages(cache / "results.json", final)
        det_by_stem = (sb.load_line_boxes(cache / "detection.json", final)
                       if (cache / "detection.json").exists() else {})
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{_ts()}] {name}: prepare-output — {len(pages)} page(s) from the OCR "
              f"cache in {final.name}/ (no OCR).", flush=True)
        _assemble(args, pages, det_by_stem, out, name, doc_source,
                  t_ocr=0.0, t_detect=0.0, t_start=time.perf_counter(), prof=prof)
        return

    # Page-number padding is ON by default (disable with '-- --no-pad-page-numbers').
    pad_on = args.pp.pad_page_numbers if args.pp is not None else True
    will_preprocess = pad_on or (args.pp is not None and (args.pp.rotate or args.pp.swap))
    if not pad_on:                              # only a risk when padding is disabled
        bad = inconsistent_number_widths(image_dir)
        if bad:
            ex = next(iter(bad.values()))
            print("WARNING: numbered filenames have inconsistent digit widths "
                  f"(e.g. {ex[0]} vs {ex[-1]} digits) and --no-pad-page-numbers is "
                  "set. Page order may come out wrong in some viewers; drop "
                  "--no-pad-page-numbers to zero-pad them.", flush=True)
            if sys.stdin.isatty():
                if input("Continue without padding? [y/N] ").strip().lower() \
                        not in ("y", "yes"):
                    raise SystemExit("Aborted. Drop --no-pad-page-numbers to pad.")
            else:
                print("(non-interactive input; continuing anyway)", flush=True)

    images = sb.list_images(image_dir)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"no page images in {image_dir}")

    opts = {"instances": args.instances, **_parse_opts(args.backend_opt)}
    backend = ocr_backend.get_backend(args.backend, **opts)

    # Overlap model/server warmup with image preprocessing: the backend's servers
    # (the GGUF load — the bulk of cold start) don't depend on the page images, so
    # start them in a thread while the rotate/swap/pad file copies run, then join
    # before OCR. Only worth it when there's preprocessing to hide behind. Servers
    # are proc_group-tracked, so a crash still reaps them; shutdown frees them early.
    warm_box, warm_thread = {}, None
    if (will_preprocess and not getattr(args, "dump_input_files", None)
            and not getattr(args, "prepare_output", False)):
        def _warm():
            try:
                warm_box["handle"] = backend.warmup(Path(out) / "_servers")
            except BaseException as e:                 # port clash, health timeout…
                warm_box["err"] = e
        warm_thread = threading.Thread(target=_warm, name=f"warmup-{name}",
                                       daemon=True)
        warm_thread.start()
        print(f"[{_ts()}] {name}: warming OCR servers in parallel with "
              "preprocessing ...", flush=True)

    def _join_warm():
        """Join the warmup thread once (idempotent); on failure fall back to the
        launch-at-OCR path (pool left unset)."""
        nonlocal warm_thread
        if warm_thread is None:
            return
        warm_thread.join()
        warm_thread = None
        if "err" in warm_box:
            print(f"[{_ts()}] {name}: server warmup failed "
                  f"({warm_box['err']}); launching at OCR instead.", flush=True)
            backend.shutdown(None)
            warm_box.pop("handle", None)

    # Image pre-processing — pad -> rotate/swap — all via the shared preprocess
    # utility, which materializes sibling dirs and repoints image_dir/images so OCR,
    # column strips, line detection and the output PDF all use the corrected images.
    # Padding runs by default (pad_on) even with no other steps, so every entry point
    # sees one canonical padded numbering. Runs in parallel with the Surya warmup.
    pp = args.pp
    if will_preprocess:
        image_dir, images = preprocess.preprocess_images(
            image_dir, images,
            pad=pad_on,
            pages=(pp.pages if pp is not None else None),
            rotate=(pp.rotate if pp is not None else None),
            swap=(pp.swap if pp is not None else None),
            order=(preprocess.step_order(pp) if pp is not None else ()),
            force=args.force,
            log=lambda m: print(f"[{_ts()}] {name}: {m}", flush=True))

    # Hand-added pages (--add-pages-after): a durable INPUT layered on AFTER
    # preprocessing (so they are never rotated/swapped). They live in <source>/_added/
    # — outside the preprocess step-dir chain, so --force can't clear them. Presence
    # in _added/ is what includes them; the flag just puts them there (idempotently).
    # OCR'd on the warm pool and merged into `pages`/detection further below.
    if getattr(args, "add_pages_after", None):
        for group in args.add_pages_after:
            add_pages.add(source, images, group[0], group[1:],
                          log=lambda m: print(f"[{_ts()}] {name}: {m}", flush=True))
    _added_dir = add_pages.added_dir(source)
    if _added_dir.is_dir():                    # heal any exact-duplicate inserts
        add_pages.dedupe_added(
            _added_dir, log=lambda m: print(f"[{_ts()}] {name}: {m}", flush=True))
    added_imgs = sb.list_images(_added_dir) if _added_dir.is_dir() else []
    if added_imgs:
        # Fold hand-added pages into the OCR INPUT: copy the durable _added/ images
        # into the final image dir so they are OCR'd in the MAIN pass and land in the
        # ONE cache (add-pages is a pre-OCR input step; the OCR stage then produces a
        # single output for assembly). _added/ stays the durable source — these copies
        # are re-derived after --force (which rebuilds the step dir under them).
        for a in added_imgs:
            dst = image_dir / a.name
            if not (dst.exists() and dst.stat().st_mtime >= a.stat().st_mtime):
                shutil.copy2(a, dst)
        images = sb.list_images(image_dir)     # now includes the added pages
        print(f"[{_ts()}] {name}: folded {len(added_imgs)} hand-added page(s) from "
              f"{_added_dir.name}/ into the OCR input.", flush=True)

    # --dump-input-files START N: copy N input images (the exact post-preprocess +
    # post-add sequence) starting at START into ./input_dump/, then quit (no OCR).
    if getattr(args, "dump_input_files", None):
        merged = list(images)                  # already includes added pages, sorted
        start_raw, n_raw = args.dump_input_files
        try:
            n = int(n_raw)
        except ValueError:
            raise SystemExit(f"--dump-input-files: N must be an integer, got {n_raw!r}")
        if n < 1:
            raise SystemExit("--dump-input-files: N must be >= 1")
        idx = add_pages.resolve_start([p.name for p in merged], start_raw)
        chosen = merged[idx:idx + n]
        dump = Path.cwd() / "input_dump"
        if dump.exists():
            shutil.rmtree(dump)
        dump.mkdir(parents=True)
        for seq, p in enumerate(chosen, 1):
            shutil.copy2(p, dump / f"{seq:04d}_{p.name}")
        print(f"\n[{name}] dumped {len(chosen)} of {len(merged)} input image(s) "
              f"from #{idx + 1} ({merged[idx].name!r}) -> {dump}", flush=True)
        for seq, p in enumerate(chosen, 1):
            print(f"  {seq:04d}_{p.name}", flush=True)
        sys.exit(0)

    # Skip OCR on verso/script pages (drop-pages). 'even'/'odd' skip by 1-based
    # parity; 'non-latin' decides PER PAGE from the PDF's own embedded text (skip a
    # page with no meaningful Latin/English). An optional :LO-HI range limits where
    # the rule applies — front/back matter outside it is always OCR'd. Skipped pages
    # stay image-only: Surya loops on unreadable script and the text drops anyway.
    ocr_imgs, skip_imgs = images, []
    _dp_mode, _dp_range = page_lang.parse_drop_pages(drop_pages)
    if _dp_mode in ("even", "odd", "non-latin"):
        import fitz
        ocr_imgs, skip_imgs = [], []

        def _pno(img):
            m = re.search(r"(\d+)", Path(img).stem)
            return int(m.group(1)) if m else None

        texts = {}
        if _dp_mode == "non-latin" and is_pdf:
            try:
                with fitz.open(source) as doc:
                    for img in images:
                        p = _pno(img)
                        if p and 1 <= p <= doc.page_count:
                            texts[p] = doc[p - 1].get_text()
            except Exception:
                texts = {}
        for img in images:
            p = _pno(img)
            if _dp_range and p is not None and not (_dp_range[0] <= p <= _dp_range[1]):
                ocr_imgs.append(img)                   # outside the range -> OCR
                continue
            if _dp_mode == "even":
                skip = p is not None and p % 2 == 0
            elif _dp_mode == "odd":
                skip = p is not None and p % 2 == 1
            else:                                       # non-latin: per-page content
                skip = page_lang.page_skip_nonlatin(texts.get(p, ""))
            (skip_imgs if skip else ocr_imgs).append(img)
        _rng = f" within pages {_dp_range[0]}-{_dp_range[1]}" if _dp_range else ""
        print(f"Skipping OCR of {len(skip_imgs)} {_dp_mode} page(s){_rng} "
              f"(script/non-Latin verso kept image-only); OCR'ing {len(ocr_imgs)}.",
              flush=True)

    # Ensure the warmup thread is joined (idempotent — auto-unscramble may already
    # have): the servers are warm, so OCR reuses them with no cold start.
    _join_warm()
    warm_handle = warm_box.get("handle")

    print(f"Backend: {backend.name}  ({len(ocr_imgs)} pages, --lines {args.lines}, "
          f"--text {args.text})", flush=True)
    for k, v in backend.config().items():
        print(f"  {k}: {v}", flush=True)

    t_start = time.perf_counter()
    t0 = time.perf_counter()
    if prof:
        prof.enter_stage("ocr")
    print(f"[{_ts()}] {name}: STAGE ocr — {len(ocr_imgs)} page(s)"
          + (f", skipping {len(skip_imgs)}" if skip_imgs else "") + " ...",
          flush=True)
    if getattr(args, "text_source", "ocr") == "native":
        # Detect-only path for digital-first books: read the clean embedded text
        # (no recognition -> no diacritic degradation) and label regions with
        # surya_layout, so the doc.json matches the scanned books'.
        import native_text as nt
        if not is_pdf:
            raise SystemExit("--text-source native needs PDF input (the embedded "
                             "text layer); image folders have none.")
        print("Native-text path: embedded text + surya_layout regions "
              "(no recognition).", flush=True)
        want = {Path(p).name for p in ocr_imgs}
        pages = [p for p in nt.native_pages(source, dpi=args.dpi) if p.name in want]
        layout = backend.layout_regions(ocr_imgs, image_dir, force=args.force)
        pages, n_lab = nt.merge_layout(pages, layout)
        print(f"  {len(pages)} page(s) native text; {n_lab} blocks labeled by "
              f"surya_layout.", flush=True)
    elif columns >= 2:
        # Column-strip re-OCR: OCR each column on its own (a single column never
        # scrambles), then recombine onto the original page in column order.
        # Detection (PDF line boxes, below) still runs on the whole page.
        import column_strips as cs
        strips_dir = image_dir / "_strips"
        mapping = cs.make_strips(ocr_imgs, strips_dir, columns, force=args.force)
        strip_imgs = sb.list_images(strips_dir)
        print(f"Column-strip OCR: {len(ocr_imgs)} pages -> {len(strip_imgs)} "
              f"{columns}-column strips.", flush=True)
        strip_pages = backend.ocr_pages(strip_imgs, strips_dir, force=args.force)
        pages = cs.recombine(strip_pages, mapping, ocr_imgs)
    else:
        pages = backend.ocr_pages(ocr_imgs, image_dir, force=args.force,
                                  reocr=args.reocr)
    if skip_imgs:                          # re-insert skipped pages as image-only
        from PIL import Image as _Image
        for img in skip_imgs:
            try:
                w, h = _Image.open(img).size
            except Exception:
                w, h = 1000, 1000
            pages.append(sb.Page(img, img.name, int(w), int(h), [], None))
        pages.sort(key=lambda p: sb.natural_key(p.name))
    t_ocr = time.perf_counter() - t0
    if prof:
        prof.record("ocr", t0, t0 + t_ocr, match=_OCR_PROC_MATCH)

    t0 = time.perf_counter()
    if prof:
        prof.enter_stage("detection")
    det_by_stem = {}
    if getattr(args, "text_source", "ocr") == "native":
        if args.lines == "detect":           # per-line boxes from the PDF's own text
            import native_text as nt
            stems = {Path(p).stem for p in ocr_imgs}
            det_by_stem = {k: v for k, v in
                           nt.native_line_boxes(source, dpi=args.dpi).items()
                           if k in stems}
    elif args.lines == "detect":
        print(f"[{_ts()}] {name}: STAGE detection — line boxes ...", flush=True)
        det_by_stem = backend.line_boxes(ocr_imgs, image_dir, force=args.force,
                                         reocr=args.reocr) or {}
    t_detect = time.perf_counter() - t0
    if prof:
        prof.record("detection", t0, t0 + t_detect, match=_OCR_PROC_MATCH)
    # OCR + detection done — free the warm servers before the (CPU/IO-bound)
    # assembly stage. A crash before here still reaps them via proc_group.
    if warm_handle is not None:
        backend.shutdown(warm_handle)
    _assemble(args, pages, det_by_stem, out, name, doc_source,
              t_ocr=t_ocr, t_detect=t_detect, t_start=t_start, prof=prof)


def _assemble(args, pages, det_by_stem, out, name, doc_source, *,
              t_ocr, t_detect, t_start, prof):
    """Assembly stage — turn the OCR stage's OUTPUT (pages + line boxes) into the
    deliverables: body/full text, doc.json, and the searchable PDF. Purely downstream
    of OCR (takes only what the OCR stage produced); this is also the entry the
    standalone create_output.py / --prepare-output path drives."""
    t0 = time.perf_counter()
    if prof:
        prof.enter_stage("assembly")
    print(f"[{_ts()}] {name}: STAGE assembly — text/doc/pdf ...", flush=True)
    if args.drop_nonlatin:
        blanked = drop_nonlatin_pages(pages, threshold=args.nonlatin_threshold)
        if blanked:
            print(f"Dropped {len(blanked)} non-Latin/hallucinated page(s) "
                  f"(kept image-only, no text layer).", flush=True)

    body_labels = {s.strip() for s in args.drop_labels.split(",") if s.strip()}
    text_mode = args.text
    out.mkdir(parents=True, exist_ok=True)

    emit_opts = dict(ascii_dashes=args.ascii_dashes,
                     strip_footnote_refs=args.strip_footnote_refs)
    if text_mode == "both":
        build_text(pages, out / "full", drop=set(), **emit_opts)
        npages, dropped = build_text(pages, out / "body", drop=body_labels,
                                     **emit_opts)
        text_note = (f"full text -> full/, body text -> body/ "
                     f"(dropped {dropped} non-body regions)")
    elif text_mode == "body":
        npages, dropped = build_text(pages, out, drop=body_labels, **emit_opts)
        text_note = f"body-only text (dropped {dropped} non-body regions)"
    else:  # full
        npages, dropped = build_text(pages, out, drop=set(), **emit_opts)
        text_note = "full text"

    if args.doc:
        doc = build_doc(pages, out / f"{name}.doc.json",
                        source=doc_source, title=args.title,
                        det_by_stem=det_by_stem)
        ntypes = {}
        for b in doc["blocks"]:
            ntypes[b["type"]] = ntypes.get(b["type"], 0) + 1
        print(f"Structured doc: {name}.doc.json ({len(doc['blocks'])} blocks: "
              + ", ".join(f"{k}={v}" for k, v in sorted(ntypes.items())) + ")",
              flush=True)

    common = dict(font_path=args.font, det_by_stem=det_by_stem, lines=args.lines,
                  uniform=args.uniform, uniform_except=args.uniform_except,
                  contrast=args.contrast)
    if args.all_pdf_sizes:
        per_tier = max(1, (os.cpu_count() or 6) // 3)

        def build_tier(tier):
            p = PDF_SIZE_PRESETS[tier]
            tout = out / f"{name}.{tier}.pdf"
            build_pdf(pages, tout, color=p["color"], max_dim=p["pdf_max_dim"],
                      image_quality=p["pdf_image_quality"], quiet=True,
                      encode_workers=per_tier, **common)
            return tier, tout
        with ThreadPoolExecutor(max_workers=3) as ex:
            made = list(ex.map(build_tier, ("small", "medium", "large")))
        pdf_note = "PDFs: " + ", ".join(
            f"{o.name} ({o.stat().st_size/1048576:.0f} MB)" for _, o in made)
    else:
        pdf_out = out / f"{name}.pdf"
        color = "grey" if args.color == "gray" else args.color
        build_pdf(pages, pdf_out, color=color, max_dim=args.pdf_max_dim,
                  image_quality=args.pdf_image_quality, **common)
        pdf_note = f"{pdf_out.name} ({pdf_out.stat().st_size/1048576:.0f} MB)"
    t_assemble = time.perf_counter() - t0
    total = time.perf_counter() - t_start
    if prof:
        prof.record("assembly", t0, t0 + t_assemble)

    print(f"\nDone: {npages} pages -> {out}/  ({pdf_note} + {text_note})", flush=True)
    print("Timing:", flush=True)
    print(f"  OCR pass:       {t_ocr:7.1f}s  ({t_ocr/npages:.2f}s/page)", flush=True)
    print(f"  Detection pass: {t_detect:7.1f}s"
          + ("  (skipped, --lines split)" if args.lines != "detect" else ""),
          flush=True)
    print(f"  Assembly:       {t_assemble:7.1f}s", flush=True)
    print(f"  Total:          {total:7.1f}s", flush=True)
    try:                               # per-book timing sidecar for the driver
        (out / ".run_timing.json").write_text(json.dumps(
            {"pages": npages, "ocr_secs": round(t_ocr, 2),
             "detect_secs": round(t_detect, 2),
             "assemble_secs": round(t_assemble, 2), "total_secs": round(total, 2)}))
    except Exception:
        pass
    if args.doc:                       # append to the ingestion manifest (tailable)
        append_manifest(out, name, doc, doc_source)
        # Verify the run actually produced usable output, then mirror the durable
        # artifacts into the corpus. A broken/empty run (e.g. an interrupted OCR
        # that left an empty results.json -> 0-block doc) FAILS LOUDLY here and is
        # not exported, so it can never be silently treated as "done" — the e3
        # failure mode. Health is content-based, not file-existence.
        import corpus
        h = corpus.health(out)
        corpus.write_marker(out, h)
        if not h.ok:
            print(f"\n*** OCR HEALTH CHECK FAILED for {name}: "
                  + "; ".join(h.reasons), flush=True)
            print("    Not exported to corpus; this book needs re-OCR and will "
                  "NOT be treated as done.", flush=True)
            _perf_finish(prof, name, npages)
            raise SystemExit(3)
        dest = corpus.corpus_dir(getattr(args, "corpus_dir", None))
        if dest is not None:
            try:
                row = corpus.export_book(out, dest)
                corpus.upsert_manifest_row(dest, row)
                print(f"Corpus += {name} ({h.n_blocks} blocks) -> {dest}", flush=True)
            except Exception as e:
                print(f"  (corpus export skipped: {e})", flush=True)
    _perf_finish(prof, name, npages)
    return npages


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=_HelpFormatter, epilog=__doc__)
    g_io = ap.add_argument_group("input & preprocessing")
    g_io.add_argument("inputs", type=Path, nargs="+", metavar="INPUT",
                      help="One or more inputs: a folder of page images, a .pdf "
                           "file, a folder of PDFs, or a .txt worklist of paths. "
                           "More than one book (multiple inputs / a folder of "
                           "PDFs / a worklist) runs in BATCH: each book goes to "
                           "<--out>/<name>/, failures are isolated, and a summary "
                           "is printed at the end.")
    g_io.add_argument("--skip-done", action="store_true",
                      help="In batch, skip a book whose output PDF already exists.")
    g_io.add_argument("--dpi", type=int, default=300,
                      help="When the input is a PDF, render resolution for "
                           "rasterizing pages (zoom = dpi/72). Keep high (300+); "
                           "Surya upscales and low-res mangles diacritics.")
    g_io.add_argument("--two-up", choices=["auto", "on", "off"], default="off",
                      help="For PDF input, split two-up scans (two book pages on "
                           "one landscape sheet) into separate page images. off "
                           "(default) = never — the safe default; gutter detection "
                           "mis-splits ordinary landscape plates/tables. on = split "
                           "every landscape page (use for a known two-up book); "
                           "auto = only landscape pages with a detected central "
                           "gutter (heuristic, mistake-prone). Prefer naming two-up "
                           "books in a --layout-overrides manifest.")
    g_io.add_argument("--columns", type=int, default=1, metavar="N",
                      help="Column-strip re-OCR: OCR each page one column at a "
                           "time (N columns) and recombine in column order, fixing "
                           "the multi-column reading-order scramble at the source. "
                           "1 (default) = whole-page OCR. Use 2 for two-column "
                           "body text; prefer a --layout-overrides manifest.")
    g_io.add_argument("--drop-pages", default=None, metavar="SPEC",
                      help="Skip OCR of certain pages, keeping them image-only. SPEC: "
                           "'even'/'odd' (skip that 1-based parity), or 'non-latin' "
                           "(skip pages whose embedded text has no meaningful Latin/"
                           "English — auto-detected per page), with an optional page "
                           "range ':LO-HI' (e.g. 'even:10-300', 'non-latin'). For "
                           "bilingual books with Tibetan/CJK on the verso, OCR'ing "
                           "those pages loops on unreadable script and the text is "
                           "dropped anyway. 'non-latin' is safest — it won't skip "
                           "English front/back matter on the verso side.")
    # Image preprocessing: everything after a `--` (or `--preprocess`) marker is
    # forwarded verbatim to the shared preprocess utility, applied post-rasterization,
    # before OCR (peeled off with preprocess.split_forward before argparse). E.g.
    # `ocr_one.py <src> --out DIR -- --rotate 180:CW --swap 9:end`.
    g_io.add_argument("--layout-overrides", type=Path, default=None, metavar="FILE",
                      help="Per-book layout manifest (two-up / columns / drop-pages "
                           "directives keyed by edition:ID or filename stem; see "
                           "lib/layout_overrides.py). A matching entry overrides "
                           "the --two-up/--columns/--drop-pages defaults for that book.")
    g_io.add_argument("--out", type=Path, default=None,
                      help="Output directory for the .pdf, per-page text, "
                           "all_pages.txt and pages.jsonl. Required.")
    g_io.add_argument("--limit", type=int, default=None,
                      help="Only process the first N images (smoke test); "
                           "None = all pages.")

    g_ocr = ap.add_argument_group("OCR engine & caching")
    g_ocr.add_argument("--backend", default="surya", choices=ocr_backend.available(),
                       help="OCR engine.")
    g_ocr.add_argument("--instances", type=int, default=3,
                       help="Parallelism hint forwarded to the backend "
                            "(surya: number of llama-server instances).")
    g_ocr.add_argument("--backend-opt", action="append", metavar="KEY=VALUE",
                       help="Extra engine-specific option, repeatable. Surya: "
                            "detector_batch=N (surya_detect images per GPU pass, "
                            "default per-device Metal 8 / CUDA 36).")
    g_ocr.add_argument("--force", action="store_true",
                       help="Re-OCR every page, ignoring the cache (default: "
                            "incremental — only new/modified pages are re-OCR'd).")
    g_ocr.add_argument("--reocr", nargs="+", metavar="FILENAME", default=None,
                       help="Force re-OCR of these specific page files (by name, "
                            "with or without extension), even if unchanged; other "
                            "pages still come from cache. Put image_dir first.")
    g_ocr.add_argument("--text-source", choices=["ocr", "native"], default="ocr",
                       help="'ocr' (default): recognize text from the page images. "
                            "'native': for digital-first PDFs — take the clean "
                            "embedded text instead of OCR'ing (no diacritic "
                            "degradation) and label regions with surya_layout, so "
                            "the doc.json matches the OCR path. PDF input only.")
    g_ocr.add_argument("--perf-stats", action="store_true",
                       help="Stream [PERF] machine-resource samples (CPU/mem/GPU + "
                            "per-process llama/surya) tagged by stage into the log, "
                            "interleaved with the task output so any slice is "
                            "self-describing. A per-stage [PERF] SUMMARY (incl. GPU "
                            "duty-cycle / idle episodes) prints at book end. See "
                            "lib/perfprofile.py.")
    g_ocr.add_argument("--perf-interval", type=float, default=5.0, metavar="S",
                       help="Seconds between --perf-stats samples (default 5).")

    g_txt = ap.add_argument_group("text output")
    g_txt.add_argument("--text", choices=["full", "body", "both"], default="both",
                       help="Which text to emit: full (everything), body (drop "
                            "running headers/footers + page numbers), or both "
                            "(full in <out>/full/, body in <out>/body/). The PDF "
                            "always keeps the full text layer.")
    g_txt.add_argument("--drop-labels", default=",".join(sorted(sb.NON_BODY_LABELS)),
                       help="Labels dropped for body text.")
    g_txt.add_argument("--drop-nonlatin", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="Blank pages whose OCR text is mostly non-Latin script "
                            "(Surya hallucinates Indic/Tibetan garbage on script it "
                            "can't read) — the scan image is kept, only the bogus "
                            "text dropped. --no-drop-nonlatin keeps everything.")
    g_txt.add_argument("--nonlatin-threshold", type=float, default=0.5,
                       help="Fraction of non-Latin letters above which a page is "
                            "treated as hallucinated.")
    g_txt.add_argument("--ascii-dashes", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="Emit em dashes (—) as '--' in the bare text for "
                            "chunking; the PDF text layer keeps the Unicode. "
                            "--no-ascii-dashes keeps —.")
    g_txt.add_argument("--strip-footnote-refs", action="store_true",
                       help="Remove in-text footnote markers [^N] from the bare "
                            "text (Surya tags them as <sup>; kept by default so "
                            "they stay recoverable for chunking/relocation).")
    g_txt.add_argument("--doc", action=argparse.BooleanOptionalAction, default=True,
                       help="Also emit <name>.doc.json — the structure-preserving "
                            "RAG contract (blocks in reading order, typed/labeled, "
                            "unicode-normalized; see ARCHITECTURE.md). "
                            "--no-doc skips it.")
    g_txt.add_argument("--title", default=None,
                       help="Book title recorded in doc.json (else null).")
    g_txt.add_argument("--corpus-dir", default=None, metavar="DIR",
                       help="After a health-verified run, mirror the durable "
                            "artifacts (doc.json, body/full text, pages.jsonl) into "
                            "DIR/<edition>/ and update DIR/manifest.jsonl. Default: "
                            "$OCR_CORPUS_DIR or <repo>/_ocr_corpus. 'none' disables. "
                            "A run that fails the health check is never exported.")

    g_size = ap.add_argument_group("PDF file size (page images)")
    g_size.add_argument("--pdf-size", choices=["small", "medium", "large"],
                        default=None,
                        help="Preset bundling the three knobs below: small=1600px, "
                             "medium=2000px, large=0 (full res); all grey at "
                             "quality 90. Overrides --pdf-max-dim / "
                             "--pdf-image-quality / --color.")
    g_size.add_argument("--color", choices=["bw", "grey", "gray", "color"],
                        default="bw",
                        help="Embedded-image colour depth, the main size lever. "
                             "File size: bw=small (1-bit B&W — default, best for "
                             "text/line-art; crisp AT FULL RES but PIXELATES when "
                             "also downscaled, so keep --pdf-max-dim high), "
                             "grey=medium (keeps anti-aliasing/tone; use for pages "
                             "with photos), color=large. Cover (page 1) always "
                             "stays colour.")
    g_size.add_argument("--pdf-max-dim", type=int, default=0, metavar="PX",
                        help="Cap page images to PX on the long edge; smaller = "
                             "smaller file but softer. 0 (default) = full res — "
                             "keep it high, since downscaling 1-bit bw is what "
                             "pixelates the text (1200 was the cause of the earlier "
                             "pixelation; bw stays tiny even at full res). Set "
                             "1600/2000 only if you need a hard size cap.")
    g_size.add_argument("--pdf-image-quality", type=int, default=None, metavar="Q",
                        help="JPEG quality (1-95) for color/grey images when "
                             "re-encoded; lower = smaller, softer. File size: "
                             "~60 small / 75 medium / 90 large (default). N/A to bw.")
    g_size.add_argument("--all-pdf-sizes", action="store_true",
                        help="Build all three --pdf-size presets at once "
                             "(<name>.small.pdf / .medium.pdf / .large.pdf), in "
                             "parallel, instead of one PDF.")

    g_app = ap.add_argument_group("PDF appearance & text layer")
    g_app.add_argument("--contrast", type=float, default=1.25, metavar="F",
                       help="Contrast boost for text pages (1.0 = none; 1.5/2.0 = "
                            "punchier black-on-white); the cover (page 1) is left "
                            "untouched. NOTE: blur is usually --pdf-max-dim "
                            "downscaling, not contrast.")
    g_app.add_argument("--no-uniform-pages", nargs="?", const="__ALL__",
                       default=None, metavar="I,J,K-L",
                       help="Uniform page sizing is ON by default — every PDF page is "
                            "the same size (scans scaled to fit, centered) so facing "
                            "pages of unequal scan size don't jump. Pass this flag with "
                            "a comma list of 1-based page indices (ranges ok: '3,5,"
                            "100-110') to keep THOSE pages at their ORIGINAL size (e.g. "
                            "a foldout/plate). Pass it bare (no list) to turn uniform "
                            "sizing off entirely.")
    g_app.add_argument("--font", default=None,
                       help="PDF text-layer font (TTF/OTF/TTC); None = Arial "
                            "Unicode.")
    g_app.add_argument("--lines", choices=["detect", "split"], default="detect",
                       help="PDF text-layer line boxes: detect (per-line boxes "
                            "from the backend) or split (band-slice OCR blocks).")
    g_app.add_argument("--add-pages-after", nargs="+", action="append",
                       default=None, metavar="ANCHOR SOURCE",
                       help="Insert hand-scanned page image(s) into the book AFTER "
                            "preprocessing (never rotated/swapped). ANCHOR is an "
                            "existing page's FILENAME (no path); the SOURCE image(s) "
                            "after it — file paths in order, or a single folder "
                            "(natural-sorted) — are copied into <source>/_added/ and "
                            "sorted right after the anchor page. Repeatable for "
                            "several anchors. Idempotent (content-hash); delete a file "
                            "from _added/ to remove a page. Survives --force.")
    g_app.add_argument("--prepare-output", action="store_true",
                       help="Assembly-only: skip OCR and (re)build the deliverables "
                            "(PDF + text + doc.json) from the OCR stage's cached "
                            "output. Lets you regenerate with different appearance "
                            "options (--color/--pdf-max-dim/--no-uniform-pages…) without "
                            "re-OCR. Errors if the book has no OCR cache yet. Same "
                            "stage the standalone create_output.py runs.")
    g_app.add_argument("--dump-input-files", nargs=2, default=None,
                       metavar=("START", "N"),
                       help="DEBUG: after preprocessing + --add-pages-after, copy N "
                            "input images starting at file START (a filename, matched "
                            "by name or page number) into ./input_dump/ (as "
                            "NNNN_<name>), then QUIT before OCR — to eyeball the exact "
                            "input order, e.g. around an added page. No servers start.")

    own, preproc_tail = preprocess.split_forward()   # peel off '-- <preprocess args>'
    args = ap.parse_args(own)
    # Uniform page sizing is ON by default. --no-uniform-pages with a list exempts
    # those 1-based pages (kept original size); bare = uniform off entirely.
    if args.no_uniform_pages is None:
        args.uniform, args.uniform_except = True, frozenset()
    elif args.no_uniform_pages == "__ALL__":
        args.uniform, args.uniform_except = False, frozenset()
    else:
        args.uniform, args.uniform_except = True, _parse_page_list(args.no_uniform_pages)
    args.pp = preprocess.parse(preproc_tail)         # validates the forwarded tail
    args.force = args.force or (args.pp is not None and args.pp.force)  # --force either side
    proc_group.install()   # tear down surya_ocr / llama-server on signal/exit
    if args.pdf_size:                              # preset overrides the 3 knobs
        preset = PDF_SIZE_PRESETS[args.pdf_size]
        args.color = preset["color"]
        args.pdf_max_dim = preset["pdf_max_dim"]
        args.pdf_image_quality = preset["pdf_image_quality"]

    sources = resolve_inputs(args.inputs)
    if len(sources) <= 1:                          # single book — current behavior
        run_one(args, sources[0] if sources else args.inputs[0], args.out)
        return

    # Batch: one book per source -> <out_root>/<name>/, failures isolated.
    out_root = args.out or Path("out")
    print(f"Batch: {len(sources)} books -> {out_root}/", flush=True)
    results = []
    for i, src in enumerate(sources, 1):
        name = src.stem if src.suffix.lower() == ".pdf" else src.name
        out = out_root / name
        if args.skip_done and (out / f"{name}.pdf").exists():
            print(f"[{i}/{len(sources)}] skip {name} (done)", flush=True)
            results.append((name, "skip"))
            continue
        print(f"\n===== [{i}/{len(sources)}] {name} =====", flush=True)
        try:
            run_one(args, src, out)
            results.append((name, "ok"))
        except (SystemExit, Exception) as e:        # isolate per-book failures
            print(f"  FAILED: {e}", flush=True)
            results.append((name, f"FAIL: {e}"))

    ok = sum(1 for _, s in results if s == "ok")
    skip = sum(1 for _, s in results if s == "skip")
    print(f"\nBatch done: {ok} ok, {skip} skipped, "
          f"{len(results) - ok - skip} failed (of {len(results)}).", flush=True)
    for name, s in results:
        if s not in ("ok", "skip"):
            print(f"  {s}  ({name})", flush=True)
    return


if __name__ == "__main__":
    sys.exit(main())

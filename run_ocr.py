#!/usr/bin/env python3
"""Rasterize + re-OCR PDFs / image folders / catalogue editions through
``ocr_pipeline/ocr_one.py`` (Surya OCR -> searchable PDF + layout-labelled text).

Handles image-only scans and font-mojibake born-digital PDFs alike, and is fully
CLI-driven, so it's reusable for any OCR / re-OCR batch. OCR is
cached per book; re-runs are cheap and ``--force`` re-OCRs. Each source gets its
own ``--out`` dir under --out-root with ``full/`` + furniture-removed ``body/``
text and a per-block-labelled ``pages.jsonl`` (for downstream LLM furniture
removal).

WORKFLOW — pick ONE (you're prompted interactively if you omit it):
  --digital-first      born-digital PDF with a clean text layer: use the embedded
                       text (no OCR, no diacritic loss) + surya_layout structure.
  --scanned-no-text    image-only scan, no text layer: OCR the page images.
  --scanned-with-text  scan/PDF with a bad/mojibake text layer: re-OCR the rendered
                       pages and discard the layer.
  (The workflow sets the low-level --text-source for you. When run interactively,
  anything else needed but omitted is also asked for: the workflow, the source, the
  page layout (single / two-up / multi-column, unless --layout-overrides is given),
  and a confirm of the output directory.)

SOURCES (combine freely):
  paths            PDFs or image folders               a.pdf "b book.pdf" scans/
  --editions ID..  catalogue editions (path via DB)    --editions 60 248
  --from-file F    list file: one entry per line; '#'/blank ignored; each line is
                   a path or 'edition:ID', with an optional front-matter limit
                   after a tab or ' :: '  (e.g. 'Book.pdf :: 16', 'edition:248 :: 16')
  --preset backfill   a built-in named set of catalogue editions (example set)
  (no source)      errors with usage — give an explicit source (no implicit preset)

FRONT MATTER:
  --front-matter N   OCR only the first N pages of every source lacking its own
                     per-line limit (PDFs only; recovers a copyright page / ISBN
                     without rasterizing a whole book).

PARALLEL RUNS (no port collisions):
  --workers-per-book N  parallel llama-server OCR workers on ONE book (default 3,
                        GPU sweet spot 2-3). Alias: --instances.
  --parallel-books K    books OCR'd concurrently within this invocation (default
                        1). Alias: --jobs.
  --base-port P    where to START probing for llama-server ports (default 8091);
                   each job leases a block of N consecutive FREE ports, skipping
                   any in use. Separate invocations auto-avoid each other's ports,
                   so you need NOT hand-pick distinct --base-port values. NOTE:
                   total live servers = parallel-books * workers-per-book; >3 on
                   one GPU regresses.

ADD A WORKER TO A RUNNING CAMPAIGN (cooperative, on by default):
  Each book is claimed with a lock file (out-root/<name>/.claim) before OCR, so
  you can throw more compute at a campaign already in flight: just start ANOTHER
  invocation against the SAME --from-file and --out-root. It picks up only books
  no live worker holds; a dead worker's claim is reclaimed and its half-OCR'd
  book resumes from the per-page cache. Ports are auto-selected (it steps over the
  ports a running worker already holds), so you DON'T need a distinct --base-port.
  Example — add a second worker (keep total live instances <= 3 on one GPU):
     python3 run_ocr.py --from-file worklist.txt \
       --layout-overrides layout_overrides.txt --out-root out/backfill \
       --workers-per-book 1 --skip-done
  Remove a worker by Ctrl-C / kill; another worker finishes its in-flight book.
  --no-claim disables claiming (single-worker only).

  See / control what's running (act on the claim files, then exit):
     python3 run_ocr.py --out-root out/backfill --list-active     # what's running
     python3 run_ocr.py --out-root out/backfill --kill e49        # stop one book
  --kill SIGTERMs just that book's OCR process group; its worker moves on and the
  book resumes from the page cache on the next --skip-done run.

HOW TO RUN (from the repo root, with python3):
  cd /path/to/ocr_pipeline

  # a built-in named set of editions (must be explicit — no implicit default)
  python3 run_ocr.py --preset backfill

  # specific catalogue editions (paths resolved from the DB)
  python3 run_ocr.py --editions 60 179

  # arbitrary PDFs / an image folder
  python3 run_ocr.py a.pdf "b book.pdf" scans/

  # front matter only (e.g. just to recover a copyright page / ISBN)
  python3 run_ocr.py some.pdf --front-matter 16

  # a batch list (one entry/line: a path or 'edition:ID', optional ' :: N' limit)
  python3 run_ocr.py --from-file batch.txt --layout-overrides layout_overrides.txt

  # run 2 books at once on separate, non-overlapping port ranges
  python3 run_ocr.py a.pdf b.pdf --parallel-books 2

Edition resolution needs the catalogue DB (the one catalogue-aware dependency);
it defaults to ../library_cataloging/catalogue-db/catalogue.db, override with
--db or $CATALOGUE_DB. Path/folder runs don't touch the DB.

USEFUL FLAGS:
  --editions ID...   catalogue edition ids (path looked up in the DB)
  --from-file F      batch list: paths and/or 'edition:ID', optional ' :: N' limit
  --layout-overrides F  per-book two-up / columns manifest (lib/layout_overrides.py)
  -- <preprocess args>  everything after a `--` marker is forwarded to the image
                     preprocessor for EVERY book (rotate/swap/pad/page-window); put
                     it LAST, e.g. `… -- --rotate 180:CW --swap 9:end`. --force may
                     go on either side of the `--`. See preprocess.py --help.
  --db PATH          catalogue DB for edition:ID resolution ($CATALOGUE_DB)
  --preset backfill  a built-in named set of catalogue editions (example set)
  --front-matter N   OCR only the first N pages of sources without their own limit
  --workers-per-book N  parallel OCR workers on ONE book (default 3; alias --instances)
  --parallel-books K    books OCR'd at once in this run (default 1; alias --jobs)
  --base-port P      first llama-server port; slot s uses P + s*instances
  --out-root DIR     output root (default: ocr_pipeline/out/backfill)
  --force            re-OCR everything, ignoring the per-page cache
  --skip-done        skip books whose final PDF exists; resume the rest from cache
  --dry-run          print what would run without OCR'ing
  -h / --help        full option help

RESUME / CHECKPOINTING (a kill == pause; the next --skip-done run continues):
  OCR progress is flushed to the page cache in chunks, so a kill (memory, Ctrl-C,
  --kill) loses at most one in-flight chunk; everything before it resumes from
  cache. To resume, just re-run the SAME campaign command with --skip-done.
  --checkpoint-pages N   flush the page cache every N pages (default: one wave =
                         instances*slots, min 16). Smaller = less lost per kill,
                         more subprocess relaunches.

ADD / REMOVE WORKERS (cooperative, on by default — see "ADD A WORKER" above):
  --claim / --no-claim   claim each book with a lock so a SECOND invocation (own
                         --base-port, same --from-file/--out-root) joins without
                         collisions and only takes unclaimed books. A dead
                         worker's book is reclaimed and resumes from cache.
                         --no-claim disables (single-worker only).
  --claim-ttl SECONDS    cross-host staleness window (same-host uses pid liveness).

STATUS / CONTROL (act on the claim files under --out-root, then exit — no OCR):
  --list-active      books being processed right now (worker pid, OCR pid, age)
  --list-resumable   books with partial OCR on disk that a re-run would resume
                     (started but not finished, and not currently running)
  --kill NAME...     stop just the named in-progress book(s) (e.g. --kill e49) by
                     SIGTERM'ing their OCR process group; the worker moves on and
                     the book resumes from cache on the next --skip-done run

Output per source -> <out-root>/<name>/ : <name>.pdf (searchable), full/ and
furniture-removed body/ text, each with a per-block-labelled pages.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF — to slice front matter

HERE = Path(__file__).resolve().parent                  # the ocr_pipeline repo
# Child-process teardown + the layout manifest loader live in this package's
# lib/, shared with ocr_one.
sys.path.insert(0, str(HERE / "lib"))
import proc_group
import layout_overrides as lo
import corpus
import preprocess               # pad / pages / rotate / swap image steps (shared)
from claimlock import ClaimLock, _pid_alive

OCR_ONE = HERE / "ocr_one.py"
OUT_ROOT = HERE / "out" / "backfill"
# This driver is the one catalogue-aware part of ocr_pipeline: it resolves
# `edition:ID` -> on-disk PDF via the catalogue DB, which lives in the sibling
# library_cataloging repo. Override with --db or $CATALOGUE_DB; only `--editions`
# / `edition:` entries need it (path/folder runs don't touch the DB).
DEFAULT_DB = Path(os.environ.get(
    "CATALOGUE_DB",
    HERE.parent / "library_cataloging" / "catalogue-db" / "catalogue.db"))

# Named preset: an example built-in set of catalogue editions. id -> front-matter
# limit (None = whole book; a number OCRs only the first N pages of that edition).
# Edit or add your own sets here; resolution needs the catalogue DB.
PRESETS = {
    "backfill": {60: None, 179: None, 67: None, 75: None, 248: 16},
}


def _slug(name: str) -> str:
    """Filesystem-safe label from a file/folder stem."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "job"


def holding_path(con, edition_id):
    """First existing on-disk PDF holding for an edition (skips placeholders)."""
    rows = con.execute(
        "SELECT DISTINCT file_path FROM v_holding_files WHERE edition_id=?", (edition_id,)
    ).fetchall()
    for (p,) in rows:
        fp = Path(p)
        if fp.suffix.lower() == ".pdf" and fp.is_file() and fp.stat().st_size > 4096:
            return fp
    return None


def slice_frontmatter(src: Path, pages: int, dest: Path) -> Path:
    """Write the first ``pages`` pages of ``src`` to ``dest`` (cached)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return dest
    with fitz.open(src) as doc:
        n = min(pages, doc.page_count)
        out = fitz.open()
        out.insert_pdf(doc, from_page=0, to_page=n - 1)
        out.save(dest)
        out.close()
    print(f"[{dest.parent.name}] sliced first {n} pages -> {dest.name}", flush=True)
    return dest


_EVAL_RE = re.compile(r'task (\d+) \|\s+eval time =\s+[\d.]+ ms /\s+(\d+) tokens')


def image_dir_of(target: Path) -> Path:
    """Where ocr_one rasterized/looked for page images (so we can find the
    Surya/llama logs): a folder src is itself; a PDF -> sibling <stem>/ folder."""
    return target if target.is_dir() else target.with_suffix("")


def flag_capped_pages(target: Path, out_dir: Path, name, max_decode):
    """Record OCR requests that hit the token cap (so they can be re-OCR'd later
    at a higher --max-decode if they're valid dense pages rather than garbage).

    The reliable signal is the model's own per-request decoded-token count in
    the llama-server logs: a request that reached ~max_decode was truncated. We
    can't see this from the OCR *output* (loops get trimmed short; coherent
    truncations look normal), and request<->page is not 1:1 (Surya retries /
    falls back from looping full-page output to block OCR), so we report the
    COUNT and token values rather than guessing page names. count==0 means the
    cap truncated nothing (it's above every real page) — the common case.

    Writes <out_dir>/capped_pages.json and returns the hit count."""
    logs = sorted((image_dir_of(target) / ".surya_ocr" / "_parallel").glob("llama_*.log"))
    per_task = {}                       # final decoded tokens per (log, task)
    for f in logs:
        for line in f.read_text(errors="ignore").splitlines():
            if "eval time" in line and "prompt eval" not in line:
                m = _EVAL_RE.search(line)
                if m:
                    key = (f.name, int(m.group(1)))
                    per_task[key] = max(per_task.get(key, 0), int(m.group(2)))
    margin = 50                         # tokens; budget is capped exactly at max_decode
    hits = sorted((v for v in per_task.values() if v >= max_decode - margin),
                  reverse=True)
    report = {
        "book": name,
        "max_decode": max_decode,
        "ocr_requests_seen": len(per_task),
        "cap_hit_count": len(hits),
        "cap_hit_tokens": hits,
        "note": ("Requests whose decode reached the cap (=truncated). Some may "
                 "be hallucination loops (garbage — ignore); others may be valid "
                 "dense pages. To recover the valid ones, re-OCR this book at a "
                 "higher cap: run_ocr.py <src> --force --max-decode 8192. "
                 "Exact page identity isn't recoverable from logs (request<->page "
                 "is not 1:1 due to loop fallback/retries)."),
    }
    (out_dir / "capped_pages.json").write_text(json.dumps(report, indent=2))
    if hits:
        print(f"[{name}] {len(hits)} OCR request(s) hit the {max_decode}-token cap "
              f"(of {len(per_task)} seen) -> {out_dir/'capped_pages.json'}; "
              f"re-OCR at a higher cap to recover any valid ones.", flush=True)
    else:
        print(f"[{name}] no requests hit the {max_decode}-token cap "
              f"({len(per_task)} seen) — cap truncated nothing.", flush=True)
    return len(hits)


def run_one(name, src: Path, limit, *, out_root, instances, base_port,
            max_decode, force, dry, two_up="off", columns=1, drop_pages=None,
            preproc_tail=(), preproc_label="", checkpoint_pages=None,
            claim=None, progress_line=None, add_pages_after=None,
            dump_input_files=None, img_limit=None, output_tail=(),
            prepare_output=False,
            perf_stats=False, perf_interval=5.0, text_source="ocr"):
    """OCR one source. ``limit`` slices front matter (PDF only); ``base_port`` is
    this job's first llama-server port (instances bind base_port..+instances-1);
    ``max_decode`` (if set) caps per-page OCR tokens to kill runaway loops.
    ``two_up``/``columns`` come from the layout manifest (page-split / column-
    strip re-OCR) and are passed through to ocr_one for this book only."""
    out_dir = out_root / name
    target = src
    if limit is not None:
        if src.is_dir():
            print(f"[{name}] (ignoring front-matter limit: image folder)", flush=True)
        else:
            target = slice_frontmatter(src, limit, out_dir / f"{name}_frontmatter.pdf")

    cmd = [
        "caffeinate", "-i", "-s",
        sys.executable, str(OCR_ONE), str(target),
        "--out", str(out_dir),
        "--instances", str(instances),
        "--backend-opt", f"base_port={base_port}",
        *( ["--backend-opt", f"max_decode={max_decode}"] if max_decode else [] ),
        *( ["--backend-opt", f"checkpoint_pages={checkpoint_pages}"]
           if checkpoint_pages else [] ),
        *( ["--two-up", two_up] if two_up and two_up != "off" else [] ),
        *( ["--columns", str(columns)] if columns and columns >= 2 else [] ),
        *( ["--drop-pages", drop_pages] if drop_pages else [] ),
        # Layout classification for downstream LLM furniture removal:
        #   --lines detect -> Surya layout pass (per-region labels + line boxes)
        #   --text both    -> full/ (everything) AND body/ (headers/footers/page
        #                     numbers dropped); both write pages.jsonl with
        #                     per-block {text, label, bbox, confidence}.
        "--lines", "detect",
        "--text", "both",
        *( ["--perf-stats", "--perf-interval", str(perf_interval)]
           if perf_stats else [] ),
        *( ["--text-source", text_source] if text_source != "ocr" else [] ),
    ]
    if force:
        cmd.append("--force")
    # Hand-added pages: an ocr_one OWN arg (applied after preprocessing), so it goes
    # BEFORE the `--` preprocess-tail marker.
    for group in (add_pages_after or []):
        cmd += ["--add-pages-after", *group]
    if dump_input_files:
        cmd += ["--dump-input-files", *dump_input_files]
    if img_limit:
        cmd += ["--limit", str(img_limit)]
    if prepare_output:
        cmd.append("--prepare-output")
    # create_output (output-stage) flags from the SECOND `--` section — ocr_one OWN
    # args, so they go BEFORE the preprocess `--`.
    cmd += list(output_tail)
    # Forward the preprocess args after a `--` marker; ocr_one peels them off with
    # preprocess.split_forward. Must be LAST (everything after `--` is the tail).
    if preproc_tail:
        cmd += ["--", *preproc_tail]

    layout = ((" two-up" if two_up and two_up != "off" else "")
              + (f" columns={columns}" if columns and columns >= 2 else "")
              + (f" drop-{drop_pages}" if drop_pages else "")
              + preproc_label)          # " rotate-… swap-… pad-…"
    kind = (f"front matter {limit}pp" if (limit and not src.is_dir())
            else "image folder" if src.is_dir() else "full book") + layout
    print(f"\n=== [{time.strftime('%Y-%m-%d %H:%M:%S')}] {name} "
          f"({kind}, ports {base_port}-{base_port + instances - 1}) ==="
          + (f"\n    {progress_line}" if progress_line else "")
          + f"\n    src: {src}\n    out: {out_dir}", flush=True)
    if dry:
        print(f"    cmd: {' '.join(cmd)}", flush=True)
        return True, target
    # start_new_session=True -> this ocr_one leads its own process group, so the
    # cleanup handler can SIGTERM/SIGKILL the whole subtree (incl. surya_ocr) at
    # once. Track it in the registry for the lifetime of the call.
    # group=True: ocr_one leads its own process group, so the shared cleanup can
    # killpg the whole subtree (ocr_one -> surya_ocr -> llama-server) at once.
    # stdin=DEVNULL: the engine never reads from the terminal, so a batch cranks
    # through without any mid-run prompt (all questions are asked up front, before
    # the first book is spawned).
    proc = proc_group.popen(cmd, group=True, stdin=subprocess.DEVNULL)
    # Record the per-book OCR pid (== its process-group id, since group=True) in
    # the claim so `--list-active` can show it and `--kill NAME` can stop just
    # this book without touching the rest of the worker.
    if claim is not None:
        claim.update(ocr_pid=proc.pid, started_at=time.time())
    try:
        rc = proc.wait()
    finally:
        proc_group.reap(proc)
    ok = rc == 0
    print(f"[{name}] -> {'OK' if ok else 'FAILED (exit %d)' % rc}", flush=True)
    return ok, target


def parse_list_file(path: Path, default_limit):
    """List file -> [(entry, limit)]; entry is a path or 'edition:ID'."""
    jobs = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        limit = default_limit
        for sep in ("\t", " :: "):
            if sep in line:
                line, _, lim = line.partition(sep)
                line, lim = line.strip(), lim.strip()
                limit = int(lim) if lim else default_limit
                break
        jobs.append((line, limit))
    return jobs


def resolve_edition(con, eid, limit):
    """(name, src_or_None, limit) for an edition id."""
    src = holding_path(con, eid)
    if src is None:
        print(f"=== e{eid}: SKIP — no usable on-disk PDF holding "
              "(missing or cloud placeholder)", flush=True)
    return (f"e{eid}", src, limit)


def active_claims(out_root: Path):
    """Yield (name, info) for every live book claim under ``out_root`` — the
    books some worker is currently OCR'ing. Skips stale claims whose owning
    worker pid is dead on this host (the OOM-kill leftovers)."""
    for claim_path in sorted(out_root.glob("*/.claim")):
        info = ClaimLock.read(claim_path)
        if not info:
            continue
        same_host = info.get("host") == socket.gethostname()
        worker_alive = _pid_alive(int(info.get("pid", -1)))
        if same_host and not worker_alive:
            continue                       # dead worker — its claim is reclaimable
        yield claim_path.parent.name, info


def list_active(out_root: Path) -> int:
    """Print the books currently being processed (one per live claim)."""
    rows = list(active_claims(out_root))
    if not rows:
        print(f"No books are being processed under {out_root}.", flush=True)
        return 0
    now = time.time()
    print(f"{'BOOK':<16} {'WORKER':>7} {'OCR_PID':>7} {'AGE':>7}  HOST", flush=True)
    for name, info in rows:
        started = info.get("started_at") or info.get("claimed_at") or now
        age = int(now - started)
        ocr = info.get("ocr_pid", "-")
        running = isinstance(ocr, int) and _pid_alive(ocr)
        ocr_disp = str(ocr) + ("" if running else "?")
        print(f"{name:<16} {info.get('pid', '?'):>7} {ocr_disp:>7} "
              f"{age:>6}s  {info.get('host', '?')}", flush=True)
    print(f"\n{len(rows)} book(s) in progress.", flush=True)
    return 0


def kill_books(out_root: Path, names) -> int:
    """SIGTERM the OCR subprocess group of each named book, stopping just that
    book (its worker moves on to the next; the book resumes from the page cache
    on a later --skip-done run). Returns a process exit code."""
    rc = 0
    for name in names:
        info = ClaimLock.read(out_root / name / ".claim")
        if not info:
            print(f"{name}: no active claim (not being processed).", flush=True)
            rc = 1
            continue
        ocr_pid = info.get("ocr_pid")
        if not isinstance(ocr_pid, int) or not _pid_alive(ocr_pid):
            print(f"{name}: claimed but no live OCR process "
                  f"(starting up, or already gone).", flush=True)
            rc = 1
            continue
        try:
            # group=True gave the ocr_one subprocess its own process group, so a
            # killpg takes down ocr_one -> surya_ocr -> llama-server together.
            os.killpg(ocr_pid, signal.SIGTERM)
            print(f"{name}: sent SIGTERM to OCR process group {ocr_pid}. "
                  f"It will resume from the page cache on the next --skip-done run.",
                  flush=True)
        except ProcessLookupError:
            print(f"{name}: OCR process {ocr_pid} already gone.", flush=True)
            rc = 1
    return rc


def _port_free(port: int) -> bool:
    """True if nothing is bound on 127.0.0.1:port right now (so a llama-server
    could take it). Best-effort — there's an inherent race between probing and
    actually binding, fine for spacing a handful of local workers apart."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def find_port_ranges(start: int, count: int, width: int):
    """``count`` non-overlapping blocks of ``width`` consecutive free ports,
    probing upward from ``start``. Lets a second worker launch without picking a
    ``--base-port`` by hand: it just steps over the ports a running worker already
    holds. Returns (bases, skipped_busy) where skipped_busy flags that some ports
    were in use (i.e. another worker is probably running)."""
    bases, port, skipped = [], int(start), False
    while len(bases) < count:
        if port > 65535 - width:
            raise SystemExit(f"no run of {width} free port(s) at/above {start}")
        if all(_port_free(p) for p in range(port, port + width)):
            bases.append(port)
            port += width                      # next block starts past this one
        else:
            skipped = True
            port += 1                          # slide past the busy port
    return bases, skipped


def book_state(book: Path):
    """Classify one output dir: (status, ocr_cached, rasterized, final_pdf).

    Status is one of: 'done' (final PDF exists), 'active' (a live worker holds
    its claim), 'resume' (some OCR cached but not finished — a re-run continues
    it), or 'not-started' (no OCR cached yet). Only the canonical per-book OCR
    cache (<book>/_pages/.surya_ocr) counts toward resume — a stale front-matter
    slice cache doesn't make the full book look half-done."""
    pdfs = [p for p in book.glob("*.pdf") if not p.name.endswith("_frontmatter.pdf")]
    cache_glob = list((book / "_pages" / ".surya_ocr").glob("results*.json"))
    cached = 0
    for f in cache_glob:
        try:
            cached += len(json.loads(f.read_text()))
        except (OSError, ValueError):
            pass
    imgs = len(list((book / "_pages").glob("*.png"))
               + list((book / "_pages").glob("*.jpg")))
    info = ClaimLock.read(book / ".claim")
    active = bool(info and _pid_alive(int(info.get("pid", -1))))
    if pdfs and corpus.health(book).ok:
        status = "done"
    elif active:
        status = "active"
    elif pdfs:
        # A final PDF exists but the doc.json is empty/broken (e.g. an interrupted
        # OCR that left a {} results stub) — NOT done; a re-run must redo it.
        status = "broken"
    elif cached == 0:
        status = "not-started"
    else:
        status = "resume"
    return status, cached, imgs, (pdfs[0].name if pdfs else None)


def list_resumable(out_root: Path) -> int:
    """List books with partial OCR on disk that a re-run would resume (not done,
    not currently running). These pick up where a kill/crash left off."""
    rows = []
    for book in sorted(p for p in out_root.iterdir() if p.is_dir()):
        status, cached, imgs, _ = book_state(book)
        if status in ("resume", "broken"):
            rows.append((book.name, cached, imgs, status))
    if not rows:
        print(f"No books are in a resume state under {out_root} "
              "(none half-OCR'd, broken, or idle).", flush=True)
        return 0
    print(f"{'BOOK':<20} {'OCR_CACHED':>10} {'RASTERIZED':>10} {'STATE':>8}  "
          "(re-run resumes/redoes)", flush=True)
    for name, cached, imgs, status in rows:
        print(f"{name:<20} {cached:>10} {imgs or '?':>10} {status:>8}", flush=True)
    n_broken = sum(1 for *_, s in rows if s == "broken")
    tail = f" ({n_broken} broken — PDF exists but doc.json empty)" if n_broken else ""
    print(f"\n{len(rows)} book(s) in resume/broken state.{tail}", flush=True)
    return 0


def _perf_line(sample, stage):
    """Compact [PERF] machine line for the interleave pipeline (the servers are
    children of this process, so one campaign-level sampler covers the tree)."""
    procs = sample.get("procs") or {}
    llama = sum(r["cpu_pct"] for r in procs.values()
                if "llama-server" in r.get("name", "").lower())
    rec = {"iso": time.strftime("%H:%M:%S",
                                time.localtime(sample.get("t_wall", 0))),
           "stage": stage, "llama_cpu": round(llama, 1),
           "tree_cpu": round(sum(r["cpu_pct"] for r in procs.values()), 1),
           "tree_rss": round(sum(r["rss_gb"] for r in procs.values()), 2)}
    for k, v in sample.items():
        if k.startswith("sys.") or k.startswith("gpu."):
            rec[k] = round(v, 2)
    print("[PERF] " + json.dumps(rec), flush=True)


def run_interleaved(args, jobs) -> int:
    """--interleave: one persistent server set + a rasterize/ocr/assemble pipeline
    over all books (lib/interleave), instead of a fresh per-book ocr_one. Keeps
    the GPU continuously fed and loads the GGUF once. Single in-process worker —
    no claiming / --parallel-books (those are for the per-book subprocess path)."""
    from types import SimpleNamespace
    import ocr_backend
    import ocr_surya  # noqa: F401 — registers the "surya" backend
    import interleave

    if args.workers_per_book > 3:
        print(f"note: {args.workers_per_book} persistent servers may swap; "
              "2-3 is the GPU sweet spot.", flush=True)
    bases, _ = find_port_ranges(args.base_port, 1, args.workers_per_book)
    backend = ocr_backend.get_backend("surya", instances=args.workers_per_book,
                                      base_port=bases[0], max_decode=args.max_decode)
    job_objs = []
    for name, src, _lim, lay in jobs:
        if src is None:
            continue
        src_p = Path(src)
        stem = src_p.stem if src_p.suffix.lower() == ".pdf" else src_p.name
        job_objs.append(SimpleNamespace(name=name, src=src_p, stem=stem,
                                        out=args.out_root / name, layout=lay or {}))
    if not job_objs:
        sys.exit("nothing to interleave (no usable sources)")
    if args.dry_run:
        print(f"[dry-run] interleave {len(job_objs)} book(s) through "
              f"{args.workers_per_book} persistent server(s) @ base {bases[0]}:",
              flush=True)
        for j in job_objs:
            print(f"  {j.name}  <- {j.src}  layout={j.layout or '{}'}", flush=True)
        return 0

    prof = None
    if args.perf_stats:
        import perfprofile
        prof = perfprofile.StageProfiler(
            root_pid=os.getpid(), interval=args.perf_interval,
            on_sample=_perf_line)
        prof.add_detector(perfprofile.DipDetector())
        prof.start()
    opts = interleave.default_opts(force=args.force, skip_done=args.skip_done,
                                   profiler=prof)
    log_dir = args.out_root / "_servers"
    print(f"=== interleave: {len(job_objs)} book(s), {args.workers_per_book} "
          f"persistent server(s) on base port {bases[0]} ===", flush=True)
    try:
        results = interleave.run_books(job_objs, backend, opts, log_dir)
    finally:
        if prof is not None:
            prof.stop()
            for f in prof.report().get("findings", []):
                print("[PERF] " + json.dumps({"finding": True, **f}), flush=True)

    print("\n=== interleave summary ===", flush=True)
    for n, st in results.items():
        print(f"  {n}: {st}", flush=True)
    return 0 if all(v in ("ok", "skip") for v in results.values()) else 1


def report_book_timing(name, out_root, stats, lock):
    """After a book is OCR'd, print its OCR s/pg and the running AVERAGE over all
    books processed this run (total OCR seconds / total pages). Reads the per-book
    timing sidecar ocr_one wrote. Goes to stdout -> the campaign log."""
    try:
        t = json.loads((out_root / name / ".run_timing.json").read_text())
    except Exception:
        return
    pages, secs = t.get("pages") or 0, t.get("ocr_secs") or 0.0
    if not pages:
        return
    with lock:
        stats["books"] += 1
        stats["pages"] += pages
        stats["ocr_secs"] += secs
        b, p, avg = stats["books"], stats["pages"], stats["ocr_secs"] / stats["pages"]
    print(f"[{name}] {pages} pages, {secs / pages:.2f} s/pg  "
          f"(run avg {avg:.2f} s/pg over {b} books, {p} pages)", flush=True)


def count_pages(src, limit) -> int:
    """Pages a source will OCR: PDF page count (capped by a front-matter limit)
    or image-folder file count. 0 if missing/unreadable — so a book we can't
    measure just doesn't inflate the campaign total."""
    if src is None:
        return 0
    try:
        if src.is_dir():
            return sum(1 for p in src.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        with fitz.open(src) as doc:
            return min(limit, doc.page_count) if limit else doc.page_count
    except Exception:
        return 0


class Progress:
    """Campaign-wide progress shown on each book's start banner: book N/M, total
    pages done/total, the wall-clock secs/page seen SO FAR this run, and a
    projected time-left. The rate is derived from how much the OCR page cache has
    grown since this invocation started (pages OCR'd this run / elapsed), so it
    reflects the live throughput at the current --workers-per-book and naturally
    excludes pages resumed from a previous run's cache. Read-only after init
    (safe to call from concurrent --parallel-books workers)."""

    def __init__(self, out_root: Path, page_totals: dict, names, start_time):
        self.out_root = out_root
        self.page_totals = page_totals              # name -> pages it will OCR
        self.order = {n: i + 1 for i, n in enumerate(names)}   # name -> 1-based N
        self.total_pages = sum(page_totals.values())
        self.total_books = len(names)
        self.t0 = start_time
        self.done_at_start = self._pages_done()     # cached before we OCR'd anything

    def _pages_done(self) -> int:
        """Pages already OCR'd across the whole campaign: finished books count
        their full page total; in-flight/resumable books count their cached
        pages (capped at the total, so a noisy cache can't exceed 100%)."""
        done = 0
        for name, total in self.page_totals.items():
            book = self.out_root / name
            if not book.is_dir():
                continue
            status, cached, _imgs, _ = book_state(book)
            if status == "done":
                done += total or cached
            else:
                done += min(cached, total) if total else cached
        return done

    def banner(self, name: str) -> str:
        """One-line progress string for the book ``name`` as it starts."""
        done = self._pages_done()
        elapsed = time.time() - self.t0
        ocrd = done - self.done_at_start               # pages OCR'd this run
        remaining = max(0, self.total_pages - done)
        n, m = self.order.get(name, 0), self.total_books
        if ocrd > 0 and elapsed > 0:
            spp = elapsed / ocrd
            eta = remaining * spp
            tail = (f"{spp:.1f}s/pg · ~{eta / 3600:.1f}h "
                    f"({eta / 86400:.1f}d) left")
        else:
            tail = "rate pending (warming up)"
        return f"book {n}/{m} · pages {done}/{self.total_pages} · {tail}"


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


_WORKFLOW_ALIASES = {
    "1": "digital-first", "digital-first": "digital-first",
    "2": "scanned-no-text", "scanned-no-text": "scanned-no-text",
    "3": "scanned-with-text", "scanned-with-text": "scanned-with-text",
}


def _prompt_workflow() -> str:
    print("\nHow should the book(s) be processed?", file=sys.stderr)
    print("  1) digital-first       born-digital PDF, clean text layer "
          "(use embedded text, no OCR)", file=sys.stderr)
    print("  2) scanned-no-text     image-only scan, no text layer (OCR the pages)",
          file=sys.stderr)
    print("  3) scanned-with-text   scan/PDF with a bad/mojibake text layer "
          "(re-OCR, discard it)", file=sys.stderr)
    while True:
        c = input("Choose 1, 2, or 3: ").strip().lower()
        if c in _WORKFLOW_ALIASES:
            return _WORKFLOW_ALIASES[c]
        print("  enter 1, 2, or 3.", file=sys.stderr)


def resolve_workflow(args) -> str:
    """Resolve the 3-way processing workflow (prompt if interactive, else error)
    and set args.text_source from it."""
    wf = args.workflow
    if wf is None and args.text_source == "native":
        wf = "digital-first"                        # back-compat: --text-source native
    if wf is None:
        if getattr(args, "dump_input_files", None):
            wf = "scanned-no-text"           # dump quits before OCR; workflow is moot
        elif _interactive():
            wf = _prompt_workflow()
        else:
            sys.exit("No workflow given. Pass one of --digital-first / "
                     "--scanned-no-text / --scanned-with-text.")
    args.workflow = wf
    args.text_source = "native" if wf == "digital-first" else "ocr"
    print(f"Workflow: {wf}  (text source: {args.text_source})", file=sys.stderr)
    return wf


def prompt_source(front_matter):
    """Ask for a source when none was given on the CLI (interactive only)."""
    print("\nNo source given. Enter a PDF path, a folder, or 'edition:ID'.",
          file=sys.stderr)
    while True:
        s = input("Source: ").strip()
        if s:
            return [(s, front_matter)]
        print("  enter a path or edition:ID (Ctrl-C to abort).", file=sys.stderr)


_LAYOUT_CHOICES = {
    "1": {}, "single": {},
    "2": {"two_up": "on"}, "two-up": {"two_up": "on"},
    "3": {"columns": 2}, "columns": {"columns": 2}, "multi-column": {"columns": 2},
}


def prompt_layout():
    """Ask the page layout when no --layout-overrides file was given. Returns a
    run-wide layout dict applied to every book (per-book file entries still win)."""
    print("\nPage layout of the book(s)?", file=sys.stderr)
    print("  1) single        one page per image, single column (normal)",
          file=sys.stderr)
    print("  2) two-up         two book pages on one landscape sheet (split L/R)",
          file=sys.stderr)
    print("  3) multi-column   2+ text columns per page (column-strip OCR)",
          file=sys.stderr)
    while True:
        c = input("Choose 1/2/3 [1]: ").strip().lower() or "1"
        if c in _LAYOUT_CHOICES:
            layout = dict(_LAYOUT_CHOICES[c])
            break
        print("  enter 1, 2, or 3.", file=sys.stderr)
    # Verso / non-readable script pages (bilingual books — Tibetan/CJK on one side).
    print("\nSkip OCR on script pages Surya can't read (e.g. Tibetan/CJK verso)?",
          file=sys.stderr)
    print("  1) no — OCR every page", file=sys.stderr)
    print("  2) auto-detect the non-Latin pages (recommended)", file=sys.stderr)
    print("  3) even-numbered pages are the script side", file=sys.stderr)
    print("  4) odd-numbered pages are the script side", file=sys.stderr)
    while True:
        d = input("Choose 1/2/3/4 [1]: ").strip() or "1"
        mode = {"1": None, "2": "non-latin", "3": "even", "4": "odd"}.get(d, "?")
        if mode != "?":
            break
        print("  enter 1, 2, 3, or 4.", file=sys.stderr)
    if mode:
        spec = mode
        if mode in ("even", "odd"):           # optional page range (else front/back
            r = input("  Apply to which pages? e.g. 10-300  (Enter = all): ").strip()
            if r and re.fullmatch(r"\d+\s*-\s*\d+", r):
                spec = f"{mode}:{r.replace(' ', '')}"
            elif r:
                print("  (couldn't parse a range — applying to all pages)",
                      file=sys.stderr)
        layout["drop_pages"] = spec
    return layout


def confirm_out_root(out_root):
    """Confirm the default output dir, or take a new path (interactive)."""
    s = input(f"\nOutput directory [{out_root}]?  (Enter to accept): ").strip()
    return Path(s).expanduser() if s else out_root


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("paths", nargs="*", help="PDFs or image folders to OCR.")
    ap.add_argument("--editions", nargs="+", type=int, metavar="ID", default=[],
                    help="Catalogue edition ids (path resolved from the DB).")
    ap.add_argument("--from-file", type=Path, metavar="LIST",
                    help="Read entries (path or 'edition:ID', + optional limit) from a file.")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="Named built-in set.")
    ap.add_argument("--front-matter", type=int, default=None, metavar="N",
                    help="OCR only the first N pages of each source lacking its own limit.")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="Process only the first N page images (after any rasterize) — "
                         "works for image folders too, unlike --front-matter (PDF-only). "
                         "Handy with --dump-input-files for a fast preview.")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT,
                    help=f"Output root dir (default: {OUT_ROOT}).")
    ap.add_argument("--workers-per-book", "--instances", dest="workers_per_book",
                    type=int, default=3,
                    help="Parallel llama-server OCR workers on ONE book (default 3; "
                         "2-3 is the GPU sweet spot). Alias: --instances.")
    ap.add_argument("--parallel-books", "--jobs", dest="parallel_books",
                    type=int, default=1,
                    help="How many books to OCR at once within this run (default 1). "
                         "Alias: --jobs.")
    ap.add_argument("--base-port", type=int, default=8091,
                    help="Where to START probing for llama-server ports (default "
                         "8091). Each job leases a block of `instances` consecutive "
                         "FREE ports; busy ports (e.g. another worker's) are skipped "
                         "automatically. So a second worker needs no hand-picked "
                         "--base-port to avoid collisions — just keep total live "
                         "instances across workers <= 3 on one GPU.")
    ap.add_argument("--max-decode", type=int, default=3500, metavar="N",
                    help="Per-page OCR token ceiling, default 3500 (caps runaway "
                         "hallucination loops; a sensible default so every worker "
                         "matches without passing the "
                         "flag). Raise toward Surya's 8192 for token-dense books; "
                         "set ABOVE the densest real page (measure first) to avoid "
                         "truncating legitimate text.")
    ap.add_argument("--layout-overrides", type=Path, metavar="FILE",
                    help="Per-book layout manifest (two-up / columns directives keyed "
                         "by edition:ID or filename stem; see ocr_pipeline "
                         "lib/layout_overrides.py). Applied per book, so the right "
                         "books get page-split / column-strip re-OCR and the rest don't.")
    # Image preprocessing: everything after a `--` (or `--preprocess`) marker is
    # forwarded verbatim to the shared preprocess utility for EVERY book, e.g.
    # `run_ocr.py <src> --scanned-no-text -- --rotate 180:CW --swap 9:end`. The
    # marker + tail are peeled off with preprocess.split_forward before argparse.
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, metavar="PATH",
                    help=f"Catalogue SQLite DB for edition:ID resolution "
                         f"($CATALOGUE_DB; default {DEFAULT_DB}). Only needed when "
                         f"a run includes editions.")
    ap.add_argument("--force", action="store_true", help="Re-OCR, ignoring the cache.")
    ap.add_argument("--skip-done", action="store_true",
                    help="Skip a book whose final output PDF already exists "
                         "(<out-root>/<name>/<src-stem>.pdf). Use on a re-run after "
                         "a crash so finished books aren't re-assembled.")
    ap.add_argument("--checkpoint-pages", type=int, default=None, metavar="N",
                    help="Flush the OCR page cache every N pages (default: one "
                         "wave across the live servers, min 16) so a kill mid-book "
                         "loses at most one chunk and the rest resume from cache. "
                         "Larger = fewer subprocess relaunches, more lost per kill.")
    ap.add_argument("--claim", action=argparse.BooleanOptionalAction, default=True,
                    help="Claim each book with a lock file so a SECOND worker "
                         "(another invocation on its own --base-port) can join a "
                         "running campaign and pick up only unclaimed books — no "
                         "collisions. A dead worker's claim is reclaimed (its "
                         "partial book resumes from the page cache). --no-claim "
                         "disables (single-worker only).")
    ap.add_argument("--claim-ttl", type=float, default=900.0, metavar="SECONDS",
                    help="Cross-host claim staleness window (same-host uses pid "
                         "liveness, so this only matters across machines).")
    ap.add_argument("--list-active", action="store_true",
                    help="List the books currently being processed (live claims "
                         "under --out-root) and exit, without OCR'ing anything.")
    ap.add_argument("--list-resumable", action="store_true",
                    help="List books with partial OCR on disk that a re-run would "
                         "resume from cache (started but not finished, and not "
                         "currently running), then exit.")
    ap.add_argument("--kill", nargs="+", metavar="NAME", default=None,
                    help="Stop the named in-progress book(s) (e.g. e49) by "
                         "SIGTERM'ing their OCR process group, then exit. The "
                         "worker moves on; the book resumes from the page cache "
                         "on a later --skip-done run. Use --list-active for names.")
    ap.add_argument("--perf-stats", action="store_true",
                    help="Pass --perf-stats to each ocr_one: stream [PERF] machine "
                         "samples (CPU/mem/GPU + per-process) tagged by stage into "
                         "the book's log, with a per-stage SUMMARY (incl. GPU "
                         "duty-cycle) at book end. For profiling stalls / "
                         "interleaving. See lib/perfprofile.py.")
    ap.add_argument("--perf-interval", type=float, default=5.0, metavar="S",
                    help="Seconds between --perf-stats samples (default 5).")
    g_wf = ap.add_argument_group(
        "WORKFLOW — pick ONE (you'll be prompted if you omit it)")
    wf = g_wf.add_mutually_exclusive_group()
    wf.add_argument("--digital-first", dest="workflow", action="store_const",
                    const="digital-first",
                    help="Born-digital PDF with a clean embedded text layer: use that "
                         "text (no OCR, no diacritic loss) + surya_layout structure.")
    wf.add_argument("--scanned-no-text", dest="workflow", action="store_const",
                    const="scanned-no-text",
                    help="Image-only scan, no text layer: OCR the page images.")
    wf.add_argument("--scanned-with-text", dest="workflow", action="store_const",
                    const="scanned-with-text",
                    help="Scan/PDF whose existing text layer is unreliable (mojibake / "
                         "bad prior OCR): re-OCR the rendered pages, discard the layer.")
    ap.add_argument("--text-source", choices=["ocr", "native"], default="ocr",
                    help="(low-level; prefer the WORKFLOW flags above) text source "
                         "passed to ocr_one. The workflow flags set this for you.")
    ap.add_argument("--prepare-output", action="store_true",
                    help="Assembly-only: skip OCR and rebuild the deliverables (PDF + "
                         "text + doc.json) from the OCR cache. Put output options after "
                         "a SECOND '--': run_ocr <book> … -- <preprocess flags> -- "
                         "<output flags e.g. --color grey --pdf-max-dim 2000>. "
                         "(Preprocess section may be empty: '… -- -- <output flags>'.)")
    ap.add_argument("--dump-input-files", nargs=2, default=None, metavar=("START", "N"),
                    help="DEBUG: copy N input images (the exact post-preprocess + "
                         "post-add sequence) starting at file START into "
                         "./input_dump/, then quit before OCR. Single-book runs only.")
    ap.add_argument("--add-pages-after", nargs="+", action="append", default=None,
                    metavar="ANCHOR SOURCE",
                    help="Insert hand-scanned page image(s) into the book AFTER "
                         "preprocessing (never rotated/swapped). ANCHOR is an existing "
                         "page's FILENAME (no path); the SOURCE image(s) after it — "
                         "file paths in order, or a single folder — sort right after "
                         "that page. Copied into <source>/_added/ (survives --force); "
                         "idempotent. Single-book runs only (the anchor is book-"
                         "specific); not supported with --interleave.")
    ap.add_argument("--interleave", action="store_true",
                    help="Pipeline all books through ONE persistent llama-server "
                         "set with overlapped rasterize/ocr/assemble stages "
                         "(lib/interleave) — loads the GGUF once and keeps the GPU "
                         "fed across books, instead of a fresh per-book ocr_one. "
                         "Single in-process worker (no claiming / --parallel-books). "
                         "v1: column-strip books fall back to whole-page OCR.")
    ap.add_argument("--dry-run", action="store_true", help="Show, don't OCR.")
    own, tail = preprocess.split_forward()           # peel off '-- <preprocess args>…'
    # A SECOND '--' splits the tail into preprocess args | create_output (output) args:
    #   run_ocr … -- <preprocess flags> -- <output flags>
    # The output flags are forwarded verbatim to ocr_one as its OWN args (before its
    # '--'), so run_ocr needn't know them — any ocr_one output flag works.
    if "--" in tail:
        k = tail.index("--")
        preproc_tail, output_tail = tail[:k], tail[k + 1:]
    else:
        preproc_tail, output_tail = tail, []
    args = ap.parse_args(own)
    pp = preprocess.parse(preproc_tail)              # validates the forwarded tail
    args.force = args.force or (pp is not None and pp.force)   # --force either side

    # Status / control modes: act on the claim files under --out-root and exit
    # (no OCR, no DB, no worklist needed).
    if args.list_active:
        sys.exit(list_active(args.out_root))
    if args.list_resumable:
        sys.exit(list_resumable(args.out_root))
    if args.kill:
        sys.exit(kill_books(args.out_root, args.kill))

    if not OCR_ONE.is_file():
        sys.exit(f"ocr_one.py not found at {OCR_ONE}")
    if shutil.which("caffeinate") is None:
        sys.exit("caffeinate not found (expected on macOS)")
    if pp is not None and preprocess.has_any(pp) and args.interleave:
        sys.exit("preprocessing (-- …) is not yet supported with --interleave (the "
                 "persistent-server pipeline bypasses ocr_one); run without "
                 "--interleave.")
    preproc_label = preprocess.label(pp) if pp is not None else ""
    resolve_workflow(args)        # digital-first / scanned-no-text / scanned-with-text
    args.run_layout = {}
    # A --dump-input-files run quits before OCR (no layout/output dir needed), so skip
    # the interactive questionnaire — it's a quick read-only inspection.
    if _interactive() and not args.dump_input_files:
        if not args.layout_overrides:            # no per-book file -> ask the layout
            args.run_layout = prompt_layout()
        if args.out_root == OUT_ROOT:            # confirm the default output dir
            args.out_root = confirm_out_root(args.out_root)
    overrides = lo.load(args.layout_overrides) if args.layout_overrides else []

    proc_group.install()   # Ctrl-C / SIGTERM tears down ocr_one + surya_ocr + llama-server

    # Gather raw entries as (entry_str, limit): paths/edition refs from CLI + file
    # + preset. A bare invocation (no source) used to silently default to the
    # backfill preset and start OCR'ing — a footgun; now it errors with usage.
    entries = [(p, args.front_matter) for p in args.paths]
    entries += [(f"edition:{i}", args.front_matter) for i in args.editions]
    if args.from_file:
        entries += parse_list_file(args.from_file, args.front_matter)
    preset = args.preset                # only when explicitly asked, never implicit
    if preset:
        entries += [(f"edition:{i}", lim) for i, lim in PRESETS[preset].items()]

    if not entries:
        if _interactive():
            entries = prompt_source(args.front_matter)
        else:
            sys.exit("no work source. Give PDFs/folders, --editions ID..., "
                     "--from-file LIST, or --preset backfill.  (-h for full help)")

    # Resolve to (name, src, limit). Editions hit the DB; paths are taken as-is.
    # Connect only if some entry is an edition (path/folder runs need no DB).
    needs_db = any(isinstance(e, str) and e.startswith("edition:")
                   for e, _ in entries)
    con = None
    if needs_db:
        if not args.db.is_file():
            sys.exit(f"catalogue DB not found: {args.db}\n"
                     "(needed to resolve edition:ID — pass --db or set "
                     "$CATALOGUE_DB, or run on paths/folders instead)")
        con = sqlite3.connect(args.db)
    jobs = []
    for entry, limit in entries:
        if isinstance(entry, str) and entry.startswith("edition:"):
            eid = int(entry.split(":", 1)[1].split("#", 1)[0].strip())
            name, src, lim = resolve_edition(con, eid, limit)
            lay = {**args.run_layout,
                   **lo.lookup(overrides, edition=eid,
                               stem=src.stem if src else None)}
            jobs.append((name, src, lim, lay))
        else:
            p = Path(entry).expanduser()
            lay = {**args.run_layout,
                   **lo.lookup(overrides, stem=p.stem if p.suffix else p.name)}
            jobs.append((_slug(p.stem if p.suffix else p.name), p, limit, lay))

    if args.add_pages_after or args.dump_input_files:   # book-specific -> one book
        flag = "--add-pages-after" if args.add_pages_after else "--dump-input-files"
        if args.interleave:
            sys.exit(f"{flag} is not supported with --interleave.")
        if len(jobs) != 1:
            sys.exit(f"{flag} applies to a single book, but {len(jobs)} were given.")

    if args.interleave:                    # persistent servers + stage pipeline
        sys.exit(run_interleaved(args, jobs))

    # Port slots: one base port per concurrent job, each a block of `instances`
    # consecutive ports. Probe upward from --base-port for FREE blocks, so a
    # second worker started without a hand-picked --base-port just steps over the
    # ports a running worker already holds (no collisions). Jobs lease a slot,
    # run, and return it for the next.
    k = max(1, args.parallel_books)
    if k * args.workers_per_book > 3:
        print(f"note: parallel-books * workers-per-book = {k * args.workers_per_book} "
              "live servers; >3 on a single GPU regresses throughput (see the "
              "Performance section of README.md).", flush=True)
    bases, skipped = find_port_ranges(args.base_port, k, args.workers_per_book)
    if skipped:
        print(f"note: some ports at/above {args.base_port} are in use (another "
              f"worker running?) — using base port(s) {bases} instead. Keep TOTAL "
              "live instances across workers <= 3 on one GPU.", flush=True)
    slots = queue.Queue()
    for b in bases:
        slots.put(b)

    # Campaign progress: measure each book's pages up front (skipped/None sources
    # count 0) so the banner can show book N/M, pages done/total, and a live
    # secs/page ETA driven by the OCR page-cache growth this run.
    page_totals = {name: count_pages(src, limit) for name, src, limit, _ in jobs}
    progress = Progress(args.out_root, page_totals, [j[0] for j in jobs],
                        time.time())

    results = {}
    run_stats = {"books": 0, "pages": 0, "ocr_secs": 0.0}   # per-run s/pg average
    run_lock = threading.Lock()

    def work(job):
        name, src, limit, layout = job
        if src is None:
            return name, False
        if not (src.is_file() or src.is_dir()):
            print(f"=== {name}: SKIP — not found: {src}", flush=True)
            return name, False
        if args.skip_done and corpus.is_done(args.out_root / name):
            print(f"=== {name}: SKIP — done ({src.stem}.pdf + verified doc.json)",
                  flush=True)
            return name, True
        # Claim the book so a second worker (own port range) can join without
        # collisions; a dead worker's claim is stolen and resumed from cache.
        claim = (None if (not args.claim or args.dry_run)
                 else ClaimLock(args.out_root / name / ".claim", ttl=args.claim_ttl))
        if claim is not None and not claim.acquire():
            print(f"=== {name}: SKIP — claimed by another live worker", flush=True)
            return name, True
        port = slots.get()
        try:
            ok, target = run_one(name, src, limit, out_root=args.out_root,
                                 instances=args.workers_per_book, base_port=port,
                                 max_decode=args.max_decode,
                                 checkpoint_pages=args.checkpoint_pages,
                                 force=args.force, dry=args.dry_run,
                                 two_up=layout.get("two_up", "off"),
                                 columns=layout.get("columns", 1),
                                 drop_pages=layout.get("drop_pages"),
                                 preproc_tail=preproc_tail,
                                 preproc_label=preproc_label,
                                 add_pages_after=args.add_pages_after,
                                 dump_input_files=args.dump_input_files,
                                 img_limit=args.limit, output_tail=output_tail,
                                 prepare_output=args.prepare_output,
                                 claim=claim, progress_line=progress.banner(name),
                                 perf_stats=args.perf_stats,
                                 perf_interval=args.perf_interval,
                                 text_source=args.text_source)
        finally:
            slots.put(port)
            if claim is not None:
                claim.release()
        if ok and not args.dry_run and not args.dump_input_files:
            if args.max_decode:
                flag_capped_pages(target, args.out_root / name, name, args.max_decode)
            report_book_timing(name, args.out_root, run_stats, run_lock)
        return name, ok

    if k == 1:
        for job in jobs:
            n, ok = work(job)
            results[n] = ok
    else:
        with ThreadPoolExecutor(max_workers=k) as ex:
            for n, ok in ex.map(work, jobs):
                results[n] = ok

    print("\n=== summary ===", flush=True)
    for n, ok in results.items():
        print(f"  {n}: {'ok' if ok else 'FAILED/skipped'}", flush=True)

    done = [n for n, ok in results.items() if ok]
    if done:
        root = args.out_root
        print(f"""
=== output: {len(done)} book(s) -> {root}/ ===
Per book, under {root}/<name>/:
  <name>.pdf          searchable PDF (page images + recognized/native text layer)
  <name>.doc.json     STRUCTURED CONTRACT — blocks (type/page/bbox/refs), footnotes
                      map, features. This is the source for chunking + embedding.
  body/all_pages.txt  body reading text  (+ per-page NNNN_*.txt and body/pages.jsonl)
  full/all_pages.txt  full text incl. running heads / page numbers
  {root}/ingest_manifest.jsonl   one JSON row per book (paths + schema_version + features)

To consume the output:
  - USING_THE_OUTPUT.md  how to read doc.json — type policy, structure-aware chunking,
                      embedding, citation. Use lib/docjson.py; don't hand-parse.
  - ARCHITECTURE.md   the doc.json contract + RAG handoff — native (digital-first) vs
                      scanned, the dedup rule, and the deduplicated rag_handoff_manifest.jsonl.
""", flush=True)
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())

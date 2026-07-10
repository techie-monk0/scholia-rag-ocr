"""Surya OCR backend (the first ``OcrBackend`` implementation).

Owns its own parallelism/scaling: it OCRs by launching ``instances`` independent
llama-server processes and striping pages across them (the GPU is ~40% idle with
one; 2-3 instances ≈ 1.4-1.9x, and it regresses past ~3). Detection (for the PDF
text-layer line boxes) shards the same way without servers. Results are cached in
``<image_dir>/.surya_ocr`` using the surya_backend naming so other tools reuse them.

``instances`` is the parallelism knob the generic layer passes in; a different
backend would interpret its own options however it likes.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path

import surya_backend as sb
import parallel_ocr as po
from ocr_backend import OcrBackend, register


def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON via a temp file + atomic rename, so a kill (e.g. an OOM
    SIGKILL) mid-write can't leave a truncated cache that fails to parse and
    forces a full re-OCR. Same-dir temp keeps the rename on one filesystem."""
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def pending_pages(images, cached_stems, cache_mtime, *, force=False, reocr=None):
    """The pages that still need OCR/detection — the resume contract.

    A page is pending if it's forced, not yet checkpointed in the cache, or was
    modified after the cache was last written. So a re-run after a kill does ONLY
    the pages not already on disk: killing a book == pausing it, and the next
    --skip-done run picks up exactly where the checkpoint left off (at most one
    in-flight chunk is redone)."""
    forced = {Path(r).stem for r in (reocr or [])}

    def stale(img):
        return (force or img.stem in forced or img.stem not in cached_stems
                or img.stat().st_mtime > cache_mtime)

    return [img for img in images if stale(img)]


@register("surya")
class SuryaBackend(OcrBackend):
    """Options (via --instances / --backend-opt KEY=VALUE):
      instances        # parallel llama-server OCR instances (default 3, sweet spot 2-3)
      detector_batch   # surya_detect images per GPU pass; default per-device
                       # (Metal 8, CUDA 36). On Metal 8 is fine; --lines detect only.
      base_port        # first llama-server port (default 8091); instance k binds
                       # base_port+k. Set distinct ranges to run several ocr_one
                       # processes concurrently without port collisions.
      max_decode       # per-page (per-block) OCR token ceiling, default Surya's
                       # 8192. Lower it (e.g. 4000) to cap runaway hallucination
                       # loops on pages the model can't parse: a looping page
                       # bails at max_decode instead of grinding to 8192. Set it
                       # ABOVE the densest real page's token count or you'll
                       # truncate legitimate text (measure first).
      checkpoint_pages # OCR checkpoint granularity: OCR the book in chunks of N
                       # pages, flushing the cache after each so a kill mid-book
                       # loses at most one chunk (the rest resume from cache).
                       # Each chunk RE-SPAWNS the surya_ocr workers — a ~1-2 min
                       # model cold-start that idles the GPU — so chunking trades
                       # that warm-start tax for resume granularity. DEFAULT =
                       # unset = the whole book in ONE warm pass (surya cold-starts
                       # once, then stays hot; best throughput). Set N (e.g. 150)
                       # only if you need bounded re-OCR on a kill.
    """

    def __init__(self, instances: int = 3, detector_batch: int = None,
                 base_port: int = 8091, max_decode: int = None,
                 checkpoint_pages: int = None, **_ignored):
        self.instances = max(1, int(instances))
        self.detector_batch = int(detector_batch) if detector_batch else None
        self.base_port = int(base_port)
        self.max_decode = int(max_decode) if max_decode else None
        self.checkpoint_pages = int(checkpoint_pages) if checkpoint_pages else None
        self._pool = None       # persistent llama-server pool (interleave mode)
        if instances and int(instances) > 3:
            print(f"note: {instances} instances saturates the GPU; 2-3 is the "
                  f"sweet spot (see the Performance section of README.md).", flush=True)

    def config(self) -> dict:
        from surya.settings import settings
        dev = settings.TORCH_DEVICE_MODEL
        det = self.detector_batch or \
            f"device default ({ {'cpu': 8, 'mps': 8, 'cuda': 36}.get(dev, 8)} on {dev})"
        return {
            "instances": self.instances,
            "base_port": self.base_port,
            "max_decode": self.max_decode or "default (8192)",
            "checkpoint_pages": self.checkpoint_pages
            or "whole book (one warm surya pass)",
            "slots_per_instance": po.SLOTS_PER_SERVER,
            "ctx_size_per_instance": po._ctx_size(),
            "detector_batch": det,
            "device": dev,
        }

    def warmup(self, log_dir):
        """Launch the llama-servers and wait until they're healthy (the GGUF load —
        the bulk of a book's cold start), then keep them as the persistent pool that
        ocr_pages()/line_boxes() reuse. Meant to run in a thread OVERLAPPING image
        preprocessing. Returns the server handles for shutdown(). Idempotent-ish: if
        a pool is already up, returns None and leaves it."""
        if self._pool is not None:
            return None
        ports = [self.base_port + i for i in range(self.instances)]
        handles = po.launch_servers(ports, Path(log_dir))
        for p in ports:
            po.wait_health(p)
        self._pool = {"ports": ports,
                      "urls": [f"http://127.0.0.1:{p}/v1" for p in ports]}
        return handles

    def shutdown(self, handle):
        """Tear down the warm pool (clears it so a later call launches fresh)."""
        self._pool = None
        if handle:
            po.kill_servers(handle)

    @contextlib.contextmanager
    def persistent_servers(self, log_dir):
        """Launch the llama-servers ONCE and keep them warm across many books,
        instead of per ocr_pages() call. ocr_pages() inside this context reuses
        the pool, eliminating the per-book GGUF reload (the bulk of each book's
        cold-start). Detection (surya_detect) is a separate model, unaffected.
        Used by the interleave pipeline; the normal per-book path never enters
        here, so its behaviour is unchanged."""
        ports = [self.base_port + i for i in range(self.instances)]
        handles = po.launch_servers(ports, Path(log_dir))
        try:
            for p in ports:
                po.wait_health(p)
            self._pool = {"ports": ports,
                          "urls": [f"http://127.0.0.1:{p}/v1" for p in ports]}
            print(f"persistent OCR servers up on ports {ports} "
                  f"(warm across books)", flush=True)
            yield self._pool
        finally:
            self._pool = None
            po.kill_servers(handles)

    # -- internal: ensure a merged results/detection json, OCR'ing only the
    #    pages that are new / modified since the cache (or forced) --
    def _ensure(self, images, image_dir: Path, *, tool: str, force: bool,
                reocr=None) -> Path:
        image_dir = Path(image_dir)
        partial = len(images) < len(sb.list_images(image_dir))
        base = "results" if tool == "ocr" else "detection"
        suffix = f".subset{len(images)}" if partial else ""
        cache_dir = image_dir / ".surya_ocr"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{base}{suffix}.json"

        # Load the prior cache (keyed by image stem) for an incremental merge.
        existing, cache_mtime = {}, -1.0
        if path.exists():
            cache_mtime = path.stat().st_mtime
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = {}
        cached = {Path(k).stem for k in existing}
        todo = pending_pages(images, cached, cache_mtime, force=force, reocr=reocr)
        if not todo:
            print(f"Reusing {tool} cache: {path.name} ({len(images)} pages)",
                  flush=True)
            return path
        if len(todo) < len(images):
            print(f"{tool}: {len(todo)} of {len(images)} pages new/changed — "
                  f"reusing {len(images) - len(todo)} cached.", flush=True)

        work = cache_dir / "_parallel"
        n = max(1, min(self.instances, len(todo)))
        if tool == "ocr":
            if self.max_decode:
                # Read by the surya_ocr subprocess (env-inherited): cap the OCR
                # token budget so runaway pages bail early. Set BOTH paths —
                # full-page recognition (the one these scans use) and per-block —
                # so a loop can't escape via whichever path Surya picks.
                os.environ["SURYA_MAX_TOKENS_FULL_PAGE"] = str(self.max_decode)
                os.environ["SURYA_MAX_TOKENS_BLOCK_CEILING"] = str(self.max_decode)
            ports = [self.base_port + i for i in range(n)]
            # Checkpoint granularity: OCR `todo` in chunks and flush the cache
            # after each, so a kill (e.g. for memory) loses at most one in-flight
            # chunk — the rest resume from cache on the next run (`stale()` skips
            # cached pages). Default one full wave across the live servers.
            # Default: OCR the whole book in ONE run_sharded call so surya_ocr is
            # spawned once and stays warm (no per-chunk model cold-start). An
            # explicit checkpoint_pages subdivides it, re-spawning per chunk to
            # bound re-OCR on a kill (at the cost of that cold-start each chunk).
            chunk = self.checkpoint_pages or len(todo)
            mode = (f"checkpoint every {chunk}" if chunk < len(todo)
                    else "one warm pass (no mid-book checkpoint)")
            # Reuse the persistent pool (interleave mode) if present; else launch
            # a per-book server set as before.
            persistent = self._pool is not None
            if persistent:
                ports, urls, n = (self._pool["ports"], self._pool["urls"],
                                  len(self._pool["ports"]))
            print(f"OCR: {len(todo)} page(s) on {n} instance(s) (ports {ports}), "
                  f"{mode}{' [persistent servers]' if persistent else ''} ...",
                  flush=True)
            handles = None if persistent else po.launch_servers(ports, work)
            try:
                if not persistent:
                    for p in ports:
                        po.wait_health(p)
                    urls = [f"http://127.0.0.1:{p}/v1" for p in ports]
                t0 = time.perf_counter()
                done = 0
                for i in range(0, len(todo), chunk):
                    batch = todo[i:i + chunk]
                    new = po.run_sharded(batch, n, work, tool="ocr",
                                         server_urls=urls)
                    existing.update(new)            # merge: only `batch` changes
                    _atomic_write_json(path, existing)      # checkpoint to disk
                    done += len(new)
                    if chunk < len(todo):            # only narrate if >1 chunk
                        print(f"  OCR checkpoint: {done}/{len(todo)} page(s) "
                              f"cached -> {path.name}", flush=True)
                dt = time.perf_counter() - t0
            finally:
                if handles is not None:
                    po.kill_servers(handles)
            print(f"OCR done: {done} page(s) in {dt:.0f}s "
                  f"({dt/len(todo):.1f}s/page) -> {path.name}", flush=True)
            return path                              # cache already flushed above
        elif tool == "layout":
            # Region/layout labels via surya_layout — a VLM prompt on the same
            # llama-server as OCR (cheap: it emits a small JSON of region boxes,
            # not full text). Used by the detect-only path for digital-first books.
            ports = [self.base_port + i for i in range(n)]
            persistent = self._pool is not None
            if persistent:
                ports, urls, n = (self._pool["ports"], self._pool["urls"],
                                  len(self._pool["ports"]))
            print(f"Layout: {len(todo)} page(s) on {n} instance(s) (ports {ports})"
                  f"{' [persistent servers]' if persistent else ''} ...", flush=True)
            handles = None if persistent else po.launch_servers(ports, work)
            try:
                if not persistent:
                    for p in ports:
                        po.wait_health(p)
                    urls = [f"http://127.0.0.1:{p}/v1" for p in ports]
                new = po.run_sharded(todo, n, work, tool="layout", server_urls=urls)
            finally:
                if handles is not None:
                    po.kill_servers(handles)
        else:
            if self.detector_batch:
                # surya_detect (a subprocess) reads this from the env.
                os.environ["DETECTOR_BATCH_SIZE"] = str(self.detector_batch)
            print(f"Detection: {len(todo)} page(s) on {n} shard(s) ...", flush=True)
            new = po.run_sharded(todo, n, work, tool="detect")

        existing.update(new)                       # merge: only `todo` pages change
        _atomic_write_json(path, existing)
        return path

    def ocr_pages(self, images, image_dir, *, force=False, reocr=None):
        path = self._ensure(images, image_dir, tool="ocr", force=force, reocr=reocr)
        return sb.load_pages(path, image_dir)

    def line_boxes(self, images, image_dir, *, force=False, reocr=None):
        path = self._ensure(images, image_dir, tool="detect", force=force,
                            reocr=reocr)
        return sb.load_line_boxes(path, image_dir)

    def layout_regions(self, images, image_dir, *, force=False):
        """{image-stem: [(bbox, label), ...]} via surya_layout — region labels
        for the detect-only (native-text) path."""
        path = self._ensure(images, image_dir, tool="layout", force=force)
        return sb.load_layout(path, image_dir)

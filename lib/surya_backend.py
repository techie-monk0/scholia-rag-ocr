"""Shared Surya-OCR backend.

Runs Surya OCR over a folder of page images *once*, caches the raw
``results.json``, and exposes the result as a normalized list of ``Page``
objects.  Both ``ocr_to_pdf.py`` (searchable PDF) and ``ocr_to_text.py``
(bare text) import this module, so the expensive GPU OCR pass is shared:
whichever script runs first does the OCR, the second reuses the cache.

Surya >= 0.20 does full-page VLM OCR and returns *blocks* (paragraph/table
level) carrying ``html`` + ``bbox`` + ``confidence``.  We drive Surya through
its CLI (the most stable interface across versions) and parse the JSON
defensively so older ``text_lines``/``text`` schemas keep working too.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

# The normalized Page/Block model is engine-neutral and lives in ocr_backend;
# re-exported here so existing `sb.Page` / `sb.Block` references keep working.
from ocr_backend import Block, Page  # noqa: F401

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


# Layout labels dropped by the --body-only text path (running headers/footers,
# which also carry page numbers). Everything else (Text, SectionHeader, Footnote,
# Caption, ListGroup, ...) is kept as body content.
NON_BODY_LABELS = frozenset({"PageHeader", "PageFooter"})


# --------------------------------------------------------------------------- #
# HTML -> plain text (Surya blocks carry html)
# --------------------------------------------------------------------------- #
_BREAK_TAGS = {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5",
               "h6", "table", "thead", "tbody", "ul", "ol", "blockquote"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._sup_depth = 0          # inside <sup>…</sup> (footnote refs)

    def handle_data(self, data):
        if self._sup_depth:
            # Surya tags in-text footnote markers as <sup>316</sup>; emit them
            # as a delimited Markdown-style ref [^316] instead of bare digits
            # glued to the preceding word ("says:316"), so a chunker can strip
            # or relocate them with one regex.
            ref = data.strip()
            if ref:
                self.parts.append(f"[^{ref}]")
        else:
            self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag == "sup":
            self._sup_depth += 1
        elif tag in _BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "sup":
            self._sup_depth = max(0, self._sup_depth - 1)
        elif tag in _BREAK_TAGS:
            self.parts.append("\n")


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html or "")
    text = "".join(p.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]


def list_images(image_dir: Path) -> list[Path]:
    return sorted((p for p in image_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                  key=lambda p: natural_key(p.name))


def _poly_bbox(poly):
    if not poly:
        return None
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


# --------------------------------------------------------------------------- #
# Running Surya (cached)
# --------------------------------------------------------------------------- #
def _bin_dirs() -> list[str]:
    """Likely locations of console scripts (pip --user installs land off PATH)."""
    import site
    import sysconfig
    dirs = []
    try:
        dirs.append(sysconfig.get_path("scripts", f"{os.name}_user"))
    except Exception:
        pass
    try:
        dirs.append(os.path.join(site.getuserbase(), "bin"))
    except Exception:
        pass
    dirs += [
        os.path.join(sys.prefix, "bin"),
        os.path.expanduser("~/Library/Python/3.12/bin"),
        "/usr/local/bin", "/opt/homebrew/bin",
    ]
    seen, out = set(), []
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def _resolve_exe(name: str, search_path: str) -> str | None:
    return shutil.which(name, path=search_path)


def _find_llama_server() -> str | None:
    """Locate a llama-server binary for Surya's llamacpp backend.

    Order: explicit env, anything on PATH, then the arm64 build vendored under
    the package's ``.llama/`` directory (this module may live in a lib/ subdir,
    so look in its own dir and the parent).
    """
    env_bin = os.environ.get("LLAMA_CPP_BINARY")
    if env_bin:
        p = Path(env_bin)
        if p.is_file() and os.access(p, os.X_OK):
            return env_bin
        # Set but unusable (typo, wrong path, non-executable): don't ignore it
        # silently — a misconfiguration should be visible, then fall back.
        print(f"WARNING: LLAMA_CPP_BINARY={env_bin!r} is not a runnable file; "
              "ignoring it and falling back to PATH / the vendored .llama/ build.",
              flush=True)
    on_path = shutil.which("llama-server")
    if on_path:
        return on_path
    here = Path(__file__).resolve().parent
    for base in (here, here.parent):
        for cand in sorted((base / ".llama").rglob("llama-server")):
            if os.access(cand, os.X_OK):
                return str(cand)
    return None


# Each Surya CLI tool, with the command-name/flag forms tried in order
# (covers drift across versions). "ocr" needs llama-server; "detect" does not.
_TOOL_VARIANTS = {
    "ocr": lambda inp, out: [
        ["surya_ocr", inp, "--output_dir", out],
        ["surya_ocr", inp, "--results_dir", out],
        ["surya", "ocr", inp, "--output_dir", out],
        ["surya_ocr", inp],
        ["surya", "ocr", inp],
    ],
    "detect": lambda inp, out: [
        ["surya_detect", inp, "--output_dir", out],
        ["surya", "detect", inp, "--output_dir", out],
        ["surya_detect", inp],
    ],
}


def _find_results_json(roots) -> Path | None:
    candidates: list[Path] = []
    for root in roots:
        root = Path(root)
        if root.is_dir():
            candidates += list(root.rglob("results.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_tool(tool: str, input_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # MPS lacks a few ops; let torch fall back to CPU for those instead of crashing.
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Make pip --user script dirs reachable (surya_* often lands off PATH).
    search_path = os.pathsep.join(_bin_dirs() + [env.get("PATH", "")])
    env["PATH"] = search_path
    # The OCR model is served by llama-server; detection uses its own small model.
    if tool == "ocr":
        llama = _find_llama_server()
        if llama:
            env["LLAMA_CPP_BINARY"] = llama
            env["PATH"] = os.path.dirname(llama) + os.pathsep + env["PATH"]
            print(f"llama-server: {llama}", flush=True)
        else:
            print("WARNING: no llama-server found; Surya OCR will fail. "
                  "See README (llama.cpp).", flush=True)
    # Avoid system sleep stalling a long pass (display-off -> sleep) on macOS.
    prefix = (["caffeinate", "-i", "-s"]
              if sys.platform == "darwin" and shutil.which("caffeinate") else [])

    last_err = None
    for variant in _TOOL_VARIANTS[tool](str(input_path), str(out_dir)):
        exe = _resolve_exe(variant[0], search_path)
        if not exe:
            last_err = f"command not found: {variant[0]}"
            continue
        cmd = prefix + [exe] + variant[1:]
        print(f"\n>>> {' '.join(cmd)}\n", flush=True)
        try:
            proc = subprocess.run(cmd, env=env)
        except FileNotFoundError:
            last_err = f"command not found: {variant[0]}"
            continue
        if proc.returncode == 0:
            found = _find_results_json([out_dir, Path.cwd() / "results", Path.cwd()])
            if found:
                return found
            last_err = "exited 0 but no results.json was found"
            break
        # Exit 2 == click usage/flag error -> a different flag form may work, so
        # try the next variant. Any other nonzero is a real runtime failure;
        # stop now rather than burn another full model load.
        last_err = f"exit code {proc.returncode}"
        if proc.returncode != 2:
            break
    raise SystemExit(
        f"Surya {tool} failed (" + str(last_err) + ").\n"
        "Install it with:  pip install surya-ocr\n"
        "First run downloads model weights from Hugging Face.")


class _DirLock:
    """Cross-process lock so concurrent script runs OCR/detect only once."""

    def __init__(self, path: Path):
        self._path = path
        self._fh = None

    def __enter__(self):
        import fcntl
        self._fh = open(self._path, "w")
        fcntl.flock(self._fh, fcntl.LOCK_EX)  # blocks until acquired
        return self

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()


def _subset_dir(cache_dir: Path, images: list[Path], limit: int) -> Path:
    """A folder of symlinks to the first N images (Surya wants a path)."""
    subset = cache_dir / f"_subset{limit}"
    if subset.exists():
        shutil.rmtree(subset)
    subset.mkdir(parents=True)
    for img in images:
        try:
            (subset / img.name).symlink_to(img.resolve())
        except OSError:
            shutil.copy2(img, subset / img.name)
    return subset


def _ensure_cached(tool: str, base: str, image_dir: Path, cache_dir: Path, *,
                   force: bool, limit: int | None) -> Path:
    image_dir = Path(image_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir)
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    if limit:
        images = images[:limit]
        canonical = cache_dir / f"{base}.subset{limit}.json"
        run_input = _subset_dir(cache_dir, images, limit)
        raw_out = cache_dir / f"_raw_{base}_subset{limit}"
    else:
        canonical = cache_dir / f"{base}.json"
        run_input = image_dir
        raw_out = cache_dir / f"_raw_{base}"

    newest_img = max(img.stat().st_mtime for img in images)

    def fresh() -> bool:
        return (canonical.exists() and not force
                and canonical.stat().st_mtime >= newest_img)

    if fresh():
        print(f"Reusing cached {tool}: {canonical}", flush=True)
        return canonical

    # Serialize concurrent runs: the loser waits here, then finds the cache fresh.
    with _DirLock(cache_dir / ".surya.lock"):
        if fresh():
            print(f"Reusing cached {tool} (built by parallel run): {canonical}",
                  flush=True)
            return canonical
        found = _run_tool(tool, run_input, raw_out)
        shutil.copy2(found, canonical)
    print(f"Cached {tool} -> {canonical}", flush=True)
    return canonical


def ensure_results(image_dir: Path, cache_dir: Path, *, force: bool = False,
                   limit: int | None = None) -> Path:
    """Path to full-page OCR results.json for ``image_dir`` (runs Surya if needed)."""
    return _ensure_cached("ocr", "results", image_dir, cache_dir,
                          force=force, limit=limit)


def ensure_detection(image_dir: Path, cache_dir: Path, *, force: bool = False,
                     limit: int | None = None) -> Path:
    """Path to line-detection results.json for ``image_dir`` (runs surya_detect)."""
    return _ensure_cached("detect", "detection", image_dir, cache_dir,
                          force=force, limit=limit)


# --------------------------------------------------------------------------- #
# Parsing results.json -> Page list
# --------------------------------------------------------------------------- #
def usable_cache(path) -> bool:
    """True only if a surya cache JSON (results/detection/layout) exists AND holds
    at least one page. An interrupted OCR/merge leaves an empty ``{}`` stub that is
    present on disk but carries no pages; guard on this, not mere file existence, so
    such a book is treated as needing re-OCR rather than silently rebuilt empty."""
    if not path:
        return False
    p = Path(path)
    if not p.is_file():
        return False
    try:
        data = json.loads(p.read_text())
    except (ValueError, OSError):
        return False
    return isinstance(data, dict) and len(data) > 0


def load_pages(results_path: Path, image_dir: Path) -> list[Page]:
    data = json.loads(Path(results_path).read_text())

    by_name, by_stem = {}, {}
    for img in list_images(Path(image_dir)):
        by_name[img.name] = img
        by_stem[img.stem] = img

    def match_image(key: str):
        return by_name.get(key) or by_stem.get(key) or by_stem.get(Path(key).stem)

    pages: list[Page] = []
    for key, page_list in data.items():
        if not isinstance(page_list, list):
            page_list = [page_list]
        img = match_image(str(key))
        for pd in page_list:
            blocks_raw = (pd.get("blocks") or pd.get("text_lines")
                          or pd.get("lines") or [])
            width = height = None
            if img:
                try:
                    width, height = Image.open(img).size
                except Exception:
                    pass
            if not width:
                ib = pd.get("image_bbox") or [0, 0, 1000, 1000]
                width, height = int(ib[2]), int(ib[3])

            blocks: list[Block] = []
            for b in blocks_raw:
                html = b.get("html")
                txt = b.get("text")
                if txt is None:
                    txt = html_to_text(html or "")
                txt = (txt or "").strip()
                if not txt:
                    continue
                bbox = b.get("bbox") or _poly_bbox(b.get("polygon")) \
                    or [0, 0, width, height]
                blocks.append(Block(txt, [float(v) for v in bbox],
                                    b.get("confidence"), b.get("label"), html))

            pages.append(Page(img, str(key), int(width), int(height),
                              blocks, pd.get("page")))

    pages.sort(key=lambda p: (natural_key(p.name), p.page_no or 0))
    return pages


def load_line_boxes(detection_path: Path, image_dir: Path) -> dict[str, list]:
    """Parse surya_detect output into {image-stem: [line bbox, ...]}."""
    data = json.loads(Path(detection_path).read_text())
    by_stem: dict[str, list] = {}
    for key, page_list in data.items():
        if not isinstance(page_list, list):
            page_list = [page_list]
        boxes: list[list[float]] = []
        for pd in page_list:
            for b in (pd.get("bboxes") or pd.get("text_lines") or []):
                bb = b.get("bbox") or _poly_bbox(b.get("polygon"))
                if bb:
                    boxes.append([float(v) for v in bb])
        by_stem[Path(str(key)).stem] = boxes
    return by_stem


def load_layout(layout_path: Path, image_dir: Path) -> dict[str, list]:
    """Parse surya_layout output into {image-stem: [(bbox, label), ...]} — the
    region/furniture labels (PageHeader, Picture, Table, Figure, ...) the
    detect-only path merges onto native-text blocks."""
    data = json.loads(Path(layout_path).read_text())
    by_stem: dict[str, list] = {}
    for key, pd in data.items():
        regions: list = []
        for g in (pd if isinstance(pd, list) else [pd]):
            for b in (g.get("bboxes") or []):
                bb = b.get("bbox") or _poly_bbox(b.get("polygon"))
                lbl = b.get("label") or b.get("raw_label")
                if bb and lbl:
                    regions.append(([float(v) for v in bb], lbl))
        by_stem[Path(str(key)).stem] = regions
    return by_stem


def block_line_boxes(blk, det_boxes: list | None = None):
    """Per-visual-line boxes for ONE block, but ONLY from real detection boxes
    (returns None if their count doesn't match the block's lines). Used for the
    optional `line_bbox` on verse/ambiguous blocks — a band-slice would give every
    line the same x and defeat the per-line alignment check, so we don't emit one."""
    bb = getattr(blk, "bbox", None)
    vlines = [ln for ln in (getattr(blk, "text", "") or "").split("\n") if ln.strip()]
    if not bb or len(bb) < 4 or not vlines:
        return None
    x0, y0, x1, y1 = bb[:4]
    inside = sorted(
        (b for b in (det_boxes or [])
         if x0 - 2 <= (b[0] + b[2]) / 2 <= x1 + 2
         and y0 - 2 <= (b[1] + b[3]) / 2 <= y1 + 2),
        key=lambda b: b[1])
    if inside and len(inside) == len(vlines):
        return [[round(float(v), 1) for v in b[:4]] for b in inside]
    return None


def line_items(page: Page, det_boxes: list | None = None) -> list[tuple]:
    """Yield (text, bbox) at *line* granularity for one page.

    Each OCR block is split into its visual lines (Surya marks them with
    ``<br/>``, already turned into ``\\n`` by ``html_to_text``).  When detection
    boxes are supplied and their count inside a block matches the block's visual
    lines, those precise per-line boxes are used; otherwise the block bbox is
    sliced into equal horizontal bands (one per line).
    """
    det = sorted(det_boxes or [], key=lambda b: (b[1], b[0]))
    items: list[tuple] = []
    for blk in page.blocks:
        vlines = [ln for ln in blk.text.split("\n") if ln.strip()]
        if not vlines:
            continue
        x0, y0, x1, y1 = blk.bbox
        inside = sorted(
            (b for b in det
             if x0 - 2 <= (b[0] + b[2]) / 2 <= x1 + 2
             and y0 - 2 <= (b[1] + b[3]) / 2 <= y1 + 2),
            key=lambda b: b[1])
        if det and len(inside) == len(vlines):
            items.extend(zip(vlines, inside))
        else:
            band = (y1 - y0) / len(vlines)
            for i, ln in enumerate(vlines):
                items.append((ln, [x0, y0 + i * band, x1, y0 + (i + 1) * band]))
    return items


# --------------------------------------------------------------------------- #
# Shared CLI scaffolding
# --------------------------------------------------------------------------- #
def common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("image_dir", type=Path,
                   help="Folder of page images (jpg/png/tif).")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Where to store/reuse OCR results "
                        "(default: <image_dir>/.surya_ocr). Shared between scripts.")
    p.add_argument("--force", action="store_true",
                   help="Re-run OCR even if a cache exists.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only OCR the first N images (quick smoke test).")
    p.add_argument("--parallel", type=int, default=None,
                   help="Concurrent OCR slots in the shared llama-server "
                        "(SURYA_INFERENCE_PARALLEL; default 8). Raise to saturate "
                        "the GPU — this, not more processes, is the speed knob.")
    p.add_argument("--detector-batch", type=int, default=None,
                   help="How many page images the text-line DETECTION model "
                        "(surya_detect) runs per GPU forward pass — Surya's "
                        "DETECTOR_BATCH_SIZE. Bigger = better GPU use but more "
                        "memory. Default is Surya's per-device value (Metal/MPS: "
                        "8, CUDA: 36). On Metal 8 is fine; raising it speeds the "
                        "detection pass only slightly (it's a small model, not "
                        "the bottleneck). Affects --lines detect only.")
    return p


def resolve_cache_dir(args) -> Path:
    return args.cache_dir or (args.image_dir / ".surya_ocr")


def apply_perf_env(args) -> None:
    """Push --parallel / --detector-batch into the env Surya reads (Pydantic
    Settings → env var of the same name). Set before any tool run."""
    if getattr(args, "parallel", None):
        os.environ["SURYA_INFERENCE_PARALLEL"] = str(args.parallel)
    if getattr(args, "detector_batch", None):
        os.environ["DETECTOR_BATCH_SIZE"] = str(args.detector_batch)


def get_pages(args) -> list[Page]:
    """Run/reuse OCR per parsed args and return the normalized pages."""
    apply_perf_env(args)
    cache_dir = resolve_cache_dir(args)
    results = ensure_results(args.image_dir, cache_dir,
                             force=args.force, limit=args.limit)
    pages = load_pages(results, args.image_dir)
    if args.limit:
        pages = pages[:args.limit]
    if not pages:
        raise SystemExit("OCR produced no pages — check the results.json.")
    return pages

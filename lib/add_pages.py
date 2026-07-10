"""Insert hand-added page images into a book AFTER preprocessing — the
``--add-pages-after`` step of run_ocr / ocr_one.

Unlike rotate/swap (preprocess steps that regenerate their step dirs), added pages
are a durable INPUT. They live in a sibling ``_added/`` dir under the SOURCE folder,
which is NOT part of the preprocess step-dir chain, so they survive ``--force``
(that only wipes the pad/rotate/swap dirs). At OCR time ocr_one OCRs ``_added/``
alongside the preprocessed pages and merges the two in natural-sort order — so an
added page is never itself rotated or swapped.

Naming (generated, not user-supplied): the caller gives an ANCHOR — an existing
page's FILENAME (no path) — and the new images IN ORDER. Each is copied in as
``<book-prefix><anchor-page-number><letter><ext>``: after ``… - 7.jpg`` in a book
padded to 3 digits the inserts become ``… - 007a.jpg``, ``… - 007b.jpg``, … The
letter suffix carries no trailing digit, so ``--pad-page-numbers`` / ``--swap`` leave
it alone, and it natural-sorts immediately after the anchor page. The source files'
own names are irrelevant — only their order. Re-adding the same bytes is a no-op
(content-hash dedup), so it is safe to run repeatedly; delete a file from ``_added/``
to remove an inserted page.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import string
from pathlib import Path

import surya_backend as sb

_NUM_RE = re.compile(r"(\d+)\s*$")          # trailing integer of a filename stem


def _page_no(stem: str):
    m = _NUM_RE.search(stem)
    return int(m.group(1)) if m else None


def _label(k: int) -> str:
    """Sequence label: 0->'a', 1->'b', … 25->'z', 26->'aa', 27->'ab', …"""
    s, k = "", k + 1
    while k:
        k, r = divmod(k - 1, 26)
        s = string.ascii_lowercase[r] + s
    return s


def added_dir(source_dir) -> Path:
    """The durable folder hand-added pages live in (a sibling of the step dirs, but
    NOT one of them — clean_step_dirs / --force never touches it)."""
    return Path(source_dir) / "_added"


def resolve_anchor(anchor: str, book_images):
    """Resolve an ANCHOR (a bare filename, NO path) to the book page it names.

    Returns ``(prefix, width, number)`` — the filename text before the page number,
    the number's digit width, and the page number — so inserts sort right after it.
    Raises ``SystemExit`` if the anchor contains a path, has no page number, or names
    a page that is not in the book."""
    if "/" in anchor or "\\" in anchor:
        raise SystemExit(
            f"--add-pages-after: ANCHOR must be a filename, not a path: {anchor!r}")
    num = _page_no(Path(anchor).stem)
    if num is None:
        raise SystemExit(
            f"--add-pages-after: ANCHOR {anchor!r} has no page number to anchor to")
    matches = [Path(p) for p in book_images if _page_no(Path(p).stem) == num]
    if not matches:
        present = sorted({_page_no(Path(p).stem) for p in book_images
                          if _page_no(Path(p).stem) is not None})
        near = [n for n in present if abs(n - num) <= 3]
        raise SystemExit(
            f"--add-pages-after: anchor page {num} not found in the book"
            + (f" (nearby pages: {near})" if near else ""))
    target = matches[0]
    m = _NUM_RE.search(target.stem)
    return target.stem[:m.start()], len(m.group(1)), num


def dedupe_added(dst, *, log=None) -> list:
    """Remove exact-content-duplicate images from the ``_added/`` dir, keeping the
    natural-sort-first of each identical-bytes group. Heals a page added under two
    names at different widths (e.g. '9b' from a pre-padding run and '009b' padded) —
    since padding never touches _added/, the stale copy would otherwise linger and
    get inserted twice. Returns the removed filenames. Content-hash exact match, so
    it can only ever drop a byte-identical copy."""
    log = log or (lambda _m: None)
    dst = Path(dst)
    if not dst.is_dir():
        return []
    groups: dict = {}
    for f in sb.list_images(dst):
        groups.setdefault(_sha(f), []).append(f)

    def _num_digits(f) -> int:               # digits in the trailing "<number><letters>"
        m = re.search(r"(\d+)[a-z]*$", Path(f).stem)
        return len(m.group(1)) if m else 0

    removed = []
    for files in groups.values():
        if len(files) == 1:
            continue
        # keep the MOST-padded name (most digits before the letter suffix, e.g. '009b'
        # over '9b'); ties broken by name. '9b' and '009b' natural-sort EQUAL, so this
        # explicit rule is what makes the choice deterministic.
        files.sort(key=lambda f: (-_num_digits(f), Path(f).name))
        keep = files[0]
        for f in files[1:]:
            f.unlink()
            removed.append(f.name)
            log(f"add-pages: removed duplicate {f.name!r} "
                f"(identical content to {keep.name!r})")
    return removed


def resolve_start(names, start):
    """Index into ``names`` (an ordered list of filenames) of the file ``start`` —
    a bare filename, matched exactly, else by stem, else by trailing page number.
    Used by --dump-input-files. Raises ``SystemExit`` if not found."""
    start = Path(start).name
    if start in names:
        return names.index(start)
    stems = [Path(x).stem for x in names]
    if Path(start).stem in stems:
        return stems.index(Path(start).stem)
    sp = _page_no(Path(start).stem)
    if sp is not None:
        for i, x in enumerate(names):
            if _page_no(Path(x).stem) == sp:
                return i
    raise SystemExit(f"--dump-input-files: start file {start!r} not found in the "
                     f"input ({len(names)} pages); first few: {names[:5]}")


def _expand_sources(sources):
    """A list of file paths (kept IN ORDER) or a single folder (natural-sorted) ->
    the ordered list of source image Paths. Raises ``SystemExit`` on a missing path
    or if nothing resolves to an image."""
    paths: list[Path] = []
    for s in sources:
        p = Path(s).expanduser()
        if p.is_dir():
            paths.extend(sb.list_images(p))
        elif p.is_file():
            paths.append(p)
        else:
            raise SystemExit(f"--add-pages-after: source not found: {s}")
    if not paths:
        raise SystemExit("--add-pages-after: no source images to add")
    return paths


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add(source_dir, book_images, anchor, sources, *, log=None) -> Path:
    """Copy ``sources`` (ordered files, or a single folder) into
    ``<source_dir>/_added/``, named to sort right after ``anchor`` (see module doc).
    Idempotent by content hash. Returns the ``_added/`` dir."""
    log = log or (lambda _m: None)
    if not sources:
        raise SystemExit(f"--add-pages-after {anchor!r}: give at least one source "
                         "image (or a folder) to add after it")
    prefix, width, num = resolve_anchor(anchor, book_images)
    paths = _expand_sources(sources)
    dst = added_dir(source_dir)
    dst.mkdir(parents=True, exist_ok=True)
    base = f"{prefix}{str(num).zfill(width)}"

    used, by_hash = set(), {}                # letters-in-use + ALL content hashes
    for f in sb.list_images(dst):
        # Dedup against EVERY already-added page, whatever its name/width — so the
        # same bytes can't be re-added as e.g. both '9a' (unpadded run) and '009a'
        # (padded run). Letter tracking stays scoped to this anchor's base.
        by_hash[_sha(f)] = Path(f).name
        suf = Path(f).stem[len(base):]
        if Path(f).stem.startswith(base) and suf and suf.isalpha() and suf.islower():
            used.add(suf)

    li = 0
    def _next() -> str:
        nonlocal li
        while True:
            lbl = _label(li)
            li += 1
            if lbl not in used:
                used.add(lbl)
                return lbl

    for f in paths:
        h = _sha(f)
        if h in by_hash:                     # same bytes already inserted -> no-op
            log(f"add-pages: {f.name!r} already present as {by_hash[h]!r} — skip")
            continue
        name = f"{base}{_next()}{f.suffix.lower()}"
        shutil.copy2(f, dst / name)
        by_hash[h] = name
        log(f"add-pages: {f.name!r} -> {name!r} (after page {num})")
    return dst

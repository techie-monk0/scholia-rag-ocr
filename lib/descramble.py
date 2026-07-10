"""De-scramble page order from the printed page numbers (running heads).

vFlat-style phone scans can come back in a wrong, non-uniform order — facing
pairs swapped, plus stray duplicate / blank re-scans wedged in — which no fixed
adjacent-swap can undo. This module holds the ENGINE-FREE logic: parse a running
head into its page number, then reorder the pages into true reading order,
dropping duplicates. The OCR that reads the running heads lives in descramble.py
(it feeds this parsed text); everything here is pure and unit-tested.

Running heads look like:
    verso (even, left):  "8 / The Great Hūṃ"              -> number leads
    recto (odd, right):  "Translator's Introduction / 3"  -> number trails
    front matter:        "viii / The Great Hūṃ"           -> roman numeral
A page with no number (blank / plate / washed scan) parses to None.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_R = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_ROMAN_VALS = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"),
               (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"),
               (5, "v"), (4, "iv"), (1, "i")]


def _int_to_roman(n: int) -> str:
    out = []
    for v, sym in _ROMAN_VALS:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def roman_to_int(s):
    """A valid lower/upper roman numeral -> its int; ``None`` otherwise. Strict:
    only canonical forms round-trip (``'iiii'`` and ``'vx'`` are rejected), so a
    stray word is never mistaken for a numeral."""
    s = (s or "").strip().lower()
    if not s or any(c not in _R for c in s):
        return None
    total, prev = 0, 0
    for c in reversed(s):
        v = _R[c]
        total += -v if v < prev else v
        prev = v
    return total if _int_to_roman(total) == s else None


def parse_running_head(text):
    """Extract a page number from running-head text: ``('arabic', n)`` /
    ``('roman', n)`` / ``None``.

    A running head reads ``N / Title`` (verso) or ``Title / N`` (recto), so the
    page number is the token on one side of the FIRST ``/``. That stays reliable
    even when the crop also grabs the first line of body text (which pushes the
    number out of the string's edges). Falls back to the very first token (verso
    heads whose slash the OCR dropped)."""
    if not text:
        return None
    cands = []
    parts = re.split(r"\s*/\s*", text.strip(), maxsplit=1)
    if len(parts) == 2:
        left = re.findall(r"[A-Za-z]+|\d+", parts[0])
        right = re.findall(r"[A-Za-z]+|\d+", parts[1])
        if left:
            cands.append(left[-1])          # token just before the slash (verso)
        if right:
            cands.append(right[0])          # token just after the slash (recto)
    toks = re.findall(r"[A-Za-z]+|\d+", text)
    if toks:
        cands.append(toks[0])               # fallback: very first token
    for tok in cands:
        if tok.isdigit():
            return ("arabic", int(tok))
    for tok in cands:
        r = roman_to_int(tok)
        if r is not None:
            return ("roman", r)
    return None


def _fmt(label) -> str:
    kind, value = label
    return _int_to_roman(value) if kind == "roman" else str(value)


def plan_descramble(records):
    """``records`` = ``[(name, label), ...]`` in FILE order (label from
    ``parse_running_head``, or ``None`` for an unnumbered page). Returns
    ``(order, drops)``:

      order  names in true reading order — numbered pages sorted (roman before
             arabic, then by value); each unnumbered page kept, anchored right
             after the numbered page that PRECEDES it in file order (leading
             unnumbered pages go first).
      drops  ``[(name, reason)]`` — a repeated page number keeps its FIRST scan
             and drops the rest (the duplicate re-scans).
    """
    seen, drops, kept = {}, [], []
    for name, label in records:
        if label is None:
            continue
        if label in seen:
            drops.append((name, f"duplicate of page {_fmt(label)} "
                                f"(kept {seen[label]!r})"))
            continue
        seen[label] = name
        kept.append((0 if label[0] == "roman" else 1, label[1], name))
    kept.sort(key=lambda t: (t[0], t[1]))
    ordered_numbered = [name for _, _, name in kept]
    kept_names = set(ordered_numbered)

    # Anchor each unnumbered page to the numbered page preceding it in file order.
    after, leading, prev = defaultdict(list), [], None
    dropped_names = {n for n, _ in drops}
    for name, label in records:
        if name in kept_names:
            prev = name
        elif label is None and name not in dropped_names:
            (leading if prev is None else after[prev]).append(name)

    order = list(leading)
    for name in ordered_numbered:
        order.append(name)
        order.extend(after.get(name, []))
    return order, drops


def flag_suspects(records, window=5, thresh=10):
    """Names whose read page number is far from EVERY nearby (same-kind) numbered
    page in FILE order — almost certainly an OCR misread (e.g. '47' read as '7'
    among pages in the 40s).

    A scan is locally coherent: consecutive files carry near-consecutive page
    numbers (facing-pair swaps and dropped openers only nudge it by a few). So a
    real number is always close to at least one file-order neighbor. A value
    isolated from ALL of them isn't trustworthy. A page at a GENUINE gap (missing
    scans) is still supported by neighbors on the far side of the gap, so it is not
    flagged — only truly isolated reads are."""
    numbered = [(name, lab) for name, lab in records if lab is not None]
    suspects = set()
    for pos, (name, (kind, val)) in enumerate(numbered):
        lo, hi = max(0, pos - window), min(len(numbered), pos + window + 1)
        near = [numbered[p][1][1] for p in range(lo, hi)
                if p != pos and numbered[p][1][0] == kind]
        if near and min(abs(val - v) for v in near) > thresh:
            suspects.add(name)
    return suspects


def _infer_run(run, prev, nxt, inferred):
    """Assign numbers to a run of consecutive unnumbered pages between numbered
    pages ``prev`` and ``nxt`` (either may be None at the ends). Only fills a real
    GAP in the numbered sequence, from ``prev``+1 upward — so a chapter opener
    between printed 16 and 18 becomes 17, but an extra plate where the numbers are
    already contiguous stays unnumbered."""
    if prev is not None and nxt is not None and prev[0] == nxt[0]:
        gap = nxt[1] - prev[1] - 1                  # missing numbers between them
        for k, name in enumerate(run):
            if k < gap:                             # within the gap -> infer
                inferred[name] = (prev[0], prev[1] + 1 + k)
            # else: more unnumbered pages than the gap -> extra leaves (plates)
    elif prev is not None and nxt is None:
        for k, name in enumerate(run):              # trailing run: best-guess run-on
            inferred[name] = (prev[0], prev[1] + 1 + k)
    # leading run, or a roman/arabic boundary -> can't infer; left None


def infer_missing(order, label_by):
    """Infer concrete page numbers for the unnumbered pages in reading ``order``
    (chapter openers etc.) by gap-filling the numbered sequence. Returns
    ``{name: (kind, value)}`` for the pages it could infer; the rest stay None.
    Annotation only — it does not reorder."""
    inferred, i, n = {}, 0, len(order)
    while i < n:
        if label_by[order[i]] is not None:
            i += 1
            continue
        j = i
        while j < n and label_by[order[j]] is None:
            j += 1
        _infer_run(order[i:j],
                   label_by[order[i - 1]] if i > 0 else None,
                   label_by[order[j]] if j < n else None,
                   inferred)
        i = j
    return inferred


def verify_increasing(labels):
    """``labels`` = the parsed labels in final reading order. Returns a list of
    human-readable violations where a numbered page isn't strictly greater than
    the previous numbered page of the same kind (empty == in order)."""
    bad, last = [], {}
    for i, label in enumerate(labels):
        if label is None:
            continue
        kind, value = label
        if kind in last and value <= last[kind]:
            bad.append(f"position {i}: {kind} {_fmt(label)} follows "
                       f"{kind} {_int_to_roman(last[kind]) if kind=='roman' else last[kind]}")
        last[kind] = value
    return bad


# --------------------------------------------------------------------------- #
# Reading the strips (default: tesseract — a printed-digit task it's suited to,
# and it needs no GPU/servers, so the reorder is decoupled from the Surya stack)
# --------------------------------------------------------------------------- #
def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def default_ocr_strips():
    """The default strip reader for --auto-unscramble-pages: tesseract. Raises a
    clear error if it isn't installed (the caller can inject its own reader —
    e.g. a Surya-backed one — instead)."""
    if not tesseract_available():
        raise SystemExit(
            "--auto-unscramble-pages needs tesseract to read page numbers "
            "(brew install tesseract / apt install tesseract-ocr), or inject a "
            "reader. It's a CPU-only printed-digit OCR, separate from Surya.")
    return tesseract_ocr_strips


def tesseract_ocr_strips(strip_images, strips_dir):
    """OCR the cropped page-number strips with tesseract, in parallel. Returns
    ``{filename: text}`` — the injectable ``ocr_strips`` the descrambler expects.
    Reading a printed page number is tesseract's sweet spot and needs no GPU."""
    def _one(s):
        s = Path(s)
        r = subprocess.run(["tesseract", str(s), "stdout", "--psm", "6"],
                           capture_output=True, text=True)
        return s.name, " ".join(r.stdout.split())
    strip_images = [Path(s) for s in strip_images]
    workers = max(1, min(8, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(_one, strip_images))


# --------------------------------------------------------------------------- #
# Orchestration (OCR is injected, so the reorder logic above stays pure/tested)
# --------------------------------------------------------------------------- #
def _crop_strips(images, strips_dir: Path, strip_frac: float, side: str, force: bool):
    """Crop the top (``side='top'``) or bottom (``'bottom'``) ``strip_frac`` of each
    page — wherever the page number sits — into ``strips_dir`` under the same
    filename; incremental. Returns strips_dir."""
    from PIL import Image
    strips_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        img = Path(img)
        out = strips_dir / img.name
        if (not force and out.exists()
                and out.stat().st_mtime >= img.stat().st_mtime):
            continue
        with Image.open(img) as im:
            w, h = im.size
            f = max(1, int(h * strip_frac))
            box = (0, 0, w, f) if side == "top" else (0, h - f, w, h)
            im.crop(box).save(out)
    return strips_dir


def _labels_for_side(images, image_dir, side, ocr_strips, strip_frac, force):
    """OCR one side's strips and parse each into a page label. Returns
    ``{filename: label|None}``."""
    import surya_backend as sb
    sd = _crop_strips(images, image_dir / f"_headstrips_{side}", strip_frac,
                      side, force)
    text = ocr_strips(sb.list_images(sd), sd)
    return {Path(i).name: parse_running_head(text.get(Path(i).name, ""))
            for i in images}


def detect_side(images, image_dir, *, ocr_strips, strip_frac, force, log,
                n_sample=16):
    """Sample ~``n_sample`` evenly-spaced pages, OCR BOTH the header and footer
    strip, and return whichever side ('top'/'bottom') carries the page numbers
    (per your hint: find where the numbers are, then trust that layout)."""
    k = min(n_sample, len(images))
    idx = sorted({round(i * (len(images) - 1) / max(1, k - 1)) for i in range(k)})
    sample = [images[i] for i in idx]
    top = _labels_for_side(sample, image_dir, "top", ocr_strips, strip_frac, force)
    bot = _labels_for_side(sample, image_dir, "bottom", ocr_strips, strip_frac, force)
    nt = sum(1 for v in top.values() if v is not None)
    nb = sum(1 for v in bot.values() if v is not None)
    side = "top" if nt >= nb else "bottom"
    log(f"auto-unscramble: page numbers are in the "
        f"{'header (top)' if side == 'top' else 'footer (bottom)'} "
        f"— sampled {len(sample)} pages: header {nt}, footer {nb}.")
    return side


def unscramble(image_dir, images, *, ocr_strips, side="auto", force=False,
               strip_frac=0.12, log=None):
    """Reorder ``images`` into true reading order from their printed page numbers.

    ``side`` is 'top'/'bottom' or 'auto' (sample to find header vs footer, then use
    it). ``ocr_strips(strip_images, strips_dir)`` is injected — it OCRs the cropped
    number strips and returns ``{filename: recognized_text}`` (ocr_one passes a
    Surya-backed closure; tests pass a fake). Pages with no number on the chosen
    side (chapter openers, plates) fall back to the OTHER side, then are anchored to
    a neighbor. Materializes a ``_reordered/`` dir (files renamed ``NNNN.ext`` in
    reading order), moves dropped duplicates to ``_reordered/_dropped/``, writes a
    ``manifest.json``, and returns ``(reordered_dir, reordered_images)``. Verifies
    the result is monotonic and logs any violation. If NO page numbers are read at
    all, it leaves the folder untouched (a bad OCR pass can't scramble a good one)."""
    import surya_backend as sb
    log = log or (lambda _m: None)
    image_dir = Path(image_dir)

    if side == "auto":
        side = detect_side(images, image_dir, ocr_strips=ocr_strips,
                           strip_frac=strip_frac, force=force, log=log)
    labels = _labels_for_side(images, image_dir, side, ocr_strips, strip_frac, force)

    # Per-page fallback to the other side (a chapter opener numbered at the foot
    # while the body numbers the head, or vice versa).
    other = "bottom" if side == "top" else "top"
    missing = [i for i in images if labels[Path(i).name] is None]
    if missing:
        alt = _labels_for_side(missing, image_dir, other, ocr_strips, strip_frac,
                               force)
        for nm, lab in alt.items():
            if lab is not None:
                labels[nm] = lab
    records = [(Path(i).name, labels[Path(i).name]) for i in images]

    n_num = sum(1 for _, lab in records if lab is not None)
    if n_num == 0:
        log("auto-unscramble: no page numbers read (header or footer) — "
            "leaving order unchanged.")
        return image_dir, images

    # Neighbor-consistency: a number far from ALL its file-order neighbors is a
    # misread (e.g. '47'->'7'). Treat it as unnumbered so it can't mis-sort — it
    # then anchors by scan position and may be recovered by gap-fill inference.
    raw_label_by = dict(records)
    suspects = flag_suspects(records)
    if suspects:
        log(f"auto-unscramble: {len(suspects)} page number(s) look like misreads "
            "(isolated from their neighbors) — treated as unnumbered so they can't "
            "mis-sort; flagged suspect=true in the manifest.")
        records = [(n, None if n in suspects else l) for n, l in records]

    order, drops = plan_descramble(records)

    label_by = dict(records)
    inferred = infer_missing(order, label_by)      # concrete numbers for openers
    ext_by = {Path(i).name: Path(i).suffix for i in images}
    src_by = {Path(i).name: Path(i) for i in images}
    width = max(4, len(str(len(order))))

    out_dir = image_dir / "_reordered"
    if force:
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, name in enumerate(order, 1):
        if label_by[name] is not None:
            page, is_inf = _fmt(label_by[name]), False
        elif name in inferred:
            page, is_inf = _fmt(inferred[name]), True
        else:
            page, is_inf = None, False
        entry = {"seq": i, "file": f"{i:0{width}d}{ext_by[name]}",
                 "from": name, "page": page, "inferred": is_inf}
        if name in suspects:                       # OCR looked wrong; not trusted
            entry["suspect"] = True
            entry["read"] = _fmt(raw_label_by[name]) if raw_label_by[name] else None
        manifest.append(entry)
        shutil.copy2(src_by[name], out_dir / f"{i:0{width}d}{ext_by[name]}")
    n_inf = sum(1 for m in manifest if m["inferred"])
    if n_inf:
        log(f"auto-unscramble: inferred a page number for {n_inf} unnumbered "
            "page(s) (openers) by gap-filling — see manifest.json (inferred=true).")
    if drops:
        drop_dir = out_dir / "_dropped"
        drop_dir.mkdir(exist_ok=True)
        for name, reason in drops:
            shutil.copy2(src_by[name], drop_dir / name)
        log(f"auto-unscramble: dropped {len(drops)} duplicate/blank page(s) "
            f"-> {drop_dir.name}/")
        for name, reason in drops[:8]:
            log(f"    drop {name}: {reason}")

    (out_dir / "manifest.json").write_text(json.dumps(
        {"count": len(order), "dropped": len(drops), "pages": manifest}, indent=2))

    violations = verify_increasing(
        [label_by[n] or inferred.get(n) for n in order])   # incl. inferred numbers
    if violations:
        log(f"auto-unscramble: WARNING — {len(violations)} order violation(s) "
            "after reorder (some page numbers may be misread):")
        for v in violations[:8]:
            log(f"    {v}")
    else:
        log(f"auto-unscramble: {len(order)} page(s) in reading order (verified "
            "monotonic).")

    return out_dir, sb.list_images(out_dir)

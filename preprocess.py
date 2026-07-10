#!/usr/bin/env python3
"""Image PRE-processing for the OCR pipeline — rotate, swap page pairs, zero-pad
page numbers — as a shared utility AND a standalone preview CLI.

This is the ONE place the jpg-preprocessing steps live. Both run_ocr.py and
ocr_one.py wire up their flags with ``add_preprocess_args`` and run the steps with
``preprocess_images``; running this file directly previews the same steps on a
folder without OCR, so you can eyeball the corrected images first.

A ``--pages START:END`` window is the DEFAULT range ``--rotate`` and ``--swap`` act
on (write ``START:end`` for "page START through the last"); pages outside it pass
through untouched. Either step can OVERRIDE that default with its own range —
``--swap 9:end`` / ``--swap 9:20``, or ``--rotate 180:CW@9:20`` — affecting only
that step; the other still uses ``--pages``. A range ALWAYS has a colon, so it can
never be confused with a folder name.

STEPS (each a no-op when its flag is unset):
    (padding)               zero-pad ALL page numbers to a uniform width (the digit
                            count of the largest page number) — ON BY DEFAULT, ALWAYS
                            runs FIRST. Turn off with --no-pad-page-numbers.
    --pages S:E             default page window for --rotate/--swap (E may be 'end')
    --rotate D:CW/CCW[@S:E]  rotate (in @-range, else --pages, else all) D°
    --swap [S:E]            swap adjacent page pairs (in its range, else --pages,
                            else the whole book) — the fix for two-up / upside-down
                            scans whose facing pairs were captured reversed

This module does pure image ops — it never invokes the Surya OCR backend.

Padding always runs first; --rotate and --swap then run in the ORDER THEY APPEAR
ON THE COMMAND LINE (so '--swap --rotate …' swaps then rotates, and '--rotate …
--swap' rotates then swaps). Each step is non-destructive: corrected copies go to a
sibling ``_renumbered/`` -> then the two step dirs (``_rotated/`` / ``_swapped/``)
in the chosen order, under the same filenames except where a step deliberately
renames (padding) or moves content between a pair's two names (swap). Page
numbering keys off the filename's trailing integer, so it is preserved throughout.
All passes are incremental, so a preview run is not wasted — a later
``run_ocr.py``/``ocr_one.py`` with the same flags reuses these dirs. ``--force``
wipes the step subdirs and rebuilds from scratch (use it when you CHANGE a range —
the incremental check keys off source mtime, not the flags). A marker file
(``.preprocess_final.json``) in the source dir records where the final images
landed, so a consumer can pick them up via ``final_dir(source)``.

Library API:
    add_preprocess_args(parser)     # register --pad-page-numbers/--pages/--rotate/--swap
    parse(argv) -> namespace | None # parse forwarded '--preprocess' tokens (validates)
    parse_pages(spec)               # 'S:E'/'S:end' -> (start, end|None) | None
    validate(args)                  # raise SystemExit on a bad spec (call after parse)
    has_any(args) -> bool           # was any preprocessing step requested?
    step_order(args) -> list[str]   # ['rotate','swap'] subset in command-line order
    label(args) -> str              # compact " pad pages-… rotate-… swap" banner suffix
    preprocess_images(image_dir, images, *, pad, pages, rotate, swap, order,
                      force, log, verbose) -> (final_dir, final_images)
    final_dir(source)               # where a prior run's output landed (per marker)
    clean_step_dirs(source)         # wipe the step subdirs + marker (what --force does)

run_ocr.py / ocr_one.py expose a single ``--preprocess ARGS…`` flag: everything
after it is forwarded here verbatim (the folder isn't — the caller already has it).

Preview CLI (a range always has a colon — 'S:E'/'S:end' — so --swap never mistakes
a folder path or a numeric folder name for its range):
    python3 preprocess.py "~/…/Great hum" --rotate 180:CW@2:end
    python3 preprocess.py "~/…/Great hum" --pages 9:end --rotate 180:CW --swap
    python3 preprocess.py --swap 9:20 --limit 20 "~/…/Great hum"
    python3 preprocess.py --swap "~/…/Great hum"
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# The sibling dirs a step materializes into (nested in this order). A marker file
# in the SOURCE dir records where the final images ended up, so any consumer can
# find them without knowing which steps ran.
_STEP_DIRS = ("_renumbered", "_rotated", "_swapped", "_reordered",
              "_headstrips_top", "_headstrips_bottom")
_MARKER = ".preprocess_final.json"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import surya_backend as sb          # noqa: E402  (list_images, same as the pipeline)
import rotate_images                # noqa: E402
import swap_pages                   # noqa: E402
import pad_pages                    # noqa: E402
import descramble                   # noqa: E402  (--auto-unscramble-pages; tesseract)


# --------------------------------------------------------------------------- #
# Page window
# --------------------------------------------------------------------------- #
def parse_pages(spec):
    """``'START:END'`` -> ``(start, end)`` with ``1 <= start <= end``; ``'START:end'``
    -> ``(start, None)`` (through the last page). The colon is REQUIRED — a bare
    number is NOT a range (so a folder named like a number is never mistaken for
    one). Returns ``None`` for a falsy spec. Raises ``SystemExit`` if malformed.
    Accepts an already-parsed ``(start, end)`` tuple unchanged."""
    if not spec:
        return None
    if isinstance(spec, tuple):
        return spec
    m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+|end)\s*", spec, re.I)
    if not m:
        raise SystemExit(
            "page range must be START:END or START:end (e.g. 9:200 or 9:end); "
            f"got {spec!r}")
    start = int(m.group(1))
    end = None if m.group(2).lower() == "end" else int(m.group(2))   # end = last page
    if start < 1:
        raise SystemExit(f"page range START must be >= 1; got {start}")
    if end is not None and end < start:
        raise SystemExit(f"page range END ({end}) must be >= START ({start})")
    return start, end


def _fmt_window(win) -> str:
    if not win:
        return "all pages"
    start, end = win
    return f"pages {start}..{end if end is not None else 'end'}"


def split_rotate(spec):
    """Split a --rotate value into its ``D:CW/CCW`` core and an optional per-command
    page range given as an ``@S:E`` suffix: ``'180:CW@9:20'`` -> ``('180:CW',
    '9:20')``; ``'90:CCW'`` -> ``('90:CCW', None)``. ``None`` spec -> ``(None,
    None)``. Does not validate — callers validate the two parts."""
    if not spec:
        return None, None
    core, sep, rng = spec.partition("@")
    return core.strip(), (rng.strip() or None if sep else None)


# --------------------------------------------------------------------------- #
# Argument plumbing — shared so run_ocr / ocr_one / this CLI stay in lock-step
# --------------------------------------------------------------------------- #
class _StepAction(argparse.Action):
    """Store a step flag's value AND record its position among the ordered steps
    (--rotate / --swap), so they execute in the order given on the command line.
    argparse invokes actions left-to-right, so the append order IS the CLI order."""
    def __call__(self, parser, namespace, values, option_string=None):
        seen = list(getattr(namespace, "preproc_order", None) or ())
        if self.dest not in seen:
            seen.append(self.dest)
        setattr(namespace, "preproc_order", seen)
        if self.nargs == 0:                                # store_true
            values = self.const
        elif values in (None, "") and self.const is not None:  # '?' with no/empty arg
            values = self.const                            # -> bare (e.g. --swap)
        setattr(namespace, self.dest, values)


_RANGE_RE = re.compile(r"\d+:(?:\d+|end)", re.I)    # 'S:E' or 'S:end' (a page range)


def normalize_argv(argv=None):
    """Return ``argv`` with ``--swap`` normalized so it only absorbs the next token
    when that token actually is a page range (``9:20`` / ``9:end`` — a range always
    has a colon, so it can't be confused with a folder name like ``9``).

    argparse's ``nargs='?'`` would otherwise greedily swallow ANY following token
    (e.g. a folder path) as ``--swap``'s value. We rewrite ``--swap <range>`` to
    ``--swap=<range>`` and ``--swap <non-range>`` to ``--swap=`` (an empty value =
    bare ``--swap``, leaving the token for the positional). Only ``--swap`` is
    touched; every other token passes through unchanged."""
    argv = list(sys.argv[1:] if argv is None else argv)
    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--swap":
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is not None and _RANGE_RE.fullmatch(nxt):
                out.append(f"--swap={nxt}")           # own range
                i += 2
                continue
            if nxt is not None and not nxt.startswith("-"):
                out.append("--swap=")                 # bare; don't eat the next token
                i += 1
                continue
            out.append(tok)                           # bare; next is an option / end
        else:
            out.append(tok)
        i += 1
    return out


def step_order(args) -> list:
    """The ordered step names (subset of ['rotate', 'swap']) in command-line
    order. Falls back to the default order for a hand-built namespace that never
    went through the ordered action (e.g. a test)."""
    order = getattr(args, "preproc_order", None)
    if order:
        return [s for s in order if s in ("rotate", "swap")]
    return [s for s in ("rotate", "swap")
            if (s == "rotate" and args.rotate) or (s == "swap" and args.swap)]


def add_preprocess_args(parser):
    """Register --no-pad-page-numbers / --pages / --rotate / --swap on ``parser``
    (an ArgumentParser or an argument group). Returns ``parser``. Padding is ON by
    default and always runs first; --rotate/--swap run in command-line order."""
    parser.add_argument("--pages", default=None, metavar="START:END",
                        help="DEFAULT page-number window for --rotate and --swap "
                             "(1-based, inclusive; write START:end for through the "
                             "last page). The colon is required — a bare number is "
                             "not a range. Page numbers are the trailing integer of "
                             "each filename; pages outside the window pass through "
                             "unchanged. Each step can override this with its own "
                             "range (see --swap / --rotate). No effect on "
                             "page-number padding (always covers all pages).")
    parser.add_argument("--rotate", default=None, metavar="D:CW/CCW[@S:E]",
                        action=_StepAction,
                        help="Rotate page images by D degrees before OCR (D<=180; "
                             "CW=clockwise, CCW=counter-clockwise), e.g. '180:CW' for "
                             "upside-down scans or '90:CCW' for a sideways one. Surya "
                             "expects upright text — a rotated scan OCRs to garbage "
                             "(180 can slip past as low-quality text). Add an '@S:E' "
                             "suffix (e.g. '180:CW@2:end') to rotate only that page "
                             "range, overriding --pages for rotate; otherwise --pages "
                             "(else all pages) applies. Copies go to _rotated/ and "
                             "drive OCR, detection AND the output PDF.")
    parser.add_argument("--swap", nargs="?", action=_StepAction, const=True,
                        default=None, metavar="S:E",
                        help="Swap adjacent page-image pairs (S<->S+1, S+2<->S+3, …; "
                             "an odd tail page is left alone), for scans whose facing "
                             "pairs were captured out of order. Give a range 'S:E' "
                             "(e.g. '--swap 9:20' or '--swap 9:end') to override "
                             "--pages for swap; bare '--swap' uses --pages (else the "
                             "whole book). A range needs the colon, so it's never "
                             "confused with a folder name. Swapped copies go to "
                             "_swapped/.")
    parser.add_argument("--no-pad-page-numbers", dest="pad_page_numbers",
                        action="store_false", default=True,
                        help="Turn OFF the default page-number zero-padding. By "
                             "DEFAULT (on), the trailing page number in EVERY image "
                             "filename is zero-padded to a uniform width — the digit "
                             "count of the LARGEST page number (a book ending at 976 "
                             "pads <prefix>1.jpg -> <prefix>001.jpg; already-wide "
                             "numbers are unchanged). It ALWAYS runs FIRST, so every "
                             "stage (rotate/swap/add-pages/OCR) sees one canonical "
                             "padded numbering regardless of entry point; copies go to "
                             "_renumbered/. Pass this flag only if you must keep the "
                             "original filenames (e.g. widths are already uniform).")
    # NOTE: --auto-unscramble-pages was removed — it read printed page numbers to
    # reorder pages but OCR misreads mis-sorted the front matter AND dropped real
    # pages as false duplicates (data loss). The reorder engine lives on in
    # descramble.py (imported below, callable programmatically) but is no longer a
    # CLI option; for two-up / upside-down scans use the deterministic --swap.
    return parser


def validate(args):
    """Validate the preprocessing specs up front (raise SystemExit on a bad one),
    so a typo fails before any images are written."""
    parse_pages(args.pages)
    core, rng = split_rotate(args.rotate)
    if core is not None:
        rotate_images.parse_rotate(core)      # the D:CW/CCW part
    parse_pages(rng)                          # rotate's own @-range, if any
    if isinstance(args.swap, str):
        parse_pages(args.swap)                # swap's own range, if any


def has_any(args) -> bool:
    """True if any preprocessing step was requested (``--pages`` alone is a no-op)."""
    return bool(args.pad_page_numbers or args.rotate or args.swap)


def split_forward(argv=None):
    """Split a run_ocr / ocr_one command line into ``(own_args, preprocess_tail)`` at
    the first ``--`` or ``--preprocess`` marker: everything after it is forwarded
    verbatim to the image preprocessor (the folder is NOT — the caller has it).
    Tolerates ``-- --preprocess …`` (both). Returns ``(argv, [])`` if no marker."""
    argv = list(sys.argv[1:] if argv is None else argv)
    idxs = [argv.index(m) for m in ("--", "--preprocess") if m in argv]
    if not idxs:
        return argv, []
    i = min(idxs)
    tail = argv[i + 1:]
    if tail and tail[0] == "--preprocess":        # tolerate '-- --preprocess …'
        tail = tail[1:]
    return argv[:i], tail


def parse(argv):
    """Parse a FORWARDED preprocess token list (the args after ``--`` / ``--preprocess``
    on a run_ocr / ocr_one command line — step flags only, NO folder positional) into
    a validated namespace. Returns ``None`` for an empty list. ``-h``/``--help`` in the
    list prints the preprocess help and exits."""
    if not argv:
        return None
    ap = argparse.ArgumentParser(
        prog="… --", add_help=True,
        description="Image preprocessing steps, forwarded verbatim after "
                    "-- / --preprocess (the folder is supplied by the calling tool).")
    add_preprocess_args(ap)
    # --force is really a run_ocr/ocr_one flag, but users naturally lump it in with
    # the preprocess args after `--`; accept it here too and let the caller hoist it.
    ap.add_argument("--force", action="store_true",
                    help="Wipe cached preprocess dirs + rebuild and re-OCR "
                         "(same as putting --force BEFORE the '--').")
    ns = ap.parse_args(normalize_argv(argv))
    validate(ns)
    return ns


def _step_label(step, args):
    if step == "rotate":
        return f"rotate-{args.rotate}"
    return "swap" if args.swap is True else f"swap-{args.swap}"


def label(args) -> str:
    """Compact banner suffix, e.g. ' pages-9:20 rotate-180:CW swap' (empty if nothing
    requested); parts are in execution order. Padding is on by default, so only its
    absence ('no-pad') is worth noting."""
    parts = []
    if not args.pad_page_numbers:
        parts.append("no-pad")
    if args.pages:
        parts.append(f"pages-{args.pages}")
    parts += [_step_label(s, args) for s in step_order(args)]
    return (" " + " ".join(parts)) if parts else ""


# --------------------------------------------------------------------------- #
# The steps
# --------------------------------------------------------------------------- #
def clean_step_dirs(source):
    """Remove every step subdir (_renumbered/_rotated/_swapped, with their nested
    chains) under ``source``, plus the final-dir marker. Used by --force to start
    from a clean slate."""
    source = Path(source)
    for name in _STEP_DIRS:
        shutil.rmtree(source / name, ignore_errors=True)
    (source / _MARKER).unlink(missing_ok=True)


def final_dir(source):
    """Where a prior preprocess run left its output (the dir a consumer should read
    images from), per the marker in ``source``. Returns ``source`` itself if there
    is no marker (nothing was preprocessed, or it was a no-op)."""
    source = Path(source)
    try:
        return Path(json.loads((source / _MARKER).read_text())["final_dir"])
    except (OSError, ValueError, KeyError):
        return source


def preprocess_images(image_dir, images, *, pad=False, pages=None, rotate=None,
                      swap=None, auto_unscramble=False, ocr_strips=None,
                      order=None, force=False, log=None, verbose=False):
    """Run page-number padding, then rotate/swap, then auto-unscramble over
    ``images`` (from ``image_dir``), materializing each requested step into a
    sibling dir and repointing. Padding ALWAYS runs FIRST and covers every page
    (width = digits of the largest page number); rotate and swap then run in
    ``order`` (a list like ``['swap', 'rotate']`` — command-line order), defaulting
    to ``rotate`` then ``swap``. ``pages`` (a ``'S:E'`` spec) is the DEFAULT window
    for rotate and swap; either can override it with its own range — ``rotate`` via
    a ``'@S:E'`` suffix on its value, ``swap`` via a range string instead of
    ``True``. ``auto_unscramble`` runs LAST (needs upright pages): it reads printed
    page numbers and reorders the whole book (see descramble.unscramble);
    ``ocr_strips`` is the injected reader (default: tesseract). Returns
    ``(final_dir, final_images)`` — what OCR should consume.

    ``force`` first wipes the step subdirs (a clean rebuild) rather than rewriting
    in place. A marker file in the SOURCE dir records the final dir (see final_dir).

    ``log`` is an optional ``callable(str)`` for progress lines (the caller decides
    prefix/timestamp); ``verbose`` additionally logs per-page swap/rename detail."""
    log = log or (lambda _m: None)
    source = Path(image_dir)
    if force:
        clean_step_dirs(source)            # drop stale subdirs -> full clean rebuild
    image_dir = source
    default_win = parse_pages(pages)
    rot_core, rot_range = split_rotate(rotate)
    rot_win = parse_pages(rot_range) if rot_range else default_win
    swap_range = swap if isinstance(swap, str) else None
    swap_win = parse_pages(swap_range) if swap_range else default_win

    if pad:
        # Width spans the WHOLE book (the largest page number in the folder), not
        # just the pages being processed — so a --limit spot-check still pads to the
        # book's true width. Padding runs first, so image_dir is the source folder.
        digits = pad_pages.auto_width(sb.list_images(image_dir))
        pad_dir = image_dir / "_renumbered"
        renamed = [(p.name, pad_pages.padded_name(p, digits)) for p in images
                   if pad_pages.padded_name(p, digits) != p.name]
        log(f"zero-pad page numbers to {digits} digit(s) (widest page) "
            f"-> {pad_dir.name}/  ({len(renamed)} of {len(images)} filename(s) "
            f"change)")
        pad_pages.make_padded(images, pad_dir, digits, force=force)
        image_dir = pad_dir
        images = sb.list_images(pad_dir)
        if verbose:
            for old, new in renamed[:3]:
                log(f"    {old}  ->  {new}")
            if len(renamed) > 3:
                log(f"    … and {len(renamed) - 3} more")

    def _rotate(image_dir, images):
        rot = rotate_images.parse_rotate(rot_core)
        if not rot:
            return image_dir, images
        degrees, cw = rot
        rot_dir = image_dir / "_rotated"
        log(f"rotate {degrees}° {'CW' if cw else 'CCW'} on {_fmt_window(rot_win)} "
            f"-> {rot_dir.name}/")
        rotate_images.make_rotated(images, rot_dir, degrees, cw, force=force,
                                   pages=rot_win)
        return rot_dir, sb.list_images(rot_dir)

    def _swap(image_dir, images):
        if not swap:
            return image_dir, images
        start, end = swap_win if swap_win else (1, None)  # no window -> whole book
        swap_dir = image_dir / "_swapped"
        plan = swap_pages.swap_plan(images, start, end)
        changed = sorted(((p.name, plan[p].name) for p in images if plan[p] != p),
                         key=lambda t: sb.natural_key(t[0]))
        log(f"swap adjacent pairs from {start} through "
            f"{end if end is not None else 'the last page'} -> {swap_dir.name}/  "
            f"({len(changed)} page file(s) change content)")
        swap_pages.make_swapped(images, swap_dir, start, end, force=force)
        if verbose:
            for name, src_name in changed:
                log(f"    {name}  <- content of {src_name}")
        return swap_dir, sb.list_images(swap_dir)

    steps = {"rotate": _rotate, "swap": _swap}
    for name in (order or ("rotate", "swap")):            # CLI order; pad already ran
        image_dir, images = steps[name](image_dir, images)

    if auto_unscramble:                                   # last: needs upright pages
        strip_ocr = ocr_strips or descramble.default_ocr_strips()
        image_dir, images = descramble.unscramble(
            image_dir, images, ocr_strips=strip_ocr, force=force, log=log)

    # Record where the output landed (or clear a stale marker on a no-op run), so a
    # consumer can find the final dir via final_dir(source) without guessing subdirs.
    if image_dir != source:
        (source / _MARKER).write_text(json.dumps(
            {"final_dir": str(image_dir), "source": str(source)}, indent=2))
    else:
        (source / _MARKER).unlink(missing_ok=True)
    return image_dir, images


# --------------------------------------------------------------------------- #
# Standalone preview CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("folders", nargs="+", type=Path,
                    help="Image folder(s) to preprocess (PDFs not supported — this "
                         "operates on already-rasterized page images).")
    add_preprocess_args(ap)
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="Only preprocess the first N pages (by natural filename "
                         "order) — handy for a quick spot-check.")
    ap.add_argument("--force", action="store_true",
                    help="Wipe the step subdirs (_renumbered/_rotated/_swapped) and "
                         "rebuild from scratch. Use this whenever you CHANGE a "
                         "range/rotation — the incremental cache keys off source "
                         "mtime, not the flags, so a plain re-run would keep stale "
                         "copies.")
    args = ap.parse_args(normalize_argv())

    validate(args)
    if not has_any(args):                       # only when --no-pad-page-numbers + no step
        sys.exit("nothing to do: padding is off (--no-pad-page-numbers) and no "
                 "--rotate/--swap given (--pages only scopes --rotate/--swap).")

    for folder in args.folders:
        folder = folder.expanduser()
        if not folder.is_dir():
            print(f"SKIP {folder}: not a folder", flush=True)
            continue
        images = sb.list_images(folder)
        if args.limit:
            images = images[:args.limit]
        if not images:
            print(f"SKIP {folder}: no page images", flush=True)
            continue
        print(f"{folder.name}: {len(images)} page image(s)"
              + (f" (limited to first {args.limit})" if args.limit else ""),
              flush=True)
        final, _ = preprocess_images(
            folder, images, pad=args.pad_page_numbers, pages=args.pages,
            rotate=args.rotate, swap=args.swap, order=step_order(args),
            force=args.force, log=lambda m: print(f"  {m}", flush=True),
            verbose=True)
        print(f"  -> preview these images in: {final}\n", flush=True)


if __name__ == "__main__":
    sys.exit(main())

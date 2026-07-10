"""Per-book OCR layout directives (two-up page-split, column-strip re-OCR).

Some scanned books need layout handling that is wrong for the rest of the
corpus — splitting two-up sheets into two pages, or OCR'ing a two-column page
one column at a time so the text doesn't come out scrambled. Auto-detecting
this per page is mistake-prone (it mis-splits the odd landscape plate / index
page), so instead we name the few books that need it in a standalone manifest,
decoupled from the run command. Both ``ocr_one`` (``--layout-overrides``) and
``run_ocr`` (edition-aware) read it.

Format — one book per line, ``<key> :: <directives>`` (``::`` or a TAB
separates key from directives; the key may contain spaces)::

    edition:129  :: two-up           # catalogue edition id
    edition:314  :: two-up
    edition:655  :: columns=2         # two-column body -> column-strip re-OCR
    Mapping the Tibetan World :: columns=2   # or match a path/stem substring

  key        ``edition:ID`` (resolved by run_ocr via the DB) OR a filename
             stem / substring (matched case-insensitively for path inputs).
  directive  ``two-up[=on|auto|off]`` (default on) — landscape page-split;
             ``columns=N`` (N>=2) — column-strip re-OCR into N columns;
             ``drop-pages=even|odd`` — skip OCR of that 1-based page-index parity
             (for bilingual books with native script on the verso pages — they
             loop on script Surya can't read and get dropped anyway; kept in the
             output as image-only). VERIFY the parity per book — skipping the
             wrong one deletes real content.
  '#' starts a comment; blank lines ignored; directives are whitespace/comma
  separated.
"""

from __future__ import annotations

from pathlib import Path

_SEPS = ("\t", " :: ", "::")


def _parse_directives(s: str) -> dict:
    """``"two-up columns=2"`` / ``"two-up=on, columns=2"`` -> normalized dict
    ``{"two_up": "on"|"auto"|"off", "columns": int}`` (only the keys present)."""
    out: dict = {}
    for tok in s.replace(",", " ").split():
        key, _, val = tok.partition("=")
        key = key.strip().lower().replace("-", "_")
        val = val.strip().lower()
        if key == "two_up":
            mode = val or "on"
            if mode not in ("on", "auto", "off"):
                raise ValueError(f"two-up must be on/auto/off, got {val!r}")
            out["two_up"] = mode
        elif key in ("columns", "two_column", "two_columns", "column"):
            if key.startswith("two_column"):
                out["columns"] = 2
            else:
                n = int(val)
                if n < 1:
                    raise ValueError(f"columns must be >=1, got {n}")
                out["columns"] = n
        elif key in ("drop_pages", "skip_pages"):
            mode = val.split(":", 1)[0].strip().lower()    # 'even'|'odd'|'non-latin'[:LO-HI]
            if mode not in ("even", "odd", "non-latin"):
                raise ValueError(
                    f"drop-pages must be even/odd/non-latin (optional :LO-HI range), "
                    f"got {val!r}")
            out["drop_pages"] = val
        else:
            raise ValueError(f"unknown layout directive {tok!r}")
    return out


def load(path) -> list:
    """Parse a manifest into ``[(key, directives_dict), ...]`` (order preserved).
    Raises ``SystemExit`` with the line number on a malformed line."""
    entries = []
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, rest = line, ""
        for sep in _SEPS:
            if sep in line:
                key, _, rest = line.partition(sep)
                break
        key = key.strip()
        try:
            directives = _parse_directives(rest)
        except ValueError as e:
            raise SystemExit(f"{path}:{n}: {e}")
        if not directives:
            raise SystemExit(f"{path}:{n}: no directives for {key!r}")
        entries.append((key, directives))
    return entries


def _matches(key: str, *, edition=None, stem=None) -> bool:
    klow = key.lower()
    if klow.startswith("edition:"):
        try:
            return edition is not None and int(klow.split(":", 1)[1]) == int(edition)
        except ValueError:
            return False
    # bare key: match a path/stem exactly or as a case-insensitive substring
    return stem is not None and klow in stem.lower()


def lookup(entries, *, edition=None, stem=None) -> dict:
    """Merged directives for a book identified by ``edition`` id and/or ``stem``
    (later matching entries win). Empty dict if nothing matches."""
    found: dict = {}
    for key, directives in entries or []:
        if _matches(key, edition=edition, stem=stem):
            found.update(directives)
    return found

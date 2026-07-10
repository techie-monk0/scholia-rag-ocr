"""Per-page script gate for bilingual books (e.g. Tibetan/Chinese on verso pages,
English on recto).

Surya can't read Tibetan/CJK; OCR'ing those pages loops to the token cap and the
text is dropped anyway (e3 burned ~2h this way). A blanket even/odd parity skip is
unsafe — front/back matter on the verso side is often English. This gate decides
per page from the PDF's OWN embedded text: skip OCR only on pages with no
meaningful Latin/English content (predominantly non-Latin script, or effectively
empty of Latin). English pages — even with font-mojibake — keep their Latin
letters and are OCR'd. Used via the `non-latin` drop-pages mode.

`page_skip_nonlatin(text)` -> True means "skip OCR on this page".
"""

from __future__ import annotations


def _is_latin_letter(c: str) -> bool:
    o = ord(c)
    return (0x41 <= o <= 0x5A or 0x61 <= o <= 0x7A      # ASCII A-Z a-z
            or 0xC0 <= o <= 0x24F                        # Latin-1 + Ext-A/B (IAST: ā ī ś ...)
            or 0x1E00 <= o <= 0x1EFF)                    # Latin Ext Additional (ṛ ṣ ṭ ḥ ...)


def _is_nonlatin_letter(c: str) -> bool:
    o = ord(c)
    return (0x0F00 <= o <= 0x0FFF        # Tibetan
            or 0x3400 <= o <= 0x9FFF     # CJK ext-A + unified
            or 0x3040 <= o <= 0x30FF     # Hiragana / Katakana
            or 0xAC00 <= o <= 0xD7A3     # Hangul
            or 0x0E00 <= o <= 0x0E7F)    # Thai


def latin_nonlatin_counts(text: str) -> tuple[int, int]:
    latin = nonlatin = 0
    for c in text or "":
        if _is_latin_letter(c):
            latin += 1
        elif _is_nonlatin_letter(c):
            nonlatin += 1
    return latin, nonlatin


def parse_drop_pages(spec):
    """Parse a drop-pages spec into (mode, (lo, hi) | None).

    mode is 'even' | 'odd' | 'non-latin' (or None); an optional 1-based page range
    is appended as ':LO-HI' and limits where the rule applies (pages outside it are
    always OCR'd). Examples: 'even', 'odd:10-300', 'non-latin', 'non-latin:5-200'.
    Returns (None, None) for an empty/falsey spec."""
    if not spec:
        return None, None
    mode, _, rng = str(spec).partition(":")
    mode = mode.strip().lower()
    span = None
    if rng.strip():
        lo, _, hi = rng.partition("-")
        span = (int(lo.strip()), int(hi.strip()))
    return (mode or None), span


def page_skip_nonlatin(text: str, *, min_latin: int = 10,
                       min_latin_frac: float = 0.15) -> bool:
    """True if a page should be SKIPPED for OCR — it has no meaningful Latin/English
    content. A page is KEPT (OCR'd) only when it has at least ``min_latin`` Latin
    letters AND Latin is at least ``min_latin_frac`` of its letters (so a lone folio
    number on a Tibetan page doesn't rescue it). Mojibake English still has Latin
    letters, so it's kept."""
    latin, nonlatin = latin_nonlatin_counts(text)
    has_english = latin >= min_latin and latin >= min_latin_frac * (latin + nonlatin)
    return not has_english

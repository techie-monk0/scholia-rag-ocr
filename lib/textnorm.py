#!/usr/bin/env python3
"""Unicode normalization for emitted block text (ARCHITECTURE.md §"Unicode
normalization spec"). Must match the RAG digital-first (Docling) path exactly.

Principle: normalize invisible / typographic NOISE; preserve meaning-bearing
punctuation. In particular we PRESERVE em/en/hyphen as three distinct chars,
keep apostrophes (the Wylie a-chung `'jug`, `ba'i` is a *letter*), keep
diacritics (`śūnyatā`), editorial brackets `[…]`, and ellipsis `…`.
"""

from __future__ import annotations

import re
import unicodedata

# 2. invisible characters to delete outright: soft hyphen + zero-width family
_DELETE_CPS = [0x00AD, 0x200B, 0x200C, 0x200D, 0xFEFF]
_DELETE_MAP = dict.fromkeys(_DELETE_CPS, None)

# 2. Unicode spaces (incl. NBSP, en/em/thin spaces, ideographic) -> plain space
_SPACE_CPS = [0x00A0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005,
              0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x202F, 0x205F, 0x3000]
_SPACE_MAP = {cp: " " for cp in _SPACE_CPS}

# 3. ligatures
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}
_LIG_MAP = {ord(k): v for k, v in _LIGATURES.items()}

# 4. curly/smart quotes -> straight. NOTE: U+2019 (’) maps to the ASCII
# apostrophe ' — it is NOT deleted (a-chung is a letter); only the *shape* is
# normalized. Straight ' and " already pass through unchanged.
_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',          # prime / double-prime
}
_QUOTE_MAP = {ord(k): v for k, v in _QUOTES.items()}

# 6. de-hyphenate line-break splits only: "exam-\nple" -> "example".
# A hyphen at end of a line followed by a lowercase continuation is a soft
# (line-wrap) hyphen; an in-line hyphen ("well-being") never matches because it
# has no newline. Continuation must start lowercase so we don't merge
# "Indo-\nEuropean" (a real compound) wrongly.
_LINEBREAK_HYPHEN = re.compile(r"(\w)-\n[ \t]*([a-zà-ÿ])")

_SPACE_RUN = re.compile(r"[ \t]{2,}")
_EDGE_SPACE = re.compile(r"\n[ \t]+|[ \t]+\n")
_NL_RUN = re.compile(r"\n{3,}")


def normalize_block_text(text: str, *, dehyphenate: bool = True) -> str:
    """Apply the 6-step normalization spec to one block's text.

    ``dehyphenate`` (default True) joins line-break-split words ("exam-\\nple" ->
    "example"). Pass ``dehyphenate=False`` for ``verse`` blocks, where the line
    breaks are semantic and joining them destroys the verse (spec §126)."""
    if not text:
        return text
    # 1. NFC
    text = unicodedata.normalize("NFC", text)
    # 2. delete zero-width / soft hyphen, then Unicode spaces -> plain space
    text = text.translate(_DELETE_MAP).translate(_SPACE_MAP)
    # 3. ligatures
    text = text.translate(_LIG_MAP)
    # 4. quotes (apostrophes preserved, not dropped)
    text = text.translate(_QUOTE_MAP)
    # 5. dashes: PRESERVE — em — / en – / hyphen - stay distinct (no transform)
    # 6. de-hyphenate line-break splits only (must run while \n still present) —
    #    skipped for verse blocks, whose line integrity is meaning-bearing.
    if dehyphenate:
        text = _LINEBREAK_HYPHEN.sub(r"\1\2", text)
    # collapse whitespace runs (after de-hyphenation)
    text = _SPACE_RUN.sub(" ", text)
    text = _EDGE_SPACE.sub("\n", text)
    text = _NL_RUN.sub("\n\n", text)
    return text.strip()

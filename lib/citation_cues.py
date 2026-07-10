#!/usr/bin/env python3
"""Citation-cue detection — SHARED producer-side signals (vendor like docjson).

In Buddhist scholarly prose a quoted verse/passage is bracketed by language: a
lead-in ("As Śāntideva says:") and/or a trailing source ("(MMK 24.18)", "— Nāgārjuna",
"vv. 65–66"). These cues LOCALIZE and ATTRIBUTE the quote — more robustly than
geometry, and they survive even when OCR flattens the verse's line breaks (the one
thing nothing downstream can recover). Attribution is the actual RAG win.

The two structure producers — Surya-OCR (scanned) and the digital-first readers
(PDF/EPUB) — emit the SAME signals into the doc.json contract from THIS one module
(no parallel re-implementation → no drift). The single consumer adjudicates the
final verse/quote/body label using the emitted signals.

Pure, stdlib-only. Field contract (all optional, additive):
  preceded_by_colon : bool      -- previous (reading-order) block ends in ':'
  lead_in_verb      : bool       -- previous block is a citation lead-in (cue word + ':')
  trailing_source   : str|None   -- attribution at the END of this block (-> ref_label)
"""

import re

# Citation lead-in vocabulary: speech/attribution verbs + canonical text words.
_CUE_WORDS = (
    r"say|says|said|state|states|stated|writ|wrote|writes|written|explain|"
    r"explains|teach|teaches|declare|declares|accord|quot|cit|compos|propound|"
    r"as it is said|in the words of|root text|"
    r"s[uū]tra|tantra|[śs][aā]stra|k[aā]rik[aā]|doha|verse|stanza|commentary|"
    r"treatise"
)
_CUE_RE = re.compile(rf"\b(?:{_CUE_WORDS})\b", re.I)
_COLON_END = re.compile(r"[:：]\s*$")

# Trailing attribution at the END of a quote block:
#   (MMK 24.18) | (D4182) | (vv. 65–66) | (Toh 4158) | — Nāgārjuna
_TRAIL_RE = re.compile(
    r"(\([A-Za-zŚśĀ][\w’.\s]*?[\dIVXLC]+[.:][\d.:–-]+\)"          # (MMK 24.18)
    r"|\([^)]*\b(?:vv?|fol|folio|pp?|D|Q|P|Toh|Derge)\b\.?[^)]*\)"  # (vv. 65-66)/(D4182)
    r"|—\s*[A-ZŚĀ][\wāīūṇṭḍṣṅñ’]+(?:\s+[A-ZŚĀ][\wāīūṇṭḍṣṅñ’]+){0,2})"  # — Author
    r"\s*$")


def preceded_by_colon(prev_text: str) -> bool:
    return bool(prev_text and _COLON_END.search(prev_text))


def lead_in_verb(prev_text: str) -> bool:
    """Previous block is a citation lead-in: ends in ':' AND names a cue word."""
    t = prev_text or ""
    return bool(_COLON_END.search(t) and _CUE_RE.search(t))


def trailing_source(text: str):
    """A source attribution at the END of the block, or None."""
    m = _TRAIL_RE.search((text or "").strip())
    return m.group(1).strip() if m else None


def cues(prev_text: str, text: str) -> dict:
    """All citation signals for a block given the PREVIOUS (reading-order) block.
    Only present keys are returned (additive — absent means 'computed, none')."""
    out: dict = {}
    if preceded_by_colon(prev_text):
        out["preceded_by_colon"] = True
    if lead_in_verb(prev_text):
        out["lead_in_verb"] = True
    ts = trailing_source(text)
    if ts:
        out["trailing_source"] = ts
    return out


def is_bracketed(cue: dict) -> bool:
    """Does this block's cues suggest it is a quotation (lead-in and/or source)?"""
    return bool(cue.get("lead_in_verb") or cue.get("trailing_source")
                or cue.get("preceded_by_colon"))

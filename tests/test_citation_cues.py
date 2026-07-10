"""Unit tests for the shared citation-cue detector (lib/citation_cues.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import citation_cues as cc


def test_preceded_by_colon():
    assert cc.preceded_by_colon("As Śāntideva says:")
    assert cc.preceded_by_colon("the text states: ")
    assert not cc.preceded_by_colon("a normal sentence.")


def test_lead_in_verb_requires_colon_and_cue_word():
    assert cc.lead_in_verb("As the Bodhicaryāvatāra says:")
    assert cc.lead_in_verb("It is stated in the Sūtra:")
    assert not cc.lead_in_verb("As he walked into the room:")   # colon but no cue word
    assert not cc.lead_in_verb("Nāgārjuna says so plainly.")    # cue word but no colon


def test_trailing_source():
    assert cc.trailing_source("…has burning hatred for enemies. (MMK 24.18)") == "(MMK 24.18)"
    assert cc.trailing_source("…through abandoning bad objects (vv. 65–66)") == "(vv. 65–66)"
    assert cc.trailing_source("…the perfection of wisdom — Nāgārjuna") == "— Nāgārjuna"
    assert cc.trailing_source("an ordinary sentence ending in prose.") is None
    assert cc.trailing_source("a count of items (3)") is None     # not a citation


def test_cues_combines_and_is_additive():
    c = cc.cues("As Śāntideva says:", "Whatever pride… (BCA 6.1)")
    assert c["preceded_by_colon"] and c["lead_in_verb"]
    assert c["trailing_source"] == "(BCA 6.1)"
    assert cc.is_bracketed(c)
    assert cc.cues("plain prose.", "more plain prose.") == {}     # nothing emitted

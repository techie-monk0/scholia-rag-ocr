"""Phase 1 — unicode normalization spec (ARCHITECTURE.md §Unicode)."""
import unicodedata

from textnorm import normalize_block_text as N


def test_nfc_composes_combining_marks():
    decomposed = "s" + "́"            # s + combining acute
    assert N(decomposed) == unicodedata.normalize("NFC", decomposed)
    # IAST stays composed/verbatim
    assert N("śūnyatā") == "śūnyatā"


def test_removes_soft_hyphen_and_zero_width():
    assert N("soft­hyphen") == "softhyphen"
    assert N("zero​width‌‍﻿join") == "zerowidthjoin"


def test_unicode_spaces_become_plain_space_and_collapse():
    assert N("a b") == "a b"          # NBSP
    assert N("a b") == "a b"          # em space
    assert N("a   b") == "a b"             # collapse run


def test_ligatures_expand():
    assert N("ﬁle") == "file"
    assert N("ﬂour") == "flour"
    assert N("eﬀort") == "effort"
    assert N("baﬄe") == "baffle"
    assert N("ﬃx") == "ffix"


def test_quotes_curly_to_straight():
    assert N("“hi”") == '"hi"'
    assert N("it’s") == "it's"


def test_apostrophe_preserved_not_deleted():
    # Wylie a-chung is a letter, not a quote — straight apostrophe untouched,
    # curly apostrophe normalized to straight (shape only), never dropped.
    assert N("ba'i") == "ba'i"
    assert N("'jug") == "'jug"
    assert N("ba’i") == "ba'i"
    assert "'" in N("rJe ’brom ston")


def test_dashes_preserved_distinct():
    s = "compound-word, range 1–5, aside—here"
    assert N(s) == s                       # —, –, - all survive unchanged
    assert "—" in N("a—b")
    assert "–" in N("a–b")


def test_dehyphenate_linebreak_only():
    assert N("exam-\nple") == "example"
    assert N("contin-\n  uation") == "continuation"
    # inline hyphen never touched
    assert N("well-being") == "well-being"
    # capitalized continuation (proper compound) is NOT merged
    assert "Indo-" in N("Indo-\nEuropean") and "European" in N("Indo-\nEuropean")


def test_editorial_brackets_and_ellipsis_preserved():
    assert N("[sic] and … omitted") == "[sic] and … omitted"
    assert N("the king [Songtsen] said") == "the king [Songtsen] said"


def test_empty_and_idempotent():
    assert N("") == ""
    once = N("ﬁrst­line’s “quote”")
    assert N(once) == once                 # idempotent

"""Per-page script gate for bilingual books (lib/page_lang.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import page_lang as pl


def test_parse_drop_pages():
    assert pl.parse_drop_pages("even") == ("even", None)
    assert pl.parse_drop_pages("odd:10-300") == ("odd", (10, 300))
    assert pl.parse_drop_pages("non-latin") == ("non-latin", None)
    assert pl.parse_drop_pages("non-latin:5-200") == ("non-latin", (5, 200))
    assert pl.parse_drop_pages("") == (None, None)
    assert pl.parse_drop_pages(None) == (None, None)


def test_skip_keeps_english():
    assert pl.page_skip_nonlatin("This is a normal English paragraph, plenty of words.") is False
    # IAST diacritics are Latin Extended -> still English
    assert pl.page_skip_nonlatin("Śāntideva's Bodhicāryāvatāra discusses śūnyatā at length.") is False
    # mostly English with a little Tibetan -> keep
    assert pl.page_skip_nonlatin("This chapter treats emptiness; the term སྟོང་པ་ཉིད་ appears once.") is False


def test_skip_drops_nonlatin():
    assert pl.page_skip_nonlatin("བྱང་ཆུབ་སེམས་དཔའ་སྤྱོད་པ་ལ་འཇུག་པ་" * 3) is True   # Tibetan
    assert pl.page_skip_nonlatin("菩提薩埵般若波羅蜜多時照見五蘊皆空度一切苦厄") is True      # Chinese
    assert pl.page_skip_nonlatin("42  བྱང་ཆུབ་སེམས་དཔའ་" * 2) is True                  # folio no. only
    assert pl.page_skip_nonlatin("") is True                                          # empty

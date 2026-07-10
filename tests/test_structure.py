"""Unit tests for the structure-analysis helpers (lib/structure.py): printer's
key, copyright, enumerator/list_item, raggedness."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import structure as st


# --- printer's key ---------------------------------------------------------- #
def test_printer_key_patterns():
    assert st.is_printer_key("10 9 8 7 6 5 4 3 2 1")
    assert st.is_printer_key("1 3 5 7 9")
    assert st.is_printer_key("04 03 02 01 00")
    assert st.is_printer_key("6 5 4 3")


def test_printer_key_negatives():
    assert not st.is_printer_key("3 It is a practice of Bodhisattvas")  # prose
    assert not st.is_printer_key("1 2")                                  # too few
    assert not st.is_printer_key("100 200 300")                          # not small
    assert not st.is_printer_key("1 2 4 3")                              # not monotonic


def test_copyright_keywords():
    assert st.is_copyright("ISBN 978-0-86171-452-8")
    assert st.is_copyright("Library of Congress Cataloging-in-Publication Data")
    assert st.is_copyright("All rights reserved.")
    assert not st.is_copyright("The practice of compassion.")


# --- enumerator / list_item / raggedness ------------------------------------ #
def test_enumerator():
    for s in ("c) Silk ribbon initiation", "1* Actual", "a* Four", "b. Understanding",
              "1. First", "1) First", "(a) alpha", "iv. fourth", "• bullet", "– dash"):
        assert st.has_enumerator(s), s
    assert not st.has_enumerator("3 It is a practice")   # bare stanza number
    assert not st.has_enumerator("That, like fire")


def test_ragged_distinguishes_verse_from_flow():
    verse = "Through abandoning bad objects\nthe afflictive emotions\nfade away\nslowly"
    assert st.is_ragged(verse)                  # multiple short lines
    flow = ("a long flowing prose line that nearly fills the column width here\n"
            "another long flowing prose line that also nearly fills the column\n"
            "short tail")
    assert not st.is_ragged(flow)               # only the last line is short


def test_list_item_vs_verse():
    assert st.is_list_item("c) Silk ribbon initiation, Kay 333.7/Bu 218.4")  # 1 line
    # enumerated BUT ragged (verse stanza set) -> not a list item
    assert not st.is_list_item("1 First verse line here\nshort\nanother line\ntiny")


def test_toc_block():
    assert st.is_toc_block("Preface . . . . . 7")            # leader dots
    assert st.is_toc_block("Chapter One ...... 12")
    assert st.is_toc_block("Introduction 5\nMethod 9\nResults 14")  # >=2 title+num
    assert not st.is_toc_block("a normal sentence without page refs")
    assert not st.is_toc_block("The year was 1990")          # single trailing number


# --- running heads / folios (repetition) ------------------------------------ #
def test_running_heads_by_repetition():
    from ocr_backend import Block, Page
    pages = []
    for i in range(4):
        head = Block(f"{14 + i} A Sample Book Title: Introduction",
                     [180, 20, 1200, 60], label="SectionHeader")     # top band
        folio = Block(str(300 + i), [720, 2470, 770, 2495], label="Text")  # bottom
        body = Block("Some ordinary body text on the page.",
                     [180, 400, 1300, 520], label="Text")
        pages.append(Page(None, f"page-{i:04d}", 1500, 2540,
                          [head, folio, body], i + 1))
    rh = st.running_heads(pages)
    for p in pages:
        assert rh[id(p.blocks[0])] == "page_header"   # repeated head (masked) detected
        assert rh[id(p.blocks[1])] == "page_number"   # repeated folio -> page_number
        assert id(p.blocks[2]) not in rh              # body untouched


def test_running_head_needs_repetition():
    from ocr_backend import Block, Page
    # a one-off top-band block is NOT a running head
    pages = [Page(None, "page-0001", 1500, 2540,
                  [Block("A Unique Chapter Title", [180, 20, 900, 60],
                         label="SectionHeader")], 1)]
    assert st.running_heads(pages) == {}


# --- region-aware furniture: title_page / dedication / colophon ------------- #
def _body(n):
    from ocr_backend import Block, Page
    return Page(None, f"page-{n:04d}", 1500, 2500, [Block(
        "Body prose paragraph that fills the column and runs on long enough to "
        "be counted as real prose content on the page.",
        [180, 300, 1320, 520], label="Text")], n)


def test_region_furniture_types():
    from ocr_backend import Block, Page
    from build_doc import build_doc_dict
    pages = [
        Page(None, "page-0001", 1500, 2500,
             [Block("THE BOOK TITLE", [400, 300, 1100, 360], label="SectionHeader"),
              Block("An Author Name", [550, 500, 950, 540], label="Text")], 1),
        Page(None, "page-0002", 1500, 2500,
             [Block("Dedicated to my root teacher.", [450, 400, 1050, 460],
                    label="Text")], 2),
    ] + [_body(i) for i in range(3, 11)] + [
        Page(None, "page-0011", 1500, 2500,
             [Block("Index", [180, 200, 400, 260], label="SectionHeader")], 11),
        Page(None, "page-0012", 1500, 2500,
             [Block("10 9 8 7 6 5 4 3 2 1", [600, 400, 900, 440], label="Text")], 12),
    ]
    byp = {}
    for b in build_doc_dict(pages)["blocks"]:
        byp.setdefault(b["page"], []).append(b["type"])
    assert all(t == "title_page" for t in byp[1])   # title + author dropped
    assert "dedication" in byp[2]
    assert "colophon" in byp[12]                     # back-matter printer's key


def test_epigraph_kept_in_backmatter():
    from ocr_backend import Block, Page
    from build_doc import build_doc_dict
    pages = [_body(i) for i in range(1, 9)]
    pages.append(Page(None, "page-0009", 1500, 2500,
                      [Block("Appendix", [180, 200, 500, 260],
                             label="SectionHeader")], 9))
    epi = Block("A quoted opening passage line one\nand a shorter second\nand third",
                [400, 300, 1100, 460], label="Text")        # centered 3-line quote
    m1 = Block("full width margin line that reaches across the whole column here now",
               [180, 600, 1320, 660], label="Text")
    m2 = Block("another full width margin line reaching across the whole column too",
               [180, 700, 1320, 760], label="Text")
    pages.append(Page(None, "page-0010", 1500, 2500, [epi, m1, m2], 10))
    p10 = [b for b in build_doc_dict(pages)["blocks"] if b["page"] == 10]
    # in back matter the 3-line quote is KEPT as epigraph, not dropped to body
    assert any(b["type"] == "epigraph" for b in p10)

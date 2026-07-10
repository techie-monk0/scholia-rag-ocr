"""Regression fixtures from ARCHITECTURE.md §"Regression fixtures" — real
pages from a re-OCR'd scholarly book (fixture set `e10`) that the RAG-side detector
mis-handled. Block geometry is committed in fixtures/regression_pages/e10_pages.json.
Assertions are added phase by phase; the p42 verse positive-control runs from the
start so precision fixes can't kill recall.

Page → PDF page (1-based) = doc.json `page`."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from ocr_backend import Block, Page
from build_doc import build_doc_dict

_FX = json.loads((Path(__file__).resolve().parent / "fixtures" /
                  "regression_pages" / "e10_pages.json").read_text())


def _blocks_for(page_num: int):
    """Build the doc.json blocks for one fixture page."""
    d = _FX[str(page_num)]
    blocks = [Block(b["text"], b["bbox"], b.get("confidence"), b.get("label"),
                    b.get("html")) for b in d["blocks"]]
    page = Page(None, d["name"], d["width"], d["height"], blocks, page_num)
    return build_doc_dict([page])["blocks"]


def _types(page_num):
    return [b["type"] for b in _blocks_for(page_num)]


def _full_doc_blocks():
    """All fixture pages as ONE doc (in page order) so the region pre-pass can run
    (front=title/copyright/TOC run, body=the rest). Returns {page_num: [blocks]}."""
    order = sorted(_FX, key=int)
    pages = []
    for k in order:
        d = _FX[k]
        blocks = [Block(b["text"], b["bbox"], b.get("confidence"), b.get("label"),
                        b.get("html")) for b in d["blocks"]]
        pages.append(Page(None, d["name"], d["width"], d["height"], blocks, int(k)))
    doc = build_doc_dict(pages)
    by_page = {}
    for b in doc["blocks"]:
        by_page.setdefault(b["page"], []).append(b)
    return by_page


# --- verse positive control (recall) — must NEVER regress ------------------- #
def test_p42_real_verse_detected():
    # Body-region page; the re-tuned detector (lower indent threshold + margin from
    # all text-ish blocks + flow-based verse/quote) recovers this real verse.
    assert "verse" in _types(42), \
        "p42 verse recall regressed — a precision fix killed a real verse"


# --- Phase 1: printer's key + copyright ------------------------------------- #
def test_p5_printer_key_is_copyright_not_footnote():
    blocks = _blocks_for(5)
    # the printer's keys (e.g. '04 03 02 01 00', '6 5 4 3') must be copyright,
    # and must NOT have been turned into a footnote / carry note markers.
    import structure
    keys = [b for b in blocks if structure.is_printer_key(b["text"])]
    assert keys, "printer's-key line not present in fixture p5"
    for b in keys:
        assert b["type"] == "copyright"
        assert b["type"] not in ("footnote", "endnote")
        assert not b.get("refs")


def test_p5_has_copyright_block():
    # ISBN/LCCN/© content on the copyright page is typed copyright (dropped later).
    assert "copyright" in _types(5)


# --- Phase 2: list_item vs numbered-verse ----------------------------------- #
def test_p152_numbered_list_is_list_item_not_verse():
    t = _types(152)
    assert "list_item" in t                 # enumerated outline items labeled
    assert "verse" not in t                 # …and NOT mis-read as verse


# --- Phase 3: TOC ----------------------------------------------------------- #
def test_p6_toc_not_verse():
    t = _types(6)
    assert "toc" in t and "verse" not in t   # Contents page typed toc, never verse


# --- Phase 4: region (front/back gate) -------------------------------------- #
def test_p1_title_is_front_and_not_verse():
    by_page = _full_doc_blocks()
    p1 = by_page[1]
    assert all(b.get("region") == "front" for b in p1)   # title page in front matter
    assert all(b["type"] != "verse" for b in p1)         # front matter never verse
    assert all(b["type"] == "title_page" for b in p1)    # …and dropped as title_page
    # and a body page is region=body, where the verse survives
    assert any(b.get("region") == "body" and b["type"] == "verse"
               for b in by_page[42])


# --- Phase 5: running head (control) ---------------------------------------- #
def test_p14_running_header_not_verse():
    # In the isolated fixture the head appears once (repetition needs the full
    # book), so it stays a heading — the control is simply that it's NOT a verse.
    assert "verse" not in _types(14)

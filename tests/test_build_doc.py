"""Phase 1 — structured-output builder (lib/build_doc.py)."""
import pytest

from build_doc import build_doc_dict, build_doc
from conftest import mk_block, mk_page


def _doc(pages, **kw):
    return build_doc_dict(pages, **kw)


def test_label_to_type_mapping():
    pages = [mk_page([
        mk_block("Body text.", label="Text"),
        mk_block("A Heading", label="SectionHeader"),
        mk_block("running head", label="PageHeader"),
        mk_block("13 a note", label="Footnote"),
        mk_block("<table><tr><td>x</td></tr></table>", label="Table",
                 html="<table><tr><td>x</td></tr></table>"),
        mk_block("a list", label="ListGroup"),
        mk_block("mystery", label="WeirdNewLabel"),
    ])]
    # order-independent (reading-order sorts the footnote to the end)
    from collections import Counter
    types = Counter(b["type"] for b in _doc(pages)["blocks"])
    assert types == Counter(["body", "heading", "page_header", "footnote",
                             "table", "list_item", "body"])   # unknown -> body


def test_heading_level_from_html_then_fallback():
    pages = [mk_page([
        mk_block("H1", label="SectionHeader", html="<h1>H1</h1>"),
        mk_block("H3", label="SectionHeader", html="<h3>H3</h3>"),
        mk_block("plain", label="SectionHeader"),       # no html -> default 2
        mk_block("title", label="Title"),               # Title -> 1
    ])]
    levels = [b["level"] for b in _doc(pages)["blocks"]]
    assert levels == [1, 3, 2, 1]
    # non-headings carry null level
    assert _doc([mk_page([mk_block("x", label="Text")])])["blocks"][0]["level"] is None


def test_page_number_retyped_from_furniture():
    pages = [mk_page([
        mk_block("57", label="PageHeader"),
        mk_block("xiv", label="PageFooter"),
        mk_block("528 Steps on the Path", label="PageHeader"),   # not just a number
    ])]
    types = [b["type"] for b in _doc(pages)["blocks"]]
    assert types == ["page_number", "page_number", "page_header"]


def test_body_refs_extracted():
    pages = [mk_page([mk_block("Nāgārjuna says[^316] and also[^317].",
                               label="Text")])]
    b = _doc(pages)["blocks"][0]
    assert b["refs"] == ["316", "317"]
    assert b["marker"] is None


@pytest.mark.parametrize("text,marker,body", [
    ("[^13]A yojana is a unit.", "13", "A yojana is a unit."),
    ("13. A yojana is a unit.", "13", "A yojana is a unit."),
    ("13) A yojana.", "13", "A yojana."),
    ("* A symbol note.", "*", "A symbol note."),
])
def test_footnote_marker_extracted(text, marker, body):
    pages = [mk_page([mk_block(text, label="Footnote")])]
    b = _doc(pages)["blocks"][0]
    assert b["type"] == "footnote"
    assert b["marker"] == marker
    assert b["text"] == body
    assert b["refs"] == []                  # a note body doesn't "reference"


def test_table_html_preserved_not_flattened():
    html = "<table><tr><td>a</td><td>b</td></tr></table>"
    pages = [mk_page([mk_block("a b", label="Table", html=html)])]
    b = _doc(pages)["blocks"][0]
    assert b["type"] == "table"
    assert b["table"] == {"format": "html", "content": html}


def test_chapter_tracking_from_heading():
    pages = [mk_page([
        mk_block("intro body", label="Text"),               # before any chapter
        mk_block("Chapter 13", label="SectionHeader"),
        mk_block("ch13 body", label="Text"),
    ])]
    blocks = _doc(pages)["blocks"]
    assert blocks[0]["chapter"] is None
    assert blocks[2]["chapter"] == "13"


@pytest.mark.parametrize("heading,nav_type", [
    ("Bibliography", "bibliography"),
    ("Index", "index"),
    ("Contents", "toc"),
])
def test_navigation_section_labeling(heading, nav_type):
    pages = [mk_page([
        mk_block(heading, label="SectionHeader"),
        mk_block("an entry under it", label="Text"),
        mk_block("another entry", label="Text"),
    ])]
    blocks = _doc(pages)["blocks"]
    assert blocks[0]["type"] == "heading"
    assert blocks[1]["type"] == nav_type
    assert blocks[2]["type"] == nav_type


def test_navigation_from_running_header():
    # Back-matter index page: the section marker is the running HEADER, not a
    # SectionHeader. Body entries on the page should be typed `index`.
    pages = [mk_page([
        mk_block("INDEX OF ENGLISH AND ANGLICIZED WORDS", label="PageHeader"),
        mk_block("377", label="PageHeader"),
        mk_block("Avalokiteśvara, 12, 84, 233.", label="Text"),
    ], name="p0387")]
    blocks = _doc(pages)["blocks"]
    assert blocks[0]["type"] == "page_header"
    assert blocks[1]["type"] == "page_number"
    assert blocks[2]["type"] == "index"


def test_needs_review_from_confidence():
    pages = [mk_page([
        mk_block("low", label="Text", confidence=0.4),
        mk_block("high", label="Text", confidence=0.95),
        mk_block("none", label="Text", confidence=None),
    ])]
    flags = [b["needs_review"] for b in _doc(pages)["blocks"]]
    assert flags == [True, False, False]


def test_page_number_from_image_name():
    pages = [mk_page([mk_block("x", label="Text")], name="p0057")]
    assert _doc(pages)["blocks"][0]["page"] == 57
    pages2 = [mk_page([mk_block("x", label="Text")], name="page-0001")]
    assert _doc(pages2)["blocks"][0]["page"] == 1


def test_block_ids_unique_and_sequential():
    pages = [mk_page([mk_block(f"b{i}", label="Text") for i in range(3)])]
    ids = [b["id"] for b in _doc(pages)["blocks"]]
    assert ids == ["b00001", "b00002", "b00003"]
    assert len(set(ids)) == len(ids)


def test_doc_id_is_content_hash(tmp_path):
    f = tmp_path / "book.pdf"
    f.write_bytes(b"%PDF-1.5 hello")
    d1 = _doc([mk_page([mk_block("x")])], source=f)
    d2 = _doc([mk_page([mk_block("y")])], source=f)
    assert d1["doc_id"] == d2["doc_id"]              # same bytes -> same id
    assert len(d1["doc_id"]) == 16
    f.write_bytes(b"%PDF-1.5 different")
    assert _doc([mk_page([mk_block("x")])], source=f)["doc_id"] != d1["doc_id"]


def test_contract_top_level_shape():
    doc = _doc([mk_page([mk_block("x")])], title="My Book")
    assert set(doc) >= {"doc_id", "source", "title", "blocks", "footnotes"}
    assert doc["title"] == "My Book"
    assert doc["footnotes"] == {}                    # empty until Phase 2
    b = doc["blocks"][0]
    assert set(b) >= {"id", "type", "text", "page", "level", "chapter", "refs",
                      "marker", "table", "bbox", "confidence", "needs_review"}


def test_build_doc_writes_json(tmp_path):
    out = tmp_path / "out" / "book.doc.json"
    build_doc([mk_page([mk_block("hello", label="Text")])], out, title="T")
    assert out.exists()
    import json
    doc = json.loads(out.read_text())
    assert doc["blocks"][0]["text"] == "hello"


def test_normalization_applied_to_block_text():
    # ligature + curly quote inside a block must be normalized in the doc
    pages = [mk_page([mk_block("the ﬁle’s name", label="Text")])]
    assert _doc(pages)["blocks"][0]["text"] == "the file's name"

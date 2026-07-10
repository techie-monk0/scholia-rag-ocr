"""Tests for the docjson read-only access layer — focused on the forward-compat
guarantees clients rely on."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import docjson
from docjson import Doc, Block


def _doc(blocks, **kw):
    d = {"schema_version": "1.2", "features": ["region", "verse"],
         "doc_id": "d1", "title": "T", "blocks": blocks, "footnotes": {}}
    d.update(kw)
    return Doc.from_dict(d)


def test_metadata_and_features():
    d = _doc([])
    assert d.schema_version == "1.2"
    assert d.has_feature("region") and not d.has_feature("glossary")
    assert d.features == frozenset({"region", "verse"})


def test_content_drops_furniture_and_notes_keeps_unknown():
    d = _doc([
        {"id": "b1", "type": "title_page", "text": "TITLE", "region": "front"},
        {"id": "b2", "type": "body", "text": "real prose", "region": "body"},
        {"id": "b3", "type": "footnote", "text": "1 a note", "region": "body"},
        {"id": "b4", "type": "page_header", "text": "running head", "region": "body"},
        {"id": "b5", "type": "sidebar", "text": "future type", "region": "body"},  # UNKNOWN
    ])
    got = {b.id for b in d.content_blocks(region=None)}
    assert got == {"b2", "b5"}            # body + unknown kept; furniture+note dropped


def test_region_filtering_when_feature_present():
    d = _doc([
        {"id": "b1", "type": "body", "text": "front bit", "region": "front"},
        {"id": "b2", "type": "body", "text": "body bit", "region": "body"},
        {"id": "b3", "type": "body", "text": "back bit", "region": "back"},
    ])
    assert [b.id for b in d.content_blocks(region="body")] == ["b2"]
    assert {b.id for b in d.content_blocks(region=None)} == {"b1", "b2", "b3"}


def test_region_default_body_for_pre_region_doc():
    # No `region` feature and no region keys: region filtering must NOT silently
    # drop everything — content() yields all content blocks, region defaults body.
    d = _doc([{"id": "b1", "type": "body", "text": "x"},
              {"id": "b2", "type": "body", "text": "y"}],
             features=["verse"])           # note: no "region"
    assert d.blocks[0].region == "body"    # default
    assert [b.id for b in d.content_blocks(region="body")] == ["b1", "b2"]


def test_verse_access_and_optional_keys():
    d = _doc([{"id": "b1", "type": "verse", "text": "a\nb", "ref_label": "24",
               "indent": 0.12, "centered": True, "line_bbox": [[1, 2, 3, 4]]}])
    v = d.verses()[0]
    assert v.ref_label == "24" and v.centered and v.indent == 0.12
    assert v.line_bbox == [[1, 2, 3, 4]]
    # a block missing those keys returns safe defaults, never KeyError
    plain = Block({"id": "x", "type": "body", "text": "t"})
    assert plain.ref_label is None and plain.centered is False and plain.region == "body"


def test_footnote_backlink():
    d = _doc(
        [{"id": "b9", "type": "footnote", "text": "the note", "marker": "1"}],
        footnotes={"p5:1": {"marker": "1", "kind": "footnote", "block_id": "b9"}},
    )
    assert d.note("p5:1")["kind"] == "footnote"
    assert d.note_source_block("p5:1").id == "b9"
    assert d.note_source_block("nope") is None


def test_clean_text_join():
    d = _doc([
        {"id": "b1", "type": "title_page", "text": "DROP ME", "region": "front"},
        {"id": "b2", "type": "body", "text": "para one", "region": "body"},
        {"id": "b3", "type": "body", "text": "para two", "region": "body"},
    ])
    assert d.text(region="body") == "para one\n\npara two"


def test_manifest_loader_tolerates_partial_lines(tmp_path):
    m = tmp_path / "ingest_manifest.jsonl"
    m.write_text(
        json.dumps({"edition": "e5", "doc_json": "/x/e5.doc.json"}) + "\n"
        + "\n"                                           # blank line
        + json.dumps({"edition": "e7", "doc_json": "/x/e7.doc.json"}) + "\n"
        + '{"edition": "e9", "doc_json": "/x/'          # half-written trailing line
    )
    rows = list(docjson.load_manifest(m))
    assert [r["edition"] for r in rows] == ["e5", "e7"]   # partial line skipped
    assert docjson.open_doc({"doc_json": "/does/not/exist"}) is None


def test_integration_reads_a_built_doc():
    # Build a real doc through the producer and read it back via the layer.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
    from ocr_backend import Block as PB, Page
    from build_doc import build_doc_dict
    pages = [Page(None, "page-0001", 800, 1400, [
        PB("A heading", [100, 100, 700, 140], label="SectionHeader"),
        PB("Some body prose that is long enough to read as real content here.",
           [100, 200, 700, 300], label="Text"),
    ], 1)]
    doc = Doc.from_dict(build_doc_dict(pages))
    assert doc.schema_version and "verse" in doc.features
    assert doc.text(region="body")            # produces clean text without error

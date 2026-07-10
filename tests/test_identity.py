"""identity — the single source of truth for a book's stable identity.

`corpus_key` is the abstraction the corpus export keys on: it must return the
`pub_id` UUID (the catalogue<->LLM contract token) when available, and degrade
predictably so an export never collides or crashes on an unstamped book.
"""
import json

import identity


def _book(tmp_path, name, meta=None):
    d = tmp_path / name
    d.mkdir()
    if meta is not None:
        (d / "edition.json").write_text(json.dumps(meta))
    return d


def test_sidecar_identity_reads_all_fields(tmp_path):
    d = _book(tmp_path, "Some Book Title", {
        "edition": 10, "pub_id": "u-10", "content_hash": "t:abc",
        "source": "/x.pdf", "doc_id": "d10"})
    ident = identity.sidecar_identity(d)
    assert ident == identity.Identity("u-10", "t:abc", 10, "/x.pdf", "d10")


def test_corpus_key_prefers_pub_id(tmp_path):
    d = _book(tmp_path, "Titled Dir", {"edition": 10, "pub_id": "u-10"})
    assert identity.corpus_key(d) == "u-10"


def test_corpus_key_falls_back_to_edition_when_unstamped(tmp_path):
    d = _book(tmp_path, "Titled Dir", {"edition": 42})   # no pub_id yet
    assert identity.corpus_key(d) == "e42"


def test_corpus_key_falls_back_to_legacy_ennn_dir(tmp_path):
    d = _book(tmp_path, "e77")                           # no sidecar at all
    assert identity.sidecar_identity(d).edition == 77
    assert identity.corpus_key(d) == "e77"


def test_corpus_key_last_resort_is_dir_name(tmp_path):
    d = _book(tmp_path, "Unresolvable Book")             # no sidecar, non-eNNN name
    assert identity.corpus_key(d) == "Unresolvable Book"

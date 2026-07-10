"""Tests for lib/add_pages.py (--add-pages-after) and its --force safety."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import surya_backend as sb          # noqa: E402
import add_pages                    # noqa: E402
import preprocess                   # noqa: E402


def _touch(path: Path, content: bytes = b"img"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _book(tmp_path, pages=(7, 8, 9), prefix="Book - ", width=3):
    d = tmp_path / "book"
    for n in pages:
        _touch(d / f"{prefix}{str(n).zfill(width)}.jpg")
    return d


# --- resolve_anchor ---------------------------------------------------------
def test_resolve_anchor_ok(tmp_path):
    book = sb.list_images(_book(tmp_path))
    assert add_pages.resolve_anchor("Book - 7.jpg", book) == ("Book - ", 3, 7)
    # tolerant: unpadded / bare-number forms resolve to the same page
    assert add_pages.resolve_anchor("Book - 007.jpg", book) == ("Book - ", 3, 7)
    assert add_pages.resolve_anchor("7.jpg", book) == ("Book - ", 3, 7)


def test_resolve_anchor_rejects_path(tmp_path):
    book = sb.list_images(_book(tmp_path))
    with pytest.raises(SystemExit, match="must be a filename, not a path"):
        add_pages.resolve_anchor("sub/Book - 7.jpg", book)


def test_resolve_anchor_missing_page(tmp_path):
    book = sb.list_images(_book(tmp_path))
    with pytest.raises(SystemExit, match="not found in the book"):
        add_pages.resolve_anchor("Book - 99.jpg", book)


def test_resolve_anchor_no_number(tmp_path):
    book = sb.list_images(_book(tmp_path))
    with pytest.raises(SystemExit, match="no page number"):
        add_pages.resolve_anchor("cover.jpg", book)


def test_label_sequence():
    labels = [add_pages._label(i) for i in range(28)]
    assert labels[:4] == ["a", "b", "c", "d"]
    assert labels[25:28] == ["z", "aa", "ab"]


# --- add: naming, order, sort ----------------------------------------------
def test_add_ordered_files_sort_after_anchor(tmp_path):
    book = _book(tmp_path)
    s1 = _touch(tmp_path / "src/first.jpg", b"one")
    s2 = _touch(tmp_path / "src/second.jpg", b"two")
    dst = add_pages.add(book, sb.list_images(book), "Book - 7.jpg", [s1, s2])
    added = [p.name for p in sb.list_images(dst)]
    assert added == ["Book - 007a.jpg", "Book - 007b.jpg"]      # in given order
    # merged natural sort places them right after 007 and before 008
    merged = sorted([p.name for p in sb.list_images(book)] + added, key=sb.natural_key)
    assert merged == ["Book - 007.jpg", "Book - 007a.jpg",
                      "Book - 007b.jpg", "Book - 008.jpg", "Book - 009.jpg"]


def test_add_folder_input_natural_sorted(tmp_path):
    book = _book(tmp_path)
    folder = tmp_path / "inserts"
    _touch(folder / "z.jpg", b"zz")
    _touch(folder / "a.jpg", b"aa")
    dst = add_pages.add(book, sb.list_images(book), "Book - 7.jpg", [folder])
    added = [p.name for p in sb.list_images(dst)]
    # folder taken in natural-sort order: a.jpg -> 007a, z.jpg -> 007b
    assert added == ["Book - 007a.jpg", "Book - 007b.jpg"]
    assert (dst / "Book - 007a.jpg").read_bytes() == b"aa"


def test_add_preserves_extension(tmp_path):
    book = _book(tmp_path)
    s = _touch(tmp_path / "src/p.png", b"png")
    dst = add_pages.add(book, sb.list_images(book), "Book - 7.jpg", [s])
    assert [p.name for p in sb.list_images(dst)] == ["Book - 007a.png"]


# --- add: idempotency / re-runs --------------------------------------------
def test_add_idempotent_same_bytes(tmp_path):
    book = _book(tmp_path)
    s = _touch(tmp_path / "src/x.jpg", b"same")
    add_pages.add(book, sb.list_images(book), "Book - 7.jpg", [s])
    dst = add_pages.add(book, sb.list_images(book), "Book - 7.jpg", [s])   # again
    assert [p.name for p in sb.list_images(dst)] == ["Book - 007a.jpg"]    # no dup


def test_add_next_free_letter_across_runs(tmp_path):
    book = _book(tmp_path)
    s1 = _touch(tmp_path / "src/x.jpg", b"first")
    s2 = _touch(tmp_path / "src/y.jpg", b"second")
    add_pages.add(book, sb.list_images(book), "Book - 7.jpg", [s1])
    dst = add_pages.add(book, sb.list_images(book), "Book - 7.jpg", [s2])
    assert [p.name for p in sb.list_images(dst)] == ["Book - 007a.jpg",
                                                     "Book - 007b.jpg"]


def test_add_dedup_across_padding_width(tmp_path):
    # An earlier UNPADDED run inserted 'Book - 9a.jpg'; now the book is padded and the
    # same bytes must NOT be re-added as 'Book - 009a.jpg'.
    book = _book(tmp_path)                      # padded pages: Book - 007/008/009.jpg
    _touch(book / "_added" / "Book - 9a.jpg", b"insert")     # stale unpadded insert
    s = _touch(tmp_path / "src/x.jpg", b"insert")            # same bytes
    dst = add_pages.add(book, sb.list_images(book), "Book - 9.jpg", [s])
    assert [p.name for p in sb.list_images(dst)] == ["Book - 9a.jpg"]   # no 009a dup


def test_add_requires_a_source(tmp_path):
    book = _book(tmp_path)
    with pytest.raises(SystemExit, match="at least one source"):
        add_pages.add(book, sb.list_images(book), "Book - 7.jpg", [])


# --- --force safety: _added survives clean_step_dirs ------------------------
def test_added_dir_not_a_preprocess_step_dir():
    assert "_added" not in preprocess._STEP_DIRS


# --- resolve_start (--dump-input-files) ------------------------------------
def test_resolve_start_exact_and_stem_and_number():
    names = ["Book - 007.jpg", "Book - 007a.jpg", "Book - 008.jpg"]
    assert add_pages.resolve_start(names, "Book - 007a.jpg") == 1      # exact
    assert add_pages.resolve_start(names, "Book - 007a") == 1          # stem
    assert add_pages.resolve_start(names, "Book - 7.jpg") == 0         # by page number
    assert add_pages.resolve_start(names, "7") == 0                    # bare number
    assert add_pages.resolve_start(names, "sub/Book - 8.jpg") == 2     # basename only


def test_dedupe_added_removes_stale_width_duplicate(tmp_path):
    d = tmp_path / "_added"
    _touch(d / "Yamantaka 1 - 9b.jpg", b"same")       # stale unpadded
    _touch(d / "Yamantaka 1 - 009b.jpg", b"same")     # padded, identical bytes
    _touch(d / "Yamantaka 1 - 009a.jpg", b"unique")   # kept (unique)
    removed = add_pages.dedupe_added(d)
    assert removed == ["Yamantaka 1 - 9b.jpg"]        # padded name sorts first -> kept
    assert sorted(p.name for p in sb.list_images(d)) == [
        "Yamantaka 1 - 009a.jpg", "Yamantaka 1 - 009b.jpg"]


def test_resolve_start_not_found():
    with pytest.raises(SystemExit, match="not found in the input"):
        add_pages.resolve_start(["Book - 007.jpg"], "Book - 99.jpg")


def test_clean_step_dirs_leaves_added(tmp_path):
    book = _book(tmp_path)
    s = _touch(tmp_path / "src/x.jpg", b"keep")
    add_pages.add(book, sb.list_images(book), "Book - 7.jpg", [s])
    # simulate a real step dir that --force SHOULD wipe
    _touch(book / "_renumbered" / "Book - 007.jpg")
    preprocess.clean_step_dirs(book)                       # what --force runs
    assert not (book / "_renumbered").exists()             # step dir wiped
    assert (book / "_added" / "Book - 007a.jpg").exists()  # added pages survive

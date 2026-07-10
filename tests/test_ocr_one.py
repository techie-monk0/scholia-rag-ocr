"""Batch input resolution (ocr_one.resolve_inputs) — the fold of re-OCR batch
into ocr_one. No OCR is run here; only the input-expansion logic is tested."""
import ocr_one


def test_single_pdf(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"%PDF")
    assert ocr_one.resolve_inputs([p]) == [p]


def test_folder_of_pdfs_is_batch(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "b.pdf").write_bytes(b"%PDF")
    got = ocr_one.resolve_inputs([tmp_path])
    assert sorted(p.name for p in got) == ["a.pdf", "b.pdf"]


def test_folder_of_images_is_single_book(tmp_path):
    (tmp_path / "p1.png").write_bytes(b"x")
    (tmp_path / "p2.png").write_bytes(b"x")
    assert ocr_one.resolve_inputs([tmp_path]) == [tmp_path]


def test_worklist_txt(tmp_path):
    wl = tmp_path / "books.txt"
    wl.write_text("/x/a.pdf\n# a comment\n\n/y/b.pdf\n")
    got = ocr_one.resolve_inputs([wl])
    assert [str(p) for p in got] == ["/x/a.pdf", "/y/b.pdf"]


def test_multiple_inputs_combined(tmp_path):
    a = tmp_path / "a.pdf"; a.write_bytes(b"%PDF")
    b = tmp_path / "b.pdf"; b.write_bytes(b"%PDF")
    assert ocr_one.resolve_inputs([a, b]) == [a, b]

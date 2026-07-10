"""The pause/resume contract: after a kill, a re-run OCRs only the pages not
already checkpointed (lib/ocr_surya.py::pending_pages). This is what makes
`--kill` == pause, and `--skip-done` == resume-where-it-stopped."""
import os

import pytest
from ocr_surya import pending_pages


def _imgs(tmp_path, n, *, mtime=1000.0):
    """n page images with a fixed, old mtime (so they're not 'modified since
    the cache' by accident)."""
    out = []
    for i in range(1, n + 1):
        p = tmp_path / f"page-{i:04d}.png"
        p.write_bytes(b"x")
        os.utime(p, (mtime, mtime))
        out.append(p)
    return out


def test_nothing_cached_does_all(tmp_path):
    imgs = _imgs(tmp_path, 5)
    todo = pending_pages(imgs, cached_stems=set(), cache_mtime=2000.0)
    assert [p.name for p in todo] == [p.name for p in imgs]


def test_all_cached_does_nothing(tmp_path):
    """Fully-OCR'd book: a re-run is a no-op (reuse), not a re-OCR."""
    imgs = _imgs(tmp_path, 5)
    cached = {p.stem for p in imgs}
    assert pending_pages(imgs, cached, cache_mtime=2000.0) == []


def test_partial_cache_resumes_remainder(tmp_path):
    """The core guarantee: killed after page 2 of 5 -> resume does pages 3-5."""
    imgs = _imgs(tmp_path, 5)
    cached = {imgs[0].stem, imgs[1].stem}          # pages 1-2 checkpointed
    todo = pending_pages(imgs, cached, cache_mtime=2000.0)
    assert [p.name for p in todo] == [imgs[2].name, imgs[3].name, imgs[4].name]


def test_force_redoes_everything(tmp_path):
    imgs = _imgs(tmp_path, 3)
    cached = {p.stem for p in imgs}
    assert pending_pages(imgs, cached, cache_mtime=2000.0, force=True) == imgs


def test_reocr_targets_specific_pages(tmp_path):
    imgs = _imgs(tmp_path, 4)
    cached = {p.stem for p in imgs}
    todo = pending_pages(imgs, cached, cache_mtime=2000.0,
                         reocr=[imgs[1].name, "page-0003"])
    assert {p.stem for p in todo} == {"page-0002", "page-0003"}


def test_image_modified_after_cache_is_redone(tmp_path):
    """A page re-rasterized after the checkpoint is treated as stale."""
    imgs = _imgs(tmp_path, 3, mtime=1000.0)
    cached = {p.stem for p in imgs}
    os.utime(imgs[1], (3000.0, 3000.0))            # newer than cache_mtime
    todo = pending_pages(imgs, cached, cache_mtime=2000.0)
    assert [p.name for p in todo] == [imgs[1].name]

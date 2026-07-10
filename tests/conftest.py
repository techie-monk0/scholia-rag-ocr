"""Shared test fixtures: put the repo root + lib/ on sys.path and provide tiny
Page/Block builders so tests can construct synthetic OCR output without running
Surya."""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "lib"))
sys.path.insert(0, str(_ROOT))                    # repo-root modules (e.g. preprocess)

import pytest
from ocr_backend import Block, Page


def mk_block(text="", *, label="Text", bbox=None, confidence=None, html=None):
    # bbox left None by default; mk_page stacks them into a clean single column.
    return Block(text=text, bbox=bbox, confidence=confidence, label=label,
                 html=html)


def mk_page(blocks, *, name="page-0001", page_no=None, width=1000, height=1400):
    blocks = list(blocks)
    # Give bbox-less blocks a non-overlapping vertical stack (single column) so
    # reading-order is unambiguous unless a test sets bboxes on purpose.
    for i, b in enumerate(blocks):
        if b.bbox is None:
            b.bbox = [0, i * 100, 800, i * 100 + 60]
    return Page(image_path=None, name=name, width=width, height=height,
                blocks=blocks, page_no=page_no)


@pytest.fixture
def block():
    return mk_block


@pytest.fixture
def page():
    return mk_page

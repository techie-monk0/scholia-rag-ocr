"""Pluggable OCR backend interface + the normalized page model.

The pipeline is split at one seam: a **backend** turns a folder of page images
into normalized ``Page`` objects (text + layout blocks); the **assembly** stage
(``ocr_to_text.build_text`` / ``ocr_to_pdf.build_pdf``) consumes only that model.
So swapping OCR engines — Surya today, Tesseract / a cloud OCR / a different VLM
tomorrow — touches neither the assembly nor ``ocr_one.py``.

A backend self-registers with ``@register("name")``; callers pick one with
``get_backend(name, **opts)``. The only required method is ``ocr_pages``;
``line_boxes`` is optional (return None and the searchable-PDF layer falls back
to band-slicing the OCR block boxes).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Normalized model (engine-agnostic). Backends emit these; assembly reads them.
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    text: str
    bbox: list                       # [x0, y0, x1, y1], image px, origin top-left
    confidence: float | None = None
    label: str | None = None         # layout label: Text/SectionHeader/PageHeader/...
    html: str | None = None          # backend's raw HTML (kept for tables + <sup> markers)


@dataclass
class Page:
    image_path: Path | None
    name: str                        # sort/identity key (usually the image filename)
    width: int
    height: int
    blocks: list = field(default_factory=list)
    page_no: int | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())


# --------------------------------------------------------------------------- #
# Backend interface
# --------------------------------------------------------------------------- #
class OcrBackend(abc.ABC):
    """Turns page images into normalized ``Page`` objects.

    Implementations own their caching, parallelism, and engine specifics.
    ``images`` is the (possibly limited) ordered list of image paths to process;
    ``image_dir`` is their folder (for caching / matching).
    """

    name: str = "base"

    def config(self) -> dict:
        """The knob values in effect, for reporting (override per backend)."""
        return {}

    @abc.abstractmethod
    def ocr_pages(self, images, image_dir, *, force: bool = False,
                  reocr=None) -> list[Page]:
        """``force`` re-OCRs everything; ``reocr`` is an iterable of filenames to
        re-OCR even if cached/unchanged. Backends with a cache should otherwise
        only (re)process images that are new or modified since the cache."""
        ...

    def line_boxes(self, images, image_dir, *, force: bool = False, reocr=None):
        """Per-line boxes ``{image-stem: [bbox, ...]}`` for the PDF text layer,
        or ``None`` if the engine doesn't provide them (assembly then band-slices
        the OCR block boxes)."""
        return None

    def warmup(self, log_dir):
        """Optionally pre-load models / start servers so a caller can OVERLAP the
        cold start with other work (e.g. image preprocessing). Returns an opaque
        handle to pass to ``shutdown``. Default: a no-op (engine loads lazily in
        ``ocr_pages``), so overlapping simply has no effect."""
        return None

    def shutdown(self, handle):
        """Tear down whatever ``warmup`` started (no-op if it returned None)."""
        pass


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, type[OcrBackend]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str, **opts) -> OcrBackend:
    if name not in _REGISTRY:
        raise SystemExit(f"unknown OCR backend {name!r}; "
                         f"available: {', '.join(available()) or '(none)'}")
    return _REGISTRY[name](**opts)

"""Zero-pad page numbers in image filenames to a uniform width (the
``--pad-page-numbers`` step).

Filenames are ``<prefix><page number><ext>`` (the trailing integer of the stem).
The width is ``auto_width(images)`` — the digit count of the LARGEST page number —
so every page pads to the same width: a book whose highest page is 976 rewrites
``Great hum - 1.jpg`` -> ``Great hum - 001.jpg``. A number already that wide is
unchanged (zero-padding never truncates).

Like the rotate/swap passes, copies go to a sibling ``_renumbered/`` dir
(non-destructive, chainable). Content is a byte copy — only the filename changes.
A file whose stem has no trailing number is copied under its original name.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_NUM_RE = re.compile(r"(\d+)\s*$")          # trailing integer of the filename stem


def _page_no(path):
    """Trailing page number in a filename stem, or ``None`` if it has none."""
    m = _NUM_RE.search(Path(path).stem)
    return int(m.group(1)) if m else None


def auto_width(images) -> int:
    """Digits needed for the largest page number among ``images`` (>=1). This is
    the width used by the ``--pad-page-numbers`` flag: e.g. a book whose highest
    page is 976 pads every page to 3 digits."""
    nums = [n for n in (_page_no(p) for p in images) if n is not None]
    return len(str(max(nums))) if nums else 1


def padded_name(path: Path, digits: int) -> str:
    """Filename for ``path`` with its trailing page number zero-padded to
    ``digits`` (returned unchanged if the stem has no trailing number)."""
    path = Path(path)
    m = _NUM_RE.search(path.stem)
    if not m:
        return path.name
    prefix = path.stem[:m.start()]
    return f"{prefix}{str(int(m.group(1))).zfill(digits)}{path.suffix}"


def make_padded(images, pad_dir: Path, digits: int, *, force=False):
    """Copy each image into ``pad_dir`` under its zero-padded name. Returns
    ``pad_dir``. Incremental: skips a target already current. Raises ``SystemExit``
    if two source pages collide on one padded name (e.g. ``p1.jpg`` + ``p01.jpg``
    at N>=1) — pick a naming that doesn't, or a different width."""
    if digits < 1:
        raise SystemExit(f"pad width must be >= 1; got {digits}")
    pad_dir = Path(pad_dir)
    pad_dir.mkdir(parents=True, exist_ok=True)
    claimed = {}                             # padded name -> source it came from
    for img in images:
        img = Path(img)
        name = padded_name(img, digits)
        if name in claimed and claimed[name] != img.name:
            raise SystemExit(
                f"pad to {digits} digit(s): {img.name!r} and {claimed[name]!r} "
                f"both map to {name!r}. Rename the sources so page numbers are "
                "distinct.")
        claimed[name] = img.name
        out = pad_dir / name
        if (not force and out.exists()
                and out.stat().st_mtime >= img.stat().st_mtime):
            continue
        shutil.copy2(img, out)               # byte copy (no re-encode); keeps mtime
    return pad_dir

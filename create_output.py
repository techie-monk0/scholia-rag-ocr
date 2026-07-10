#!/usr/bin/env python3
"""prepare-output stage, run on its own — (re)build a book's deliverables (searchable
PDF + body/full text + doc.json) from the OCR stage's cached OUTPUT, with NO re-OCR.

This is the back-of-the-pipeline counterpart to preprocess.py at the front: an
independent stage whose INPUT is the OCR stage's output (the cached results.json +
detection.json) and whose OUTPUT is the deliverables. So you can regenerate with
different appearance options (--color / --pdf-max-dim / --uniform-pages / --text / …)
in seconds instead of re-running OCR.

    python3 create_output.py "<book source folder>" --out <dir> \\
        --color grey --pdf-max-dim 2000 --uniform-pages

It is exactly ``ocr_one.py … --prepare-output`` — every ocr_one output option applies;
run ``python3 create_output.py --help`` (or ``ocr_one.py --help``) for the full list.
The book must already have been OCR'd (its cache must exist); otherwise it errors.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ocr_one  # noqa: E402


def main():
    argv = sys.argv[1:]
    if "--prepare-output" not in argv:          # imply the flag; this IS that stage
        argv = ["--prepare-output", *argv]
    sys.argv = [sys.argv[0], *argv]
    return ocr_one.main()


if __name__ == "__main__":
    sys.exit(main())

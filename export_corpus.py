#!/usr/bin/env python3
"""Export every verified book's durable artifacts into a self-contained corpus
folder for future LLM / RAG use — the small text/structure (doc.json, body/full
text, pages.jsonl), NOT the 48 GB of regenerable page images in ``out/``.

  python3 export_corpus.py                 # -> ./_ocr_corpus  (or $OCR_CORPUS_DIR)
  python3 export_corpus.py --corpus-dir DIR
  python3 export_corpus.py --dry-run       # verify only; report broken books

Book set: the deduplicated ``out/rag_handoff_manifest.jsonl`` if present (the
authoritative one-row-per-book list — respects the native/scanned dedup), else a
scan of ``out/backfill`` + ``out/detectonly``. Each book is HEALTH-CHECKED first
(``corpus.health``): a broken one (empty results stub / 0-block doc — the e3
failure mode) is NOT exported and is reported loudly so it can be re-OCR'd.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import corpus  # noqa: E402


def book_dirs():
    """The authoritative per-book output dirs. Prefer the deduped rag_handoff
    manifest (one row per book); else scan the two output trees."""
    man = HERE / "out" / "rag_handoff_manifest.jsonl"
    if man.is_file():
        seen = set()
        for line in man.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            d = (HERE / row["doc_json"]).resolve().parent
            if d not in seen:
                seen.add(d)
                yield d
        return
    for tree in ("out/backfill", "out/detectonly"):
        base = HERE / tree
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if d.is_dir() and list(d.glob("*.doc.json")):
                yield d


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-dir", default=None,
                    help="destination (default: $OCR_CORPUS_DIR or <repo>/_ocr_corpus)")
    ap.add_argument("--dry-run", action="store_true",
                    help="health-check and report only; copy nothing")
    args = ap.parse_args()

    dest = corpus.corpus_dir(args.corpus_dir)
    if dest is None:
        sys.exit("corpus export is disabled (corpus dir set to 'none')")

    rows, broken, exported = [], [], 0
    for d in book_dirs():
        h = corpus.health(d)
        corpus.write_marker(d, h)
        if not h.ok:
            broken.append((d.name, h.reasons))
            print(f"  BROKEN {d.name}: {'; '.join(h.reasons)}", flush=True)
            continue
        if args.dry_run:
            rows.append({"edition": d.name})   # count only
        else:
            rows.append(corpus.export_book(d, dest))
            exported += 1

    if not args.dry_run and rows:
        mp = corpus.write_manifest(dest, rows)
        print(f"\nmanifest -> {mp}", flush=True)

    print(f"\n{'DRY-RUN: ' if args.dry_run else ''}"
          f"verified {len(rows)} book(s)"
          + (f", exported {exported}" if not args.dry_run else "")
          + f", broken {len(broken)}.", flush=True)
    if broken:
        print("\nBROKEN books (not exported — re-OCR with "
              "`run_ocr.py --editions N --force`):", flush=True)
        for name, reasons in broken:
            print(f"  {name}: {'; '.join(reasons)}", flush=True)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())

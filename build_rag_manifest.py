#!/usr/bin/env python3
"""Build the deduplicated RAG handoff manifest from the OCR pipeline output.

The pipeline writes two trees: scanned (OCR) books under one out-root and
digital-first (native-text) books under another. A digital-first book that was
OCR'd before reclassification exists in BOTH — a degraded OCR copy and a clean
native copy. This tool emits ONE authoritative row per book: the **native** copy
wins over a scanned copy of the same edition, so the degraded duplicates are
excluded. See docs/ARCHITECTURE.md §"The RAG handoff" for how a consumer reads it.

    python3 build_rag_manifest.py
        [--native out/detectonly] [--scanned out/backfill]
        [--out out/rag_handoff_manifest.jsonl]

Each line: edition, doc_id, title, kind (native|scanned), source, schema_version,
doc_json, body_text, full_text, pages_jsonl, pdf, n_blocks, nfc, pub_id, content_hash.

`pub_id` is the catalogue's stable, opaque edition-identity token (the RAG consumer keys on it and
looks up everything else live); `content_hash` is the version (for re-embed staleness). Both are read
from the catalogue's `v_holding_files` external read-contract — see the catalogue's
external_tool_dependency_contract.md. Rows go unstamped (pub_id=null) if the catalogue DB is absent.
"""
import argparse
import glob
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import identity  # noqa: E402  — single source of truth for pub_id / content_hash / edition

# The catalogue is the system-of-record for edition identity. Default to the sibling checkout.
_DEFAULT_CAT_DB = Path(__file__).resolve().parents[1] / "library_cataloging" / "catalogue-db" / "catalogue.db"


def _first(base, *patterns):
    for pat in patterns:
        hit = glob.glob(f"{base}/{pat}")
        if hit:
            return hit[0]
    return None


def _edition_ids(root):
    """Map catalogue edition-id -> book dir path for every finished book under `root`.

    A book is any dir holding a `*.doc.json`. The edition integer is resolved by the
    shared identity layer (`edition.json` sidecar, or a legacy `eNNN` dir name). Dirs
    with a doc.json but no resolvable edition-id are skipped (no catalogue identity to
    hand off)."""
    out = {}
    for p in glob.glob(f"{root}/*"):
        if not (Path(p).is_dir() and glob.glob(p + "/*.doc.json")):
            continue
        e = identity.sidecar_identity(p).edition
        if e is not None:
            out[int(e)] = p
    return out


def _row(e, kind, base):
    dj = _first(base, "*.doc.json")
    d = json.load(open(dj))
    blocks = [b for b in d.get("blocks", []) if isinstance(b, dict)]
    sample = "".join((b.get("text") or "") for b in blocks[:300])
    return {
        "edition": f"e{e}",
        "doc_id": d.get("doc_id"),
        "title": d.get("title"),
        "kind": kind,                                  # native = pristine; scanned = OCR
        "source": d.get("source"),
        "schema_version": d.get("schema_version"),
        "doc_json": dj,
        "body_text": _first(base, "body/all_pages.txt"),
        "full_text": _first(base, "full/all_pages.txt"),
        "pages_jsonl": _first(base, "body/pages.jsonl"),
        "pdf": _first(base, "*.pdf"),
        "n_blocks": len(blocks),
        "nfc": unicodedata.normalize("NFC", sample) == sample,
        "pub_hint": identity.sidecar_identity(base).pub_id,  # sidecar snapshot; last-resort pub_id
    }


def _stamp_identity(rows, catalogue):
    """Stamp each row with the catalogue's stable `pub_id` (the edition-identity token the RAG
    consumer keys on) + a `content_hash` (the version, best-effort matched by source path). Returns
    the editions whose pub_id could not be resolved. `catalogue` is the map from
    `identity.catalogue_identity`.

    Primary key: the edition-id embedded in `e<N>` -> catalogue edition_id. Fallback: when that id
    no longer resolves (edition renumbered or its file re-pathed since OCR), reconnect by matching
    the row's original `source` path against the catalogue holding file paths. The two routes miss
    different rows, so trying both maximizes resolution; only editions absent from the catalogue by
    BOTH id and path stay unstamped."""
    # reverse index for the fallback route: original holding file path -> (pub_id, content_hash)
    by_path = {}
    for rec in catalogue.values():
        for fp, ch in rec["hashes"].items():
            by_path[fp] = (rec["pub_id"], ch)
    missing = []
    for r in rows:
        m = re.fullmatch(r"e(\d+)", r["edition"])
        rec = catalogue.get(int(m.group(1))) if m else None
        if rec and rec["pub_id"]:                          # primary: e<N> -> edition_id
            hashes = rec["hashes"]
            r["pub_id"] = rec["pub_id"]
            # match the version to this row's source file; else the sole holding hash; else None
            r["content_hash"] = hashes.get(r.get("source")) or (
                next(iter(hashes.values())) if len(hashes) == 1 else None)
        else:                                              # fallback: original source path -> holding
            r["pub_id"], r["content_hash"] = by_path.get(r.get("source"), (None, None))
        if not r["pub_id"]:                                # last resort: edition.json snapshot
            r["pub_id"] = r.get("pub_hint")
        if not r["pub_id"]:
            missing.append(r["edition"])
    return missing


def build(native_root, scanned_root, out, catalogue_db=_DEFAULT_CAT_DB):
    native_ids = _edition_ids(native_root)             # {edition_int: dir_path}
    scanned_ids = _edition_ids(scanned_root)
    rows, non_nfc, incomplete = [], [], []
    for e in sorted(native_ids):                       # native wins
        r = _row(e, "native", native_ids[e])
        rows.append(r)
    for e in sorted(set(scanned_ids) - set(native_ids)):   # scanned-only
        r = _row(e, "scanned", scanned_ids[e])
        rows.append(r)
    cat_identity = identity.catalogue_identity(catalogue_db)
    missing_pub = _stamp_identity(rows, cat_identity)
    for r in rows:
        r.pop("pub_hint", None)                        # internal helper — keep out of the manifest
    for r in rows:
        if not r["nfc"]:
            non_nfc.append(r["edition"])
        if not (r["doc_json"] and r["body_text"]):
            incomplete.append(r["edition"])
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_native = sum(1 for r in rows if r["kind"] == "native")
    print(f"wrote {out}: {len(rows)} books "
          f"({n_native} native + {len(rows) - n_native} scanned)")
    print(f"  excluded {len(set(native_ids) & set(scanned_ids))} degraded-OCR duplicates "
          f"(native copy kept)")
    print(f"  non-NFC: {non_nfc or 'none'}")
    print(f"  missing doc_json/body_text: {incomplete or 'none'}")
    if not cat_identity:
        print(f"  pub_id: catalogue DB not found at {catalogue_db} — rows NOT stamped")
    else:
        print(f"  pub_id: stamped from catalogue; unresolved: {missing_pub or 'none'}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--native", default="out/detectonly",
                    help="root of native (digital-first) book outputs")
    ap.add_argument("--scanned", default="out/backfill",
                    help="root of scanned (OCR) book outputs")
    ap.add_argument("--out", default="out/rag_handoff_manifest.jsonl",
                    help="manifest output path")
    ap.add_argument("--catalogue-db", default=str(_DEFAULT_CAT_DB),
                    help="catalogue.db to read pub_id/content_hash from (v_holding_files); "
                         "rows go unstamped if absent")
    args = ap.parse_args(argv)
    build(args.native, args.scanned, args.out, args.catalogue_db)


if __name__ == "__main__":
    sys.exit(main())

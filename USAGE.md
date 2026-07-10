# USAGE — running OCR

Workhorse: **`run_ocr.py`** (Surya OCR → searchable PDF + layout-labelled text + `doc.json`).
See `README.md` for the full picture; this is the operational cheat-sheet.

## Run

```bash
cd /path/to/ocr_pipeline

# one book (PDF or image folder)
python3 run_ocr.py "/path/to/Book.pdf"

# many books at once; resume finished ones from cache
python3 run_ocr.py --from-file worklist.txt --skip-done

# ad-hoc: mixed inputs / catalogue editions / front matter only
python3 run_ocr.py a.pdf "b book.pdf" scans/
python3 run_ocr.py --editions 60 179                 # needs the catalogue DB (optional)
python3 run_ocr.py some.pdf --front-matter 16        # just recover a copyright/ISBN page
```

Output goes to `<--out-root>/<name>/` (default out-root `out/backfill/`):
`<name>.pdf`, furniture-removed `body/` + full `full/` text, a per-block-labelled
`pages.jsonl`, and `doc.json`.

## Environment variables — none required

**You don't need to set anything to run the pipeline.** It's self-contained: the
`llama-server` binary is vendored (or found on `PATH`), the model weights download
themselves on first run, and there are no API keys. If the `llama-server` binary is
genuinely missing, the run stops with a clear error pointing at `README.md` — it
doesn't fail silently.

These variables are **optional overrides** only:

| variable | default | override it to… |
|---|---|---|
| `LLAMA_CPP_BINARY` | vendored `.llama/`, else `llama-server` on `PATH` | point at a specific `llama-server` build |
| `CATALOGUE_DB` | `../library_cataloging/catalogue-db/catalogue.db` | use your own catalogue DB (only for `--editions` / `pub_id` stamping — ignored on path/folder runs) |
| `OCR_CORPUS_DIR` | `<repo>/_ocr_corpus` | send `export_corpus.py` output elsewhere |
| `HF_TOKEN` | unset | authenticate model-weight downloads (rarely needed — the weights are public) |

`SURYA_MAX_TOKENS_*`, `DETECTOR_BATCH_SIZE`, `SURYA_INFERENCE_PARALLEL` are set by the
pipeline itself from your CLI flags — don't set them by hand.

## Preprocess / output flags (the four stages)

The pipeline is `preprocess → add-pages → OCR → output`. `run_ocr.py` forwards
extra flags to the front and back stages via `--` markers:

```bash
# flags after the 1st `--` go to preprocess; after a 2nd `--` go to output
python3 run_ocr.py book.pdf -- --rotate 180:CW --swap 9:end -- --color grey --pdf-max-dim 2000
```

Rebuild deliverables later with **no re-OCR** (change color/size/text): run
`python3 create_output.py "<book>" --out <dir> …` (same as `ocr_one.py … --prepare-output`).

## Resume / control

- **Resumable:** OCR is cached per page. A kill (Ctrl-C, `--kill`, OOM) is a *pause* —
  re-run the **same** command with `--skip-done` and it continues from cache.
  `--force` re-OCRs from scratch.
- **Status / control:** `--list-active` (what's running), `--list-resumable` (partial,
  not running), `--kill NAME` (stop one book; it resumes later).

## Best performance

The short version: **leave the defaults alone on one machine.** A single GPU is
already fully used by one OCR run, so throwing more workers or more books at it in
parallel won't make it faster (and can make it slower). A dense book is a one-time
~50 min; after that it's cached, so rebuilds are seconds. To actually cut wall-clock,
OCR fewer pages (`--front-matter N`).

### Technical details

- Keep total live `llama-server`s **≤ 3** (`--workers-per-book` × `--parallel-books`);
  Surya's default 8 decode slots already saturate one GPU. See the "Performance"
  section of `README.md`.
- `--parallel-books K` / a second cooperative worker (`--claim`, on by default) only
  helps across **separate GPUs / hosts**, not one GPU.

# scholia-rag-ocr

**Structure-preserving OCR for scholarly books — built for RAG.** Turn scanned page
images (or a PDF) into two things in one pass:

1. **a searchable PDF** — the original scan with an invisible, selectable text layer, and
2. **a citation-grade structured document** (`doc.json`) for LLM/RAG — labeled blocks in
   true reading order, footnotes resolved to their markers, verses and quotations
   labeled and attributed, running heads and page numbers stripped — plus plain text
   (`all_pages.txt`, `pages.jsonl`).

Runs **entirely on your machine — no API keys.** The OCR engine is
[Surya](https://pypi.org/project/surya-ocr/); what's distinctive here is everything
*after* OCR.

## Why this, not plain OCR

Plain OCR gives you *text* — and a RAG system fed raw OCR text mis-cites and
mis-attributes. This pipeline keeps the structure citations depend on:

- **Footnotes resolved** — `[^N]` in the body links to its note, even endnotes hundreds of pages away.
- **Right voice, right source** — verses/quotes are attributed (not pinned on the author); opponent positions (pūrvapakṣa) are flagged so a refuted view isn't retrieved as the teaching.
- **Clean embedding corpus** — running heads, page numbers, and TOCs dropped; body text in reading order across columns.
- **Scholarly-multilingual** — IAST, Tibetan, Devanagari, verse line breaks, and canonical references survive verbatim.

### Why diacritics drive the design

The real OCR failure here isn't visible garbage — it's *silent* diacritic loss
(`Nāgārjuna → Nagarjuna`): the stripped word still looks plausible, passes a spellcheck,
and quietly breaks author/title matching, citation extraction, and search. A
**confidently-wrong** mark is worse than a missing one — it doesn't fold away in a
diacritic-folded index and it survives validation, so once the scan is the only copy it's
permanent. That failure mode is why the pipeline (a) preserves IAST/Wylie **verbatim**
(NFC, never stripped, never "autocorrected"), (b) **flags** doubtful regions with
per-block `confidence` / `needs_review` instead of silently substituting, and (c) **keeps
the scan image** in the PDF so any page can be re-OCR'd by a better engine later — the
transcription is never the only copy. The right way to judge this OCR is diacritic
*survival*, not dictionary hit-rate.

Details: [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) (the contract) · [`INTEGRATIONS.md`](docs/INTEGRATIONS.md) (reading it).

## Self-contained

No dependency on any surrounding repo — copy or move this folder anywhere. Each entry
script puts its own directory on `sys.path`, the `llama-server` binary is vendored under
`.llama/`, and the only external requirements are the Python packages in
`requirements.txt` (plus Surya's model weights, auto-downloaded on first run). No API
keys are needed — see "External API keys (all optional)" below.

## The scripts

One package, a handful of small top-level entry points. Every one is runnable
directly and puts `lib/` on its own `sys.path`, so you can copy the folder anywhere.

| script | what it does |
|---|---|
| **`run_ocr.py`** | the **workhorse** — OCR one or many books (PDFs, image folders, or catalogue editions) with resume-from-cache, cooperative multi-worker / multi-book scheduling, and live status / kill control. **Start here.** |
| `ocr_one.py` | the single-book core that `run_ocr` drives: OCR a folder or PDF → searchable PDF + text + `doc.json`. Call it directly for a one-off book or to reach an output option. |
| `preprocess.py` | **front stage** — rotate / swap facing pairs / zero-pad page numbers on the raw page images *before* OCR. |
| `create_output.py` | **back stage** — rebuild a book's deliverables from the OCR cache with **no re-OCR** (e.g. to change `--color` or size). Same as `ocr_one.py … --prepare-output`. |
| `build_rag_manifest.py` | build the deduplicated RAG handoff manifest, stamping each book's stable `pub_id` / `content_hash`. |
| `export_corpus.py` | copy every health-checked book's durable artifacts into a portable, self-contained `_ocr_corpus/`. |

### The pipeline is four independent stages

`preprocess → add-pages → OCR → output`. Each has one input and one output and can
run on its own, so you never re-OCR to redo a cheap step. `run_ocr.py` chains them
for you: flags after a `--` go to **preprocess**, flags after a **second** `--` go
to **output** (`create_output`):

```sh
python3 run_ocr.py book.pdf -- --rotate 180:CW --swap 9:end -- --color grey --pdf-max-dim 2000
#                           └────── preprocess ──────┘  └──────── output ────────┘
```

## Quick start

```sh
# ONE book, end to end: PDF or image folder -> searchable PDF + text + doc.json
python3 run_ocr.py "/path/to/Book.pdf"
python3 run_ocr.py "/path/to/scans/"

# smoke-test the first few pages first (first run also downloads model weights)
python3 run_ocr.py "/path/to/Book.pdf" --front-matter 3

# MANY books at once (each isolated — one failure doesn't sink the batch)
python3 run_ocr.py a.pdf "b book.pdf" scans/
python3 run_ocr.py --from-file worklist.txt --skip-done    # resumes finished books from cache

# long runs: wrap in caffeinate so the Mac sleeping doesn't stall them
caffeinate -i -s python3 run_ocr.py "/path/to/Book.pdf"
```

Output lands under `<--out-root>/<book-name>/` (default out-root `out/backfill/`):
`<book-name>.pdf` (searchable), a per-page `.txt`, `all_pages.txt`, `pages.jsonl`,
and the structured `doc.json`. **OCR is cached per page**, so a re-run rebuilds the
text/PDF in seconds — and a kill (Ctrl-C, OOM, `--kill`) is just a *pause*: re-run
the same command with `--skip-done` to continue from cache. `--force` re-OCRs.

For a single book you can also call the core script directly —
`python3 ocr_one.py "/path/to/Book.pdf" --out out/book` — it takes the same input
and every output option documented below; `run_ocr.py` adds batching, resume, and
worker control on top. Status / control while a batch runs: `--list-active` (what's
running), `--list-resumable` (partial, not running), `--kill NAME` (stop one book;
it resumes later).

### PDF input (rasterize → OCR)

`ocr_one.py` natively takes a folder of page images. If you hand it a **`.pdf`**
instead, it first rasterizes every page to a sibling image folder (named after
the PDF) and then proceeds down the normal image path — so the output PDF is
named after the book, not a cache dir. This works for both:

- **image-only (scanned) PDFs** — the embedded page bitmaps are rasterized, and
- **mojibake born-digital PDFs** — the *visually correct glyphs* are rendered to
  pixels and the garbage text layer is discarded; re-OCR'ing the pixels is
  exactly how rasterize-then-OCR repairs mojibake.

Pages render at `--dpi 300` by default (zoom = dpi/72). Keep it high — Surya
upscales internally and low-res input mangles diacritics; file-size trimming
happens later in PDF assembly, not here. Rasterization is incremental (cached
sibling pages are reused unless `--force`).

```sh
# auto-detect: PDF in, searchable PDF + text out
caffeinate -i -s python3 run_ocr.py "/path/to/Book.pdf"

# render only (no OCR) — standalone, e.g. for batch prep
python3 lib/pdf_to_images.py "/path/to/Book.pdf"          # -> Book/
python3 lib/pdf_to_images.py /folder/of/pdfs -o out/pages # batch: one <stem>/ each
```

## Install (one time)

```sh
pip install -r requirements.txt   # surya-ocr, PyMuPDF, Pillow
```

Surya runs its OCR model as a **VLM served by `llama-server` (llama.cpp)** — there
is no in-process MPS path on macOS, so a `llama-server` binary is required. A
Metal (arm64) build is vendored under `.llama/` and found automatically (it also
honors `LLAMA_CPP_BINARY` or any `llama-server` on `PATH`). To refresh it,
download the latest `*-bin-macos-arm64.tar.gz` from
https://github.com/ggml-org/llama.cpp/releases, extract into `.llama/`, and run
`xattr -dr com.apple.quarantine .llama`. The first OCR run downloads the model
weights (a few GB) from Hugging Face into `~/.cache/huggingface`.

## External API keys (all optional)

**None are required.** The pipeline runs entirely on your machine — OCR happens
locally through the vendored `llama-server`, and there are no calls to any paid or
cloud API, so you never need to set up an account or a key to use it.

The only thing it fetches from the internet is Surya's OCR **model weights**, pulled
once from Hugging Face on the first run. Those weights are **public**, so no token is
needed. The one optional credential:

| key | needed? | what for |
|---|---|---|
| `HF_TOKEN` | almost never | The `huggingface_hub` library reads this env var if it's set. You only need it if your network blocks anonymous downloads, you hit Hugging Face's anonymous rate limit, or you point Surya at a private/gated model mirror. For the normal public weights, leave it unset. |

Everything else the pipeline reads from the environment (`LLAMA_CPP_BINARY`,
`SURYA_INFERENCE_URL`, `CATALOGUE_DB`, `OCR_CORPUS_DIR`, the `SURYA_*` tuning knobs)
is **local configuration — paths and settings, not credentials**. The optional
catalogue integration below uses a local SQLite file, not an API.

## Optional: catalogue integration (`run_ocr.py`)

`ocr_one.py` (above) is fully self-contained and needs nothing external. The
separate `run_ocr.py` multi-book driver adds an **optional** hook to resolve
`--editions ID` and stamp a stable `pub_id` from a sibling library-catalogue
SQLite DB. That DB is **not** part of this repo and is not required: run on
PDFs or image folders directly and the catalogue is never touched (manifest
rows simply get `pub_id=null`). Point at your own DB with `--db` or
`$CATALOGUE_DB` only if you have one.

## Driver options (`ocr_one.py`)

| flag | default | meaning |
|---|---|---|
| `image_dir` | *(positional)* | folder of page images (jpg/png/tif), **or a `.pdf`** (auto-rasterized to a sibling image folder first) |
| `--out DIR` | *(required)* | output directory |
| `--dpi N` | `300` | when the input is a PDF, page render resolution (zoom = dpi/72); keep high (300+) — Surya upscales and low-res mangles diacritics |
| *(page-number padding)* | on | trailing page numbers are zero-padded to a uniform width so `p9.jpg`/`p200.jpg` sort right. This now lives in the **preprocess** stage (`preprocess.py`, on by default; opt out with `--no-pad-page-numbers`) so every stage sees one canonical numbering — see "The scripts" above |
| `--backend NAME` | `surya` | OCR engine (see "Pluggable backends") |
| `--instances N` | `3` | parallelism hint forwarded to the backend |
| `--backend-opt KEY=VALUE` | — | extra engine-specific option (repeatable) |
| `--lines detect\|split` | `detect` | PDF line boxes: per-line from the backend, or band-slice OCR blocks |
| `--text full\|body\|both` | `both` | text to emit: full, body (drop headers/footers/page numbers), or both (full in `<out>/full/`, body in `<out>/body/`); PDF always keeps the full layer |
| `--font PATH` | Arial Unicode | font for the invisible PDF text layer (see below) |
| `--drop-nonlatin` / `--no-drop-nonlatin` | on | blank pages whose OCR is mostly non-Latin script (hallucinated garbage — see below); scan image kept, only the text dropped |
| `--nonlatin-threshold F` | `0.5` | non-Latin-letter fraction above which a page is treated as hallucinated |
| `--pdf-size small\|medium\|large` | — | **preset** bundling the next three knobs: small=1600px, medium=2000px, large=0 (full res); all grey at quality 90. Overrides `--pdf-max-dim`/`--pdf-image-quality`/`--color` |
| `--color bw\|grey\|color` | `bw` | embedded-image color depth (size: bw small / grey medium / color large), no sharpness cost. Cover (page 1) always stays color |
| `--pdf-max-dim PX` | `1200` | cap images to PX on the long edge (size/sharpness: 1200 small / 2000 medium / 0 = full-res largest & sharpest) |
| `--pdf-image-quality Q` | `90` | JPEG quality (1–95) for color/grey images *when re-encoded* (size: ~60 small / 75 medium / 90 large). N/A to `--color bw` |
| `--all-pdf-sizes` | off | build all three `--pdf-size` presets at once (`<name>.small/medium/large.pdf`), in parallel |
| `--contrast F` | `1.25` | contrast boost for text pages (1.0 = none); cover left untouched |
| `--no-uniform-pages [I,J,K-L]` | *(uniform is on)* | uniform page sizing is **on by default** — every PDF page is the same size (scans scaled-to-fit, centered) so unequal-size facing pages don't jump; pass `--no-uniform-pages` to turn it off (optionally for only pages `I,J,K-L`) |
| `--limit N` | — | first N pages only (smoke test) |
| `--reocr FILE …` | — | force re-OCR of these page files (by name, ±extension) even if unchanged; other pages stay cached |
| `--force` | off | re-OCR *every* page, ignoring the cache |

### Incremental OCR (only re-OCR what changed)

OCR is cached per page in `results.json` / `detection.json` (keyed by filename).
A normal run re-OCRs **only the pages that are new or whose file was modified
since the cache** (compared by mtime), merges them in, and reuses the rest — so
if you rotate or rescan one page and rebuild, it re-OCRs just that page (~15–25 s)
and rebuilds the PDF, not the whole book. `--reocr "p-012.jpg" "p-013.jpg"` forces
specific pages even if unchanged; `--force` re-OCRs everything.

### Non-Latin / hallucination filter

Surya can't read non-Latin script (Tibetan, Devanagari, Odia, …) — on such a page
the VLM **hallucinates** other Indic scripts and loops, producing garbage text
(and burning decode time). `--drop-nonlatin` (on by default) blanks any page whose
OCR text is mostly non-Latin: the **scan image stays in the PDF**, only the bogus
text layer / text-file content is dropped, so you never store fabricated text.
Real Latin/IAST pages are untouched. `--no-drop-nonlatin` keeps everything.

### File size (without sacrificing sharpness)

Two levers shrink the PDF, and they cost different things:

- **`--color`** removes colour depth — and costs *nothing* in sharpness (text has
  no useful colour). This is the lever to use. At full resolution on a 200-page
  book: `bw` ≈ 14 MB, `grey` ≈ 110 MB, `color` ≈ 164 MB. So **`--color bw`
  (default) at full resolution is small *and* sharp.** (bw is lossy for
  photos/halftones; the cover (page 1) is always kept in colour.)
- **`--pdf-max-dim`** removes *pixels* (resolution) — which **does** cost
  sharpness (this is the usual cause of blurry pages). Default is 1200 px; pass
  `--pdf-max-dim 0` for full-resolution (sharpest), and only lower it if a file is
  still too big after choosing `bw` and you'll accept some softness.

`--pdf-size small|medium|large` is a one-flag preset (1600/2000/0 px, grey,
quality 90); `--all-pdf-sizes` builds all three of those presets at once (in
parallel).
Embedded fonts are auto-subset, so even tiny PDFs aren't bloated by the ~15 MB
Arial Unicode. Because OCR is cached, changing any of this and rebuilding takes
seconds — no re-OCR.

**Contrast.** `--contrast F` (e.g. 1.5) darkens faded/grey text to crisp
black-on-white (text pages only; cover untouched) — separate from resolution.
(Alternatively,
post-process any finished PDF with Ghostscript, which keeps the text layer:
`gs -sDEVICE=pdfwrite -dPDFSETTINGS=/ebook -o small.pdf big.pdf`.)

### About `--font`

The PDF text layer is **invisible** (it sits over the scan image for
selection/search), so the font's appearance is irrelevant — what matters is
**glyph coverage**. A character can only enter the searchable layer if the chosen
font has a glyph for it; anything missing is dropped (counted on stderr) and
becomes unsearchable. The default Arial Unicode covers IAST diacritics + Tibetan +
Devanagari; only override it with a *wider*-coverage font, never a prettier one.

## Code layout

The top-level scripts are listed in "The scripts" above (`run_ocr.py`,
`ocr_one.py`, `preprocess.py`, `create_output.py`, `build_rag_manifest.py`,
`export_corpus.py`); the modules they share live in **`lib/`**
(`surya_backend.py`, `ocr_backend.py`, `ocr_surya.py`, `parallel_ocr.py`,
`ocr_to_text.py`, `ocr_to_pdf.py`, `pdf_to_images.py`, the preprocess helpers
`pad_pages.py` / `rotate_images.py` / `swap_pages.py` / `add_pages.py`, plus the
identity/corpus/doc modules `identity.py` / `corpus.py` / `docjson.py`). Each
top-level script puts `lib/` on `sys.path`, and every `lib/` script is also
runnable directly (e.g. `python3 lib/ocr_to_pdf.py …`,
`python3 lib/pdf_to_images.py Book.pdf`).

## Pluggable backends

The OCR engine is swappable. The pipeline talks to a **backend** that turns page
images into a common block model; everything after that (the text and PDF assembly)
only sees that model, so you can drop in a different engine — Tesseract, a cloud
OCR — without touching the rest. Surya is the built-in one.

### Technical details

The seam: `lib/ocr_backend.py` defines the `Block`/`Page` model, the `OcrBackend`
ABC (`ocr_pages()` required; `line_boxes()` optional → `None` falls back to
band-slicing), and a name→class registry (`register` / `get_backend` / `available`).
`lib/ocr_surya.py` is `SuryaBackend`, the current implementation, wrapping the
parallel orchestration in `lib/parallel_ocr.py`. Each backend owns its own
parallelism, scaling, and caching; generic options (`--instances`, `--backend-opt`)
are forwarded to it, and it reads what it understands and ignores the rest.

To add an engine: write a module with a `@register("name")` class implementing
`ocr_pages` (and `line_boxes` if it has word/line boxes), add one `import` to
`ocr_one.py`, and `--backend name` works — no change to the driver or assembly.

Surya-specific options go through `--backend-opt`:

- **`detector_batch=N`** — how many page images the text-line detection model
  (`surya_detect`) runs per GPU forward pass. Bigger = better GPU use but more
  memory; default is Surya's per-device value (Metal/MPS 8, CUDA 36). On Metal 8 is
  fine — detection is a small model (~2 s/page), not the bottleneck. Affects
  `--lines detect` only. Example: `--backend-opt detector_batch=16`.
- `instances=N` — same as `--instances` (parallel OCR servers; default 3). This is
  the real speed lever (see Performance).

## Performance — parallelism is the speed lever

Speed comes from running several OCR workers at once. **Three is the sweet spot**
(`--instances 3`, the default): it keeps the GPU busy without oversubscribing it,
and going higher actually gets *slower*. A dense ~200-page book takes roughly
50 min on one worker and ~26 min on three — and the result is cached, so any later
rebuild is seconds. Lowering image resolution does **not** help (see below), so
just leave the default of three and let the cache do the rest.

### Technical details

OCR cost is dominated by autoregressive text **decode**, and a single
`llama-server` leaves the GPU **~40% idle** (its vision pipeline serializes), so
raising `--parallel` slots within one server does nothing. The fix is running
several independent server instances. Measured on an M5 Max (40-core GPU, 128 GB),
same workload per instance:

| instances | GPU util | throughput vs 1 |
|--:|--:|--:|
| 1 | ~40% | 1.00× |
| 2 | ~64% | 1.70× |
| **3** | ~82% | **1.95× (peak)** |
| 4 | ~86% | 1.62× (regresses) |

So 3 is the peak; 4+ saturates the GPU and regresses. (Surya normally shares one
server via a sentinel/lock; `parallel_ocr.py` launches the instances itself and
pins each shard with `SURYA_INFERENCE_URL`.)

Lowering image resolution is **not** a useful lever: Surya internally upscales
inputs to a fixed working resolution, so downscaling barely speeds anything up
(~1.09×) while it mangles diacritics.

## Caching

OCR is the only slow step, so its result is cached and reused. Once a book is OCR'd,
rebuilding the PDF or text from that cache takes seconds; `--force` throws the cache
away and re-OCRs.

### Technical details

`SuryaBackend` caches OCR/detection under `<image_dir>/.surya_ocr/` (`results.json` /
`detection.json`, with a `.subsetN` suffix when only some pages are processed) and
reuses them when fresh (cache mtime ≥ newest image) unless `--force`.

## Full vs body text

surya-ocr-2 is layout-aware — it labels every region (Text, SectionHeader,
PageHeader, PageFooter, ListGroup, Footnote, …). The **body** text variant drops
the non-body regions (running headers/footers, which carry the page numbers);
**full** keeps everything. By default (`--text both`) you get both, in
`<out>/full/` and `<out>/body/`. Tune what "body" drops with
`--drop-labels PageHeader,PageFooter,Footnote`. Every block's `label` is recorded
in `pages.jsonl` regardless, so you can also filter downstream.

## Lower-level scripts

`lib/ocr_to_text.py` and `lib/ocr_to_pdf.py` are the original single-process entry
points (no multi-instance parallelism); they share the `build_text` / `build_pdf`
assembly functions with `ocr_one.py`. Use `run_ocr.py` / `ocr_one.py` for normal work.

## The RAG handoff (structured `doc.json`)

Beyond the searchable PDF + plain text, each book gets a structured **`doc.json`**
— per-block layout labels, regions, verses, footnotes — for downstream LLM
chunking/embedding. Two scripts package it for a consumer:

- `python3 build_rag_manifest.py` — writes a **deduplicated** manifest (one row per
  book, stamped with a stable `pub_id` / `content_hash`) so a consumer reads one
  file instead of walking the output tree.
- `python3 export_corpus.py` — copies every health-checked book's durable artifacts
  into a portable, self-contained `_ocr_corpus/` with relative paths.

The `doc.json` contract and the RAG-handoff architecture are described in
`docs/ARCHITECTURE.md`; how a consumer *reads* `doc.json` (type policy, the
`lib/docjson.py` reference reader) is in `docs/INTEGRATIONS.md`.

## Known limitations & planned enhancements

The structured `doc.json` path (see `docs/ARCHITECTURE.md`) has two known gaps,
both currently handled by **flagging `needs_review`** rather than fixing at the
source:

- **Merged multi-column blocks (glossaries / indexes / dictionaries).** When
  Surya linearizes several columns into one block, the text *inside* that block
  is scrambled and block-level reading-order can't fix it. We detect it via an
  internal vertical whitespace gutter (`reading_order.multicolumn_indices`) and
  set `needs_review`. *Planned fix — column-strip re-OCR:* crop each detected
  column and OCR it separately, then concatenate in column order, removing the
  scramble at the source (a `--columns reocr|flag|off` knob; flag is today's
  default).
- **Hand annotations (pencil/pen underlining, highlighting).** Verified adequate
  on real annotated pages (the rule detector rejects wavy underlines; Surya read
  underlined text correctly at 300 DPI). *Planned hardening* if heavier marks
  appear: straightness + left-margin checks on the footnote-rule detector, and a
  pre-OCR de-annotation pass (drop grey graphite / saturated colored pen).

## Notes

- OCR is the only expensive step and is cached; everything else is fast.
- Long runs are wrapped in `caffeinate` so the Mac sleeping won't stall them.
- Surya CLI flag/name drift is handled by trying several command forms; if all
  fail, the error reminds you to `pip install surya-ocr`.

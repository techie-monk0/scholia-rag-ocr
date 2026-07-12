# Consuming doc.json — client usage guide

How a client (e.g. the RAG ingestion / embedding task) consumes the OCR
pipeline's output. The contract itself is specified in `ARCHITECTURE.md`;
this file is the consumer-facing "how to read it" guide.

## Where the output comes from (producer scripts)

The pipeline (`run_ocr.py` / `ocr_one.py` — see `README.md`) writes one book dir
per source under the output root, each containing a **`doc.json`** (the structured
contract this guide is about) alongside the searchable PDF and plain text. Two
scripts package those for a consumer:

- **`build_rag_manifest.py`** → a **deduplicated** manifest, one row per book (each
  stamped with a stable `pub_id` / `content_hash`), so you read one file instead of
  walking the tree. Default output `out/rag_handoff_manifest.jsonl`.
- **`export_corpus.py`** → a portable, self-contained `_ocr_corpus/` of every
  health-checked book's durable artifacts, indexed by `_ocr_corpus/manifest.jsonl`
  with paths relative to that folder.

See `ARCHITECTURE.md` §"The RAG handoff" for the handoff overview. Everything below is about reading a
`doc.json` once you have it.

## The output: one `doc.json` per book, no sidecar

Everything about a book lives in a single file. There is no separate sidecar — the
verses, geometry, regions, and all other structure are merged right into each
block. To find the books to process, read the manifest (default
`out/rag_handoff_manifest.jsonl`) that `build_rag_manifest.py` writes: one line per
book, already deduplicated, each pointing at that book's `doc.json` and PDF.

### Technical details

The doc declares two top-level fields so consumers can evolve safely:

```jsonc
{ "schema_version": "1.2",
  "features": ["indent","centered","verse","quote","ref_label","printer_key",
               "copyright","list_item","toc","region","running_head",
               "title_page","dedication","colophon","epigraph"],
  "doc_id": "...", "title": "...", "source": "...",
  "blocks": [ ... ], "footnotes": { ... } }
```

Each manifest line carries `schema_version` + `features`, a stable `pub_id` /
`content_hash`, and paths (`doc_json`, `pdf`, …). There is no `verses_json` field.

## The compatibility guarantee (additive-only)

The format only ever grows — nothing is renamed, removed, or repurposed. So new
producer versions won't break an old consumer, and a new consumer still reads old
documents. You never have to "upgrade" to keep working; you opt in to new fields
only when you want to read them.

### Technical details

Core block keys (`id`, `type`, `text`, `page`, `bbox`, `confidence`,
`needs_review`, `refs`, `marker`, …) never change meaning. New information always
arrives as **new optional keys** and **new `type` values**, gated by a `features`
entry (and, if the shape grows, a `schema_version` bump).

## What the client must do

1. **Drop any sidecar code.** Read verse/quote/region/etc. from the block itself.
2. **Read optional keys with `.get(key, default)`** — never index. `region`,
   `indent`, `centered`, `ref_label`, `line_bbox`, `continuation`, and anything
   added later are optional and may be absent on a block or in an older doc.
3. **Check `features` before relying on a layer.** `"region" in features` tells you
   the doc was built by a pipeline that computes regions — so a missing `region` on
   a block means "computed, none/body", not "unknown". Absence of the feature means
   the doc predates it (don't infer it).
4. **Treat `type` as an OPEN vocabulary.** Switch on the types you know; for an
   **unknown** `type`, fall back to a safe default (treat as `body`-like content,
   not drop) so a future type never silently disappears or crashes a strict enum.
5. **Expect a heterogeneous corpus during rollout.** Books built before a feature
   landed won't list it; key behaviour off `features`/`schema_version`, not the
   filename. Re-running the output stage (`create_output.py` / `ocr_one.py
   --prepare-output`) rebuilds older docs from cache, and re-running
   `build_rag_manifest.py` refreshes the manifest — no re-OCR needed.
6. **Bump nothing to receive additions.** Because changes are additive +
   feature-gated, new fields appear without breaking you; opt in only when you read
   them.

## Use the reference reader — don't hand-roll parsing

`lib/docjson.py` implements all of the above as a **read-only, stdlib-only** access
layer. Use it or vendor it (single file, no dependencies) rather than
re-implementing defensive parsing and drifting from the contract.

```python
from docjson import Doc, load_manifest

# Iterate finished books from the manifest
for row in load_manifest("out/rag_handoff_manifest.jsonl"):
    doc = Doc.from_file(row["doc_json"])

    # Clean text for embedding: content blocks only (furniture + notes dropped),
    # body region only — all in reading order.
    text = doc.text(region="body")

    # Verses (verse/quote/epigraph) with their structure
    for v in doc.verses():
        index_verse(v.text, ref=v.ref_label, lines=v.line_bbox, centered=v.centered)

    # Footnote backlink: which block carried a given note
    src = doc.note_source_block("p5:1")     # -> Block or None

    # Gate on capability when needed
    if doc.has_feature("region"):
        front = [b for b in doc if b.region == "front"]
```

Guarantees the layer enforces for you: unknown `type` is **kept as content** (never
silently dropped); missing keys get **safe defaults** (`region` → `"body"`,
`centered` → `False`, …); region filtering only applies when the doc declares the
feature; and the raw dicts stay reachable via `Block.raw` / `Doc.raw` — the layer
never hides data.

## Citation cues — producer signals, consumer adjudicates (feature `citation_cues`)

When the producer isn't sure whether a block is a quotation, it doesn't guess — it
leaves the block as `body` but attaches **hints** ("the line before this ends in a
colon", "this block ends with a source in parentheses") and lets you make the call.
The payoff is *attribution*: even when OCR flattened a verse's line breaks, these
hints tell you "this is the *Samādhirāja Sūtra* speaking, not the author," which is
the part that matters for a citation. Read the hints rather than re-deriving your
own; fall back to your own only when a doc lacks the feature.

### Technical details

The hint fields (produced by the shared `lib/citation_cues.py`, so OCR and
digital-first producers emit identical signals — vendor it like `docjson`):

- `preceded_by_colon` / `lead_in_verb` — the previous (reading-order) block is a
  quote lead-in (`"As Śāntideva says:"`).
- `trailing_source` — an attribution at the block's end (`"(MMK 24.18)"`, `"— Nāgārjuna"`),
  also copied into `ref_label`.
- `quote_candidate` (+ `needs_review`) — a `body` block the geometry missed but the
  cues bracket (e.g. a verse whose line breaks OCR flattened). **This is your set to
  adjudicate:** `quote_candidate` + agreeing cues → treat as `quote`, pull `ref_label`.

## Type policy — what to drop, exclude, or keep

The OCR stage labels every block but deletes nothing — *you* decide what reaches the
embedding corpus. Roughly: keep the real content, drop the repeated furniture
(running heads, page numbers, title/copyright pages), and set navigation (contents,
index, bibliography) aside — keep it for provenance but don't embed it. Because the
raw text is always kept under a label, changing your mind later is a re-chunk, not a
re-OCR. And treat any `type` you don't recognize as content — never drop what you
don't recognize.

### Technical details

`lib/docjson.py` encodes this policy — `content_blocks()` drops furniture + notes;
`FURNITURE_TYPES` / `VERSE_TYPES` enumerate it. The full mapping:

- **content** (chunk, with policy): `body` / `heading` / `table` / `caption` /
  `list_item` / `glossary` / `epigraph` / `footnote` / `endnote`.
- **furniture** — **DROP** from embeddings: `page_header` / `page_footer` /
  `page_number` (repeated "528 Steps on the Path" is noise that dilutes embeddings).
- **navigation** — **EXCLUDE** from embeddings, keep for provenance: `toc` /
  `index` / `bibliography` (pure locators, zero semantic value).
- **front/back furniture** — **DROP**: `title_page` / `copyright` / `colophon` /
  `dedication`.
- **non-text / junk**: `figure` / `drop`.

## What each downstream phase uses from the contract

A RAG pipeline consumes the same `doc.json` in three stages — **chunking** (grouping
blocks into retrieval units), **embedding** (turning each unit into a vector), and
**answering** (reranking, generation, and citation). Each stage leans on different
fields; the lists below say which ones and what breaks if a field is missing, so you
know what to depend on. (The contract itself is in `ARCHITECTURE.md`.)

### Chunking (structure-aware chunker v2)
The chunker turns blocks into retrieval units. It needs, in priority order:
- **Whole-book reading order** + the **`type` taxonomy** — to pack body, keep
  verses/tables whole, and exclude furniture/nav. Scrambled order is fatal
  (incoherent chunks).
- **`heading` blocks with `level`** — to build the heading stack and prepend
  breadcrumbs (`§A › B › C`) to each chunk's *embedded* text. Cheap, large
  retrieval win.
- **`verse`/`quote` labels + `ref_label` + preserved line breaks +
  `indent`/`centered`** — verses are emitted as atomic, citable chunks; the
  geometry is how they're found. A flowed verse is unrecoverable downstream.
- **Citation markers, made explicit.** The chunker must NOT guess markers from raw
  text. Markers render as **bracketed `[N]`** or **superscript digits**, and `[…]`
  is also used for **editorial insertions** (translator's `[Valid cognitions…]`)
  which are NOT markers. So: (a) every referenced marker is listed in the body
  block's **`refs`** (and on `verse`/`quote` blocks too — a verse's citation marker
  often sits on the verse); (b) **editorial `[…]` is preserved verbatim** so you can
  tell them apart; (c) the resolved **`footnotes` map** makes marker → citation body
  a dict hit, not a parse. Citation bodies feed canonical references (`Suhṛllekha
  vv. 65–66, Derge D4182`); resolve `ibid./op. cit.` anaphora across the note
  sequence (`bdrag/citations.py`), so notes arrive in reading order.
- **`table.format`/`content`** kept intact — one chunk per table, never
  prose-flattened.
- **`confidence` + `needs_review`** — to quarantine/down-weight hallucinated regions.

### Embedding
Embeddings are computed over each chunk's text (breadcrumb + passage):
- **Unicode-normalized text** exactly per the spec (NFC, soft-hyphen/zero-width
  removal, ligatures, quote/dash policy). Mojibake and stray furniture **dilute
  vectors**.
- **Diacritics preserved** (`śūnyatā` verbatim) — folding happens later in the
  *lexical* index, not here; the embedder must see the true form.
- **Furniture/nav excluded** (labeled so the chunker drops them) — repeated running
  heads embedded hundreds of times are pure noise.
- **Verse line breaks preserved** — they carry meaning the embedder should see.
- **Low-confidence text quarantined** — confidently-wrong OCR poisons a vector
  worse than missing text. Embeddings re-compute only for *changed* chunks, so
  **stable block text** keeps re-embed cost bounded across re-OCR passes.

### Reranking, generation & citation (answer-facing phases)
These produce the cited answer the user reads:
- **`page` on every block** → human citations ("p268"); **`ref_label`** → scholarly
  canonical refs.
- **`chapter` + heading path** → "§Chapter › Section" citation locations and endnote
  scoping.
- **`block_id` provenance** → backlinks (a footnote chunk links to the body block
  that referenced it; a verse links to its source).
- **Resolved footnote bodies** → the generator quotes the actual citation, and
  `verse`/`quote` attribution (`ref_label`) prevents **misattributing a quoted sutra
  to the book's author**.
- **`needs_review`/`confidence` surfaced** → the doc-level quality gate can flag
  shaky sources before they reach an answer.

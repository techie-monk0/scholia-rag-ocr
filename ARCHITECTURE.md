# Architecture — structured OCR output for RAG/LLM

This pipeline's real job isn't "get the text off the page" — it is to **recover the
structure of a book and label every piece of it**, so a downstream RAG/LLM system can
chunk, search, and cite it well. The output is one JSON file per book, `doc.json`,
recording the reading order, headings, footnotes, verses, tables, and more.

Two kinds of book feed into the same format:

- **scanned** books, read by OCR (Surya), and
- **digital-first** books, whose text is pulled straight from the PDF's own text layer.

Both end up as the same `doc.json`, so a consumer treats them identically. How a
consumer *reads* `doc.json` is a separate guide, **`USING_THE_OUTPUT.md`**; this
document is the *design* — what goes in the format and how the structure is recovered.

## Why this shape — the failure mode it's built around

Structure is one of the two things this pipeline protects; **text fidelity** is the
other, and it drives the choice of tools as much as the structure work does. For
scholarly Buddhist texts the dangerous OCR failure is **not** visible garbage — you can
see that and re-run it. It is **silent, systematic loss of diacritics**:
`Śāntideva → Santideva`, `Nāgārjuna → Nagarjuna`, `dharmakāya → dharmakaya`. The
stripped word still looks like a plausible word, so it survives a spellcheck/dictionary
pass and reads as authoritative — while it quietly poisons everything downstream that
depends on the exact string: author/title matching, citation and terminology extraction,
cross-reference resolution, and search. And a **confidently-wrong** character is worse
than a missing one: a dropped mark folds away harmlessly in a diacritic-folded search
index and can be recovered by re-OCR, but a wrong mark does not fold, survives
validation, and — once the scan is the only surviving copy — becomes permanent.

That single fact is why the pipeline is shaped the way it is:

- **Preserve, never "clean."** IAST diacritics and Wylie transliteration are carried
  through verbatim — NFC-normalized, never stripped, never "autocorrected" into
  English-looking words (`bsgrubs` and `rnam par shes pa` are not typos; a language
  prior that helps ordinary English *hurts* here). This is the reason for the strict
  normalization spec below and for choosing an engine tuned for Latin+IAST fidelity.
- **Judge OCR on diacritic *survival*, not dictionary hit-rate.** A page can score
  "clean" on a dictionary check while every diacritic is wrong, so a dictionary score is
  the wrong yardstick — the right question is whether `ā ī ū ṛ ṃ ḥ ṅ ñ ṭ ḍ ṇ ś ṣ`
  actually survived.
- **Flag, don't silently substitute.** Per-block `confidence` + `needs_review` (item 7
  below) exist so a doubtful region is surfaced for review rather than quietly
  "corrected" into a confident error.
- **Keep the scan image so the choice stays reversible.** The searchable PDF retains the
  page bitmap, so any page can be re-OCR'd by a better/higher-recall engine later — the
  transcription is never the only copy. This is the cheap insurance that makes the engine
  decision reversible.

Automated escalation on top of this — a per-page router that detects diacritic-dense
pages and sends only those to a higher-recall (e.g. cloud) engine, plus a
valid-IAST-only post-filter to catch an engine's spurious non-IAST marks — is a
deliberate **extension point**, not part of this local-only repo. The retained-image
insurance is what keeps it addable after the fact.

## Guiding principle — recover and label, don't flatten

The goal is **not "clean text."** It is **structure-preserving, labeled output.** The
OCR stage's job is to recover and label structure (reading order, headings, footnotes,
tables, confidence); the consumer's job is to render it into retrieval units. So:

- **Don't flatten** — no prose-merging of tables, columns, or footnotes.
- **Don't discard** — footnotes, indexes, and headers are *labeled*, not deleted; the
  consumer decides what to keep.
- **Don't pre-chunk** — emit blocks + structure and let the consumer choose chunk
  boundaries.

If a choice is lossy, push it downstream — keep the richer representation here.

## What matters most (ranked by impact on final answers)

1. **Reading order on complex layouts** (multi-column, sidebars, glossaries).
   Scrambled text is *fatal* — the chunk becomes incoherent. Highest priority.
2. **Section-header hierarchy** — lets each chunk carry a breadcrumb ("§Joyous
   Perseverance › How to Begin"), one of the biggest retrieval wins, and nearly free
   given Surya's `SectionHeader` label.
3. **Footnote / endnote association** — link a marker in the body to the note's text
   (which, for endnotes, can live hundreds of pages away). High value for a scholarly
   tool, where citations are primary-source references.
4. **Furniture labeling** (running heads/feet, page numbers) — so they can be dropped
   from the embedding corpus; a repeated "528 Steps on the Path" is noise that dilutes
   embeddings.
5. **Table structure** — emit HTML or markdown, never prose-flattened (that destroys
   row/column meaning).
6. **Unicode normalization** — consistent, clean text (spec below).
7. **Per-block confidence** — so the consumer can drop or flag hallucinated regions;
   confidently-wrong text is worse than missing text.
8. **Navigation labeling** (contents / index / bibliography) — so they can be excluded
   from embeddings (pure locators) while staying recoverable.

## The output: one `doc.json` per book

Each book becomes a single JSON document: a list of **blocks** in reading order — each
block is one labeled piece (a heading, a paragraph, a verse, a footnote, a table…) —
plus a **footnotes** map that resolves note markers to their text. Most block fields
already exist in the per-page `pages.jsonl`; `doc.json` adds the whole-book reading
order, the richer labels, and the resolved footnotes.

There is **no sidecar** — verse/quote, geometry, region, and all structure live inside
`doc.json`. It carries a `schema_version` and a `features` list so consumers can evolve
safely; changes are **additive-only** (see `USING_THE_OUTPUT.md`).

### Technical details — the `doc.json` schema

Top level:

```jsonc
{
  "doc_id": "<stable id, e.g. content hash>",
  "source": "<original file path>",
  "title": "<book title if known>",
  "blocks": [ /* Block[], in READING ORDER across the whole book */ ],
  "footnotes": { "<scope>:<marker>": { /* Footnote */ } }
}
```

**Block** — the unit of structure; ordered = reading order:

```jsonc
{
  "id": "b0421",                  // stable, unique within the doc (footnotes reference it)
  "type": "heading | body | verse | quote | epigraph | footnote | endnote | table | caption |
           list_item | glossary | page_header | page_footer | page_number | toc | index |
           bibliography | title_page | copyright | colophon | figure | drop",
  "text": "...",                  // reading-order-correct, unicode-normalized (see spec).
                                  // For `verse`: PRESERVE internal line breaks (\n per line).
  "ref_label": "MMK 24.18",       // verse number / canonical ref, or a quote's source, if present
  "page": 268,                    // 1-based source page
  "region": "body",              // front | body | back — macro-region (see "Front/back matter…").
                                  //   In front/back an indented/centered short block is furniture
                                  //   or nav, NOT verse, unless explicitly typed verse/quote/epigraph.
  "level": 2,                     // heading depth (1 = top); null for non-headings
  "chapter": "13",                // chapter the block belongs to (endnote scoping); null if unknown
  "refs": ["316", "317"],         // note markers REFERENCED in this block (body blocks only)
  "marker": "316",                // for footnote/endnote blocks: the note's OWN number/symbol
  "table": { "format": "html", "content": "<table>…</table>" },  // table blocks only
  "bbox": [x0, y0, x1, y1],       // source coords (debug/review); optional
  "indent": 0.07,                 // left offset from the COLUMN body margin, normalized 0..1 by
                                  //   column width (NOT the page edge). The verse/quote signal.
  "centered": false,              // true if L/R gaps are symmetric (centered verses/titles)
  "line_bbox": [[x0,y0,x1,y1]],   // OPTIONAL per-line boxes; emit only on verse/ambiguous blocks
  "confidence": 0.98,             // per-block OCR confidence 0..1 (use 1.0 for born-digital text)
  "needs_review": false,          // true if low confidence OR uncertain layout reconstruction
  // --- citation-cue SIGNALS (producer emits; CONSUMER adjudicates verse/quote/body) ---
  "preceded_by_colon": true,      // previous (reading-order) block ends in ':'
  "lead_in_verb": true,           // previous block is a citation lead-in ("As X says:")
  "trailing_source": "(MMK 24.18)", // attribution at this block's end (also -> ref_label)
  "quote_candidate": true,        // body block the geometry MISSED but cues bracket
  // --- argument-structure SIGNALS (producer emits; CONSUMER adjudicates voice) ---
  "objection_opener": true,       // block STARTS with an opponent-position opener
  "refutation_pivot": false       // block STARTS with a refutation turn
}
```

**Footnote** — a doc-level map of resolved note bodies, so the consumer can look up a
marker → body:

```jsonc
{
  "marker": "316",
  "kind": "footnote | endnote",
  "text": "Suhṛllekha, vv. 65–66, Derge D4182, folio 40b.",
  "page": 268,
  "chapter": "13",                // resolution scope (see rules)
  "block_id": "b0421"             // the source footnote/endnote block
}
```

The citation-cue and argument-structure fields are **signals**: the producer emits them
and the *consumer* decides what a block actually is. They are computed in a shared,
vendorable cue library (`lib/citation_cues.py`) so every structure producer — Surya-OCR
and the digital-first readers — emits identical signals. The producer commits
`verse`/`quote` only on high-confidence geometry; blocks the geometry missed but the
cues bracket are left `body` + `quote_candidate` + `needs_review` for the consumer to
adjudicate.

## Reading a `doc.json` (the consumer side)

How a client reads the output — the additive/feature-gated guarantee, reading optional
keys safely, treating `type` as an open vocabulary, and the reference reader
`lib/docjson.py` — lives in **`USING_THE_OUTPUT.md`**. In short: there is no
`.verses.json` sidecar, changes are additive-only, and clients should use or vendor
`lib/docjson.py` rather than hand-rolling defensive parsing.

## How the structure is recovered

The rest of this document is the *how*: the specific problems the OCR stage solves and
the algorithms behind each field.

### Footnotes and endnotes

A body block lists the note markers it references; the consumer resolves each to the
note's text via the `footnotes` map. The trick is that markers renumber — footnotes
restart per page, endnotes restart per chapter — so the map key has to say *which*
page or chapter, or two different notes both called "5" collide.

Failure is benign: a note that can't be matched simply keeps its own raw text as its
own block (nothing is lost, it just isn't linked) and gets `needs_review`.

#### Technical details — keying and matching

Keys are **scope-qualified**:

- **Page-bottom footnotes** renumber per page → key = `"p<page>:<marker>"` (e.g.
  `"p268:5"`).
- **Endnotes** renumber per chapter → key = `"ch<chapter>:<marker>"` (e.g. `"ch13:316"`).
- Put the scope on each Footnote (`page` for footnotes, `chapter` for endnotes) and
  build the map key from it. Body blocks carry `chapter` so the consumer can form the
  lookup key.

Detection: page-bottom footnotes are high-feasibility (Surya's `Footnote` label → parse
the leading number → join to the marker on the same page). Endnotes are medium — they
read as `Text`/`SectionHeader`, so detect the "Notes" / "Notes to Chapter N" section
near book-end, parse its numbered entries, and join by (chapter, number).

Fallbacks: OCR number errors (`1↔l`, `7→17`) break a join, so use a **positional /
sequential fallback** when the number lookup fails (notes are monotonic). Non-numeric
markers (`*`, `†`, `‡`) are best-effort — treat the symbol as the marker. Set
`needs_review` on unmatched notes.

### Cleaning up the text (Unicode normalization)

The rule of thumb: **normalize invisible/typographic noise, preserve meaning-bearing
punctuation.** Straighten smart quotes and expand ligatures, but never strip diacritics
(`śūnyatā` must survive verbatim), never delete apostrophes (in Wylie transliteration
the apostrophe is a *letter*, not a quote), and never collapse the three kinds of dash
into one (they mean different things). And never join lines or de-hyphenate inside a
`verse` — a verse's line breaks are part of its meaning. Diacritic *folding* happens
later, in the consumer's lexical index, not here.

#### Technical details — the normalization spec

Apply, in order, to every block's `text` (must match the RAG digital path exactly):

1. **NFC** normalization (`unicodedata.normalize("NFC", …)`).
2. **Remove soft hyphens** `U+00AD`, zero-width chars `U+200B/200C/200D/FEFF`; map NBSP
   `U+00A0` and other Unicode spaces → plain space; collapse whitespace runs.
3. **Expand ligatures**: `ﬁ→fi`, `ﬂ→fl`, `ﬀ→ff`, `ﬃ→ffi`, `ﬄ→ffl`, `ﬅ/ﬆ→st`.
4. **Quotes**: curly/smart → straight. ⚠️ **Never DROP apostrophes** — the Tibetan
   a-chung in Wylie (`'jug`, `ba'i`) is a letter; converting curly→straight is fine,
   deleting is corruption.
5. **Dashes — PRESERVE, do NOT collapse to a hyphen.** `—` (em, aside), `–` (en,
   ranges), `-` (hyphen, compounds) are three meanings; keep the distinct chars.
   (If output must be ASCII: em→`--`, en→`-`, NEVER em→`-`.)
6. **De-hyphenate ONLY line-break splits** (`exam-\nple → example`); never an in-line
   hyphen. ⚠️ **Body/prose blocks only** — never inside `verse` blocks.

Also preserve editorial brackets `[…]` (translator insertions) and ellipsis `…`
(omitted canon) — they carry scholarly meaning.

### Reading order

This is item #1 above. Reconstruct **column reading order** from the bounding boxes
before emitting blocks. When the order is uncertain (overlapping boxes, ambiguous flow),
still emit a best-effort order but set `needs_review: true` on the affected blocks so
the consumer can quarantine or down-weight them. A glossary page mis-ordered by column
interleaving is the canonical failure to catch (one such page scored 54% purely from
column interleaving).

### Verses and block quotations

Verses (root verses, kārikās, songs) are often the *primary* text, so the consumer keeps
each verse as its own atomic, citable unit — which means the OCR stage must **preserve
its line breaks** and, when present, record its number/reference in `ref_label`.

Block quotations of *other* texts are labeled `quote` so the consumer can **attribute
them to their source, not the book's author** — getting "the *Samādhirāja Sūtra* says X"
right instead of pinning X on the author. A verse keeps its line breaks; a quote flows
as an indented paragraph — that's how they're told apart.

The main signal for both is **indentation**, recovered from the bounding boxes.

#### Technical details — the indentation algorithm

Indentation is usable only as a **relative** offset, never an absolute `x0` (pages here
have different widths — sampled 1820 / 1558 / 1678 px — and scans skew/shift):

1. **Deskew first** (or `x0` is noisy), then work **per page and per column** — never
   globally.
2. **Body left margin** = the modal/median `x0` of body `Text` blocks *in that column*.
   The page edge (0) is irrelevant — books have margins.
3. **`indent` = (block.x0 − margin_left) / column_width**, emitted normalized 0..1. A
   block whose `indent` exceeds ~3–5% of column width is a `verse`/`quote` candidate; a
   block quote usually also pulls its **right** edge in (a confirming signal).
4. **Centered verses/titles** aren't left-indented — detect symmetric left/right gaps
   (`(x0+x1)/2 ≈ column center`) and set `centered: true`.
5. **verse vs prose-quote:** once flagged, decide by whether internal line breaks are
   preserved (`verse`) or the block flows as a paragraph (`quote`).

Limits: block-level `x0` catches whole-block indentation but **not** first-line-only or
hanging indents — for those, emit `line_bbox` so the consumer can inspect per-line
alignment. Drop caps can masquerade as an outdent; multi-column requires the offset
computed *within* the column. When unsure, prefer `quote` over `body` and set
`needs_review`; carry `indent`/`centered` regardless, so a later policy change is a
re-chunk, not a re-OCR.

### The opponent's view vs the author's (argument structure)

Scholarly Buddhist commentary argues like a debate: the author first states the
**opponent's position** (the *pūrvapakṣa*) fully and persuasively, *then* refutes it. A
block like "The self is a permanent, partless entity…" reads, in isolation, as the
author's own teaching when it is exactly the view being demolished. If the RAG system
retrieves that block without knowing it is a view-under-refutation, it will assert **the
opposite of what the text teaches** — and sound well-grounded doing it. This is the same
attribution problem `quote` solves one axis over: `quote` says "attribute this to its
source, not the author"; this says "attribute this to the opponent, not the author."

So the OCR stage does the same thing it does for quotes: **emit the cues, let the
consumer decide the voice.** It marks blocks that *start* with an opponent-opener or a
refutation-turn, and it keeps heading text verbatim (editions often label the debate
structure with headings like "Objection" / "Reply" — a free, high-precision signal).
It does **not** emit a semantic "opponent" type — deciding whose view a passage argues
is interpretation, not structure recovery, and belongs to the consumer.

Precision over recall: emit a cue only on a clear lexical match at block start; don't
guess. English translators smooth these formulae away inconsistently, so recall is
inherently partial — the consumer runs a semantic pass over the residue.

#### Technical details — the opener/pivot lexicons

Match at block/sentence start, case-insensitive. Feature `argument_cues`, computed in
the shared cue lib next to `citation_cues`.

**`objection_opener`** (opens an opponent position):
- Sanskrit/IAST: `syād etat`, `nanu`, `iti cet`, `kaścid āha`, `āha`, `atha` (+
  `pūrvapakṣa` as an explicit label).
- Tibetan (Wylie): `gal te … na`, `kha cig na re`, `ci ste`, `'o na`, `'di ltar … na`.
- English: "Someone might object", "One might argue/think", "If you say", "Suppose it is
  asked", "It may be objected", "Objection:".

**`refutation_pivot`** (turns back to the author's voice):
- Sanskrit/IAST: `na`, `tan na`, `maivam`, `ucyate`, `tad ayuktam`.
- Tibetan (Wylie): `de ni mi rigs te`, `mi 'thad`, `de ltar ma yin`.
- English: "But this is incorrect", "That is not so", "We reply", "In response", "That
  is untenable", "This reasoning is flawed", "Reply:".

Emit as **block-start** signals only (a formula mid-paragraph is usually a quotation of
the formula, not a live turn). The list is English-primary and corpus-validated: this
library is all English translations, so the English list is the main signal; the
consumer mines and validates the authoritative lists from the debate-dense subset and
the shared lib carries that curated list. Reading order is a hard precondition — the
consumer refuses to glue an objection to its refutation across a `needs_review` block,
so an over-optimistic order there causes an *inverted attribution*, not just an
incoherent chunk.

### Front/back matter, contents, lists, and running heads

Indentation signals verse — but a lot of indented or centered content is **not** verse:
title pages, copyright pages, dedications, tables of contents, numbered/bulleted lists,
running heads, and back matter (index/glossary/bibliography/colophon). Left labeled
`body`, these get mis-read as verses (observed: title pages, TOCs, and numbered lists
all surfaced as bogus "verses"; a copyright printer's-key line surfaced as a footnote).
The fix is to **label them** — give each block a `region` (front | body | back) and a
specific `type`. In front/back matter, an indented short block is presumed
furniture/nav, never verse, unless explicitly typed verse/quote/epigraph.

One nuance worth stating: **multi-column is not an exclusion mechanism.** What includes
or excludes a region is its `type`, not its column count — a two-column *glossary* is
included as content, a single-column *index* is excluded as nav. Column layout is
handled by reading-order reconstruction, and because `indent` is computed per column, an
entry at its column's left margin isn't falsely flagged.

#### Technical details — the type-by-type rules

- **`region` (front | body | back).** Detect the front→body boundary (first
  chapter-level heading / roman→arabic page-number transition / first sustained body
  prose after the TOC) and the body→back boundary (the "Notes"/"Appendix"/"Glossary"/
  "Bibliography"/"Index" run near the end).
- **Title page** → `title_page` (drop): sparse page, large/centered title/author.
- **Copyright / edition page** → `copyright` (drop): ISBN, LCCN, "All rights reserved",
  and the **printer's key** — a line whose tokens are all small integers in a monotonic
  run (`10 9 8 7 6 5 4 3 2 1`). The printer's key is **not always in front matter** —
  many books print it with the colophon on the last page(s). Detect it by its pattern in
  **any** region (it has no prose) and label it `copyright` (or `colophon` in a
  back-matter note); this is exactly where stray digit-runs get mistaken for note markers.
- **Dedication** → drop. **Epigraph** (a quoted verse/passage opening a chapter) IS
  content → `epigraph` (treated as quote/verse), not dropped.
- **Table of contents** → `toc` (nav, excluded): under a "Contents" heading; leader dots
  and/or trailing page numbers. Multi-line + indented, so it must be typed or it reads as
  verse.
- **Numbered / bulleted lists** → `list_item` (content, but NOT verse): a leading
  enumerator (`1.` / `1)` / `(a)` / `i.` / `•`) whose text **flows as prose**. ⚠️ A
  leading number alone doesn't mean a list — verse stanzas are numbered too. Distinguish
  by line structure: numbered + flowed ⇒ `list_item`; numbered + preserved line breaks ⇒
  `verse` (the number → `ref_label`).
- **Tables** → `table` (html/markdown, never flattened). Column alignment is the signal —
  don't read aligned columns as indentation.
- **Running heads / feet / folios** → `page_header` / `page_footer` / `page_number`
  (furniture, drop): margin-band text that repeats (digit-masked) across several pages.
- **Back matter**: `index` / `bibliography` (nav, excluded); **`glossary`** (CONTENT —
  split per entry, not dropped); `colophon` (drop); appendices = ordinary `heading`/
  `body`; "about the author" / "also by" / ads → drop.

## What gets dropped vs. labeled

The producer principle is **label, never delete**: the OCR stage emits everything with a
`type` and discards nothing — the *consumer* decides what reaches the embedding corpus.
The per-type drop/exclude/keep policy and what each downstream phase (chunking,
embedding, reranking) consumes are consumer concerns and live in **`USING_THE_OUTPUT.md`**
(§"Type policy" and §"What each downstream phase uses").

## Regression fixtures — real pages that caught these bugs

These are concrete pages from a re-OCR'd scholarly book (fixture set `e10`) that the
verse/citation detector once mis-handled; they are the labeling regression tests. Block
geometry is committed in `tests/fixtures/regression_pages/e10_pages.json` (keyed by the
page numbers below); the tests read that JSON, not page images. The `doc.json` `page`
field equals the PDF page (1-based).

| page | what's on it (layout) | Surya label now | expected | bug it caused |
|---|---|---|---|---|
| 1 | title page (title + translator line) | `body` | `title_page`, region=front (drop) | front matter read as **verse** |
| 5 | copyright page: © line, ISBN, a **printer's-key digit run**, a numbered catalogue-listing block, colophon | all `body` | `copyright`, region=front (drop); the numbered list → `list_item`/copyright | **printer's key read as footnote marker**; numbered list read as **verse** |
| 6 | table of contents (title + trailing page numbers) | `toc` ✓ | stays `toc` (excluded) | (control — already labeled; must not regress to verse) |
| 14 | running head (page number + short title) | `page_header` ✓ | stays `page_header` (drop) | (control — repeated head must not become verse) |
| 152 | a numbered/enumerated list in the body | `body` | `list_item`, region=body | numbered list read as **verse** |
| 42 | a genuine indented block quotation | `body` (indented) | `verse`, region=body | **positive control** — must STILL be detected as verse so the precision fixes don't kill recall |

The pair "fix the false positives AND keep the p42 positive control" is the whole test.

## What this adds to the raw `pages.jsonl`

The pipeline already emits per-block `label`, `bbox`, `confidence`, and `text` — most of
the way there. The deltas this contract asks for:

1. **Guaranteed whole-book reading order** (cross-column).
2. The **`type` taxonomy** above (map Surya labels → these types; add toc/index/
   bibliography detection).
3. Per-block **`refs`** (which note markers a body block references) + per-block **`chapter`**.
4. A doc-level **`footnotes` map** with resolved bodies, scope-keyed.
5. **`table.format`/`content`** instead of flattened table text.
6. The **unicode normalization** pass above.
7. **`needs_review`** on low-confidence / uncertain-layout blocks.
8. Per-block **`indent`** + **`centered`** (the verse/quote signal), optional **`line_bbox`**
   on verse/ambiguous blocks.
9. Per-block **`region`** + the expanded types (`title_page`, `copyright` incl.
   printer-key detection, `glossary`, `colophon`, `epigraph`, reliable `list_item`/`toc`).
   This is the precision fix for verse detection; recall already works from `indent`.
10. Per-block **`objection_opener`** / **`refutation_pivot`** cues (feature `argument_cues`).

Everything else (per-block confidence, bboxes) is already produced — just carried through.

## The RAG handoff — packaging the corpus

Beyond the per-book `doc.json`, two scripts package a whole corpus for a consumer, and a
few guarantees make that corpus safe to ingest.

**Two production paths, one schema.** A book arrives either as **native** (digital-first
— text pulled from the PDF, pristine, no OCR error) or **scanned** (recognized by
`surya_ocr`, recognition-level fidelity). Both run the same `build_doc` assembly, so the
`doc.json` schema is identical and a consumer treats them the same. Digital-first books
are deliberately **not** OCR'd — recognizing born-digital text only degrades it.

**One deduplicated entry point.** `build_rag_manifest.py` writes a manifest (default
`out/rag_handoff_manifest.jsonl`) with one row per book, already deduplicated to the
single authoritative copy and stamped with a stable `pub_id` / `content_hash`. A consumer
should drive ingestion from the manifest, not by walking the output tree — a book that
was OCR'd and later reclassified as native can exist in two places, and only the manifest
points at the clean copy.

**A portable export.** `export_corpus.py` copies every health-checked book's durable
artifacts (`doc.json`, body/full text, `pages.jsonl`) into a self-contained `_ocr_corpus/`
with paths relative to that folder, so it can be moved anywhere and still resolve.

**A health gate.** Every exported book has passed `corpus.health()` — a non-empty OCR
cache and a `doc.json` with at least one block. An interrupted OCR that leaves an empty
results stub (a 0-block doc) is caught, reported, and never exported; the driver also
refuses to treat a bare searchable PDF as "done" unless it passes health, so such a book
is re-OCR'd instead of silently skipped.

### Technical details — the manifest schema

One JSON object per line:

```jsonc
{
  "pub_id": "…",                        // stable catalogue identity token (may be null)
  "content_hash": "…",                  // version, for re-embed staleness (may be null)
  "edition": "e300",
  "doc_id": "e71fb98cdac298c3",         // stable, deterministic from source
  "title": "...",
  "kind": "native",                     // native = pristine; scanned = OCR
  "source": "/…/original.pdf",          // the original file
  "schema_version": "1.2",
  "doc_json": "out/…/<name>.doc.json",  // AUTHORITATIVE structure — chunk from this
  "body_text": "out/…/body/all_pages.txt",
  "full_text": "out/…/full/all_pages.txt",
  "pages_jsonl": "out/…/body/pages.jsonl",
  "pdf": "out/…/<name>.pdf",            // searchable PDF (provenance/preview)
  "n_blocks": 6445
}
```

Quality/provenance: trust native text fully (diacritics exact, no `confidence` flags);
honor `confidence`/`needs_review` on scanned books.

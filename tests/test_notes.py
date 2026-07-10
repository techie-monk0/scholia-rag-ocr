"""Phase 2 — footnote / endnote resolution (lib/notes.py).

Executable specs for the unified note engine. They SKIP until ``lib/notes.py``
exists, then run as the Phase 2 acceptance suite.

Grounded in real edition-383 data (printed pages 57-58), which revealed:
  * Surya's `Footnote` label is INCONSISTENT even between facing pages — page 58's
    footnotes came back labeled `Text`. So note detection MUST use geometry (a
    bottom-of-page band), not the label alone.
  * A page-bottom note continues across the page break with NO repeated marker:
    note 118 ends mid-phrase ("…five heaps, four") and the next page's footnote
    band opens with an unmarked, lowercase fragment ("elements, six inner …").

Expected API (provisional, finalized when implemented):
    notes.resolve(pages, *, mode="auto") -> dict     # scope-key -> Footnote dict
        operates on the rich Page model (needs bbox geometry + labels + page
        height); idempotent; never raises on bad input.
"""
import pytest

notes = pytest.importorskip("notes", reason="Phase 2: lib/notes.py not built yet")

from conftest import mk_block, mk_page


# --- scope keying ---------------------------------------------------------- #
def test_page_bottom_footnote_keyed_by_page():
    pages = [mk_page([
        mk_block("Body says[^5] something.", label="Text", bbox=[0, 100, 1000, 300]),
        mk_block("[^5]A page-bottom note.", label="Footnote",
                 bbox=[0, 1500, 1000, 1560]),
    ], name="p0042", height=1800)]
    fns = notes.resolve(pages)
    assert "p42:5" in fns
    assert fns["p42:5"]["kind"] == "footnote"
    assert fns["p42:5"]["text"] == "A page-bottom note."
    assert fns["p42:5"]["page"] == 42


def test_endnote_keyed_by_chapter():
    pages = [
        mk_page([mk_block("Chapter 13", label="SectionHeader"),
                 mk_block("Body cites[^316] a sutra.", label="Text")], name="p0120"),
        mk_page([mk_block("Notes to Chapter 13", label="SectionHeader"),
                 mk_block("316. Suhṛllekha, vv. 65–66.", label="Text")], name="p0268"),
    ]
    fns = notes.resolve(pages)
    assert "ch13:316" in fns
    assert fns["ch13:316"]["kind"] == "endnote"


def test_per_chapter_endnotes_reset_numbering():
    # Mirrors edition-221 (Kindly Bent to Ease Us): each chapter's notes restart
    # at 1 under "Notes to Chapter X" headings -> distinct chN scopes, so note 1
    # of chapter one and note 1 of chapter two don't collide.
    pages = [
        mk_page([mk_block("Notes to Chapter One", label="SectionHeader"),
                 mk_block("1. First note of chapter one.", label="Text"),
                 mk_block("2. Second note of chapter one.", label="Text"),
                 mk_block("3. Third note of chapter one.", label="Text")],
                name="p0119"),
        mk_page([mk_block("Notes to Chapter Two", label="SectionHeader"),
                 mk_block("1. First note of chapter two.", label="Text"),
                 mk_block("2. Second note of chapter two.", label="Text"),
                 mk_block("3. Third note of chapter two.", label="Text")],
                name="p0125"),
    ]
    fns = notes.resolve(pages)
    # chapter scope is case-normalized (so body "Chapter One" and a "CHAPTER ONE"
    # heading resolve to the same key); note 1 of each chapter stays distinct.
    assert "chone:1" in fns and "chtwo:1" in fns
    assert fns["chone:1"]["text"].startswith("First note of chapter one")
    assert fns["chtwo:1"]["text"].startswith("First note of chapter two")
    assert {v["kind"] for v in fns.values()} == {"endnote"}


def test_endnote_continuous_book_numbering_keyed_by_book():
    pages = [
        mk_page([mk_block("Body cites[^200].", label="Text")], name="p0100"),
        mk_page([mk_block("Notes", label="SectionHeader"),
                 mk_block("200. A continuously numbered note.", label="Text")],
                name="p0400"),
    ]
    fns = notes.resolve(pages)
    assert "book:200" in fns


# --- end-of-chapter notes (edition 138) ------------------------------------ #
# Grounded in real native-text-layer data from edition 138, "Buddhism Between
# Tibet and China" (born-digital; built via fitz `page.get_text("dict")` on the
# chapter-1 note pages — printed pp. 56–59 — NOT OCR). This is the END-OF-CHAPTER
# style no other fixture covered: each chapter's notes are printed right after
# that chapter's text, under a small-caps "notes" heading with NO "to Chapter X"
# qualifier, then numbered 1, 2, 3…. Faithful details this fixture preserves:
#   * the heading is the bare word "notes" (lowercase small-caps in print);
#   * the chapter's last body page carries an unnumbered "*" acknowledgments note
#     before the numbered run (a real symbol-marked note);
#   * note 1 (a long Beckwith/Twitchett citation) wraps across the page break:
#     p.58 ends "…Sui and T'ang China, 589–" and p.59 RESUMES with "906
#     (Cambridge…" — a bare number that is citation text, not a new marker;
#   * body reference markers are superscripts, rendered "[^121]"/"[^122]" by the
#     pipeline's OCR convention.
def _e138_chapter1_pages():
    H = 648                              # the PDF's native page height (points)
    body = mk_page([
        mk_block("56", label="PageHeader", bbox=[66, 50, 310, 61]),
        mk_block("…the others—Chinese donor inscriptions relating to later "
                 "restorations…are generally agreed to post-date the temple's "
                 "foundation by several decades and more.[^121] While the Tibetan "
                 "inscription…it is only thanks to a recent discovery…[^122]",
                 label="Text", bbox=[67, 75, 367, 300]),
    ], name="p0056", height=H)
    notes_pg = mk_page([
        mk_block("58", label="PageHeader", bbox=[66, 50, 310, 61]),
        mk_block("…diplomacy, ritual, and icon were interwoven in the tissue of "
                 "events, places, and texts according to the ideal of a sublime, "
                 "transcending plan.", label="Text", bbox=[67, 75, 367, 308]),
        mk_block("notes", label="SectionHeader", bbox=[67, 337, 99, 350]),
        mk_block("* I am grateful to many colleagues, students, and friends for "
                 "their responses to this chapter…", label="Text",
                 bbox=[67, 361, 367, 471]),
        mk_block("1 Christopher I. Beckwith, he Tibetan Empire in Central Asia "
                 "(Princeton: Princeton University Press, 1987)…summarized in brief "
                 "by Denis Twitchett, ed., he Cambridge History of China, Volume 3, "
                 "Sui and T'ang China, 589–", label="Text", bbox=[67, 536, 367, 580]),
    ], name="p0058", height=H)
    cont = mk_page([
        mk_block("59", label="PageHeader", bbox=[66, 50, 310, 61]),
        mk_block("906 (Cambridge: Cambridge University Press, 1979), pp. 35–36: "
                 "\"Tibet suddenly grew into a powerful united kingdom…\"",
                 label="Text", bbox=[67, 75, 367, 260]),
        mk_block("2 For a sustained overview of China's relations with Tibet, the "
                 "Uighurs, and others…see Pan Yihong, Son of Heaven and Heavenly "
                 "Qaghan…(Bellingham, Washington, 1997).", label="Text",
                 bbox=[67, 270, 367, 320]),
        mk_block("3 Hugh E. Richardson, \"Two Chinese Princesses in Tibet…\" in "
                 "High Peaks, Pure Earth…(London: Serindia Publications, 1998), "
                 "pp. 207–215.", label="Text", bbox=[67, 330, 367, 380]),
    ], name="p0059", height=H)
    return [body, notes_pg, cont]


def test_end_of_chapter_notes_resolved_under_bare_notes_heading():
    # The numbered run under the bare "notes" heading resolves to endnotes, the
    # "*" acknowledgments note is kept, and the chapter body's "*" / numbered runs
    # are NOT misread as body. The unnumbered ack note keeps its symbol marker.
    fns = notes.resolve(_e138_chapter1_pages())
    assert {v["kind"] for v in fns.values()} == {"endnote"}
    assert "book:1" in fns and "book:2" in fns and "book:3" in fns
    assert "book:*" in fns                       # the symbol-marked ack note
    assert fns["book:2"]["text"].startswith("For a sustained overview")
    assert fns["book:3"]["text"].startswith("Hugh E. Richardson")


def test_end_of_chapter_note_stitched_across_page_break():
    # Note 1's long citation wraps the page break: p.58 ends "…589–" and p.59
    # resumes with "906 (Cambridge…". The resumed line is a bare number that is
    # citation text, NOT a new note — so it must be appended to note 1 (spanning
    # both pages) and must NOT surface as a spurious "book:906" note.
    fns = notes.resolve(_e138_chapter1_pages())
    assert fns["book:1"]["text"].startswith("Christopher I. Beckwith")
    assert "906 (Cambridge" in fns["book:1"]["text"]
    assert fns["book:1"]["span_pages"] == [58, 59]
    assert not any(k.startswith("book:906") for k in fns)
    # numbering stays clean: exactly the ack + notes 1–3
    assert set(fns) == {"book:*", "book:1", "book:2", "book:3"}


# --- KNOWN LIMITATION: end-of-chapter notes scope to book:, not chapter ---- #
def test_end_of_chapter_notes_scope_is_book_not_chapter_KNOWN_GAP():
    # Documents a real gap in notes.py: a bare "notes" heading carries no chapter
    # token, so end-of-chapter notes key to global "book:" scope. For a book like
    # e138 — where every chapter restarts numbering at 1 under its own "notes"
    # heading — chapter 2's note 1 would COLLIDE with chapter 1's "book:1". The
    # fix needs current-chapter tracking from chapter SectionHeaders (the chapter
    # a "notes" block physically follows), which is deferred as a larger change.
    pages = _e138_chapter1_pages()
    # a SECOND chapter's notes, also numbered from 1 under a bare "notes" heading
    ch2_notes = mk_page([
        mk_block("120", label="PageHeader", bbox=[66, 50, 310, 61]),
        mk_block("notes", label="SectionHeader", bbox=[67, 337, 99, 350]),
        mk_block("1 A FIRST note belonging to chapter two.", label="Text",
                 bbox=[67, 361, 367, 420]),
        mk_block("2 A SECOND note of chapter two.", label="Text",
                 bbox=[67, 430, 367, 480]),
        mk_block("3 A THIRD note of chapter two.", label="Text",
                 bbox=[67, 490, 367, 540]),
    ], name="p0120", height=648)
    fns = notes.resolve(pages + [ch2_notes])
    # CURRENT (gap) behavior: chapter 2's note 1 overwrites chapter 1's book:1.
    assert fns["book:1"]["text"].startswith("A FIRST note belonging to chapter two")
    # If/when chapter scoping lands, this becomes "ch1:1" vs "ch2:1" and the two
    # note-1 bodies coexist; update this test then.


# --- the hard case: page-bottom footnote continuing onto the next page ----- #
# Faithful to edition-383 pp.57-58: page 58's notes are Text-labeled (not
# Footnote) and sit in the bottom band; the continuation has no marker.
def test_pagebottom_continuation_stitched_across_pages():
    H = 1803
    p57 = mk_page([
        mk_block("…reciting passages[^118].", label="Text", bbox=[192, 200, 1120, 400]),
        mk_block("[^118]These are expanded forms of the Family of One … "
                 "identifying five aspects in each of a Buddha's five heaps, four",
                 label="Footnote", bbox=[192, 1450, 1120, 1700]),
    ], name="p0057", height=H)
    p58 = mk_page([
        mk_block("If you fail to understand this point …", label="Text",
                 bbox=[192, 128, 1120, 512]),
        # footnote band — Surya MISLABELS these as Text on this page:
        mk_block("elements, six inner bases of consciousness (S: āyatanam) and "
                 "five sense objects.", label="Text", bbox=[191, 1346, 1129, 1410]),
        mk_block("[^119]In the sutra, these verses are spoken by Maitreya …",
                 label="Text", bbox=[191, 1404, 1132, 1525]),
        mk_block("[^120]S: bodhipakṣadharmaḥ. Thirty-seven forms of knowledge …",
                 label="Text", bbox=[191, 1519, 1132, 1610]),
    ], name="p0058", height=H)

    fns = notes.resolve([p57, p58])
    # note 118 stitched across the page break (geometry + unmarked continuation)
    assert "four elements, six inner bases" in fns["p57:118"]["text"]
    assert fns["p57:118"]["span_pages"] == [57, 58]
    # the genuine new notes on page 58 survive…
    assert "p58:119" in fns and "p58:120" in fns
    # …but the unmarked continuation is NOT emitted as its own note
    assert not any(k.startswith("p58:") and v["text"].startswith("elements")
                   for k, v in fns.items())


# --- matching fallbacks ---------------------------------------------------- #
def test_sequential_fallback_on_ocr_number_error():
    pages = [mk_page([
        mk_block("ref[^5] here.", label="Text", bbox=[0, 100, 1000, 300]),
        mk_block("S A note whose number misread.", label="Footnote",
                 bbox=[0, 1500, 1000, 1560]),
    ], name="p0042", height=1800)]
    fns = notes.resolve(pages)
    assert fns.get("p42:5") is not None
    assert fns["p42:5"]["needs_review"] is True


def test_unmatched_note_kept_raw_and_flagged():
    pages = [mk_page([
        mk_block("body with no refs", label="Text", bbox=[0, 100, 1000, 300]),
        mk_block("[^9]An orphan note nobody references.", label="Footnote",
                 bbox=[0, 1500, 1000, 1560]),
    ], name="p0042", height=1800)]
    fns = notes.resolve(pages)
    assert fns["p42:9"]["text"] == "An orphan note nobody references."
    assert fns["p42:9"]["needs_review"] is True


def test_symbol_marker_supported():
    pages = [mk_page([
        mk_block("a claim* worth noting.", label="Text", bbox=[0, 100, 1000, 300]),
        mk_block("* a symbol-marked note.", label="Footnote",
                 bbox=[0, 1500, 1000, 1560]),
    ], name="p0042", height=1800)]
    fns = notes.resolve(pages)
    assert "p42:*" in fns


# --- generality: conventions NOT used by the sample books ------------------ #
from notes import extract_marker, parse_notes_heading


@pytest.mark.parametrize("text,marker,body", [
    ("[^13]Bracketed sup.", "13", "Bracketed sup."),   # 383 style
    ("13. Period.", "13", "Period."),                  # 375 style
    ("13) Paren.", "13", "Paren."),                    # 221 style
    ("13: Colon.", "13", "Colon."),                    # not in samples
    ("(13) Wrapped paren.", "13", "Wrapped paren."),   # not in samples
    ("[13] Wrapped bracket.", "13", "Wrapped bracket."),
    ("†† Double dagger.", "††", "Double dagger."),     # not in samples
    ("# Hash.", "#", "Hash."),
])
def test_marker_styles_general(text, marker, body):
    assert extract_marker(text) == (marker, body)


def test_bare_number_marker_only_in_note_context():
    # "13 text" (no punctuation) is a marker ONLY when bare=True (a confirmed
    # note context) — otherwise body prose like "13 monks ..." would be eaten.
    assert extract_marker("13 monks gathered.")[0] is None
    assert extract_marker("13 First note.", bare=True) == ("13", "First note.")


@pytest.mark.parametrize("heading,is_notes,scope", [
    ("Notes", True, None),
    ("Endnotes", True, None),
    ("Reference Notes", True, None),
    ("Notes to Chapter Three", True, "Three"),
    ("N otes to C hapter Three", True, "Three"),   # OCR letter-spacing garble
    ("Notes to the Introduction", True, "Introduction"),
    ("Bibliography", False, None),
    ("Chapter Five", False, None),
])
def test_notes_heading_general(heading, is_notes, scope):
    ok, ch = parse_notes_heading(heading)
    assert ok is is_notes
    if is_notes:
        assert (ch or None) == (scope.lower() if scope else None) or ch == scope


def test_endnotes_titled_endnotes_bare_numbered():
    # A book whose back-matter is titled "Endnotes" with bare-number entries.
    pages = [mk_page([
        mk_block("Endnotes", label="SectionHeader"),
        mk_block("1 First with no punctuation.", label="Text"),
        mk_block("2 Second one here.", label="Text"),
        mk_block("3 Third entry follows.", label="Text"),
    ], name="p0300")]
    fns = notes.resolve(pages)
    assert {"book:1", "book:2", "book:3"} <= set(fns)
    assert fns["book:1"]["text"] == "First with no punctuation."


def test_resolve_is_idempotent():
    pages = [mk_page([
        mk_block("Body[^5].", label="Text", bbox=[0, 100, 1000, 300]),
        mk_block("[^5]Note.", label="Footnote", bbox=[0, 1500, 1000, 1560]),
    ], name="p0042", height=1800)]
    assert notes.resolve(pages) == notes.resolve(pages)


def test_find_refs_matches_bracket_and_caret_forms():
    # OCR renders body citation markers as [^N] (superscript) or [N]; both must be
    # captured, while editorial [insertions] and years must NOT.
    from notes import find_refs
    assert find_refs("text [^5] more [^12]") == ["5", "12"]
    assert find_refs("pride [4] and [9]") == ["4", "9"]            # the missed form
    assert find_refs("mix [^3] and [7]") == ["3", "7"]
    assert find_refs("editorial [Valid cognitions] kept") == []   # not a marker
    assert find_refs("the year [2010] here") == []                # 4 digits, not a note
    assert find_refs("symbol [^*] mark") == ["*"]

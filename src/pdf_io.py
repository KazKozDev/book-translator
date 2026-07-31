"""Reading a PDF, which the app accepts as a source of plain text and,
when the file carries a usable outline or clear chapter headings, as a
chapter list so chunking respects those breaks the way EPUB already does.

Deliberately one-directional: there is no PDF writer here. A PDF is turned into
the same shape of string (and optional chapters) the rest of the pipeline
already understands.

The work that is not trivial is undoing the page. A PDF has no paragraphs, only
lines placed on sheets, so extracted text arrives hard-wrapped at the print
column with a running header on every page and the page number under it. Fed in
raw, the chunker would treat each printed line as a paragraph and the model
would translate "CHAPTER FOUR 57" a hundred times. So the lines are stitched
back into paragraphs here, before anything else sees them.

Split out of translator.py for the same reason epub_io.py is: no database, no
model, no request state.
"""

import re
from collections import Counter
from typing import List, Optional, Tuple

from pypdf import PdfReader
from pypdf.generic import Destination


# A line that is only a page number — "57", "- 57 -", "[57]", "xiv". Removed
# before paragraphs are assembled, or it would be glued into the sentence that
# happens to run across that page break.
PAGE_NUMBER = re.compile(r'^[\s\[\(\-–—]*(?:\d{1,4}|[ivxlcdm]{1,7})[\s\]\)\-–—.]*$', re.IGNORECASE)

# The same number when it shares its line with a running head — "6 The Boy Who
# Lived" or "The Boy Who Lived 6". Only used to compare edge lines between
# pages, never to rewrite text that is kept.
PAGE_NUMBER_IN_EDGE_LINE = re.compile(r'^\s*\d{1,4}\s+|\s+\d{1,4}\s*$')

# How much shorter than the body column a line must be to read as the last line
# of a paragraph rather than a wrap. Justified body text fills the column to
# within a few characters; a paragraph's final line rarely does.
SHORT_LINE_RATIO = 0.85

# Below this many pages, a repeated first or last line is more likely to be real
# text than a running head, so the header stripper stays out of it.
MIN_PAGES_FOR_HEADER_DETECTION = 5

# A first/last line has to recur on this share of pages to count as furniture.
HEADER_FREQUENCY = 0.4

# Fewer lines than this on a page and its own column width means nothing, so the
# book's width is used instead.
MIN_LINES_TO_MEASURE_A_PAGE = 5

# Explicit chapter / part openers in novels and non-fiction. Kept conservative:
# a false chapter break only adds a boundary, but too many of them fragment the
# book into one-paragraph chapters.
CHAPTER_HEADING = re.compile(
    r'^(?:'
    r'(?:chapter|chap\.?|ch\.?)\s+(?:\d+|[ivxlcdm]+)\b'
    r'|(?:part|book|section)\s+(?:\d+|[ivxlcdm]+)\b'
    r'|(?:глава|часть|раздел)\s+(?:\d+|[ivxlcdm]+)\b'
    r'|(?:capítulo|capitulo|partie|kapitel)\s+(?:\d+|[ivxlcdm]+)\b'
    r')'
    r'(?:\s*[:.\-–—].*)?$',
    re.IGNORECASE,
)

# Short all-caps titles used as chapter heads in many print PDFs.
ALL_CAPS_HEADING = re.compile(r'^[A-Z0-9][A-Z0-9 \'\-–—:,.!?]{2,80}$')


def is_pdf_filename(filename: str) -> bool:
    return filename.lower().endswith('.pdf')


def extract_pdf_book(
    filepath: str,
) -> Tuple[str, Optional[List[str]], Optional[str], Optional[str]]:
    """Read a PDF and return (text, chapters, title, author).

    The text is plain, with paragraphs separated by blank lines — the same shape
    decode_text_file() returns for a .txt book. `chapters` is set when the PDF
    outline (bookmarks) yields at least two sections with text, or when the
    rejoined paragraphs clearly open with chapter headings; otherwise None so
    the pipeline treats the file like .txt.

    Returns empty text when the file carries no extractable text at all, which
    is what a scan looks like: the caller decides what to tell the user.
    """
    reader = PdfReader(filepath)
    if reader.is_encrypted:
        # An empty user password is the common case: a book locked against
        # printing or copying, not against reading. Anything else is a file we
        # genuinely cannot open, and PdfReader raises on the page access below.
        try:
            reader.decrypt('')
        except Exception:
            pass

    page_lines = [(page.extract_text() or '').splitlines() for page in reader.pages]
    cleaned_pages = _strip_page_furniture(page_lines)
    page_texts = [_join_lines_into_paragraphs([page]) for page in cleaned_pages]
    text = _join_lines_into_paragraphs(cleaned_pages)

    title = None
    author = None
    meta = reader.metadata
    if meta:
        # Producers write empty strings and their own name here freely, so both
        # fields are treated as hints, exactly like EPUB metadata.
        title = (meta.title or '').strip() or None
        author = (meta.author or '').strip() or None

    chapters = _chapters_from_outline(reader, page_texts)
    if chapters is None:
        chapters = _split_text_into_chapters(text)

    return text, chapters, title, author


def _chapters_from_outline(
    reader: PdfReader, page_texts: List[str],
) -> Optional[List[str]]:
    """Build chapters from PDF bookmarks when they point at real page ranges."""
    try:
        outline = reader.outline
    except Exception:
        return None
    if not outline:
        return None

    starts: List[Tuple[int, str]] = []
    for title, page_index in _flatten_outline(reader, outline):
        if page_index is None or page_index < 0 or page_index >= len(page_texts):
            continue
        if starts and starts[-1][0] == page_index:
            # Two bookmarks on the same page — keep the first title.
            continue
        starts.append((page_index, title))

    if len(starts) < 2:
        return None

    chapters: List[str] = []
    for index, (start, title) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(page_texts)
        body = '\n\n'.join(
            page for page in page_texts[start:end] if page and page.strip()
        ).strip()
        if not body:
            continue
        # Prefer the bookmark title as the opening line when the page text
        # does not already begin with it.
        heading = title.strip()
        if heading and not body.lower().startswith(heading.lower()[: min(40, len(heading))]):
            body = f'{heading}\n\n{body}'
        chapters.append(body)

    if len(chapters) < 2:
        return None
    # Outline noise: a bookmark per page produces hundreds of tiny chapters.
    if len(chapters) > max(80, len(page_texts)):
        return None
    return chapters


def _flatten_outline(
    reader: PdfReader, nodes, depth: int = 0,
) -> List[Tuple[str, Optional[int]]]:
    """Walk nested bookmarks; only top-level entries open chapters."""
    entries: List[Tuple[str, Optional[int]]] = []
    for node in nodes:
        if isinstance(node, list):
            # Nested children — ignored for chapter splits so a book's section
            # tree does not explode into one chapter per subsection.
            continue
        title = ''
        page_index = None
        try:
            if isinstance(node, Destination):
                title = (node.title or '').strip()
                page_index = reader.get_destination_page_number(node)
            else:
                title = str(getattr(node, 'title', '') or '').strip()
                try:
                    page_index = reader.get_destination_page_number(node)
                except Exception:
                    page_index = None
        except Exception:
            continue
        if title:
            entries.append((title, page_index))
    return entries


def _split_text_into_chapters(text: str) -> Optional[List[str]]:
    """Heuristic chapter split when the PDF has no usable outline."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) < 4:
        return None

    heading_indexes = [
        i for i, paragraph in enumerate(paragraphs)
        if _looks_like_chapter_heading(paragraph)
    ]
    if len(heading_indexes) < 2:
        return None
    # Too many "headings" means the detector is matching body text.
    if len(heading_indexes) > max(3, len(paragraphs) // 8):
        return None
    # First heading should not be buried after most of the book.
    if heading_indexes[0] > max(20, len(paragraphs) // 3):
        return None

    chapters: List[str] = []
    # Keep a short preface before the first chapter heading when present.
    if heading_indexes[0] > 0:
        preface = '\n\n'.join(paragraphs[:heading_indexes[0]]).strip()
        if preface:
            chapters.append(preface)

    for index, start in enumerate(heading_indexes):
        end = heading_indexes[index + 1] if index + 1 < len(heading_indexes) else len(paragraphs)
        body = '\n\n'.join(paragraphs[start:end]).strip()
        if body:
            chapters.append(body)

    return chapters if len(chapters) >= 2 else None


def _looks_like_chapter_heading(paragraph: str) -> bool:
    """True for a short paragraph that opens a chapter."""
    if '\n' in paragraph:
        return False
    stripped = paragraph.strip()
    if len(stripped) < 3 or len(stripped) > 80:
        return False
    if CHAPTER_HEADING.match(stripped):
        return True
    # All-caps titles: "THE BOY WHO LIVED". Reject lines that are mostly digits.
    letters = sum(1 for char in stripped if char.isalpha())
    if letters >= 3 and ALL_CAPS_HEADING.match(stripped) and stripped == stripped.upper():
        return True
    return False


def _strip_page_furniture(pages: List[List[str]]) -> List[List[str]]:
    """Drop running heads and folios, keeping the page grouping.

    The grouping survives only so that each page can be measured against its own
    column width; paragraphs are still assembled across pages, because a
    sentence interrupted by a page break has to close over it.
    """
    trimmed = [[line.rstrip() for line in page] for page in pages]
    trimmed = [[line for line in page if not PAGE_NUMBER.match(line)] for page in trimmed]

    repeated = _repeated_edge_lines(trimmed)

    stripped = []
    for page in trimmed:
        body = list(page)
        while body and _without_folio(body[0]) in repeated:
            body.pop(0)
        while body and _without_folio(body[-1]) in repeated:
            body.pop()
        stripped.append(body)
    return stripped


def _without_folio(line: str) -> str:
    """A running head with its page number removed, for comparing across pages.

    "6 Perplexity at Work" and "7 Perplexity at Work" are the same piece of
    furniture, and compared literally neither would ever repeat.
    """
    return PAGE_NUMBER_IN_EDGE_LINE.sub('', line).strip()


def _repeated_edge_lines(pages: List[List[str]]) -> set:
    """The first and last lines that recur across pages: running heads and feet.

    Only the edges are counted. A phrase repeated in the middle of pages is the
    author's, not the typesetter's, and stays.
    """
    if len(pages) < MIN_PAGES_FOR_HEADER_DETECTION:
        return set()

    counts: Counter = Counter()
    for page in pages:
        body = [_without_folio(line) for line in page if line.strip()]
        body = [line for line in body if line]
        if not body:
            continue
        counts.update({body[0]})
        if len(body) > 1:
            counts.update({body[-1]})

    threshold = max(2, int(len(pages) * HEADER_FREQUENCY))
    return {line for line, count in counts.items() if count >= threshold}


def _join_lines_into_paragraphs(pages: List[List[str]]) -> str:
    """Rewrap hard-wrapped print lines into paragraphs.

    Two signals end a paragraph and nothing else does: a blank line, and a line
    noticeably shorter than the column it sits in — the last line of a paragraph
    rarely reaches the margin, and a heading never does. A line ending in a
    hyphen is a word broken across the column and is rejoined without a space.

    The column is measured per page, not per book. A PDF is free to change
    layout partway through, and it does: a narrow two-column section inside an
    otherwise normal book has lines around half the width, and measured against
    the book's column every one of them reads as a paragraph end. That failure
    is the one worth avoiding, because it puts the text through the model one
    printed line at a time.

    It is still a heuristic and still wrong sometimes — a book set ragged-right
    gives more false breaks than a justified one. Wrong in that direction is
    cheap: an extra paragraph break costs one chunk boundary.
    """
    document_width = _body_column_width([line for page in pages for line in page])

    paragraphs: List[str] = []
    current = ''
    for page in pages:
        page_width = _body_column_width(page) or document_width
        # Too few lines to measure anything: a chapter opening, a page with one
        # illustration caption. The book's own column is the better guess.
        if len([line for line in page if line.strip()]) < MIN_LINES_TO_MEASURE_A_PAGE:
            page_width = document_width or page_width

        for line in page:
            stripped = line.strip()
            if not stripped:
                if current:
                    paragraphs.append(current)
                    current = ''
                continue

            if not current:
                current = stripped
            elif current.endswith('-') and not current.endswith('--'):
                current = current[:-1] + stripped
            else:
                current = f'{current} {stripped}'

            if len(stripped) < page_width * SHORT_LINE_RATIO:
                paragraphs.append(current)
                current = ''

    if current:
        paragraphs.append(current)

    return '\n\n'.join(paragraph.strip() for paragraph in paragraphs if paragraph.strip())


def _body_column_width(lines: List[str]) -> float:
    """The width a full line of body text runs to, in characters.

    The 90th percentile rather than the maximum: one over-long line — a URL, a
    table row, a page whose extraction ran two columns together — would
    otherwise raise the bar above every real line and merge everything into a
    single paragraph.
    """
    lengths = sorted(len(line.strip()) for line in lines if line.strip())
    if not lengths:
        return 0.0
    return float(lengths[int(len(lengths) * 0.9) - 1] if len(lengths) > 1 else lengths[0])

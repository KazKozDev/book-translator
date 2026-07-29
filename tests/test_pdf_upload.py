"""Reading a PDF, where the damage is done before anything else runs.

A PDF has no paragraphs — only lines placed on sheets, wrapped at the print
column, under a running head, over a page number. Uploaded as extracted, the
book reaches the chunker as thousands of one-line paragraphs and the running
head gets translated once per page. So these tests are about the shape of the
text, not about whether the file opened.

Nothing here touches the pipeline: a PDF arrives as plain text with no chapter
list, exactly like a .txt upload.
"""

from io import BytesIO

import pytest

import pdf_io
import translator


def _escape(line):
    return (line.replace('\\', r'\\').replace('(', r'\(').replace(')', r'\)')).encode('latin-1')


def _content_stream(lines):
    """One page of type: a text object that draws each line and steps down."""
    ops = [b'BT', b'/F1 11 Tf', b'14 TL', b'72 720 Td']
    for line in lines:
        ops.append(b'(%s) Tj' % _escape(line))
        ops.append(b'T*')
    ops.append(b'ET')
    return b'\n'.join(ops)


def _pdf(pages, title=None, author=None):
    """A real PDF, written here rather than with a library.

    The suite has no PDF writer and should not grow one: a dependency that
    exists only to build fixtures is a dependency that gets installed on three
    operating systems in CI so that eight tests can run. This is a few hundred
    bytes of the format — catalog, page tree, a content stream per page — and it
    is what pypdf will actually be handed.
    """
    objects = {
        1: b'<< /Type /Catalog /Pages 2 0 R >>',
        3: b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    }
    next_id = 4
    kids = []
    for lines in pages:
        content = _content_stream(lines)
        content_id, page_id = next_id, next_id + 1
        next_id += 2
        objects[content_id] = b'<< /Length %d >>\nstream\n%s\nendstream' % (len(content), content)
        objects[page_id] = (
            b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
            b'/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>' % content_id
        )
        kids.append(page_id)
    objects[2] = b'<< /Type /Pages /Count %d /Kids [%s] >>' % (
        len(kids), b' '.join(b'%d 0 R' % kid for kid in kids))

    info_id = None
    if title or author:
        entries = b''
        if title:
            entries += b'/Title (%s) ' % _escape(title)
        if author:
            entries += b'/Author (%s) ' % _escape(author)
        info_id = next_id
        next_id += 1
        objects[info_id] = b'<< %s>>' % entries

    out = bytearray(b'%PDF-1.4\n')
    offsets = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += b'%d 0 obj\n%s\nendobj\n' % (obj_id, objects[obj_id])

    xref_offset = len(out)
    size = max(objects) + 1
    out += b'xref\n0 %d\n0000000000 65535 f \n' % size
    for obj_id in range(1, size):
        out += b'%010d 00000 n \n' % offsets[obj_id]
    trailer = b'<< /Size %d /Root 1 0 R' % size
    if info_id:
        trailer += b' /Info %d 0 R' % info_id
    out += b'trailer\n%s >>\nstartxref\n%d\n%%%%EOF\n' % (trailer, xref_offset)
    return bytes(out)


def _written(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


# A justified paragraph as it comes off a page: full-width lines, then a short
# last line. Only that short line marks the end of the paragraph.
WRAPPED = [
    'Mr Dursley was the director of a firm called Grunnings, which made',
    'drills. He was a big, beefy man with hardly any neck, although he did',
    'have a very large moustache.',
]


def test_wrapped_lines_come_back_as_one_paragraph(tmp_path):
    path = _written(tmp_path, 'book.pdf', _pdf([WRAPPED]))

    text, _, _ = pdf_io.extract_pdf_book(path)

    assert text == (
        'Mr Dursley was the director of a firm called Grunnings, which made '
        'drills. He was a big, beefy man with hardly any neck, although he did '
        'have a very large moustache.'
    )


def test_a_paragraph_broken_across_a_page_closes_over_the_break(tmp_path):
    """The page boundary is not a paragraph boundary, and the pipeline has no
    way to learn that later — by then the sentence is already split in two."""
    path = _written(tmp_path, 'book.pdf', _pdf([WRAPPED[:2], WRAPPED[2:]]))

    text, _, _ = pdf_io.extract_pdf_book(path)

    assert 'he did have a very large moustache.' in text


def test_a_word_hyphenated_at_the_column_is_rejoined(tmp_path):
    path = _written(tmp_path, 'book.pdf', _pdf([[
        'She looked at the enormous, unmistakably threatening moust-',
        'ache and said nothing at all.',
    ]]))

    text, _, _ = pdf_io.extract_pdf_book(path)

    assert 'moustache' in text
    assert 'moust- ache' not in text


def test_the_running_head_and_page_number_do_not_reach_the_text(tmp_path):
    """Left in, they are translated once per page and read as body text."""
    pages = [
        ['THE BOY WHO LIVED', str(number)] + WRAPPED
        for number in range(1, 9)
    ]
    path = _written(tmp_path, 'book.pdf', _pdf(pages))

    text, _, _ = pdf_io.extract_pdf_book(path)

    assert 'THE BOY WHO LIVED' not in text
    assert 'Mr Dursley' in text


def test_the_page_number_printed_inside_the_running_head_does_not_save_it(tmp_path):
    """The common layout: "7 THE BOY WHO LIVED" as one line. Compared literally
    no two pages ever match, and the head survives on every page."""
    pages = [
        [f'{number} THE BOY WHO LIVED'] + WRAPPED
        for number in range(1, 9)
    ]
    path = _written(tmp_path, 'book.pdf', _pdf(pages))

    text, _, _ = pdf_io.extract_pdf_book(path)

    assert 'THE BOY WHO LIVED' not in text
    assert 'Mr Dursley' in text


def test_a_narrow_section_is_measured_against_its_own_column(tmp_path):
    """A book is free to change layout partway through, and a page set in a
    narrow column has lines about half the usual width. Measured against the
    book's column, every one of them looks like a paragraph end — which is the
    failure this whole module exists to prevent, one printed line per chunk."""
    narrow = [
        'She had been standing there for a',
        'long time before anyone in the hall',
        'noticed her at all, and by then it',
        'was already much too late.',
    ]
    pages = [WRAPPED * 3] * 4 + [narrow * 3]
    path = _written(tmp_path, 'book.pdf', _pdf(pages))

    text, _, _ = pdf_io.extract_pdf_book(path)

    assert 'She had been standing there for a long time before anyone in the hall' in text


def test_a_phrase_repeated_inside_the_page_is_the_author_s_and_stays(tmp_path):
    """The header stripper looks only at the first and last line of a page. A
    refrain in the middle of the text is writing, not furniture."""
    pages = [
        ['Chapter opening line that is long enough to be a body line here.',
         'All work and no play.',
         'A closing body line that is also long enough to look like prose.']
        for _ in range(8)
    ]
    path = _written(tmp_path, 'book.pdf', _pdf(pages))

    text, _, _ = pdf_io.extract_pdf_book(path)

    assert text.count('All work and no play.') == 8


def test_metadata_is_read_when_the_producer_wrote_it(tmp_path):
    path = _written(tmp_path, 'book.pdf', _pdf([WRAPPED], title='Source Book', author='Source Author'))

    _, title, author = pdf_io.extract_pdf_book(path)

    assert title == 'Source Book'
    assert author == 'Source Author'


def test_a_pdf_without_metadata_reports_none_rather_than_empty_strings(tmp_path):
    path = _written(tmp_path, 'book.pdf', _pdf([WRAPPED]))

    _, title, author = pdf_io.extract_pdf_book(path)

    assert title is None
    assert author is None


def test_a_scan_yields_no_text_rather_than_a_plausible_empty_book(tmp_path):
    """A PDF of page images extracts to nothing. Silently accepting it would
    start a translation of an empty book and bill an hour of model time to it."""
    path = _written(tmp_path, 'scan.pdf', _pdf([[], []]))

    text, _, _ = pdf_io.extract_pdf_book(path)

    assert text == ''


def test_upload_reports_a_scan_instead_of_translating_nothing(tmp_path, monkeypatch):
    upload_folder = tmp_path / 'uploads'
    upload_folder.mkdir()
    monkeypatch.setattr(translator, 'UPLOAD_FOLDER', str(upload_folder))

    class _Upload:
        filename = 'scan.pdf'

        def save(self, path):
            with open(path, 'wb') as handle:
                handle.write(_pdf([[], []]))

    with pytest.raises(translator.UploadError, match='OCR'):
        translator.read_uploaded_book(_Upload(), 'en')


def test_a_pdf_enters_the_pipeline_as_plain_text_with_no_chapters(tmp_path, monkeypatch):
    """The whole point of the format: chapters stay None, so chunking, refinement
    and download take exactly the paths a .txt book takes. Only the recorded
    source format differs, and every branch downstream asks about 'epub'."""
    upload_folder = tmp_path / 'uploads'
    upload_folder.mkdir()
    monkeypatch.setattr(translator, 'UPLOAD_FOLDER', str(upload_folder))

    class _Upload:
        filename = 'book.pdf'

        def save(self, path):
            with open(path, 'wb') as handle:
                handle.write(_pdf([WRAPPED], title='Source Book'))

    text, chapters, title, author, source_format, filepath = translator.read_uploaded_book(
        _Upload(), 'en')

    assert chapters is None
    assert source_format == 'pdf'
    assert title == 'Source Book'
    assert 'Mr Dursley' in text
    assert '=== Chapter 1 ===' not in text


def test_the_source_preview_reads_a_pdf_and_cleans_up(tmp_path, monkeypatch):
    def unexpected_ollama_check(*args, **kwargs):
        raise AssertionError('source preview must not call Ollama')

    monkeypatch.setattr(translator.requests, 'get', unexpected_ollama_check)
    upload_folder = tmp_path / 'uploads'
    upload_folder.mkdir()
    monkeypatch.setattr(translator, 'UPLOAD_FOLDER', str(upload_folder))
    translator.app.config.update(TESTING=True)

    response = translator.app.test_client().post(
        '/source-preview',
        data={
            'file': (BytesIO(_pdf([WRAPPED], title='Source Book', author='Source Author')), 'source.pdf'),
            'sourceLanguage': 'en',
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['source_format'] == 'pdf'
    assert payload['title'] == 'Source Book'
    assert payload['author'] == 'Source Author'
    assert payload['truncated'] is False
    assert 'Mr Dursley' in payload['preview']
    assert list(upload_folder.iterdir()) == []


def test_a_rejected_upload_does_not_stay_in_the_uploads_folder(tmp_path, monkeypatch):
    """The caller deletes the file it was handed back. A rejected file is never
    handed back, so it used to sit in uploads/ until somebody noticed."""
    upload_folder = tmp_path / 'uploads'
    upload_folder.mkdir()
    monkeypatch.setattr(translator, 'UPLOAD_FOLDER', str(upload_folder))

    class _Upload:
        filename = 'scan.pdf'

        def save(self, path):
            with open(path, 'wb') as handle:
                handle.write(_pdf([[], []]))

    with pytest.raises(translator.UploadError):
        translator.read_uploaded_book(_Upload(), 'en')

    assert list(upload_folder.iterdir()) == []


def test_a_file_that_is_not_a_pdf_at_all_is_refused_with_a_message(tmp_path, monkeypatch):
    """Renaming book.txt to book.pdf is a thing people do. pypdf raises
    something unhelpful; the upload path has to turn it into a 400."""
    upload_folder = tmp_path / 'uploads'
    upload_folder.mkdir()
    monkeypatch.setattr(translator, 'UPLOAD_FOLDER', str(upload_folder))

    class _Upload:
        filename = 'book.pdf'

        def save(self, path):
            with open(path, 'wb') as handle:
                handle.write(b'This is a plain text file wearing a .pdf name.')

    with pytest.raises(translator.UploadError, match='Could not read this PDF'):
        translator.read_uploaded_book(_Upload(), 'en')

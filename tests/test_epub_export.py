"""Both export paths go through one EPUB writer, because the one thing a
hand-rolled XML writer forgets is escaping.

An ampersand in a book title or a ``<`` in the body is not exotic — it is
Tolstoy's "War & Peace" and any dialogue with an angle bracket. Unescaped, the
result is not a slightly wrong EPUB but a file no reader will open, and the
failure only shows up on the reader's device.
"""

import json
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO

import pytest

import translator


HOSTILE = 'War & Peace <draft> "quoted"'


def _epub(chapters, title=HOSTILE, author='A & B'):
    return zipfile.ZipFile(BytesIO(
        translator.build_epub_from_chapters(chapters, title=title, author=author)
    ))


def test_every_xml_part_still_parses_with_hostile_metadata():
    book = _epub(['Tom & Jerry met.\n\n5 < 6 and 7 > 6.'])

    for name in book.namelist():
        if name.endswith(('.opf', '.ncx', '.xhtml', '.xml')):
            ET.fromstring(book.read(name))


def test_the_title_survives_as_text_rather_than_markup():
    book = _epub(['One paragraph.'])

    package = ET.fromstring(book.read('OEBPS/content.opf'))
    title = package.find('.//{http://purl.org/dc/elements/1.1/}title')

    assert title is not None
    # Round-tripped through the parser: escaped on the way in, and the reader
    # gets the original characters back rather than a mangled or truncated one.
    assert title.text == HOSTILE


def test_body_text_is_escaped_not_interpreted():
    book = _epub(['Tom & Jerry <b>not bold</b>'])

    chapter = book.read('OEBPS/chapter1.xhtml').decode('utf-8')

    assert '&amp;' in chapter
    assert '<b>' not in chapter
    paragraphs = ET.fromstring(chapter).findall('.//{*}p')
    assert [p.text for p in paragraphs] == ['Tom & Jerry <b>not bold</b>']


def test_the_mimetype_entry_is_first_and_stored_uncompressed():
    """EPUB requires it; a reader that checks will reject the file otherwise."""
    book = _epub(['Text.'])
    first = book.infolist()[0]

    assert first.filename == 'mimetype'
    assert first.compress_type == zipfile.ZIP_STORED
    assert book.read('mimetype') == b'application/epub+zip'


def test_chapters_keep_their_breaks_and_reading_order():
    book = _epub(['First.', 'Second.', 'Third.'])

    spine = ET.fromstring(book.read('OEBPS/content.opf')).find(
        '{http://www.idpf.org/2007/opf}spine'
    )

    assert spine is not None
    assert [item.get('idref') for item in spine] == [
        'chapter1', 'chapter2', 'chapter3',
    ]
    assert 'Third.' in book.read('OEBPS/chapter3.xhtml').decode('utf-8')


def test_a_missing_title_and_author_fall_back_rather_than_writing_none():
    book = _epub(['Text.'], title='', author='')

    package = ET.fromstring(book.read('OEBPS/content.opf'))
    title = package.find('.//{http://purl.org/dc/elements/1.1/}title')
    creator = package.find('.//{http://purl.org/dc/elements/1.1/}creator')

    assert title is not None and title.text == 'Translation'
    assert creator is not None and creator.text == 'Book Translator'


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The route, offline. The middleware only pings Ollama, and this export
    never reaches a model — but it still goes through the real request cycle."""
    class AvailableOllama:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(translator.requests, 'get', lambda *a, **k: AvailableOllama())
    monkeypatch.setattr(translator, 'TRANSLATIONS_FOLDER', str(tmp_path))
    translator.app.config.update(TESTING=True)
    return translator.app.test_client()


def test_the_route_builds_a_readable_file_from_a_hostile_title(client):
    """The regression this file exists for: /export/epub used to write its own
    XML with the title, author and body interpolated raw, so one ampersand
    anywhere produced a file that no reader could open."""
    response = client.post('/export/epub', json={
        'text': 'Tom & Jerry met.\n\nShe wrote <i>quickly</i>.',
        'title': HOSTILE,
        'author': 'A & B',
    })

    assert response.status_code == 200
    book = zipfile.ZipFile(BytesIO(response.data))
    for name in book.namelist():
        if name.endswith(('.opf', '.ncx', '.xhtml', '.xml')):
            ET.fromstring(book.read(name))

    package = ET.fromstring(book.read('OEBPS/content.opf'))
    title = package.find('.//{http://purl.org/dc/elements/1.1/}title')
    assert title is not None and title.text == HOSTILE


def test_the_route_refuses_an_empty_body(client):
    response = client.post('/export/epub', json={'text': '', 'title': 'X'})

    assert response.status_code == 400
    assert 'error' in json.loads(response.data)

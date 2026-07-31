"""DOCX uploads enter the same book pipeline as EPUB/TXT."""

from pathlib import Path

import pytest
from docx import Document

import docx_io
import translator


def _docx(path: Path, paragraphs, *, title=None, author=None):
    document = Document()
    if title:
        document.core_properties.title = title
    if author:
        document.core_properties.author = author
    for text, style in paragraphs:
        paragraph = document.add_paragraph(text)
        if style:
            paragraph.style = style
    document.save(path)
    return path


def test_docx_without_headings_is_plain_text(tmp_path):
    path = _docx(tmp_path / 'plain.docx', [
        ('First paragraph.', None),
        ('Second paragraph.', None),
    ], title='Plain Book', author='Author')

    text, chapters, title, author = docx_io.extract_docx_book(str(path))

    assert chapters is None
    assert title == 'Plain Book'
    assert author == 'Author'
    assert 'First paragraph.' in text
    assert 'Second paragraph.' in text


def test_docx_heading1_sections_become_chapters(tmp_path):
    path = _docx(tmp_path / 'chapters.docx', [
        ('Chapter 1', 'Heading 1'),
        ('Once upon a time.', None),
        ('Chapter 2', 'Heading 1'),
        ('The adventure continued.', None),
    ])

    text, chapters, _, _ = docx_io.extract_docx_book(str(path))

    assert chapters is not None
    assert len(chapters) == 2
    assert chapters[0].startswith('Chapter 1')
    assert 'Once upon a time.' in chapters[0]
    assert 'The adventure continued.' in chapters[1]
    assert 'Chapter 1' in text


def test_read_uploaded_book_accepts_docx(tmp_path, monkeypatch):
    upload_folder = tmp_path / 'uploads'
    upload_folder.mkdir()
    monkeypatch.setattr(translator, 'UPLOAD_FOLDER', str(upload_folder))
    source = _docx(tmp_path / 'source.docx', [
        ('Chapter 1', 'Heading 1'),
        ('Body one.', None),
        ('Chapter 2', 'Heading 1'),
        ('Body two.', None),
    ], title='Source Doc')

    class _Upload:
        filename = 'source.docx'

        def save(self, path):
            Path(path).write_bytes(source.read_bytes())

    text, chapters, title, author, source_format, filepath = translator.read_uploaded_book(
        _Upload(), 'en',
    )

    assert source_format == 'docx'
    assert title == 'Source Doc'
    assert chapters is not None and len(chapters) == 2
    assert '=== Chapter 1 ===' in text
    assert Path(filepath).exists()

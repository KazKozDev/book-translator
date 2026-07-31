"""PDF chapter detection from headings (outline-free fixtures)."""

from pathlib import Path

import pdf_io
from test_pdf_upload import _pdf, _written


def test_pdf_chapter_headings_become_chapters(tmp_path):
    pages = [
        ['CHAPTER 1', 'Mr Dursley was the director of a firm.', 'He was proud.'],
        ['CHAPTER 2', 'The adventure continued across the lake.'],
        ['CHAPTER 3', 'They reached the castle before nightfall.'],
    ]
    # Repeat enough pages that furniture heuristics stay quiet; headings alone
    # should still split the rejoined text into chapters.
    path = _written(tmp_path, 'chapters.pdf', _pdf(pages))

    text, chapters, _, _ = pdf_io.extract_pdf_book(path)

    assert 'Mr Dursley' in text
    assert chapters is not None
    assert len(chapters) >= 2
    assert any('CHAPTER 1' in chapter for chapter in chapters)
    assert any('CHAPTER 2' in chapter for chapter in chapters)


def test_pdf_without_headings_keeps_chapters_none(tmp_path):
    pages = [
        ['Mr Dursley was the director of a firm called Grunnings',
         'which made drills. He was a big, beefy man with hardly',
         'any neck, although he did have a very large mustache.'],
    ]
    path = _written(tmp_path, 'plain.pdf', _pdf(pages))

    _, chapters, _, _ = pdf_io.extract_pdf_book(path)

    assert chapters is None

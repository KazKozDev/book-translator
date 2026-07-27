"""Decoding an uploaded book, which is where a wrong guess is silent.

A single-byte codepage cannot fail to decode: every byte maps to some
character. So none of these cases raise — they either return the book or return
mojibake, and only a test can tell the difference.
"""

import translator


def _written(tmp_path, name, text, encoding):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return str(path)


RUSSIAN = 'Мистер Дурсль был директором фирмы «Граннингс».'
PORTUGUESE = 'Era uma vez um coração, à noite, com ção e ãs.'


def test_utf8_is_settled_without_guessing(tmp_path):
    assert translator.decode_text_file(
        _written(tmp_path, 'book.txt', RUSSIAN, 'utf-8'), 'ru') == RUSSIAN


def test_a_windows_notepad_bom_does_not_reach_the_text(tmp_path):
    """Plain utf-8 decoding keeps the BOM as an invisible first character, which
    then travels into the first chunk and the first glossary term."""
    path = _written(tmp_path, 'book.txt', '﻿Chapter One', 'utf-8')

    assert translator.decode_text_file(path, 'en') == 'Chapter One'


def test_cp1251_russian_survives_when_the_source_language_says_so(tmp_path):
    """Read as cp1252 this decodes cleanly into "Ìèñòåð Äóðñëü" — no exception,
    just a ruined book. The selected source language is what prevents it."""
    path = _written(tmp_path, 'book.txt', RUSSIAN, 'cp1251')

    assert translator.decode_text_file(path, 'ru') == RUSSIAN


def test_cp1252_portuguese_survives_too(tmp_path):
    """The same failure in the other direction: this file used to be decoded as
    cp1251, the only fallback there was, and came back as Cyrillic nonsense."""
    path = _written(tmp_path, 'book.txt', PORTUGUESE, 'cp1252')

    assert translator.decode_text_file(path, 'pt') == PORTUGUESE


def test_without_a_language_hint_the_western_codepage_wins(tmp_path):
    """Documented, not desired. Cyrillic in cp1251 and Western text in cp1252
    both decode to coherent-looking output under either codepage, so nothing but
    real language detection can separate them. Both upload endpoints require a
    source language, so the app never takes this path — a direct caller can.
    """
    path = _written(tmp_path, 'book.txt', RUSSIAN, 'cp1251')

    assert translator.decode_text_file(path) != RUSSIAN
    assert translator.decode_text_file(path).startswith('Ìèñòåð')


def test_the_language_hint_is_read_loosely(tmp_path):
    """A caller may pass 'ru', 'RU' or 'ru-RU'; all three mean cp1251."""
    path = _written(tmp_path, 'book.txt', RUSSIAN, 'cp1251')

    for hint in ('ru', 'RU', ' ru-RU '):
        assert translator.decode_text_file(path, hint) == RUSSIAN

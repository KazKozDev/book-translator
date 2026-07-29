"""Per-document editable glossary persistence."""

import sqlite3
from pathlib import Path

import pytest

import translator as app_module


FINGERPRINT = 'a' * 64
OTHER_FINGERPRINT = 'b' * 64


@pytest.fixture
def glossary_client(tmp_path, monkeypatch):
    def unexpected_ollama_check(*args, **kwargs):
        raise AssertionError('workspace glossary persistence must not call Ollama')

    monkeypatch.setattr(app_module.requests, 'get', unexpected_ollama_check)
    database_path = tmp_path / 'translations.db'
    monkeypatch.setattr(app_module, 'DB_PATH', str(database_path))
    app_module.init_db()
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client(), database_path


def _query(source='en', target='ru'):
    return {'sourceLanguage': source, 'targetLanguage': target}


def _draft(source='en', target='ru', glossary='Darcy => Дарси | exact'):
    return {
        'sourceLanguage': source,
        'targetLanguage': target,
        'glossary': glossary,
    }


def test_workspace_glossary_is_isolated_by_book_and_language_pair(glossary_client):
    client, database_path = glossary_client

    response = client.put(f'/workspace-glossary/{FINGERPRINT}', json=_draft())
    assert response.status_code == 200
    assert response.get_json() == {'status': 'saved'}

    same_book = client.get(f'/workspace-glossary/{FINGERPRINT}', query_string=_query())
    assert same_book.status_code == 200
    assert same_book.get_json() == {
        'glossary': 'Darcy => Дарси | exact',
        'found': True,
    }

    new_book = client.get(f'/workspace-glossary/{OTHER_FINGERPRINT}', query_string=_query())
    assert new_book.status_code == 200
    assert new_book.get_json() == {'glossary': '', 'found': False}

    other_language = client.get(
        f'/workspace-glossary/{FINGERPRINT}',
        query_string=_query(target='es'),
    )
    assert other_language.status_code == 200
    assert other_language.get_json() == {'glossary': '', 'found': False}

    with sqlite3.connect(database_path) as conn:
        rows = conn.execute('SELECT document_fingerprint, source_lang, target_lang, glossary FROM workspace_glossaries').fetchall()
    assert rows == [(FINGERPRINT, 'en', 'ru', 'Darcy => Дарси | exact')]


def test_workspace_glossary_can_be_replaced_with_an_empty_draft(glossary_client):
    client, _ = glossary_client

    client.put(f'/workspace-glossary/{FINGERPRINT}', json=_draft(glossary='name => имя'))
    response = client.put(f'/workspace-glossary/{FINGERPRINT}', json=_draft(glossary=''))

    assert response.status_code == 200
    stored = client.get(f'/workspace-glossary/{FINGERPRINT}', query_string=_query())
    assert stored.get_json() == {'glossary': '', 'found': True}


@pytest.mark.parametrize(
('fingerprint', 'payload', 'message'),
[
    ('short', _draft(), 'Invalid document fingerprint'),
    (FINGERPRINT, _draft(source=''), 'Source language is required'),
    (FINGERPRINT, {'sourceLanguage': 'en', 'targetLanguage': 'ru', 'glossary': 12}, 'Glossary must be text'),
])
def test_workspace_glossary_rejects_invalid_drafts(glossary_client, fingerprint, payload, message):
    client, _ = glossary_client

    response = client.put(f'/workspace-glossary/{fingerprint}', json=payload)

    assert response.status_code == 400
    assert response.get_json()['error'] == message


def test_client_drops_legacy_global_glossary_storage():
    index_html = (
        Path(app_module.__file__).parent / 'static' / 'index.html'
    ).read_text(encoding='utf-8')

    assert "glossary: 'workspaceGlossary'" not in index_html
    assert "localStorage.removeItem(LEGACY_WORKSPACE_GLOSSARY_KEY)" in index_html
    assert "crypto.subtle.digest('SHA-256'" in index_html
    assert 'workspace-glossary/' in index_html

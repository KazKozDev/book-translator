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


def _job(database_path, *, fingerprint=FINGERPRINT, terms=()):
    """A translation row as /translate would have written it."""
    with sqlite3.connect(database_path) as conn:
        cur = conn.execute(
            '''
            INSERT INTO translations (
                filename, source_lang, target_lang, model, status,
                document_fingerprint
            ) VALUES ('book.txt', 'en', 'ru', 'model', 'completed', ?)
            ''',
            (fingerprint,),
        )
        translation_id = cur.lastrowid
        conn.executemany(
            '''
            INSERT INTO translation_terms (
                translation_id, source_term, target_term, enforcement_mode, status
            ) VALUES (?, ?, ?, ?, 'verified')
            ''',
            [(translation_id, *term) for term in terms],
        )
    return translation_id


def test_a_reopened_translation_carries_its_book_and_its_glossary(glossary_client):
    """Reopening a job — which is also what a page reload does — used to leave
    the glossary editor empty: the File the fingerprint was computed from is
    gone from the tab, so nothing could find the saved draft. The job now
    records the fingerprint, which is the binding the editor rebinds through."""
    client, database_path = glossary_client
    client.put(f'/workspace-glossary/{FINGERPRINT}', json=_draft())
    translation_id = _job(database_path, terms=[('Darcy', 'Дарси', 'exact')])

    reopened = client.get(f'/translations/{translation_id}').get_json()

    assert reopened['document_fingerprint'] == FINGERPRINT
    assert reopened['glossary'] == 'Darcy => Дарси | exact'
    draft = client.get(
        f'/workspace-glossary/{reopened["document_fingerprint"]}',
        query_string=_query(reopened['source_lang'], reopened['target_lang']),
    )
    assert draft.get_json() == {'glossary': 'Darcy => Дарси | exact', 'found': True}


def test_a_job_without_a_fingerprint_still_shows_the_terms_it_ran_under(glossary_client):
    """Pasted text has no file to fingerprint, and jobs from before the column
    existed have no value in it. Those cannot rebind an editable draft, but the
    approved terms are still on the job and are what the editor falls back to."""
    client, database_path = glossary_client
    translation_id = _job(
        database_path,
        fingerprint=None,
        terms=[('Darcy', 'Дарси', 'exact'), ('Netherfield', 'Незерфилд', 'inflectable')],
    )

    reopened = client.get(f'/translations/{translation_id}').get_json()

    assert reopened['document_fingerprint'] is None
    assert reopened['glossary'] == (
        'Darcy => Дарси | exact\nNetherfield => Незерфилд | inflectable'
    )


def test_a_reopened_glossary_parses_back_into_the_terms_it_came_from(glossary_client):
    """The rebuilt text goes back into the same textarea and through the same
    parser as anything typed by hand, so it has to survive the round trip."""
    client, database_path = glossary_client
    terms = [('Darcy', 'Дарси', 'exact'), ('Mr Bennet', 'мистер Беннет', 'preferred')]
    translation_id = _job(database_path, terms=terms)

    text = client.get(f'/translations/{translation_id}').get_json()['glossary']

    assert [
        (term.source, term.target, term.mode)
        for term in app_module.TerminologyManager.from_text(text).terms
    ] == terms


def test_a_translation_with_no_glossary_reopens_with_an_empty_one(glossary_client):
    client, database_path = glossary_client
    translation_id = _job(database_path)

    assert client.get(f'/translations/{translation_id}').get_json()['glossary'] == ''


def test_client_drops_legacy_global_glossary_storage():
    index_html = (
        Path(app_module.__file__).parent / 'static' / 'index.html'
    ).read_text(encoding='utf-8')

    assert "glossary: 'workspaceGlossary'" not in index_html
    assert "localStorage.removeItem(LEGACY_WORKSPACE_GLOSSARY_KEY)" in index_html
    assert "crypto.subtle.digest('SHA-256'" in index_html
    assert 'workspace-glossary/' in index_html


def test_client_restores_the_glossary_when_it_reopens_a_translation():
    index_html = (
        Path(app_module.__file__).parent / 'static' / 'index.html'
    ).read_text(encoding='utf-8')

    assert 'loadWorkspaceGlossaryForTranslation(t);' in index_html
    assert "formData.append('documentFingerprint', fingerprintForJob);" in index_html

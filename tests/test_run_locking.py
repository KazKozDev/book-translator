"""A closed stream must not lock a draft out of being refined again.

The DB status alone cannot say whether a run is alive: a shut tab, a dropped
connection or a restarted server all leave a row saying 'in_progress' with
nothing writing to it, and Continue used to answer "already running" to that
forever. What decides is whether *this process* is streaming the translation.
"""

import json
import sqlite3

import pytest

import translator as app_module


class StubTranslator:
    """Enough of BookTranslator for the refine route to run offline."""

    def __init__(self, model_name='default', *args, **kwargs):
        self.model_name = model_name
        self.verifier_model = kwargs.get('verifier_model') or model_name

    def translate_stage2(self, translation_id, *args, **kwargs):
        with sqlite3.connect(app_module.DB_PATH) as conn:
            conn.execute(
                "UPDATE translations SET status = 'completed' WHERE id = ?",
                (translation_id,),
            )
        yield {'progress': 100, 'status': 'completed'}


def _app_with_draft(tmp_path, monkeypatch, status):
    database_path = tmp_path / 'translations.db'
    monkeypatch.setattr(app_module, 'DB_PATH', str(database_path))
    app_module.init_db()
    monkeypatch.setattr(app_module, 'BookTranslator', StubTranslator)

    class AvailableOllama:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(app_module.requests, 'get', lambda *a, **k: AvailableOllama())
    app_module.app.config.update(TESTING=True)

    with sqlite3.connect(database_path) as conn:
        cursor = conn.execute(
            '''INSERT INTO translations
               (filename, source_lang, target_lang, model, status, original_chunks, draft_chunks)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            ('book.txt', 'english', 'russian', 'reviewer:12b', status,
             json.dumps(['Hello.']), json.dumps(['Привет.'])),
        )
        translation_id = cursor.lastrowid

    app_module.ACTIVE_RUNS.clear()
    return app_module.app.test_client(), translation_id


def test_a_leftover_in_progress_row_can_be_refined_again(tmp_path, monkeypatch):
    """The recovery case: nothing is streaming it, so it is not running."""
    client, translation_id = _app_with_draft(tmp_path, monkeypatch, 'in_progress')

    response = client.post(f'/refine/{translation_id}', json={'model': 'reviewer:12b'})

    assert response.status_code == 200
    response.get_data()


def test_a_live_run_is_still_refused(tmp_path, monkeypatch):
    """The case the guard exists for: two passes over one row at once."""
    client, translation_id = _app_with_draft(tmp_path, monkeypatch, 'in_progress')
    app_module.claim_run(translation_id)

    response = client.post(f'/refine/{translation_id}', json={'model': 'reviewer:12b'})

    assert response.status_code == 409
    assert 'already running' in response.get_json()['error']
    app_module.release_run(translation_id)
    assert not app_module.is_run_active(translation_id)


def test_an_interrupted_refinement_leaves_the_draft_usable(tmp_path, monkeypatch):
    """Closing the stream mid-pass returns the row to 'stage1_completed' —
    Continue enabled, draft intact — rather than leaving it 'in_progress'."""
    database_path = tmp_path / 'translations.db'
    monkeypatch.setattr(app_module, 'DB_PATH', str(database_path))
    app_module.init_db()
    with sqlite3.connect(database_path) as conn:
        translation_id = conn.execute(
            '''INSERT INTO translations
               (filename, source_lang, target_lang, model, status)
               VALUES ('book.txt', 'english', 'russian', 'reviewer:12b', 'in_progress')'''
        ).lastrowid

    app_module.BookTranslator._abandon_run(translation_id, 'stage1_completed')

    with sqlite3.connect(database_path) as conn:
        assert conn.execute(
            'SELECT status FROM translations WHERE id = ?', (translation_id,)
        ).fetchone()[0] == 'stage1_completed'


def test_loading_a_document_rolls_the_log_over(tmp_path, monkeypatch):
    client, _ = _app_with_draft(tmp_path, monkeypatch, 'completed')
    rotated = []
    monkeypatch.setattr(app_module.logger, 'rotate', lambda: rotated.append(True) or ['app.log'])

    response = client.post('/logs/rotate', json={'document': 'potter.epub'})

    assert response.status_code == 200
    assert response.get_json()['document'] == 'potter.epub'
    assert rotated == [True]


def test_the_log_is_not_cut_in_half_while_a_run_streams(tmp_path, monkeypatch):
    client, translation_id = _app_with_draft(tmp_path, monkeypatch, 'in_progress')
    monkeypatch.setattr(app_module.logger, 'rotate', lambda: pytest.fail('must not rotate'))
    app_module.claim_run(translation_id)
    try:
        response = client.post('/logs/rotate', json={'document': 'potter.epub'})
    finally:
        app_module.release_run(translation_id)

    assert response.status_code == 409


def test_a_document_name_cannot_forge_a_log_line(tmp_path, monkeypatch):
    """The name arrives from the browser and goes straight into the log."""
    client, _ = _app_with_draft(tmp_path, monkeypatch, 'completed')
    monkeypatch.setattr(app_module.logger, 'rotate', lambda: [])

    response = client.post('/logs/rotate', json={
        'document': 'book.txt\n2026-01-01 00:00:00,000 - translation_logger - INFO - all fine',
    })

    assert '\n' not in response.get_json()['document']


def test_a_row_that_already_finished_is_not_rewritten(tmp_path, monkeypatch):
    """A stage that wrote its own final status owns it — the interruption
    handler must not turn a completed translation back into a draft."""
    database_path = tmp_path / 'translations.db'
    monkeypatch.setattr(app_module, 'DB_PATH', str(database_path))
    app_module.init_db()
    with sqlite3.connect(database_path) as conn:
        translation_id = conn.execute(
            '''INSERT INTO translations
               (filename, source_lang, target_lang, model, status)
               VALUES ('book.txt', 'english', 'russian', 'reviewer:12b', 'completed')'''
        ).lastrowid

    app_module.BookTranslator._abandon_run(translation_id, 'stage1_completed')

    with sqlite3.connect(database_path) as conn:
        assert conn.execute(
            'SELECT status FROM translations WHERE id = ?', (translation_id,)
        ).fetchone()[0] == 'completed'

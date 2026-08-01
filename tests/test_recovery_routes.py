"""Recovery of failed translations: the two routes and the daily cleanup.

These paths only ever run after something already went wrong, which is the
worst moment to discover that the retry route itself is broken. The tests use
the real Flask and SQLite paths against a temporary database.
"""

import sqlite3

import pytest

import translator as app_module


class _HealthyOllama:
    """The single response shape ``check_ollama`` looks at."""

    def raise_for_status(self):
        return None


def _insert_failed(conn, translation_id, *, age_days=0):
    conn.execute(
        '''
        INSERT INTO translations (
            id, filename, source_lang, target_lang, model, status,
            progress, current_chunk, error_message, created_at
        ) VALUES (?, ?, 'en', 'ru', 'translator:12b', 'error', 40, 3, 'boom',
                  datetime('now', ?))
        ''',
        (translation_id, f'book-{translation_id}.txt', f'-{age_days} days'),
    )


@pytest.fixture
def recovery_app(tmp_path, monkeypatch):
    database_path = tmp_path / 'translations.db'
    monkeypatch.setattr(app_module, 'DB_PATH', str(database_path))
    app_module.init_db()

    # Neither route is exempt from the Ollama health check, so without this
    # they answer 503 on any machine that is not running Ollama — which is
    # every CI runner. The check itself is not what these tests are about.
    monkeypatch.setattr(
        app_module.requests, 'get', lambda *args, **kwargs: _HealthyOllama(),
    )

    with sqlite3.connect(database_path) as conn:
        _insert_failed(conn, 1)
        conn.execute(
            '''
            INSERT INTO chunks (
                translation_id, chunk_number, original_text, status,
                error_message
            ) VALUES (1, 0, 'One', 'error', 'boom'),
                     (1, 1, 'Two', 'completed', NULL)
            ''',
        )
        conn.execute(
            '''
            INSERT INTO translations (
                id, filename, source_lang, target_lang, model, status
            ) VALUES (2, 'done.txt', 'en', 'ru', 'translator:12b', 'completed')
            ''',
        )

    return app_module.app.test_client(), database_path


def test_failed_translations_lists_only_errors(recovery_app):
    client, _ = recovery_app

    response = client.get('/failed-translations')

    assert response.status_code == 200
    assert [row['id'] for row in response.get_json()] == [1]


def test_retry_resets_the_translation_and_its_failed_chunks(recovery_app):
    client, database_path = recovery_app

    response = client.post('/retry-translation/1')

    assert response.status_code == 200
    with sqlite3.connect(database_path) as conn:
        assert conn.execute(
            'SELECT status, progress, current_chunk, error_message '
            'FROM translations WHERE id = 1'
        ).fetchone() == ('pending', 0.0, 0, None)
        # The completed chunk keeps its result: a retry resumes the run, it
        # does not translate the whole book again.
        assert conn.execute(
            'SELECT status FROM chunks WHERE translation_id = 1 '
            'ORDER BY chunk_number'
        ).fetchall() == [('pending',), ('completed',)]


def test_cleanup_removes_only_old_failures(recovery_app):
    _, database_path = recovery_app
    with sqlite3.connect(database_path) as conn:
        _insert_failed(conn, 3, age_days=30)

    app_module._cleanup_failed_translations()

    with sqlite3.connect(database_path) as conn:
        remaining = conn.execute(
            'SELECT id FROM translations ORDER BY id'
        ).fetchall()
    assert remaining == [(1,), (2,)]

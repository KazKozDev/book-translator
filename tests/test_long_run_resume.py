"""Stage 1 must survive a closed tab and resume from the last finished chunk."""

import json
import sqlite3
import time

import pytest

import translator as app_module


class FakeTranslator:
    """Stage 1 stub that records where it started and finishes two chunks."""

    calls = []

    def __init__(self, model_name='stub', *args, **kwargs):
        self.model_name = model_name
        self.verifier_model = kwargs.get('verifier_model') or model_name

    def translate_stage1(self, text, source_lang, target_lang, translation_id,
                         genre='unknown', terminology=None, chapters=None,
                         resume=False):
        FakeTranslator.calls.append({
            'translation_id': translation_id,
            'resume': resume,
        })
        with sqlite3.connect(app_module.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                'SELECT original_chunks, draft_chunks FROM translations WHERE id = ?',
                (translation_id,),
            ).fetchone()
            chunks = json.loads(row['original_chunks']) if row and row['original_chunks'] else ['A', 'B']
            drafts = json.loads(row['draft_chunks'] or '[]') if row else []
            start = len(drafts) if resume else 0
            if not resume:
                drafts = []
                conn.execute(
                    "UPDATE translations SET original_chunks = ?, draft_chunks = ?, "
                    "status = 'in_progress', total_chunks = ? WHERE id = ?",
                    (json.dumps(chunks), json.dumps(drafts), len(chunks), translation_id),
                )
            else:
                conn.execute(
                    "UPDATE translations SET status = 'in_progress', error_message = NULL "
                    "WHERE id = ?",
                    (translation_id,),
                )
            conn.commit()
            app_module.claim_run(translation_id)
            try:
                for i in range(start, len(chunks)):
                    drafts.append(f'T{chunks[i]}')
                    progress = ((i + 1) / len(chunks)) * 100
                    conn.execute(
                        "UPDATE translations SET progress = ?, draft_chunks = ?, "
                        "machine_translation = ?, current_chunk = ? WHERE id = ?",
                        (
                            progress,
                            json.dumps(drafts, ensure_ascii=False),
                            '\n\n'.join(drafts),
                            i + 1,
                            translation_id,
                        ),
                    )
                    conn.commit()
                    yield {
                        'progress': progress,
                        'stage': 'primary_translation',
                        'batch_index': i + 1,
                        'machine_translation_chunk': drafts[-1],
                        'current_chunk': i + 1,
                        'total_chunks': len(chunks),
                    }
                    time.sleep(0.05)
                conn.execute(
                    "UPDATE translations SET status = 'stage1_completed', progress = 100, "
                    "error_message = NULL WHERE id = ?",
                    (translation_id,),
                )
                conn.commit()
                yield {'progress': 100, 'status': 'stage1_completed'}
            finally:
                app_module.release_run(translation_id)


def _client(tmp_path, monkeypatch):
    database_path = tmp_path / 'translations.db'
    monkeypatch.setattr(app_module, 'DB_PATH', str(database_path))
    app_module.init_db()
    FakeTranslator.calls = []
    monkeypatch.setattr(app_module, 'BookTranslator', FakeTranslator)
    app_module.ACTIVE_RUNS.clear()
    app_module._clear_progress_queue  # ensure helpers exist
    with app_module._PROGRESS_QUEUES_LOCK:
        app_module._PROGRESS_QUEUES.clear()

    class AvailableOllama:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(app_module.requests, 'get', lambda *a, **k: AvailableOllama())
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client(), database_path


def test_orphaned_in_progress_is_listed_as_interrupted(tmp_path, monkeypatch):
    client, database_path = _client(tmp_path, monkeypatch)
    with sqlite3.connect(database_path) as conn:
        cur = conn.execute(
            '''INSERT INTO translations
               (filename, source_lang, target_lang, model, status,
                original_chunks, draft_chunks, progress)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                'book.txt', 'en', 'ru', 'stub', 'in_progress',
                json.dumps(['One.', 'Two.']), json.dumps(['Один.']), 50,
            ),
        )
        translation_id = cur.lastrowid

    response = client.get('/translations')
    assert response.status_code == 200
    rows = response.get_json()['translations']
    match = next(item for item in rows if item['id'] == translation_id)
    assert match['status'] == 'interrupted'
    assert match['running'] is False


def test_resume_continues_from_partial_draft(tmp_path, monkeypatch):
    client, database_path = _client(tmp_path, monkeypatch)
    with sqlite3.connect(database_path) as conn:
        cur = conn.execute(
            '''INSERT INTO translations
               (filename, source_lang, target_lang, model, status, original_text,
                original_chunks, draft_chunks, chunk_chapter_map, progress, genre)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                'book.txt', 'en', 'ru', 'stub', 'interrupted', 'One.\n\nTwo.',
                json.dumps(['One.', 'Two.']), json.dumps(['Один.']),
                json.dumps([0, 0]), 50, 'fiction',
            ),
        )
        translation_id = cur.lastrowid

    response = client.post(f'/resume-translation/{translation_id}')
    assert response.status_code == 200
    # Drain SSE until the detached job finishes.
    payload = response.get_data(as_text=True)
    assert 'stage1_completed' in payload or 'primary_translation' in payload

    # Wait briefly if the worker is still flushing.
    for _ in range(50):
        if not app_module.is_run_active(translation_id):
            break
        time.sleep(0.05)

    assert FakeTranslator.calls and FakeTranslator.calls[-1]['resume'] is True
    with sqlite3.connect(database_path) as conn:
        row = conn.execute(
            'SELECT status, draft_chunks FROM translations WHERE id = ?',
            (translation_id,),
        ).fetchone()
    assert row[0] == 'stage1_completed'
    assert json.loads(row[1]) == ['Один.', 'TTwo.']


def test_closing_sse_does_not_mark_detached_job_interrupted(tmp_path, monkeypatch):
    """The progress consumer can stop; the worker must keep writing chunks."""
    client, database_path = _client(tmp_path, monkeypatch)
    with sqlite3.connect(database_path) as conn:
        cur = conn.execute(
            '''INSERT INTO translations
               (filename, source_lang, target_lang, model, status, original_text,
                original_chunks, draft_chunks, chunk_chapter_map, progress, genre)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                'book.txt', 'en', 'ru', 'stub', 'interrupted', 'One.\n\nTwo.',
                json.dumps(['One.', 'Two.']), json.dumps([]),
                json.dumps([0, 0]), 0, 'fiction',
            ),
        )
        translation_id = cur.lastrowid

    response = client.post(f'/resume-translation/{translation_id}')
    assert response.status_code == 200
    # Drop the client immediately without reading — like closing a tab.
    response.close()

    for _ in range(100):
        with sqlite3.connect(database_path) as conn:
            status = conn.execute(
                'SELECT status FROM translations WHERE id = ?',
                (translation_id,),
            ).fetchone()[0]
        if status == 'stage1_completed' and not app_module.is_run_active(translation_id):
            break
        time.sleep(0.05)
    else:
        pytest.fail('detached job did not finish after the SSE client disconnected')

    with sqlite3.connect(database_path) as conn:
        status, drafts = conn.execute(
            'SELECT status, draft_chunks FROM translations WHERE id = ?',
            (translation_id,),
        ).fetchone()
    assert status == 'stage1_completed'
    assert len(json.loads(drafts)) == 2

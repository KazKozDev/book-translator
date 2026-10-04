"""What Continue is allowed to do to the draft.

'auto' patches it. 'manual' reports what it would patch and writes nothing,
so every edit is the reader's own decision in the review desk. 'skip' calls no
model and the draft becomes the final text. These tests run the real Stage 2
loop against SQLite with the model scripted.
"""

import json
import sqlite3
import threading

import pytest

import translator as app_module
from translator import BookTranslator


SOURCE = 'Mr Dursley was the director of a firm called Grunnings.'
DRAFT = 'Мистер Дурсли был директором фирмы Grunnings.'
FIX = json.dumps([{
    'span': 'Grunnings', 'replacement': 'Граннингс',
    'type': 'terminology', 'severity': 'major',
}])


class RecordingCache:
    def __init__(self, hit=None):
        self.hit = hit
        self.reads = 0
        self.writes = 0

    def get_cached_translation(self, *args, **kwargs):
        self.reads += 1
        return self.hit

    def cache_translation(self, *args, **kwargs):
        self.writes += 1


@pytest.fixture
def draft_job(tmp_path, monkeypatch):
    database_path = tmp_path / 'translations.db'
    monkeypatch.setattr(app_module, 'DB_PATH', str(database_path))
    app_module.init_db()
    app_module.ACTIVE_RUNS.clear()
    # Every temporary database numbers its first job 1, so a queue left by
    # another test's job would end this one's stream before it had started.
    with app_module._PROGRESS_QUEUES_LOCK:
        app_module._PROGRESS_QUEUES.clear()
    app_module.RUN_PAUSE_EVENTS.clear()
    monkeypatch.setattr(app_module.time, 'sleep', lambda seconds: None)
    cache = RecordingCache()
    monkeypatch.setattr(app_module, 'cache', cache)

    class AvailableOllama:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(app_module.requests, 'get', lambda *a, **k: AvailableOllama())
    app_module.app.config.update(TESTING=True)

    with sqlite3.connect(database_path) as conn:
        cursor = conn.execute(
            '''INSERT INTO translations
               (filename, source_lang, target_lang, model, status,
                original_chunks, draft_chunks)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            ('book.txt', 'english', 'russian', 'reviewer:12b', 'stage1_completed',
             json.dumps([SOURCE]), json.dumps([DRAFT], ensure_ascii=False)),
        )
        translation_id = cursor.lastrowid
    yield translation_id, cache
    app_module.ACTIVE_RUNS.clear()


def _script_model_calls(monkeypatch, answers):
    calls = []

    def fake_call(self, prompt, temperature=0.2, read_timeout=180):
        calls.append(self.model_name)
        return answers.pop(0) if answers else None

    monkeypatch.setattr(BookTranslator, '_call_model', fake_call)
    return calls


def _run(translation_id, mode):
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')
    updates = list(translator.translate_stage2(
        translation_id, 'english', 'russian', mode=mode,
    ))
    with sqlite3.connect(app_module.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT * FROM translations WHERE id = ?', (translation_id,),
        ).fetchone()
        reviews = conn.execute(
            'SELECT * FROM chunk_reviews WHERE translation_id = ?', (translation_id,),
        ).fetchall()
    return updates, row, reviews


def test_auto_mode_still_patches_the_draft(draft_job, monkeypatch):
    translation_id, cache = draft_job
    _script_model_calls(monkeypatch, [FIX])

    updates, row, _ = _run(translation_id, 'auto')

    assert json.loads(row['final_chunks']) == [DRAFT.replace('Grunnings', 'Граннингс')]
    assert row['refinement_mode'] == 'auto'
    assert updates[-1]['refinement']['errors_applied'] == 1
    assert cache.writes == 1


def test_manual_mode_reports_the_fix_and_leaves_the_draft_alone(draft_job, monkeypatch):
    translation_id, cache = draft_job
    calls = _script_model_calls(monkeypatch, [FIX])

    updates, row, reviews = _run(translation_id, 'manual')

    assert row['status'] == 'completed'
    assert json.loads(row['final_chunks']) == [DRAFT]
    assert row['translated_text'] == DRAFT
    assert calls == ['reviewer:12b']  # the estimate only — never the verifier
    summary = updates[-1]['refinement']
    assert (summary['mode'], summary['errors_found'], summary['errors_applied']) == ('manual', 1, 0)
    details = json.loads(reviews[0]['review_details'])
    assert details['mode'] == 'manual'
    assert details['issues'][0]['replacement'] == 'Граннингс'
    assert details['applied_issues'] == []


def test_manual_mode_neither_reads_nor_writes_the_patch_cache(draft_job, monkeypatch):
    """A cached Stage 2 result is an already-patched text with no findings."""
    translation_id, cache = draft_job
    cache.hit = {'translated_text': 'Патч из кэша.'}
    _script_model_calls(monkeypatch, [FIX])

    _, row, _ = _run(translation_id, 'manual')

    assert json.loads(row['final_chunks']) == [DRAFT]
    assert (cache.reads, cache.writes) == (0, 0)


def test_manual_mode_offers_the_fix_in_the_review_desk(draft_job, monkeypatch):
    translation_id, _ = draft_job
    _script_model_calls(monkeypatch, [FIX])
    _run(translation_id, 'manual')

    payload = app_module.app.test_client().get(
        f'/translations/{translation_id}/review-chunks',
    ).get_json()

    chunk = payload['chunks'][0]
    assert chunk['final'] == DRAFT
    assert chunk['problematic'] and chunk['review_status'] == 'open'
    assert chunk['issues'][0]['span'] in chunk['final']
    assert payload['refinement']['mode'] == 'manual'


def test_skip_mode_makes_the_draft_final_without_calling_a_model(draft_job, monkeypatch):
    translation_id, cache = draft_job
    with sqlite3.connect(app_module.DB_PATH) as conn:
        conn.execute(
            '''INSERT INTO chunk_reviews (translation_id, chunk_index, review_details)
               VALUES (?, 0, ?)''',
            (translation_id, json.dumps({'issues': [{'span': 'stale'}]})),
        )
    calls = _script_model_calls(monkeypatch, [FIX])

    updates, row, reviews = _run(translation_id, 'skip')

    assert calls == []
    assert (cache.reads, cache.writes) == (0, 0)
    assert row['status'] == 'completed'
    assert json.loads(row['final_chunks']) == [DRAFT]
    assert row['translated_text'] == DRAFT
    assert row['refinement_mode'] == 'skip'
    assert reviews == []  # findings about a replaced final text are gone
    assert updates[-1]['status'] == 'completed'
    assert updates[-1]['refinement']['mode'] == 'skip'


def test_an_unknown_mode_is_refused_by_the_route(draft_job):
    translation_id, _ = draft_job

    response = app_module.app.test_client().post(
        f'/refine/{translation_id}', json={'mode': 'yolo'},
    )

    assert response.status_code == 400
    assert 'mode must be one of' in response.get_json()['error']


def test_skip_mode_does_not_need_an_instruct_model(draft_job, monkeypatch):
    """TranslateGemma cannot review, but skipping asks nothing of it."""
    translation_id, _ = draft_job
    client = app_module.app.test_client()

    refused = client.post(
        f'/refine/{translation_id}', json={'model': 'translategemma:12b'},
    )
    skipped = client.post(
        f'/refine/{translation_id}',
        json={'model': 'translategemma:12b', 'mode': 'skip'},
    )
    skipped.get_data()
    for thread in threading.enumerate():
        if thread.name == f'tolmach-job-{translation_id}':
            thread.join(timeout=5)

    assert refused.status_code == 400
    assert skipped.status_code == 200
    with sqlite3.connect(app_module.DB_PATH) as conn:
        status, final = conn.execute(
            'SELECT status, final_chunks FROM translations WHERE id = ?',
            (translation_id,),
        ).fetchone()
    assert status == 'completed'
    assert json.loads(final) == [DRAFT]

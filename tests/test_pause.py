"""Pause is in-process state, so these pin the two places it used to leak:
a paused run nobody could see, and a rendering pool that kept calling the
model while the job was held."""
import threading
import time

import translator
from translator import (
    BookTranslator,
    is_run_paused,
    pause_run,
    register_pause_event,
    resume_run,
    run_pause_checkpoint,
)


def test_paused_state_is_visible_to_the_server():
    run_id = 'pause-visible'
    assert is_run_paused(run_id) is False  # never registered
    register_pause_event(run_id)
    try:
        assert is_run_paused(run_id) is False
        assert pause_run(run_id) is True
        assert is_run_paused(run_id) is True
        assert resume_run(run_id) is True
        assert is_run_paused(run_id) is False
    finally:
        translator.RUN_PAUSE_EVENTS.pop(run_id, None)


class _HealthyOllama:
    def raise_for_status(self):
        return None


def test_translation_payload_reports_a_paused_run(tmp_path, monkeypatch):
    monkeypatch.setattr(translator, 'DB_PATH', str(tmp_path / 'translations.db'))
    translator.init_db()
    monkeypatch.setattr(
        translator.requests, 'get', lambda *args, **kwargs: _HealthyOllama(),
    )
    with translator.sqlite3.connect(translator.DB_PATH) as conn:
        translation_id = conn.execute(
            '''INSERT INTO translations
               (filename, source_lang, target_lang, model, status, original_text)
               VALUES ('paused.txt', 'en', 'fr', 'test-model', 'in_progress', 'x')'''
        ).lastrowid
    client = translator.app.test_client()
    translator.claim_run(translation_id)
    register_pause_event(translation_id)
    try:
        assert client.get(f'/translations/{translation_id}').get_json()['paused'] is False
        assert client.post(f'/pause/{translation_id}').status_code == 200
        assert client.get(f'/translations/{translation_id}').get_json()['paused'] is True
        listed = client.get('/translations').get_json()['translations']
        assert next(t for t in listed if t['id'] == translation_id)['paused'] is True
    finally:
        translator.release_run(translation_id)
        translator.RUN_PAUSE_EVENTS.pop(str(translation_id), None)


def test_paused_rendering_starts_no_new_model_call(monkeypatch):
    run_id = 'pause-rendering'
    register_pause_event(run_id)
    worker = BookTranslator(model_name='test-model')
    monkeypatch.setattr(worker, 'PREPARE_BATCH_TERMS', 1)
    monkeypatch.setattr(worker, 'PREPARE_CONCURRENCY', 2)
    calls = []
    monkeypatch.setattr(
        worker, '_call_model',
        lambda *a, **k: calls.append(1) or '[]',
    )
    text = 'Alice met Bob. Alice met Carol. Bob met Carol. ' * 3
    candidates = [
        {'surface': name, 'canonical_source': name, 'count': 6, 'kind': 'person'}
        for name in ('Alice', 'Bob', 'Carol')
    ]

    pause_run(run_id)
    job = threading.Thread(
        target=worker.propose_proper_noun_records,
        args=(text, 'en', 'fr'),
        kwargs={
            'candidates': candidates,
            'before_call': lambda: run_pause_checkpoint(run_id),
        },
        daemon=True,
    )
    try:
        job.start()
        time.sleep(0.3)
        assert calls == [], 'the pool called the model while the run was paused'
        resume_run(run_id)
        job.join(timeout=10)
        assert not job.is_alive()
        assert len(calls) == 3
    finally:
        resume_run(run_id)
        translator.RUN_PAUSE_EVENTS.pop(run_id, None)

"""Chunk review desk persistence and route contracts.

The model-backed route is scripted; these tests exercise the real Flask,
SQLite, canonical-text, and EPUB regrouping paths without opening Ollama.
"""

import json
import sqlite3
from types import SimpleNamespace

import pytest

import translator as app_module


@pytest.fixture
def review_app(tmp_path, monkeypatch):
    database_path = tmp_path / 'translations.db'
    monkeypatch.setattr(app_module, 'DB_PATH', str(database_path))
    app_module.init_db()
    app_module.ACTIVE_RUNS.clear()

    with sqlite3.connect(database_path) as conn:
        cursor = conn.execute(
            '''
            INSERT INTO translations (
                filename, source_lang, target_lang, model, status,
                original_text, machine_translation, translated_text,
                source_format, translated_chapters, original_chunks,
                draft_chunks, final_chunks, chunk_chapter_map
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                'book.epub', 'en', 'ru', 'translator:12b', 'completed',
                'One\n\nTwo\n\nThree',
                'Черновик 1\n\nЧерновик 2\n\nЧерновик 3',
                'Финал 1\n\nФинал 2\n\nФинал 3',
                'epub',
                json.dumps(['Финал 1\n\nФинал 2', 'Финал 3'], ensure_ascii=False),
                json.dumps(['One', 'Two', 'Three']),
                json.dumps(['Черновик 1', 'Черновик 2', 'Черновик 3'], ensure_ascii=False),
                json.dumps(['Финал 1', 'Финал 2', 'Финал 3'], ensure_ascii=False),
                json.dumps([0, 0, 1]),
            ),
        )
        translation_id = cursor.lastrowid
        conn.execute(
            '''
            INSERT INTO chunk_reviews (
                translation_id, chunk_index, review_details, review_status,
                revision
            ) VALUES (?, 1, ?, 'open', 2)
            ''',
            (
                translation_id,
                json.dumps({
                    'issues': [{
                        'span': 'Черновик 2',
                        'replacement': 'Исправленный фрагмент',
                        'type': 'mistranslation',
                        'severity': 'major',
                    }],
                    'verified': {
                        'accepted': False,
                        'model': 'verifier:27b',
                        'verdicts': ['draft', 'patched'],
                    },
                }, ensure_ascii=False),
            ),
        )
        for test_name in ('backtranslation_chrf', 'chunk_coverage'):
            conn.execute(
                '''
                INSERT INTO evaluation_results (
                    translation_id, test_name, flagged, note, details
                ) VALUES (?, ?, 1, ?, ?)
                ''',
                (
                    translation_id,
                    test_name,
                    test_name,
                    json.dumps(
                        {'empty_final_chunks': [2]}
                        if test_name == 'chunk_coverage' else {}
                    ),
                ),
            )

    yield app_module.app.test_client(), translation_id, database_path
    app_module.ACTIVE_RUNS.clear()


def test_review_route_returns_exact_stage2_issues_and_chunk_quality_signals(review_app):
    client, translation_id, _ = review_app

    response = client.get(f'/translations/{translation_id}/review-chunks')

    assert response.status_code == 200
    payload = response.get_json()
    second = payload['chunks'][1]
    assert payload['problematic_count'] == 1
    assert payload['open_count'] == 1
    assert second['revision'] == 2
    assert second['issues'][0]['span'] == 'Черновик 2'
    assert {signal['test'] for signal in second['signals']} == {
        'chunk_coverage', 'stage2_verifier',
    }


def test_manual_edit_atomically_rebuilds_text_and_epub_and_invalidates_only_final_checks(review_app):
    client, translation_id, database_path = review_app

    response = client.patch(
        f'/translations/{translation_id}/review-chunks/1',
        json={'text': 'Ручная правка', 'expected_revision': 2},
    )

    assert response.status_code == 200
    assert response.get_json()['revision'] == 3
    with sqlite3.connect(database_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT * FROM translations WHERE id = ?', (translation_id,),
        ).fetchone()
        review = conn.execute(
            '''SELECT * FROM chunk_reviews
               WHERE translation_id = ? AND chunk_index = 1''',
            (translation_id,),
        ).fetchone()
        remaining_tests = {
            result[0] for result in conn.execute(
                'SELECT test_name FROM evaluation_results WHERE translation_id = ?',
                (translation_id,),
            )
        }

    assert json.loads(row['final_chunks']) == ['Финал 1', 'Ручная правка', 'Финал 3']
    assert row['translated_text'] == 'Финал 1\n\nРучная правка\n\nФинал 3'
    assert json.loads(row['translated_chapters']) == [
        'Финал 1\n\nРучная правка',
        'Финал 3',
    ]
    assert review['review_status'] == 'resolved'
    assert review['resolution_kind'] == 'manual'
    assert review['revision'] == 3
    assert remaining_tests == {'backtranslation_chrf'}

    stale = client.patch(
        f'/translations/{translation_id}/review-chunks/1',
        json={'text': 'Поздняя правка', 'expected_revision': 2},
    )
    assert stale.status_code == 409


def test_active_translation_cannot_be_edited(review_app):
    client, translation_id, _ = review_app
    app_module.claim_run(translation_id)

    response = client.patch(
        f'/translations/{translation_id}/review-chunks/1',
        json={'text': 'Нельзя сейчас', 'expected_revision': 2},
    )

    assert response.status_code == 409


def test_alternatives_are_generated_for_one_chunk_and_never_auto_applied(
    review_app, monkeypatch,
):
    client, translation_id, database_path = review_app
    generated_calls = []
    judged = []

    class ScriptedTranslator:
        def __init__(self, model_name, **_):
            self.model_name = model_name
            self.terminology = None

        def generate_translation_candidate(self, text, source_lang, target_lang, **kwargs):
            generated_calls.append({
                'model': self.model_name,
                'text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                **kwargs,
            })
            return f'Вариант {len(generated_calls)}', None

        def judge_translation_candidates(self, source, candidates, source_lang, target_lang):
            judged.append((self.model_name, source, candidates, source_lang, target_lang))
            return 2, 'Candidate 2 preserves the relation most precisely.', None

    monkeypatch.setattr(app_module, 'BookTranslator', ScriptedTranslator)
    monkeypatch.setattr(
        app_module.requests,
        'get',
        lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None),
    )

    response = client.post(
        f'/translations/{translation_id}/review-chunks/1/alternatives',
        json={
            'count': 2,
            'model': 'translator:12b',
            'judge_model': 'judge:27b',
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert len(generated_calls) == 2
    assert {call['text'] for call in generated_calls} == {'Two'}
    assert [option['id'] for option in payload['options']] == [
        'current', 'candidate-1', 'candidate-2',
    ]
    assert payload['recommended_id'] == 'candidate-2'
    assert judged[0][0] == 'judge:27b'

    with sqlite3.connect(database_path) as conn:
        final_chunks = json.loads(conn.execute(
            'SELECT final_chunks FROM translations WHERE id = ?',
            (translation_id,),
        ).fetchone()[0])
    assert final_chunks[1] == 'Финал 2'


@pytest.mark.parametrize('judge_model', ['translator:12b', 'translategemma:12b'])
def test_alternatives_reject_non_independent_or_translation_only_judges(
    review_app, monkeypatch, judge_model,
):
    client, translation_id, _ = review_app
    monkeypatch.setattr(
        app_module.requests,
        'get',
        lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None),
    )

    response = client.post(
        f'/translations/{translation_id}/review-chunks/0/alternatives',
        json={
            'count': 2,
            'model': 'translator:12b',
            'judge_model': judge_model,
        },
    )

    assert response.status_code == 400

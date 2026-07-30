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


def test_frontier_review_routes_send_one_chunk_or_all_open_cases(
    review_app, monkeypatch,
):
    client, translation_id, database_path = review_app
    calls = []
    with sqlite3.connect(database_path) as conn:
        conn.execute(
            '''
            UPDATE chunk_reviews
            SET review_details = ?
            WHERE translation_id = ? AND chunk_index = 1
            ''',
            (
                json.dumps({
                    'issues': [{
                        'span': 'Финал 2',
                        'replacement': 'Исправленный фрагмент',
                        'type': 'mistranslation',
                        'severity': 'major',
                    }],
                }, ensure_ascii=False),
                translation_id,
            ),
        )

    def scripted_decision(provider, cases, api_key, model):
        calls.append((provider, cases, api_key, model))
        decisions = [{
            'chunk_index': case['chunk_index'],
            'revision': case['revision'],
            'text': case['final'].replace('Финал 2', 'Исправленный фрагмент'),
            'applied_issue_indexes': [0],
            'kept_issue_indexes': [],
            'choices': [{
                'issue_index': 0,
                'apply': True,
                'reason': 'The source supports the proposed correction.',
            }],
        } for case in cases]
        return SimpleNamespace(
            provider=provider,
            provider_label='OpenAI',
            model=model,
            decisions=decisions,
        )

    monkeypatch.setattr(app_module, 'decide_review_cases', scripted_decision)
    request_body = {
        'provider': 'openai',
        'apiKey': 'session-key',
        'model': 'gpt-5.4-mini',
    }

    one = client.post(
        f'/translations/{translation_id}/review-chunks/1/frontier-decision',
        json=request_body,
    )
    all_open = client.post(
        f'/translations/{translation_id}/review-chunks/frontier-decisions',
        json=request_body,
    )

    assert one.status_code == 200
    assert all_open.status_code == 200
    assert [case['chunk_index'] for case in calls[0][1]] == [1]
    assert [case['chunk_index'] for case in calls[1][1]] == [1]
    sent = calls[0][1][0]
    assert sent['source'] == 'Two'
    assert sent['final'] == 'Финал 2'
    assert sent['revision'] == 2
    assert sent['issues'][0]['span'] == 'Финал 2'
    assert calls[0][0::3] == ('openai', 'gpt-5.4-mini')
    decision = one.get_json()['decisions'][0]
    assert decision['text'] == 'Исправленный фрагмент'

    saved = client.patch(
        f'/translations/{translation_id}/review-chunks/1',
        json={
            'text': decision['text'],
            'expected_revision': decision['revision'],
        },
    )
    assert saved.status_code == 200
    with sqlite3.connect(database_path) as conn:
        final_chunks = json.loads(conn.execute(
            'SELECT final_chunks FROM translations WHERE id = ?',
            (translation_id,),
        ).fetchone()[0])
    assert final_chunks[1] == 'Исправленный фрагмент'


def test_frontier_review_rejects_chunks_without_applicable_manual_fixes(
    review_app,
):
    client, translation_id, _ = review_app

    response = client.post(
        f'/translations/{translation_id}/review-chunks/0/frontier-decision',
        json={'provider': 'openai'},
    )

    assert response.status_code == 400
    assert 'no applicable proposed fixes' in response.get_json()['error'].lower()


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


def test_review_route_rebuilds_the_progress_rail_counters_from_stored_chunks(
    review_app,
):
    """The refinement and glossary counters used to exist only in the
    translation stream, so reopening a finished book redrew a completed
    refinement as "Not started". Everything they showed is recoverable from
    what the run already stored per chunk."""
    client, translation_id, database_path = review_app
    with sqlite3.connect(database_path) as conn:
        conn.execute(
            '''
            INSERT INTO chunk_reviews (
                translation_id, chunk_index, review_details, review_status
            ) VALUES (?, 0, ?, 'resolved')
            ''',
            (
                translation_id,
                json.dumps({
                    'errors_found': 3,
                    'errors_applied': 2,
                    'review_model': 'translator:12b',
                    'verifier_model': 'verifier:27b',
                    'verified': {
                        'accepted': True,
                        'position_bias_detected': True,
                        'neutral_check': 'accepted',
                    },
                }, ensure_ascii=False),
            ),
        )
        # A cache hit is not a review: the stream does not count it, so
        # neither does the replay.
        conn.execute(
            '''
            INSERT INTO chunk_reviews (
                translation_id, chunk_index, review_details, review_status
            ) VALUES (?, 2, ?, 'not_needed')
            ''',
            (translation_id, json.dumps({'cache_hit': True, 'issues': []})),
        )
        conn.executemany(
            '''
            INSERT INTO translation_terms (
                translation_id, source_term, target_term, enforcement_mode
            ) VALUES (?, ?, ?, ?)
            ''',
            [
                (translation_id, 'One', 'Один', 'exact'),
                (translation_id, 'Two', 'Два', 'inflectable'),
                (translation_id, 'Nowhere', 'Нигде', 'exact'),
            ],
        )

    payload = client.get(
        f'/translations/{translation_id}/review-chunks'
    ).get_json()

    assert payload['refinement'] == {
        'errors_found': 3,
        'errors_applied': 2,
        'patches_rejected': 1,
        'position_biases': 1,
        'neutral_checks': 1,
        'chunks_reviewed': 2,
        'chunks_changed': 2,
        'review_failures': 0,
        'verifier_model': 'verifier:27b',
        'review_model': 'translator:12b',
    }
    # "used" counts the terms this source text puts in play, not the ones the
    # translation got right — the same thing the stream counts.
    assert payload['terminology'] == {'total': 3, 'used': 2, 'violations': 1}

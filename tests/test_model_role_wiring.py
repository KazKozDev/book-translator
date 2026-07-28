"""The four model choices must stay independent from Settings to the API.

This is deliberately a route-level test: it records the exact model name the
server passes into ``BookTranslator`` after each real endpoint parses its
request.  No Ollama inference is needed to prove the wiring.
"""

import io
import json
import sqlite3
from pathlib import Path

import translator as app_module


class RecordingTranslator:
    """Small stand-in that records model selection and completes each route."""

    selected_models = []
    entity_resolver_models = []
    verifier_models = []
    glossary_builder_calls = 0

    def __init__(self, model_name='default', *args, **kwargs):
        self.model_name = model_name
        self.verifier_model = kwargs.get('verifier_model') or model_name
        self.selected_models.append(model_name)

    @staticmethod
    def harvest_proper_noun_candidates(text):
        return []

    @staticmethod
    def build_glossary_candidates(text):
        RecordingTranslator.glossary_builder_calls += 1
        return [], []

    @staticmethod
    def collapse_honorific_aliases(*args):
        return []

    def adjudicate_entity_clusters(self, text, source_lang, candidates, review_queue=None):
        RecordingTranslator.entity_resolver_models.append(self.model_name)
        return candidates, []

    @staticmethod
    def propose_proper_noun_records(*args, **kwargs):
        return []

    @staticmethod
    def find_rendering_conflicts(*args):
        return []

    def translate_stage1(self, text, source_lang, target_lang, translation_id, **kwargs):
        with sqlite3.connect(app_module.DB_PATH) as conn:
            conn.execute(
                """UPDATE translations SET status = 'stage1_completed',
                   original_chunks = ?, draft_chunks = ?, machine_translation = ?
                   WHERE id = ?""",
                (json.dumps([text]), json.dumps(['draft']), 'draft', translation_id),
            )
        yield {'progress': 100, 'status': 'stage1_completed'}

    def translate_stage2(self, translation_id, *args, **kwargs):
        RecordingTranslator.verifier_models.append(self.verifier_model)
        with sqlite3.connect(app_module.DB_PATH) as conn:
            conn.execute(
                """UPDATE translations SET status = 'completed', translated_text = ?,
                   final_chunks = ? WHERE id = ?""",
                ('final', json.dumps(['final']), translation_id),
            )
        yield {'progress': 100, 'status': 'completed'}

    @staticmethod
    def _evaluation_result(name):
        return {'test': name, 'value': 1, 'flagged': False, 'note': 'ok'}

    def eval_llm_judge_stage1(self, *args):
        return self._evaluation_result('llm_judge_stage1')

    def eval_llm_judge_stage2(self, *args):
        return self._evaluation_result('llm_judge_stage2')

    def eval_llm_judge_final(self, *args):
        return self._evaluation_result('llm_judge_final')

    def eval_backtranslation_chrf(self, *args):
        return self._evaluation_result('backtranslation_chrf')


def test_constructing_a_model_helper_does_not_overwrite_the_active_run(monkeypatch):
    """``/models`` creates a helper; it must not replace the live model label."""
    monkeypatch.setattr(app_module.monitor, 'active_model', 'translation-model')

    app_module.BookTranslator('qwen3:4b-instruct')

    assert app_module.monitor.active_model == 'translation-model'


def test_translation_default_matches_settings_when_no_preference_is_saved():
    """The two pages must not default Translation to different models."""
    project_root = Path(__file__).resolve().parents[1]
    settings_page = (project_root / 'static' / 'settings.html').read_text(encoding='utf-8')
    main_page = (project_root / 'static' / 'index.html').read_text(encoding='utf-8')

    assert "const PREFERRED_MODEL = 'translategemma:12b';" in settings_page
    assert "const PREFERRED_MODEL = 'translategemma:12b';" in main_page
    assert 'const API_URL = window.location.origin;' in main_page
    assert "window.addEventListener('pageshow', syncModelsFromSettings);" in main_page
    assert "window.addEventListener('storage'," in main_page


def test_each_pipeline_role_uses_its_own_requested_model(tmp_path, monkeypatch):
    database_path = tmp_path / 'translations.db'
    monkeypatch.setattr(app_module, 'DB_PATH', str(database_path))
    app_module.init_db()
    RecordingTranslator.selected_models = []
    RecordingTranslator.entity_resolver_models = []
    RecordingTranslator.verifier_models = []
    RecordingTranslator.glossary_builder_calls = 0
    monkeypatch.setattr(app_module, 'BookTranslator', RecordingTranslator)

    # The middleware only checks Ollama availability; keep this wiring test
    # offline while retaining the real Flask request lifecycle.
    class AvailableOllama:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(app_module.requests, 'get', lambda *args, **kwargs: AvailableOllama())
    app_module.app.config.update(TESTING=True)
    client = app_module.app.test_client()

    def upload(model):
        return {
            'file': (io.BytesIO(b'Hello world.'), 'sample.txt'),
            'sourceLanguage': 'english',
            'targetLanguage': 'russian',
            'model': model,
            'genre': 'fiction',
        }

    # Stage 0 runs two model calls, and the split stays reachable over the API
    # even though the interface now offers one choice for both.
    prepare_response = client.post('/prepare', data={
        **upload('rendering-model'), 'entityModel': 'entity-model',
    }, content_type='multipart/form-data')
    assert prepare_response.status_code == 200
    assert RecordingTranslator.entity_resolver_models == ['entity-model']
    assert 'rendering-model' in RecordingTranslator.selected_models
    assert prepare_response.get_json()['entity_resolution'] == {
        'clustered_candidates': 0,
        'extracted_candidates': 0,
        'review_pairs': 0,
        'cluster_decisions': [],
        'clusters_confirmed': 0,
        'clusters_split': 0,
        'added_by_model': [],
    }
    assert RecordingTranslator.glossary_builder_calls == 1
    start_response = client.post('/translate', data=upload('translation-model'), content_type='multipart/form-data')
    assert start_response.status_code == 200
    start_response.get_data()  # Consume the SSE generator so it persists the draft.

    with sqlite3.connect(database_path) as conn:
        translation_id = conn.execute('SELECT id FROM translations').fetchone()[0]

    # Reaching Start is the user's approval of the editable glossary.  The
    # persisted row must not remain in the old "proposed" limbo state.
    approved_start = client.post('/translate', data={
        **upload('translation-model'),
        'glossary': 'Dursley => Дурсль | exact',
    }, content_type='multipart/form-data')
    assert approved_start.status_code == 200
    approved_start.get_data()
    with sqlite3.connect(database_path) as conn:
        assert conn.execute(
            'SELECT status FROM translation_terms ORDER BY translation_id DESC LIMIT 1'
        ).fetchone()[0] == 'verified'

    # Stage 2 has two roles as well: one reviews and patches, another rules on
    # the patch. They must not collapse into one model — that is the
    # configuration in which the reviewer grades its own edits.
    refinement_response = client.post(f'/refine/{translation_id}', json={
        'model': 'refinement-model', 'verifier_model': 'verifier-model',
    })
    assert refinement_response.status_code == 200
    refinement_response.get_data()  # Consume the SSE generator so it persists the final text.
    assert RecordingTranslator.verifier_models == ['verifier-model']
    for test_name in ('llm_judge_stage1', 'llm_judge_stage2', 'llm_judge_final', 'backtranslation_chrf'):
        assert client.post(
            f'/evaluate/{translation_id}/{test_name}',
            json={'judge_model': 'judge-model'},
        ).status_code == 200

    assert RecordingTranslator.selected_models == [
        # Prepare constructs two translators, one per Stage 0 role.
        'rendering-model',
        'entity-model',
        'translation-model',
        'translation-model',
        'refinement-model',
        'judge-model',
        'judge-model',
        'judge-model',
        'judge-model',
    ]


def test_prepare_runs_both_of_its_passes_on_one_model_by_default(tmp_path, monkeypatch):
    """The interface offers a single Glossary preparation choice: a small
    model matched a much larger one at both passes, and splitting the roles
    only made a 32 GB machine unload one model and load the other halfway
    through Prepare. A request with no entityModel must therefore run the
    identity pass on the model it was given, not on some default."""
    monkeypatch.setattr(app_module, 'DB_PATH', str(tmp_path / 'translations.db'))
    app_module.init_db()
    RecordingTranslator.selected_models = []
    RecordingTranslator.entity_resolver_models = []
    monkeypatch.setattr(app_module, 'BookTranslator', RecordingTranslator)

    class AvailableOllama:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(app_module.requests, 'get', lambda *args, **kwargs: AvailableOllama())
    app_module.app.config.update(TESTING=True)

    response = app_module.app.test_client().post('/prepare', data={
        'file': (io.BytesIO(b'Hello world.'), 'sample.txt'),
        'sourceLanguage': 'english',
        'targetLanguage': 'russian',
        'model': 'prepare-model',
        'genre': 'fiction',
    }, content_type='multipart/form-data')

    assert response.status_code == 200
    assert RecordingTranslator.entity_resolver_models == ['prepare-model']
    # And only one translator is constructed, so the second role cannot
    # quietly bring a second set of weights into memory.
    assert RecordingTranslator.selected_models == ['prepare-model']


def test_the_settings_page_offers_one_choice_for_glossary_preparation():
    settings_page = (Path(__file__).resolve().parents[1] / 'static' / 'settings.html').read_text(
        encoding='utf-8')
    main_page = (Path(__file__).resolve().parents[1] / 'static' / 'index.html').read_text(
        encoding='utf-8')

    assert 'Glossary preparation' in settings_page
    assert 'id="entityModel"' not in settings_page
    # The main page must not send a stale saved preference for the role it no
    # longer offers, or the two passes would silently split again.
    assert 'entityModel' not in main_page

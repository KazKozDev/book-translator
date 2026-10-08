"""Shared glossaries: terminology that carries over from one document to the next."""

import io
import json
import sqlite3
from pathlib import Path

import pytest

import shared_glossary
import translator as app_module
from terminology import GlossaryTerm


NOVEL = ('My novel', 'english', 'german')


def _record(source, target, mode='inflectable'):
    return {'source': source, 'target': target, 'mode': mode}


def test_only_entries_stage_0_found_are_overridden():
    """The rule the feature stands on: a shared glossary of 500 names must not
    put 500 names into the prompt of a chapter that mentions three of them."""
    records, applied = shared_glossary.apply(
        [_record('Darcy', 'Darcy-guess')],
        [
            GlossaryTerm('Darcy', 'Darcy', 'exact'),
            GlossaryTerm('Pemberley', 'Pemberley', 'exact'),
        ],
    )

    assert [record['source'] for record in records] == ['Darcy']
    assert applied == ['Darcy']


def test_the_whole_entry_is_taken_and_settled_entries_go_last():
    records, applied = shared_glossary.apply(
        [
            _record('Darcy', 'guess'),
            _record('Wickham', 'Wickham'),
            _record('lizzy', 'guess'),
            _record('Meryton', 'Meryton'),
        ],
        [
            GlossaryTerm('Lizzy', 'Lizzy', 'preferred', 'Elizabeth, to her family'),
            GlossaryTerm('Darcy', 'Darcy', 'exact'),
        ],
    )

    # New entries first, in Stage 0's order; then the settled ones, also in
    # Stage 0's order rather than the shared glossary's.
    assert [record['source'] for record in records] == [
        'Wickham', 'Meryton', 'Darcy', 'lizzy',
    ]
    assert applied == ['Darcy', 'lizzy']
    assert records[3] == {
        'source': 'lizzy', 'target': 'Lizzy', 'mode': 'preferred',
        'note': 'Elizabeth, to her family',
    }


def test_without_a_shared_glossary_the_records_are_untouched():
    records = [_record('Darcy', 'Darcy'), _record('Wickham', 'Wickham')]

    assert shared_glossary.apply(records, []) == (records, [])


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, 'DB_PATH', str(tmp_path / 'translations.db'))
    app_module.init_db()
    with sqlite3.connect(app_module.DB_PATH) as connection:
        yield connection


def test_writing_back_overwrites_matches_and_keeps_everything_else(conn):
    shared_glossary.store(conn, NOVEL, [
        GlossaryTerm('Darcy', 'Darcy', 'exact'),
        GlossaryTerm('Pemberley', 'Pemberley', 'exact'),
    ])

    written = shared_glossary.store(conn, NOVEL, [
        GlossaryTerm('darcy', 'Herr Darcy', 'preferred', 'formal register'),
        GlossaryTerm('Wickham', 'Wickham', 'inflectable'),
    ])

    assert written == 2
    assert shared_glossary.load(conn, NOVEL) == [
        GlossaryTerm('darcy', 'Herr Darcy', 'preferred', 'formal register'),
        GlossaryTerm('Pemberley', 'Pemberley', 'exact'),
        GlossaryTerm('Wickham', 'Wickham', 'inflectable'),
    ]


def test_terms_marked_local_are_not_written_back(conn):
    written = shared_glossary.store(
        conn, NOVEL,
        [GlossaryTerm('Darcy', 'Darcy', 'exact'), GlossaryTerm('Rose', 'Rosa', 'exact')],
        local_sources=['rose'],
    )

    assert written == 1
    assert [term.source for term in shared_glossary.load(conn, NOVEL)] == ['Darcy']


def test_a_local_edit_does_not_disturb_the_shared_entry(conn):
    """The case behind the flag: a name that is used differently in this one
    text. The run gets the edited rendering; later chapters keep the old one."""
    shared_glossary.store(conn, NOVEL, [GlossaryTerm('Rose', 'Rose', 'exact')])

    shared_glossary.store(
        conn, NOVEL, [GlossaryTerm('Rose', 'die Rose', 'exact')], local_sources=['Rose'],
    )

    assert shared_glossary.load(conn, NOVEL) == [GlossaryTerm('Rose', 'Rose', 'exact')]


def test_a_glossary_belongs_to_its_name_and_its_language_pair(conn):
    shared_glossary.store(conn, NOVEL, [GlossaryTerm('Darcy', 'Darcy', 'exact')])

    assert shared_glossary.load(conn, ('Other novel', 'english', 'german')) == []
    assert shared_glossary.load(conn, ('My novel', 'english', 'french')) == []
    assert shared_glossary.listing(conn, 'english', 'french') == []
    listed = shared_glossary.listing(conn, 'english', 'german')
    assert [(entry['name'], entry['terms']) for entry in listed] == [('My novel', 1)]


@pytest.mark.parametrize(('name', 'message'), [
    ('', 'Shared glossary name is required'),
    ('   ', 'Shared glossary name is required'),
    ('x' * 81, 'Shared glossary name exceeds 80 characters'),
    ('two\nlines', 'Shared glossary name contains control characters'),
])
def test_an_unusable_name_is_refused(name, message):
    with pytest.raises(ValueError, match=message):
        shared_glossary.scope(name, 'english', 'german')


class StandInTranslator:
    """Completes each route without a model, proposing a fixed Stage 0 result."""

    proposed = []

    def __init__(self, model_name='default', *args, **kwargs):
        self.model_name = model_name

    @staticmethod
    def build_glossary_candidates(text, progress_callback=None):
        return [], []

    def adjudicate_entity_clusters(self, text, source_lang, candidates, review_queue=None):
        return candidates, []

    @classmethod
    def propose_proper_noun_records(cls, *args, **kwargs):
        return [dict(record) for record in cls.proposed]

    find_rendering_conflicts = staticmethod(
        app_module.BookTranslator.find_rendering_conflicts
    )

    def translate_stage1(self, *args, **kwargs):
        yield {'progress': 100, 'stage': 'stage1_completed'}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, 'DB_PATH', str(tmp_path / 'translations.db'))
    app_module.init_db()
    monkeypatch.setattr(app_module, 'BookTranslator', StandInTranslator)
    StandInTranslator.proposed = []

    class AvailableOllama:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(app_module.requests, 'get', lambda *args, **kwargs: AvailableOllama())
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def _upload(**extra):
    return {
        'file': (io.BytesIO(b'Darcy met Wickham at Meryton.'), 'chapter.txt'),
        'sourceLanguage': 'english',
        'targetLanguage': 'german',
        'model': 'model',
        'genre': 'fiction',
        **extra,
    }


def _post(client, route, **extra):
    response = client.post(route, data=_upload(**extra), content_type='multipart/form-data')
    body = response.get_data(as_text=True)  # Consuming the stream lets the job finish.
    return response, body


def _completed(body):
    for line in body.splitlines():
        if line.startswith('data: '):
            event = json.loads(line[len('data: '):])
            if event.get('stage') == 'completed':
                return event
    raise AssertionError('Prepare did not emit a completed event')


def _stored(name='My novel', target='german'):
    with sqlite3.connect(app_module.DB_PATH) as connection:
        return shared_glossary.to_text(
            shared_glossary.load(connection, (name, 'english', target))
        )


def test_start_creates_the_shared_glossary_and_the_next_prepare_uses_it(client):
    """The round trip the feature exists for: approve a glossary on chapter
    one, and chapter two's Prepare comes back already corrected."""
    response, _ = _post(
        client, '/translate', sharedGlossary='My novel',
        glossary='Darcy => Herr Darcy | exact {always formal}\nPemberley => Pemberley',
    )
    assert response.status_code == 200
    assert _stored() == (
        'Darcy => Herr Darcy | exact {always formal}\n'
        'Pemberley => Pemberley | inflectable'
    )

    StandInTranslator.proposed = [
        _record('Darcy', 'Darcy'), _record('Wickham', 'Wickham'),
    ]
    response, body = _post(client, '/prepare', sharedGlossary='My novel')
    assert response.status_code == 200
    event = _completed(body)

    assert event['glossary'] == (
        'Wickham => Wickham | inflectable\n'
        'Darcy => Herr Darcy | exact {always formal}'
    )
    assert event['shared_glossary'] == {'name': 'My novel', 'applied': ['Darcy']}
    assert [entity['source'] for entity in event['entities']] == ['Wickham', 'Darcy']


def test_a_conflict_is_judged_on_the_renderings_the_run_will_use(client):
    """Stage 0 proposed two distinct targets; the shared entry makes them the
    same. The warning has to see that, since that is what Start would run."""
    _post(client, '/translate', sharedGlossary='My novel', glossary='Dursley => Dursleys')
    StandInTranslator.proposed = [
        _record('Dursley', 'Dursley'), _record('Dursleys', 'Dursleys'),
    ]

    _, body = _post(client, '/prepare', sharedGlossary='My novel')

    assert _completed(body)['rendering_conflicts'] == [{
        'target': 'dursleys',
        'sources': ['Dursley', 'Dursleys'],
        'reason': 'distinct source entities have the same target rendering',
    }]


def test_prepare_without_a_shared_glossary_is_unchanged(client):
    StandInTranslator.proposed = [_record('Darcy', 'Darcy')]

    _, body = _post(client, '/prepare')
    event = _completed(body)

    assert event['glossary'] == 'Darcy => Darcy | inflectable'
    assert event['shared_glossary'] is None


def test_start_without_a_shared_glossary_writes_none(client):
    response, _ = _post(client, '/translate', glossary='Darcy => Darcy | exact')

    assert response.status_code == 200
    with sqlite3.connect(app_module.DB_PATH) as connection:
        assert connection.execute(
            'SELECT COUNT(*) FROM shared_glossary_terms'
        ).fetchone()[0] == 0


def test_start_leaves_local_terms_out_of_the_shared_glossary(client):
    response, _ = _post(
        client, '/translate', sharedGlossary='My novel',
        glossary='Darcy => Darcy | exact\nRose => Rosa | exact',
        sharedGlossaryLocal=json.dumps(['Rose']),
    )

    assert response.status_code == 200
    assert _stored() == 'Darcy => Darcy | exact'
    # Local to the shared glossary, not to the run: the job still has both.
    with sqlite3.connect(app_module.DB_PATH) as connection:
        assert connection.execute(
            'SELECT COUNT(*) FROM translation_terms'
        ).fetchone()[0] == 2


@pytest.mark.parametrize('extra', [
    {'glossary': 'Darcy is Herr Darcy'},
    {'glossary': 'Darcy => Darcy', 'sharedGlossaryLocal': 'not json'},
    {'glossary': 'Darcy => Darcy', 'sharedGlossaryLocal': '{"Darcy": true}'},
    {'glossary': 'Darcy => Darcy', 'file': (io.BytesIO(b'not a document'), 'broken.docx')},
])
def test_a_start_that_is_refused_changes_no_shared_glossary(client, extra):
    _post(client, '/translate', sharedGlossary='My novel', glossary='Darcy => Herr Darcy')

    response, _ = _post(client, '/translate', **{'sharedGlossary': 'My novel', **extra})

    assert response.status_code == 400
    assert _stored() == 'Darcy => Herr Darcy | inflectable'
    with sqlite3.connect(app_module.DB_PATH) as connection:
        assert connection.execute('SELECT COUNT(*) FROM translations').fetchone()[0] == 1


@pytest.mark.parametrize('route', ['/prepare', '/translate'])
def test_an_unusable_name_is_reported_instead_of_ignored(client, route):
    response, _ = _post(client, route, sharedGlossary='x' * 81)

    assert response.status_code == 400
    assert response.get_json()['error'] == 'Shared glossary name exceeds 80 characters'


def test_shared_glossaries_can_be_listed_read_and_deleted(client, monkeypatch):
    _post(client, '/translate', sharedGlossary='My novel', glossary='Darcy => Darcy | exact')
    _post(client, '/translate', sharedGlossary='Essays', glossary='a => b\nc => d')

    # Managing stored glossaries is local bookkeeping and must not need Ollama.
    def unexpected_ollama_check(*args, **kwargs):
        raise AssertionError('shared glossary management must not call Ollama')

    monkeypatch.setattr(app_module.requests, 'get', unexpected_ollama_check)
    pair = {'sourceLanguage': 'english', 'targetLanguage': 'german'}

    listed = client.get('/shared-glossaries', query_string=pair).get_json()['glossaries']
    assert sorted((entry['name'], entry['terms']) for entry in listed) == [
        ('Essays', 2), ('My novel', 1),
    ]
    assert client.get('/shared-glossaries').status_code == 400

    assert client.get('/shared-glossaries/My novel', query_string=pair).get_json() == {
        'name': 'My novel', 'glossary': 'Darcy => Darcy | exact', 'terms': 1, 'found': True,
    }
    assert client.get('/shared-glossaries/Nothing', query_string=pair).get_json() == {
        'name': 'Nothing', 'glossary': '', 'terms': 0, 'found': False,
    }

    deleted = client.delete('/shared-glossaries/My novel', query_string=pair)
    assert deleted.get_json() == {'status': 'deleted', 'terms': 1}
    assert client.delete('/shared-glossaries/My novel', query_string=pair).status_code == 404
    remaining = client.get('/shared-glossaries', query_string=pair).get_json()['glossaries']
    assert [entry['name'] for entry in remaining] == ['Essays']


def test_the_interface_names_the_glossary_and_sends_what_to_keep_out():
    index_html = (
        Path(app_module.__file__).parent / 'static' / 'index.html'
    ).read_text(encoding='utf-8')

    assert 'id="sharedGlossaryName"' in index_html
    # Prepare only names the glossary; Start also says which entries stay local.
    assert 'appendSharedGlossaryFields(formData);' in index_html
    assert 'appendSharedGlossaryFields(formData, { withLocal: true });' in index_html
    assert "formData.append('sharedGlossaryLocal', JSON.stringify(local));" in index_html

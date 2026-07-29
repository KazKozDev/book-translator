import os

import pytest

import frontier_glossary as frontier


ORIGINAL = (
    'Hermione => Гермиона\n'
    'Hogwarts => | exact\n'
    'Ministry of Magic'
)
CORRECTED = (
    'Hermione => Гермиона | inflectable\n'
    'Hogwarts => Хогвартс | exact\n'
    'Ministry of Magic => Министерство магии | exact'
)


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._data


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_openai_uses_web_search_and_returns_only_validated_glossary():
    client = FakeClient([FakeResponse({
        'output': [
            {'type': 'web_search_call', 'id': 'ws_1'},
            {
                'type': 'message',
                'content': [{'type': 'output_text', 'text': CORRECTED}],
            },
        ],
    })])

    result = frontier.verify_glossary(
        'openai', 'PROMPT', ORIGINAL, 'session-key', client=client,
    )

    assert result.glossary == CORRECTED
    assert result.model == 'gpt-5.6-luna'
    assert len(result.changes) == 3
    url, request = client.calls[0]
    assert url == 'https://api.openai.com/v1/responses'
    assert request['json']['tools'] == [{
        'type': 'web_search',
        'search_context_size': 'high',
    }]
    assert request['json']['store'] is False
    assert request['headers']['Authorization'] == 'Bearer session-key'
    assert request['timeout'] == (10, 600)


def test_anthropic_uses_server_web_search_and_extracts_text():
    client = FakeClient([FakeResponse({
        'content': [{'type': 'text', 'text': CORRECTED}],
        'stop_reason': 'end_turn',
        'usage': {'server_tool_use': {'web_search_requests': 2}},
    })])

    result = frontier.verify_glossary(
        'anthropic', 'PROMPT', ORIGINAL, 'session-key', client=client,
    )

    assert result.glossary == CORRECTED
    assert result.model == 'claude-haiku-4-5-20251001'
    _, request = client.calls[0]
    assert request['json']['tools'][0]['type'] == 'web_search_20250305'
    assert request['headers']['x-api-key'] == 'session-key'


def test_google_requires_grounding_metadata_and_extracts_text():
    client = FakeClient([FakeResponse({
        'candidates': [{
            'content': {'parts': [{'text': CORRECTED}]},
            'groundingMetadata': {'webSearchQueries': ['Harry Potter Russian names']},
        }],
    })])

    result = frontier.verify_glossary(
        'google', 'PROMPT', ORIGINAL, 'session-key', client=client,
    )

    assert result.glossary == CORRECTED
    assert result.model == 'gemini-3.5-flash-lite'
    url, request = client.calls[0]
    assert 'gemini-3.5-flash-lite:generateContent' in url
    assert request['json']['tools'] == [{'google_search': {}}]
    assert request['headers']['x-goog-api-key'] == 'session-key'


def test_custom_model_is_used_and_reported():
    client = FakeClient([FakeResponse({
        'output': [
            {'type': 'web_search_call', 'id': 'ws_1'},
            {
                'type': 'message',
                'content': [{'type': 'output_text', 'text': CORRECTED}],
            },
        ],
    })])

    result = frontier.verify_glossary(
        'openai',
        'PROMPT',
        ORIGINAL,
        'session-key',
        'gpt-5.4-mini',
        client=client,
    )

    assert result.model == 'gpt-5.4-mini'
    assert client.calls[0][1]['json']['model'] == 'gpt-5.4-mini'


@pytest.mark.parametrize('model', [
    '../other-model',
    'models/custom',
    'model name',
    'x' * 129,
])
def test_custom_model_rejects_unsafe_or_malformed_ids(model):
    with pytest.raises(frontier.FrontierGlossaryError, match='Model name'):
        frontier.verify_glossary(
            'google', 'PROMPT', ORIGINAL, 'session-key', model,
        )


def test_a_response_without_actual_web_search_is_rejected():
    client = FakeClient([FakeResponse({
        'output': [{
            'type': 'message',
            'content': [{'type': 'output_text', 'text': CORRECTED}],
        }],
    })])

    with pytest.raises(frontier.FrontierGlossaryError, match='did not use web search'):
        frontier.verify_glossary(
            'openai', 'PROMPT', ORIGINAL, 'session-key', client=client,
        )


@pytest.mark.parametrize('candidate, message', [
    (
        'Hermione => Гермиона | inflectable\nHogwarts => Хогвартс | exact',
        'changed the number',
    ),
    (
        CORRECTED.replace('Hermione =>', 'Hermione Granger =>', 1),
        'changed source term',
    ),
    (
        CORRECTED.replace(' | exact', '', 1),
        'malformed glossary line',
    ),
])
def test_provider_output_cannot_drop_change_or_damage_source_entries(candidate, message):
    with pytest.raises(frontier.FrontierGlossaryError, match=message):
        frontier.validate_frontier_output(ORIGINAL, candidate)


def test_environment_key_availability_never_exposes_the_secret(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'super-secret-value')

    catalog = frontier.provider_catalog()

    openai = next(item for item in catalog if item['id'] == 'openai')
    assert openai['environment_key_available'] is True
    assert 'super-secret-value' not in repr(catalog)


def test_owner_only_local_env_key_is_available_without_exposing_it(
    monkeypatch, tmp_path,
):
    env_file = tmp_path / '.env.local'
    env_file.write_text('OPENAI_API_KEY=local-secret-value\n', encoding='utf-8')
    env_file.chmod(0o600)
    monkeypatch.setattr(frontier, 'LOCAL_ENV_FILE', env_file)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    catalog = frontier.provider_catalog()

    openai = next(item for item in catalog if item['id'] == 'openai')
    assert openai['environment_key_available'] is True
    assert frontier._api_key('openai', None) == 'local-secret-value'
    assert 'local-secret-value' not in repr(catalog)


@pytest.mark.skipif(
    os.name == 'nt',
    reason='Windows does not expose POSIX owner-only permission bits',
)
def test_local_env_with_open_permissions_is_rejected(monkeypatch, tmp_path):
    env_file = tmp_path / '.env.local'
    env_file.write_text('OPENAI_API_KEY=local-secret-value\n', encoding='utf-8')
    env_file.chmod(0o644)
    monkeypatch.setattr(frontier, 'LOCAL_ENV_FILE', env_file)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    with pytest.raises(
        frontier.FrontierGlossaryError,
        match=r'permissions are too open',
    ):
        frontier._api_key('openai', None)

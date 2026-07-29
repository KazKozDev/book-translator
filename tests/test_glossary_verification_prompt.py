import translator as app_module
from frontier_glossary import FrontierResult


def test_glossary_verification_prompt_is_ready_to_copy_without_ollama(monkeypatch):
    def unexpected_ollama_check(*args, **kwargs):
        raise AssertionError('Copying a manual prompt must not contact Ollama')

    monkeypatch.setattr(app_module.requests, 'get', unexpected_ollama_check)
    app_module.app.config.update(TESTING=True)
    response = app_module.app.test_client().post(
        '/glossary-verification-prompt',
        json={
            'sourceLanguage': 'en',
            'targetLanguage': 'ru',
            'glossary': (
                '# Review notes\n'
                '  Dursley => Дурсль | inflectable  \n'
                '\n'
                'Grunnings => Граннингс | exact'
            ),
        },
    )

    assert response.status_code == 200
    prompt = response.get_json()['prompt']
    assert prompt.startswith('ORIGINAL LANGUAGE: English\nTARGET LANGUAGE: Russian\n')
    assert 'authoritative published translations of that specific work' in prompt
    assert prompt.endswith(
        'ENTITIES:\n\n'
        'Dursley => Дурсль | inflectable\n'
        'Grunnings => Граннингс | exact'
    )
    assert '# Review notes' not in prompt
    assert '[PASTE ENTITIES HERE]' not in prompt


def test_glossary_verification_prompt_rejects_an_empty_glossary():
    app_module.app.config.update(TESTING=True)
    response = app_module.app.test_client().post(
        '/glossary-verification-prompt',
        json={
            'sourceLanguage': 'en',
            'targetLanguage': 'ru',
            'glossary': '\n# comments only\n',
        },
    )

    assert response.status_code == 400
    assert response.get_json() == {'error': 'Add at least one glossary entry first'}


def test_frontier_provider_catalog_exposes_no_secret_values(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'never-return-this-secret')
    app_module.app.config.update(TESTING=True)

    response = app_module.app.test_client().get('/frontier-providers')

    assert response.status_code == 200
    payload = response.get_json()
    assert {provider['id'] for provider in payload['providers']} == {
        'openai', 'anthropic', 'google',
    }
    assert 'never-return-this-secret' not in response.get_data(as_text=True)
    assert response.headers['Cache-Control'] == 'no-store'


def test_frontier_route_returns_reviewable_changes_without_applying_them(monkeypatch):
    recorded = {}

    def fake_verify(provider, prompt, glossary, submitted_key, submitted_model):
        recorded.update({
            'provider': provider,
            'prompt': prompt,
            'glossary': glossary,
            'submitted_key': submitted_key,
            'submitted_model': submitted_model,
        })
        return FrontierResult(
            glossary='Hermione => Гермиона | inflectable',
            provider='openai',
            provider_label='OpenAI',
            model='gpt-5.6-luna',
            changes=[{
                'source': 'Hermione',
                'before': 'Hermione => Гермиона',
                'after': 'Hermione => Гермиона | inflectable',
            }],
            searched=True,
        )

    monkeypatch.setattr(app_module, 'verify_glossary', fake_verify)
    app_module.app.config.update(TESTING=True)
    response = app_module.app.test_client().post(
        '/verify-glossary-frontier',
        json={
            'provider': 'openai',
            'apiKey': 'session-only-key',
            'model': 'gpt-5.4-mini',
            'sourceLanguage': 'en',
            'targetLanguage': 'ru',
            'glossary': 'Hermione => Гермиона',
        },
    )

    assert response.status_code == 200
    assert response.get_json()['changes'][0]['source'] == 'Hermione'
    assert response.get_json()['searched'] is True
    assert 'TARGET LANGUAGE: Russian' in recorded['prompt']
    assert recorded['submitted_key'] == 'session-only-key'
    assert recorded['submitted_model'] == 'gpt-5.4-mini'
    assert response.headers['Cache-Control'] == 'no-store'

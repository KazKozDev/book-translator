"""The app has no login, so who can reach it is the whole of its access control."""

import pytest

import translator as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, 'DB_PATH', str(tmp_path / 'translations.db'))
    monkeypatch.delenv('HOST', raising=False)
    app_module.init_db()
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def test_the_server_listens_on_this_computer_unless_told_otherwise(monkeypatch):
    monkeypatch.delenv('HOST', raising=False)
    assert app_module.bind_host() == '127.0.0.1'

    monkeypatch.setenv('HOST', '0.0.0.0')
    assert app_module.bind_host() == '0.0.0.0'


@pytest.mark.parametrize('base_url', [
    'http://localhost:5001', 'http://127.0.0.1:5001', 'http://[::1]:5001',
])
def test_the_interface_reaches_its_own_server(client, base_url):
    assert client.get('/translations', base_url=base_url).status_code == 200
    # What the page's own fetch() sends on a POST or a DELETE.
    assert client.get(
        '/translations', base_url=base_url, headers={'Origin': base_url},
    ).status_code == 200


def test_another_sites_page_cannot_use_the_api(client):
    """A page on any site can send a request to localhost from the user's own
    browser. Deleting a job needs no response to be read, so withholding the
    CORS headers alone would not have stopped it."""
    refused = client.delete(
        '/translations/1', base_url='http://localhost:5001',
        headers={'Origin': 'https://example.com'},
    )

    assert refused.status_code == 403
    assert refused.get_json() == {'error': 'Cross-origin requests are not accepted'}
    assert 'Access-Control-Allow-Origin' not in refused.headers


def test_a_domain_pointed_at_this_computer_is_refused(client):
    """DNS rebinding: the attacker's own domain resolves to 127.0.0.1, so its
    Origin and Host agree with each other and only the name gives it away."""
    refused = client.get(
        '/translations', base_url='http://attacker.example:5001',
        headers={'Origin': 'http://attacker.example:5001'},
    )

    assert refused.status_code == 403
    assert refused.get_json() == {'error': 'This app only answers on this computer'}


def test_opening_the_app_to_the_network_is_an_explicit_choice(client, monkeypatch):
    monkeypatch.setenv('HOST', '0.0.0.0')
    by_address = 'http://192.168.1.20:5001'

    assert client.get('/translations', base_url=by_address).status_code == 200
    assert client.get(
        '/translations', base_url=by_address, headers={'Origin': by_address},
    ).status_code == 200
    # Still not for other sites' pages.
    assert client.get(
        '/translations', base_url=by_address, headers={'Origin': 'https://example.com'},
    ).status_code == 403


def test_no_cross_origin_headers_are_sent(client):
    response = client.get('/translations', headers={'Origin': 'http://localhost'})

    assert 'Access-Control-Allow-Origin' not in response.headers

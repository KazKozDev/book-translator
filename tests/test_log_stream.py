"""The live Log page begins with the shared Tolmach banner."""

import json

import translator as app_module


def test_live_log_stream_starts_with_the_shared_banner():
    app_module.app.config.update(TESTING=True)

    with app_module.app.test_client() as client:
        response = client.get('/logs/stream?tail=0', buffered=False)
        first_event = next(response.response).decode('utf-8')
        response.close()

    assert first_event.startswith('data: ')
    entry = json.loads(first_event.removeprefix('data: ').strip())
    assert entry == {
        'source': 'console',
        'level': 'BANNER',
        'time': '',
        'message': app_module.LOG_CONSOLE_BANNER,
    }

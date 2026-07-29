"""The live Log page begins with real log entries, not a startup banner."""

import json

import translator as app_module


def test_live_log_stream_starts_with_a_real_log_entry(tmp_path, monkeypatch):
    app_module.app.config.update(TESTING=True)
    (tmp_path / 'app.log').write_text(
        '2026-07-29 09:00:00,000 - app_logger - INFO - Log stream is ready\n'
    )
    (tmp_path / 'api.log').touch()
    (tmp_path / 'translations.log').touch()
    monkeypatch.setattr(app_module, 'LOG_FOLDER', str(tmp_path))

    with app_module.app.test_client() as client:
        response = client.get('/logs/stream?tail=1', buffered=False)
        first_event = next(response.response).decode('utf-8')
        response.close()

    assert first_event.startswith('data: ')
    entry = json.loads(first_event.removeprefix('data: ').strip())
    assert entry['source'] == 'app'
    assert entry['level'] == 'INFO'
    assert entry['message'] == 'Log stream is ready'

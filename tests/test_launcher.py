"""Unit tests for the bootstrap that decides which interpreter serves the app.

Both behaviours here are regressions, and both were macOS-only: the launcher
looked like it restarted a stale server and reported a successful start, while
in fact leaving an older process — started by hand with the system Python, and
therefore without the model packages — serving the port all day.
"""

import io
import json
import subprocess
import sys
import types
from contextlib import redirect_stdout

import pytest

import launch


class FakeAccessDenied(Exception):
    pass


class FakeNoSuchProcess(Exception):
    pass


def fake_psutil(processes):
    return types.SimpleNamespace(
        process_iter=lambda attrs=None: iter(processes),
        AccessDenied=FakeAccessDenied,
        NoSuchProcess=FakeNoSuchProcess,
    )


class FakeConnection:
    def __init__(self, port, status='LISTEN'):
        self.status = status
        self.laddr = types.SimpleNamespace(port=port)


class FakeProcess:
    """A psutil 5.9.6-shaped process: ``connections``, no ``net_connections``."""

    def __init__(self, pid, ports=(), raises=None):
        self.pid = pid
        self._ports = ports
        self._raises = raises

    def connections(self, kind=None):
        if self._raises:
            raise self._raises(self.pid)
        return [FakeConnection(port) for port in self._ports]


def snippet(monkeypatch):
    """Return the psutil program listening_pids() runs inside the venv."""
    captured = {}

    def fake_run(command, **kwargs):
        captured['code'] = command[2]
        return subprocess.CompletedProcess(command, 0, stdout='[]', stderr='')

    monkeypatch.setattr(launch, 'run', fake_run)
    launch.listening_pids()
    return captured['code']


def run_snippet(code, processes, monkeypatch):
    monkeypatch.setitem(sys.modules, 'psutil', fake_psutil(processes))
    out = io.StringIO()
    with redirect_stdout(out):
        exec(compile(code, '<listening_pids>', 'exec'), {'__name__': '__main__'})
    return json.loads(out.getvalue())


def test_listening_pids_survives_processes_it_may_not_inspect(monkeypatch):
    """macOS denies a system-wide connection scan; one refusal must not blind it.

    psutil.net_connections() raises AccessDenied on the first unreadable
    process, which returned an empty list and left free_port() with nothing to
    kill. Walking processes individually still finds our own server.
    """
    processes = [
        FakeProcess(101, raises=FakeAccessDenied),
        FakeProcess(202, ports=(launch.PORT,)),
        FakeProcess(303, ports=(9999,)),
        FakeProcess(404, raises=FakeNoSuchProcess),
    ]

    assert run_snippet(snippet(monkeypatch), processes, monkeypatch) == [202]


def test_listening_pids_reports_nothing_when_the_port_is_free(monkeypatch):
    processes = [FakeProcess(101, ports=(9999,)), FakeProcess(202, raises=FakeAccessDenied)]

    assert run_snippet(snippet(monkeypatch), processes, monkeypatch) == []


def test_listening_pids_returns_empty_when_the_probe_fails(monkeypatch):
    """A crashing probe stays non-fatal — it just cannot claim the port is held."""
    monkeypatch.setattr(
        launch, 'run',
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, stdout='', stderr='boom'),
    )

    assert launch.listening_pids() == []


class FakeServer:
    def __init__(self, exit_code):
        self._exit_code = exit_code
        self.waited = False

    def poll(self):
        return self._exit_code

    def wait(self):
        self.waited = True

    def terminate(self):
        pass


def arrange_start(monkeypatch, exit_code):
    """Start the server against a port that already answers for someone else."""
    server = FakeServer(exit_code)
    opened = []
    monkeypatch.setattr(launch, 'free_port', lambda: None)
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *args, **kwargs: server)
    monkeypatch.setattr(launch, 'port_ready', lambda: True)
    monkeypatch.setattr(launch.webbrowser, 'open', opened.append)
    monkeypatch.setattr(launch.time, 'sleep', lambda seconds: None)
    return server, opened


def test_start_translator_does_not_mistake_a_stale_server_for_its_own(monkeypatch):
    """A 200 on the port proves someone is serving, not that the child is.

    With a stale server holding the port, the new process dies on a bind error;
    the launcher used to see the old process answer, announce Ready, and open
    the browser onto exactly the environment the restart meant to replace.
    """
    server, opened = arrange_start(monkeypatch, exit_code=1)

    with pytest.raises(SystemExit):
        launch.start_translator()

    assert opened == []
    assert not server.waited


def test_start_translator_reports_ready_while_its_child_runs(monkeypatch):
    server, opened = arrange_start(monkeypatch, exit_code=None)

    launch.start_translator()

    assert server.waited
    assert len(opened) == 1
    assert opened[0].startswith(f'http://127.0.0.1:{launch.PORT}/?launched=')

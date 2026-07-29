import io
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


INSTALLER = Path(__file__).resolve().parents[1] / 'install.sh'
SHELL_TOOLS_AVAILABLE = all(shutil.which(tool) for tool in ('sh', 'curl', 'tar'))
pytestmark = pytest.mark.skipif(
    os.name == 'nt' or not SHELL_TOOLS_AVAILABLE,
    reason='install.sh supports macOS/Linux and needs sh, curl, and tar',
)


def build_archive(path: Path, *, launcher_output: str = 'launcher-ok', unsafe=False):
    files = {
        'book-translator-main/launch.py': f'print({launcher_output!r})\n'.encode(),
        'book-translator-main/requirements.txt': b'',
        'book-translator-main/src/translator.py': b'',
        'book-translator-main/uploads/.gitkeep': b'',
        'book-translator-main/translations/.gitkeep': b'',
        'book-translator-main/logs/.gitkeep': b'',
    }
    if unsafe:
        files['../escaped.txt'] = b'unsafe'

    with tarfile.open(path, 'w:gz') as archive:
        for name, content in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))


def run_installer(home: Path, archive: Path):
    env = os.environ.copy()
    env.update({
        'HOME': str(home),
        'TOLMACH_ARCHIVE_URL': archive.as_uri(),
        'TOLMACH_PYTHON': sys.executable,
    })
    return subprocess.run(
        ['sh', str(INSTALLER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_installer_keeps_runtime_data_across_releases(tmp_path):
    home = tmp_path / 'home'
    archive = tmp_path / 'tolmach.tar.gz'
    build_archive(archive)

    first = run_installer(home, archive)
    assert first.returncode == 0, first.stderr

    install_root = home / '.local/share/tolmach'
    command = home / '.local/bin/tolmach'
    first_release = (install_root / 'current').resolve()
    assert command.stat().st_mode & 0o111
    assert (first_release / 'uploads').resolve() == install_root / 'data/uploads'
    assert (first_release / 'translations').resolve() == install_root / 'data/translations'
    assert (first_release / 'logs').resolve() == install_root / 'data/logs'
    assert (first_release / 'translations.db').resolve() == install_root / 'data/translations.db'
    assert (first_release / 'cache.db').resolve() == install_root / 'data/cache.db'

    uploaded = install_root / 'data/uploads/book.txt'
    database = install_root / 'data/translations.db'
    uploaded.write_text('keep me', encoding='utf-8')
    database.write_bytes(b'sqlite-state')

    build_archive(archive, launcher_output='updated-launcher')
    second = run_installer(home, archive)
    assert second.returncode == 0, second.stderr
    second_release = (install_root / 'current').resolve()

    assert second_release != first_release
    assert uploaded.read_text(encoding='utf-8') == 'keep me'
    assert database.read_bytes() == b'sqlite-state'

    launched = subprocess.run(
        [str(command)],
        env={**os.environ, 'HOME': str(home)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert launched.returncode == 0
    assert launched.stdout.strip() == 'updated-launcher'


def test_installer_rejects_archive_path_traversal(tmp_path):
    home = tmp_path / 'home'
    archive = tmp_path / 'unsafe.tar.gz'
    build_archive(archive, unsafe=True)

    result = run_installer(home, archive)

    assert result.returncode != 0
    assert 'unsafe paths' in result.stderr
    assert not (tmp_path / 'escaped.txt').exists()

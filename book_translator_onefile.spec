# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File for Book Translator - SINGLE FILE VERSION
=================================================================
Creates a SINGLE .exe with everything included.

To create the .exe:
    pyinstaller book_translator_onefile.spec

The result will be in: dist/BookTranslator.exe
"""

import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Get the spec file directory
spec_dir = os.path.dirname(os.path.abspath(SPEC))

# Collect all data from Flask and its dependencies
flask_datas, flask_binaries, flask_hiddenimports = collect_all('flask')
jinja2_datas, jinja2_binaries, jinja2_hiddenimports = collect_all('jinja2')
werkzeug_datas, werkzeug_binaries, werkzeug_hiddenimports = collect_all('werkzeug')

# Collect PIL/Pillow data
try:
    pil_datas, pil_binaries, pil_hiddenimports = collect_all('PIL')
except Exception:
    pil_datas, pil_binaries, pil_hiddenimports = [], [], []

# Collect pystray data
try:
    pystray_datas, pystray_binaries, pystray_hiddenimports = collect_all('pystray')
except Exception:
    pystray_datas, pystray_binaries, pystray_hiddenimports = [], [], []

# Collect book_translator submodules
book_translator_imports = collect_submodules('book_translator')

# Main script analysis - using new modular entry point
a = Analysis(
    ['run.py'],
    pathex=[spec_dir],
    binaries=flask_binaries + jinja2_binaries + werkzeug_binaries + pil_binaries + pystray_binaries,
    datas=[
        # Include static files (frontend)
        ('static', 'static'),
        # Include the book_translator package
        ('book_translator', 'book_translator'),
    ] + flask_datas + jinja2_datas + werkzeug_datas + pil_datas + pystray_datas,
    hiddenimports=[
        # Book translator modules
        'book_translator',
        'book_translator.app',
        'book_translator.config',
        'book_translator.config.settings',
        'book_translator.config.constants',
        'book_translator.models',
        'book_translator.models.translation',
        'book_translator.models.schemas',
        'book_translator.services',
        'book_translator.services.ollama_client',
        'book_translator.services.cache_service',
        'book_translator.services.terminology',
        'book_translator.services.translator',
        'book_translator.database',
        'book_translator.database.connection',
        'book_translator.database.repositories',
        'book_translator.api',
        'book_translator.api.routes',
        'book_translator.api.middleware',
        'book_translator.utils',
        'book_translator.utils.language_detection',
        'book_translator.utils.text_processing',
        'book_translator.utils.validators',
        'book_translator.utils.logging',
        # Flask and dependencies
        'flask',
        'flask.app',
        'flask.blueprints',
        'flask.cli',
        'flask.config',
        'flask.ctx',
        'flask.globals',
        'flask.helpers',
        'flask.json',
        'flask.logging',
        'flask.sessions',
        'flask.signals',
        'flask.templating',
        'flask.testing',
        'flask.views',
        'flask.wrappers',
        'flask_cors',
        # Werkzeug
        'werkzeug',
        'werkzeug.datastructures',
        'werkzeug.debug',
        'werkzeug.exceptions',
        'werkzeug.formparser',
        'werkzeug.http',
        'werkzeug.local',
        'werkzeug.routing',
        'werkzeug.security',
        'werkzeug.serving',
        'werkzeug.test',
        'werkzeug.urls',
        'werkzeug.utils',
        'werkzeug.wrappers',
        # Jinja2
        'jinja2',
        'jinja2.ext',
        # Other dependencies
        'requests',
        'urllib3',
        'charset_normalizer',
        'certifi',
        'idna',
        'sqlite3',
        'json',
        'threading',
        'hashlib',
        'webbrowser',
        'dataclasses',
        'typing',
        'pathlib',
        'functools',
        'contextlib',
        'datetime',
        'time',
        'enum',
        're',
        'difflib',
        'logging',
        'logging.handlers',
        # System tray dependencies
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'pystray',
    ] + ([
        'pystray._win32'
    ] if sys.platform == 'win32' else [
        'pystray._gtk',
        'pystray._appindicator',
        'pystray._xorg'
    ] if sys.platform.startswith('linux') else [
        'pystray._darwin'
    ] if sys.platform == 'darwin' else []) + flask_hiddenimports + jinja2_hiddenimports + werkzeug_hiddenimports 
      + pil_hiddenimports + pystray_hiddenimports + book_translator_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'unittest',
        '_pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='BookTranslator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(spec_dir, 'app_icon.ico'),
)

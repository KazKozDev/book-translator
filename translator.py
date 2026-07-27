import json
import requests
import time
from typing import List, Dict, Optional, Callable, Set, Tuple
import os
# COMET pins SentencePiece below 0.2. Its generated protobuf bindings require
# Python parsing mode; set this before any library can import SentencePiece.
# Setting it lazily during a request can load the same proto descriptor twice.
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import sqlite3
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import hashlib
import traceback
import psutil
import subprocess
import threading
import signal
import atexit
import re
import sys
import random
import difflib
import unicodedata
from statistics import mean, median
from collections import deque, Counter
from dataclasses import dataclass, field
from functools import wraps

try:
    import sacrebleu
except ImportError:
    sacrebleu = None
from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import zipfile
import io
import uuid
from datetime import datetime as dt
import html as html_escape
import ebooklib
from ebooklib import epub as epub_lib
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

LANG_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese',
    'ja': 'Japanese', 'ko': 'Korean'
}

# TranslateGemma (Gemma 3 based, 4B/12B/27B) is a translation-only model: it
# was trained on one fixed prompt shape and cannot follow editor or judge
# instructions. Stage 1 gets its native prompt format; Stage 2 (refinement)
# and the LLM-judge tests have to run on a general instruct model instead.
TRANSLATEGEMMA_TEMPERATURE = 0.3  # A dedicated MT model wants near-greedy decoding.

# The banner lives in its own stdlib-only module so the launcher — which runs
# before the virtual environment exists and cannot import this file — shows the
# same logo from the same source.
from banner import TERMINAL_LOGO, print_terminal_banner  # noqa: E402


def is_translategemma(model_name: Optional[str]) -> bool:
    return 'translategemma' in (model_name or '').lower()

# Folders setup
UPLOAD_FOLDER = 'uploads'
TRANSLATIONS_FOLDER = 'translations'
STATIC_FOLDER = 'static'
# Overridable so that a test run does not write into the log a person is
# watching: three warnings about "translation 1" from a fixture, landing in the
# live console between two real chunks, is worse than no log at all.
LOG_FOLDER = os.environ.get('TOLMACH_LOG_DIR', 'logs')
DB_PATH = 'translations.db'
CACHE_DB_PATH = 'cache.db'

# Create necessary directories
for folder in [UPLOAD_FOLDER, TRANSLATIONS_FOLDER, STATIC_FOLDER, LOG_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Logger setup
class AppLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.app_logger = self._setup_logger(
            'app_logger',
            os.path.join(log_dir, 'app.log')
        )
        
        self.translation_logger = self._setup_logger(
            'translation_logger',
            os.path.join(log_dir, 'translations.log')
        )
        
        self.api_logger = self._setup_logger(
            'api_logger',
            os.path.join(log_dir, 'api.log')
        )

    def _setup_logger(self, name, log_file):
        logger = logging.getLogger(name)
        # LOG_LEVEL=DEBUG turns on the verbose records that are too noisy for a
        # normal run — notably the full prompt sent to the model for every
        # chunk, which is how you check what a model actually received.
        logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO').upper())
        if logger.handlers:
            return logger
        
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        return logger

    @property
    def loggers(self) -> List[logging.Logger]:
        return [self.app_logger, self.translation_logger, self.api_logger]

    def rotate(self) -> List[str]:
        """Start fresh log files, keeping the old ones as ``*.log.1``.

        Used when a new document is loaded: one book per log file makes the
        console readable, and rolling over rather than truncating means the
        previous run is still on disk to look at afterwards. The handlers keep
        their existing backupCount, so this cannot grow without bound.
        """
        rotated = []
        for log in self.loggers:
            for handler in log.handlers:
                if isinstance(handler, RotatingFileHandler):
                    try:
                        handler.doRollover()
                        rotated.append(os.path.basename(handler.baseFilename))
                    except OSError as e:
                        self.app_logger.error(f"Could not roll over {handler.baseFilename}: {e}")
        return rotated

# Initialize logger
logger = AppLogger(log_dir=LOG_FOLDER)

class PlainAccessFormatter(logging.Formatter):
    """Strip the ANSI coloring werkzeug puts on its own per-request access log
    lines, so the console stays plain text."""

    _ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

    def format(self, record):
        return self._ANSI_RE.sub('', record.getMessage())


def setup_access_log():
    """Attach a plain-text console handler to werkzeug's access logger,
    replacing its default colored formatting."""
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.handlers.clear()
    werkzeug_logger.propagate = False
    werkzeug_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(PlainAccessFormatter())
    werkzeug_logger.addHandler(handler)

# Monitoring setup
@dataclass
class TranslationMetrics:
    total_requests: int = 0
    successful_translations: int = 0
    failed_translations: int = 0
    average_translation_time: float = 0
    translation_times: deque = field(default_factory=lambda: deque(maxlen=100))

class AppMonitor:
    def __init__(self):
        self.metrics = TranslationMetrics()
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.active_model: Optional[str] = None

    def set_active_model(self, model_name: str):
        with self._lock:
            self.active_model = model_name

    def record_translation_attempt(self, success: bool, translation_time: float):
        with self._lock:
            self.metrics.total_requests += 1
            if success:
                self.metrics.successful_translations += 1
                self.metrics.translation_times.append(translation_time)
                self.metrics.average_translation_time = (
                    sum(self.metrics.translation_times) / len(self.metrics.translation_times)
                )
            else:
                self.metrics.failed_translations += 1
    
    def get_system_metrics(self) -> Dict:
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': time.time() - self.start_time
        }

    def get_ollama_gpu_metrics(self) -> Dict[str, str]:
        """Report Ollama's active model and processor without inventing GPU data.

        On Apple Silicon, CPU and GPU share the same physical memory.  Ollama's
        ``ps`` output is therefore the reliable source for whether a loaded
        model is actually running on the GPU; it does not expose a separate
        VRAM percentage for this architecture.
        """
        try:
            result = subprocess.run(
                ['ollama', 'ps'],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {'status': 'Unavailable', 'model': 'Ollama not reachable'}

        if result.returncode != 0:
            return {'status': 'Unavailable', 'model': 'Ollama not reachable'}

        rows = [line.strip() for line in result.stdout.splitlines()[1:] if line.strip()]
        if not rows:
            return {'status': 'Idle', 'model': 'No model loaded'}

        # Ollama can keep several models resident at once (e.g. after testing
        # with more than one). Prefer whichever row matches the model this
        # app itself last used, so switching models in the selector is
        # reflected here instead of showing an arbitrary loaded model.
        active_model = self.active_model
        chosen_row = rows[0]
        if active_model:
            for row in rows:
                if row.split()[0] == active_model:
                    chosen_row = row
                    break

        # Columns are separated by two or more spaces. The final processor
        # column is emitted by Ollama as e.g. "100% GPU" or "100% CPU".
        columns = re.split(r'\s{2,}', chosen_row)
        processor = next(
            (column for column in columns if re.fullmatch(r'\d+% (?:GPU|CPU)', column)),
            None,
        )
        if processor:
            return {'status': processor, 'model': columns[0]}
        return {'status': 'Active', 'model': columns[0]}
    
    def get_metrics(self) -> Dict:
        with self._lock:
            metrics_data = {
                'translation_metrics': {
                    'total_requests': self.metrics.total_requests,
                    'successful_translations': self.metrics.successful_translations,
                    'failed_translations': self.metrics.failed_translations,
                    'average_translation_time': self.metrics.average_translation_time
                },
                'system_metrics': self.get_system_metrics()
            }
            metrics_data['ollama_gpu'] = self.get_ollama_gpu_metrics()
            
            if self.metrics.total_requests > 0:
                metrics_data['translation_metrics']['success_rate'] = (
                    self.metrics.successful_translations / self.metrics.total_requests * 100
                )
            else:
                metrics_data['translation_metrics']['success_rate'] = 0
                
            return metrics_data

# Initialize monitor
monitor = AppMonitor()

# Translation cache setup
class TranslationCache:
    def __init__(self, db_path: str = CACHE_DB_PATH):
        self.db_path = db_path
        self._init_cache_db()
    
    def _init_cache_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS translation_cache (
                    hash_key TEXT PRIMARY KEY,
                    source_lang TEXT,
                    target_lang TEXT,
                    original_text TEXT,
                    translated_text TEXT,
                    machine_translation TEXT,
                    created_at TIMESTAMP,
                    last_used TIMESTAMP
                )
            ''')

    def _generate_hash(self, text: str, source_lang: str, target_lang: str, model: str = "") -> str:
        key = f"{text}:{source_lang}:{target_lang}:{model}".encode('utf-8')
        return hashlib.sha256(key).hexdigest()
    
    def get_cached_translation(self, text: str, source_lang: str, target_lang: str, model: str = "") -> Optional[Dict[str, str]]:
        hash_key = self._generate_hash(text, source_lang, target_lang, model)
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute('''
                SELECT translated_text, machine_translation
                FROM translation_cache
                WHERE hash_key = ?
            ''', (hash_key,))
            
            result = cur.fetchone()
            if result:
                conn.execute('''
                    UPDATE translation_cache
                    SET last_used = CURRENT_TIMESTAMP
                    WHERE hash_key = ?
                ''', (hash_key,))
                return {
                    'translated_text': result[0],
                    'machine_translation': result[1]
                }
        
        return None
    
    def cache_translation(self, text: str, translated_text: str, machine_translation: str, 
                         source_lang: str, target_lang: str, model: str = ""):
        hash_key = self._generate_hash(text, source_lang, target_lang, model)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO translation_cache
                (hash_key, source_lang, target_lang, original_text, translated_text, 
                 machine_translation, created_at, last_used)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (hash_key, source_lang, target_lang, text, translated_text, machine_translation))
    
    def cleanup_old_entries(self, days: int = 30):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"DELETE FROM translation_cache WHERE last_used < datetime('now', '-{days} days')"
            )

# Initialize cache
cache = TranslationCache()

# Terminology constraints
@dataclass(frozen=True)
class GlossaryTerm:
    source: str
    target: str
    mode: str = "inflectable"


class TerminologyManager:
    """Language-neutral, per-book terminology constraints."""

    VALID_MODES = {"exact", "inflectable", "preferred"}
    MAX_TERMS = 500
    MAX_TERM_LENGTH = 200

    def __init__(self, terms: Optional[List[GlossaryTerm]] = None):
        deduplicated = {}
        for term in terms or []:
            deduplicated[term.source.casefold()] = term
        self.terms = list(deduplicated.values())

    @classmethod
    def from_text(cls, glossary_text: str):
        """Parse `source => target | mode` or TSV lines; mode defaults to inflectable."""
        terms = []
        for line_number, raw_line in enumerate(glossary_text.splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            mode = "inflectable"
            if "\t" in line:
                parts = [part.strip() for part in line.split("\t")]
                if len(parts) not in (2, 3):
                    raise ValueError(
                        f"Glossary line {line_number}: use source<TAB>target<TAB>mode"
                    )
                source, target = parts[:2]
                if len(parts) == 3:
                    mode = parts[2].lower()
            else:
                separator = "=>" if "=>" in line else "=" if "=" in line else None
                if not separator:
                    raise ValueError(
                        f"Glossary line {line_number}: use source => target | mode"
                    )
                source, remainder = [part.strip() for part in line.split(separator, 1)]
                if "|" in remainder:
                    target, mode = [part.strip() for part in remainder.rsplit("|", 1)]
                    mode = mode.lower()
                else:
                    target = remainder.strip()

            if not source or not target:
                raise ValueError(f"Glossary line {line_number}: both terms are required")
            if len(source) > cls.MAX_TERM_LENGTH or len(target) > cls.MAX_TERM_LENGTH:
                raise ValueError(
                    f"Glossary line {line_number}: a term exceeds {cls.MAX_TERM_LENGTH} characters"
                )
            if mode not in cls.VALID_MODES:
                raise ValueError(
                    f"Glossary line {line_number}: mode must be exact, inflectable, or preferred"
                )
            terms.append(GlossaryTerm(source=source, target=target, mode=mode))

        if len(terms) > cls.MAX_TERMS:
            raise ValueError(f"Glossary supports at most {cls.MAX_TERMS} terms")
        return cls(terms)

    def relevant_terms(self, source_text: str) -> List[GlossaryTerm]:
        folded_text = source_text.casefold()
        return [term for term in self.terms if term.source.casefold() in folded_text]

    def prompt_context(self, source_text: str) -> str:
        relevant = self.relevant_terms(source_text)
        if not relevant:
            return ""

        lines = []
        for term in relevant:
            rule = {
                "exact": "use this target form exactly",
                "inflectable": "use this lexical choice; grammatical inflection is allowed",
                "preferred": "prefer this translation when it fits the context",
            }[term.mode]
            lines.append(f'- "{term.source}" => "{term.target}" ({rule})')
        return (
            "\n\nVERIFIED TERMINOLOGY FOR THIS PASS:\n"
            + "\n".join(lines)
            + "\nThese constraints apply regardless of the source and target languages."
        )

    def exact_violations(self, source_text: str, translated_text: str) -> List[Dict[str, str]]:
        translated_folded = translated_text.casefold()
        return [
            {"source": term.source, "required_target": term.target}
            for term in self.relevant_terms(source_text)
            if term.mode == "exact" and term.target.casefold() not in translated_folded
        ]

    def enforce_exact_source_forms(self, translated_text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Replace an exact term only when the model leaked its source form.

        A glossary is still provided to the model as translation context: it
        remains the only safe way to choose a rendering that is absent from
        the output.  But an ``exact`` rule has one deterministic case we can
        honour without guessing — the model translated the surrounding prose
        and left the literal source term unchanged.  Fix that case here, both
        for fresh generations and cached chunks.  ``inflectable`` and
        ``preferred`` terms are intentionally never rewritten this way.
        """
        replacements: List[Dict[str, str]] = []
        result = translated_text
        for term in self.terms:
            if term.mode != "exact" or term.source.casefold() == term.target.casefold():
                continue
            # Do not turn a source substring inside a longer word into a
            # glossary term. ``\w`` is Unicode-aware, so this works for Latin,
            # Cyrillic and CJK source terms alike.
            pattern = re.compile(rf"(?<!\w){re.escape(term.source)}(?!\w)", re.IGNORECASE)
            result, count = pattern.subn(term.target, result)
            if count:
                replacements.append({
                    "source": term.source,
                    "target": term.target,
                    "count": count,
                })
        return result, replacements

    def fingerprint(self) -> str:
        canonical = sorted(
            (term.source.casefold(), term.target, term.mode) for term in self.terms
        )
        payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

# Which translations this process is streaming right now.
#
# The DB status alone cannot answer "is this running?". A client that closes
# the stream — a shut tab, a dropped network, a killed curl — leaves the row
# saying 'in_progress' with nobody writing to it, and a server that was
# restarted mid-run leaves the same thing behind with no chance to clean up.
# Both used to make Continue answer "this translation is already running"
# forever, with the database needing an edit by hand to recover.
ACTIVE_RUNS: Set[int] = set()
ACTIVE_RUNS_LOCK = threading.Lock()


def claim_run(translation_id: int):
    with ACTIVE_RUNS_LOCK:
        ACTIVE_RUNS.add(translation_id)


def release_run(translation_id: int):
    with ACTIVE_RUNS_LOCK:
        ACTIVE_RUNS.discard(translation_id)


def is_run_active(translation_id: int) -> bool:
    """Whether this process is actually streaming that translation.

    Restart-safe by construction: a fresh process holds no claims, so every
    'in_progress' row it inherits is correctly seen as abandoned.
    """
    with ACTIVE_RUNS_LOCK:
        return translation_id in ACTIVE_RUNS


# Part of the Stage 2 cache key, so that changing what the refinement pass
# DOES invalidates results produced by the old behaviour. The inputs alone are
# not enough: same chunk, same glossary, same brief, same model — and a
# different answer, because the pass itself changed. Bump this whenever the
# estimate/patch/verify behaviour changes in a way that would alter output.
#   v2: single "review and improve" rewrite replaced by estimate/patch/verify
#   v3: style-only and minor subjective errors reported but no longer applied
#   v4: omission/addition patches skip the verifier, which now runs on its own
#       model rather than on the one that wrote the draft
STAGE2_PIPELINE_VERSION = 'v4'


def context_fingerprint(text: str) -> str:
    """Short, stable hash of a free-text prompt block (the Stage 0 brief),
    for keying the translation cache on it. Mirrors
    TerminologyManager.fingerprint, which does the same job for the glossary:
    anything that changes the prompt has to change the cache key, or a later
    run silently replays drafts produced under different instructions."""
    normalized = (text or '').strip()
    if not normalized:
        return 'none'
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


# Error handling setup
class TranslationError(Exception):
    pass

def with_error_handling(f: Callable):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except requests.Timeout as e:
            logger.app_logger.error(f"Timeout error: {str(e)}")
            raise TranslationError("Translation service timeout")
        except requests.RequestException as e:
            logger.app_logger.error(f"Request error: {str(e)}")
            raise TranslationError("Translation service unavailable")
        except sqlite3.Error as e:
            logger.app_logger.error(f"Database error: {str(e)}")
            raise TranslationError("Database error occurred")
        except Exception as e:
            logger.app_logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
            raise TranslationError("An unexpected error occurred")
    return wrapper

# Initialize database
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0,
                current_chunk INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 0,
                original_text TEXT,
                machine_translation TEXT,
                translated_text TEXT,
                detected_language TEXT,
                genre TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT,
                source_format TEXT DEFAULT 'txt',
                translated_chapters TEXT,
                book_title TEXT,
                book_author TEXT,
                original_chunks TEXT,
                draft_chunks TEXT,
                final_chunks TEXT,
                chunk_chapter_map TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                translation_id INTEGER,
                chunk_number INTEGER,
                original_text TEXT,
                machine_translation TEXT,
                translated_text TEXT,
                status TEXT,
                error_message TEXT,
                attempts INTEGER DEFAULT 0,
                FOREIGN KEY (translation_id) REFERENCES translations (id)
            );

            CREATE TABLE IF NOT EXISTS translation_terms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                translation_id INTEGER NOT NULL,
                source_term TEXT NOT NULL,
                target_term TEXT NOT NULL,
                enforcement_mode TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'verified',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (translation_id, source_term),
                FOREIGN KEY (translation_id) REFERENCES translations (id)
            );

            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                translation_id INTEGER NOT NULL,
                test_name TEXT NOT NULL,
                judge_model TEXT,
                value REAL,
                flagged INTEGER NOT NULL DEFAULT 0,
                note TEXT,
                details TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (translation_id, test_name),
                FOREIGN KEY (translation_id) REFERENCES translations (id)
            );
        ''')

        # translations existed before source_format/translated_chapters/book_title/book_author
        # were added — CREATE TABLE IF NOT EXISTS won't retrofit columns onto an existing table.
        existing_columns = {row[1] for row in conn.execute('PRAGMA table_info(translations)')}
        for column, ddl in (
            ('source_format', "ALTER TABLE translations ADD COLUMN source_format TEXT DEFAULT 'txt'"),
            ('translated_chapters', 'ALTER TABLE translations ADD COLUMN translated_chapters TEXT'),
            ('book_title', 'ALTER TABLE translations ADD COLUMN book_title TEXT'),
            ('book_author', 'ALTER TABLE translations ADD COLUMN book_author TEXT'),
            ('original_chunks', 'ALTER TABLE translations ADD COLUMN original_chunks TEXT'),
            ('draft_chunks', 'ALTER TABLE translations ADD COLUMN draft_chunks TEXT'),
            ('chunk_chapter_map', 'ALTER TABLE translations ADD COLUMN chunk_chapter_map TEXT'),
            # Stage 0 artifacts and the per-chunk final text. final_chunks is
            # what lets the Stage 2 judges compare draft and final in the same
            # units as the Stage 1 judge (chunks), instead of realigning
            # paragraph-by-position after the fact.
            ('final_chunks', 'ALTER TABLE translations ADD COLUMN final_chunks TEXT'),
        ):
            if column not in existing_columns:
                conn.execute(ddl)
        # Remove the obsolete per-document context column from existing job
        # history as well as from the new-table schema above.
        if 'doc_summary' in existing_columns:
            conn.execute('ALTER TABLE translations DROP COLUMN doc_summary')

init_db()


def is_epub_filename(filename: str) -> bool:
    return filename.lower().endswith('.epub')


def extract_epub_book(filepath: str):
    """Read an EPUB and return (chapters, title, author) in spine (reading) order.

    Each chapter is plain text with paragraphs separated by blank lines — enough
    to feed the existing chunk-based translation pipeline. Inline formatting
    (bold/italic/links) is intentionally not preserved; see build_epub_from_chapters.
    """
    book = epub_lib.read_epub(filepath, options={'ignore_ncx': True})

    chapters = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        # The auto-generated nav/TOC document is also an ITEM_DOCUMENT — skip it,
        # it's a list of links to other chapters, not book content.
        if isinstance(item, epub_lib.EpubNav):
            continue
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        blocks = [
            block.get_text(' ', strip=True)
            for block in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li', 'blockquote'])
        ]
        blocks = [block for block in blocks if block]
        text = '\n\n'.join(blocks) if blocks else soup.get_text('\n\n', strip=True)
        chapters.append(text)

    title = None
    author = None
    title_meta = book.get_metadata('DC', 'title')
    if title_meta:
        title = title_meta[0][0]
    creator_meta = book.get_metadata('DC', 'creator')
    if creator_meta:
        author = creator_meta[0][0]

    return chapters, title, author


def build_epub_from_chapters(chapters: List[str], title: str, author: str) -> bytes:
    """Assemble a minimal, valid EPUB from translated chapter texts.

    Mirrors /export/epub's hand-rolled EPUB writer but for N chapters instead
    of one, so translations of multi-chapter books keep their chapter breaks.
    """
    safe_title = html_escape.escape(title or 'Translation')
    safe_author = html_escape.escape(author or 'Book Translator')
    book_uid = str(uuid.uuid4())
    buffer = io.BytesIO()

    manifest_items = []
    spine_items = []
    nav_points = []
    for i, chapter_text in enumerate(chapters, 1):
        manifest_items.append(
            f'<item id="chapter{i}" href="chapter{i}.xhtml" media-type="application/xhtml+xml"/>'
        )
        spine_items.append(f'<itemref idref="chapter{i}"/>')
        nav_points.append(f'''<navPoint id="chapter{i}" playOrder="{i}">
            <navLabel><text>Chapter {i}</text></navLabel>
            <content src="chapter{i}.xhtml"/>
        </navPoint>''')

    content_opf = f'''<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="BookID">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:title>{safe_title}</dc:title>
        <dc:creator>{safe_author}</dc:creator>
        <dc:language>en</dc:language>
        <dc:identifier id="BookID">{book_uid}</dc:identifier>
        <meta property="dcterms:modified">{dt.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}</meta>
    </metadata>
    <manifest>
        {''.join(manifest_items)}
        <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    </manifest>
    <spine toc="ncx">
        {''.join(spine_items)}
    </spine>
</package>'''

    toc_ncx = f'''<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
    <head>
        <meta name="dtb:uid" content="{book_uid}"/>
        <meta name="dtb:depth" content="1"/>
    </head>
    <docTitle><text>{safe_title}</text></docTitle>
    <navMap>
        {''.join(nav_points)}
    </navMap>
</ncx>'''

    container_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>'''

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as epub:
        epub.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)
        epub.writestr('META-INF/container.xml', container_xml)
        epub.writestr('OEBPS/content.opf', content_opf)
        epub.writestr('OEBPS/toc.ncx', toc_ncx)
        for i, chapter_text in enumerate(chapters, 1):
            paragraphs = [p.strip() for p in chapter_text.split('\n\n') if p.strip()]
            html_paragraphs = ''.join(
                f'<p>{html_escape.escape(p)}</p>\n' for p in paragraphs
            )
            chapter_xhtml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{safe_title}</title>
    <style>
        body {{ font-family: serif; line-height: 1.6; margin: 2em; }}
        p {{ margin-bottom: 1em; text-indent: 1.5em; }}
        p:first-of-type {{ text-indent: 0; }}
    </style>
</head>
<body>
    {html_paragraphs}
</body>
</html>'''
            epub.writestr(f'OEBPS/chapter{i}.xhtml', chapter_xhtml)

    return buffer.getvalue()


class BookTranslator:
    def __init__(
        self,
        model_name: str = "llama3.3:70b-instruct-q2_K",
        chunk_size: int = 1200,
        verifier_model: Optional[str] = None,
    ):
        self.model_name = model_name
        # Who rules on whether a Stage 2 patch is an improvement. Defaults to
        # this same model, which is the arrangement that made the refinement
        # pass a no-op: a model grading its own edit is not a check, and the
        # A/B verdict it gives flips with the order the two versions are shown
        # in, so "must win both orderings" was effectively a coin toss the
        # patch had to win twice. Set this to a model clearly larger than the
        # reviewer and the double-blind vote starts measuring accuracy again.
        self.verifier_model = verifier_model or model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.chunk_size = chunk_size
        self.session = requests.Session()
        self.session.mount('http://', requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        ))
        self.terminology = TerminologyManager()
        self._verifier: Optional['BookTranslator'] = None

        # Note: Ollama should be running separately
        # Don't try to start it automatically

    @property
    def verifier(self) -> 'BookTranslator':
        """The translator that runs the Stage 2 verdict, built on first use.

        Returns ``self`` when no separate verifier was chosen, so nothing is
        constructed for a run that does not need it.
        """
        if self.verifier_model == self.model_name:
            return self
        if self._verifier is None:
            self._verifier = BookTranslator(
                model_name=self.verifier_model, chunk_size=self.chunk_size,
            )
        return self._verifier

    def split_into_chunks(self, text: str) -> list:
        """Split text into smaller chunks for translation.

        Paragraph-sized chunks (self.chunk_size, 1200 chars by default) rather
        than page-sized ones: translation quality degrades on long blocks, and
        TranslateGemma in particular is tuned for segment-level input.
        """
        MAX_LENGTH = self.chunk_size
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            if len(paragraph) + current_length > MAX_LENGTH:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                if len(paragraph) > MAX_LENGTH:
                    sentences = paragraph.split('. ')
                    temp_chunk = []
                    temp_length = 0
                    
                    for sentence in sentences:
                        if temp_length + len(sentence) > MAX_LENGTH:
                            if temp_chunk:
                                chunks.append('. '.join(temp_chunk) + '.')
                                temp_chunk = []
                                temp_length = 0
                        temp_chunk.append(sentence)
                        temp_length += len(sentence) + 2  # +2 for '. '
                        
                    if temp_chunk:
                        chunks.append('. '.join(temp_chunk) + '.')
                else:
                    current_chunk.append(paragraph)
                    current_length = len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph) + 2  # +2 for '\n\n'
                
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    # ------------------------------------------------------------------
    # Stage 0: Prepare. Runs once per document, before any translation, and
    # produces the proper-noun records every later stage is fed.
    #
    # Candidate names are harvested deterministically first and only then
    # handed to the model to render, rather than asking the model to do NER
    # over the whole book: a book is hundreds of chunks, and a frequency
    # count over capitalised runs finds the recurring names for free.
    # ------------------------------------------------------------------

    # A ceiling on how many candidates reach the rendering call. It was 40,
    # set when Stage 0 was a single model call; batching means the only cost
    # of raising it is more calls, and 40 is far too few for a text that
    # names a lot of things once each — a critical essay mentions six novels
    # and a dozen authors in as many pages.
    PNR_MAX_TERMS = 120
    # How much source text a single Stage 0 model call is shown. Kept well
    # inside _ollama_payload's num_ctx so the whole excerpt plus the answer
    # fits without silent truncation by the runtime.
    PREPARE_SOURCE_BUDGET = 6000

    # A "word" for harvesting purposes: letters only, so digits and
    # punctuation can't start a candidate, but internal apostrophes and
    # hyphens stay inside one (O'Brien, Privet-Drive).
    _WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
    # Sentence-final punctuation, optionally followed by a closing quote or
    # bracket — what has to precede a word for it to count as capitalised
    # only because it opens a sentence.
    _SENTENCE_END_RE = re.compile(r'[.!?…:;]["\'”»)\]]*\s*\Z')
    # What may sit between two words for them to still be one name:
    # a single space or a hyphen, nothing else.
    _NAME_JOIN_RE = re.compile(r'[ \-]')
    # An English possessive tail, stripped so "Dursley's" and "Dursley"
    # count as one name rather than two records that can drift apart.
    _POSSESSIVE_RE = re.compile(r"['’]s\Z", re.IGNORECASE)
    # A short capitalised word followed by a period is an abbreviation
    # (Mr., Mrs., Dr., St.), not the end of a sentence. Without this, the
    # name in "said Mr. Dursley" looks sentence-initial and loses the
    # mid-sentence evidence that identifies it as a name at all — which is
    # how most characters are mentioned most of the time in English prose.
    _ABBREVIATION_MAX_LENGTH = 4
    # Titles often look like mid-sentence proper nouns because they appear
    # directly before a name ("met Mrs. Fenwick"). They are instructions
    # about how to address a person, not independent glossary entries.
    _HONORIFICS = frozenset({
        'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'sir', 'dame',
        'lord', 'lady', 'rev', 'fr', 'sr', 'jr',
    })
    ENTITY_TYPES = frozenset({'person', 'family', 'place', 'organisation', 'work', 'term', 'other'})
    # How many candidates go into one rendering call. Small enough that every
    # candidate can carry its own quoted context and the answer still fits in
    # num_ctx, which is what a single truncated excerpt could never promise.
    PREPARE_BATCH_TERMS = 8
    # Per-quote budget for that context. Two short quotes per name say more
    # about how a name is used than the opening pages of the book do.
    PREPARE_EVIDENCE_CHARS = 240
    # An adjudication pass costs one call, so it is worth running only while
    # there is a bounded number of genuinely ambiguous groups to rule on.
    CLUSTER_REVIEW_MAX_GROUPS = 24
    # Prepare is one deliberate action per document, and the roles that reason
    # deserve a large model, so its calls get a longer read timeout than the
    # per-chunk work. A 27B model answering in 200s is not a failure; running
    # out of patience at 180s and reporting "no proper nouns found" was.
    PREPARE_READ_TIMEOUT = 600
    # Optional document diagnostics. They stay lazy because loading either
    # model is a deliberate quality-check action, never a prerequisite for
    # translating a book.
    LABSE_MODEL_ID = 'sentence-transformers/LaBSE'
    LANGUAGE_ID_MODEL_ID = 'papluca/xlm-roberta-base-language-detection'
    _labse_model = None
    _labse_lock = threading.Lock()
    _language_id_pipeline = None
    _language_id_lock = threading.Lock()

    @classmethod
    def harvest_proper_noun_candidates(cls, text: str, limit: int = PNR_MAX_TERMS) -> List[Dict]:
        """Frequency-ranked proper-noun candidates from the source, found
        without a model call. ``limit=0`` returns all of them.

        A candidate is a run of up to three consecutive capitalised words.
        The discriminator is position: a capitalised word that also occurs
        mid-sentence is a name, whereas one that only ever opens a sentence
        is probably just a sentence-initial ordinary word — and if its
        lowercase form appears elsewhere in the text, that settles it.

        Returns [] for scripts without case distinction (Chinese, Japanese,
        Arabic, …), where capitalisation carries no signal at all; the
        caller falls back to asking the model directly in that case.
        """
        words = [(m.group(0), m.start(), m.end()) for m in cls._WORD_RE.finditer(text)]
        lowercase_counts = Counter(word.casefold() for word, _, _ in words if word[:1].islower())

        candidates: Dict[str, Dict] = {}
        index = 0
        while index < len(words):
            word, start, _ = words[index]
            if not word[:1].isupper():
                index += 1
                continue

            # Grow the run over following capitalised words (max 3 words).
            run_end_index = index
            while run_end_index + 1 < len(words) and run_end_index - index + 1 < 3:
                next_word, next_start, _ = words[run_end_index + 1]
                gap = text[words[run_end_index][2]:next_start]
                if not next_word[:1].isupper() or not cls._NAME_JOIN_RE.fullmatch(gap):
                    break
                run_end_index += 1

            # Drop leading words of the run that the same text also uses in
            # lowercase — they were only capitalised because the sentence
            # started there. Without this, "The Dursleys" is harvested as a
            # unit and the name "Dursleys" is never seen on its own, which
            # is exactly the plural/singular pair that drifts apart.
            run_start_index = index
            while (
                run_start_index < run_end_index
                and lowercase_counts.get(words[run_start_index][0].casefold(), 0) > 0
            ):
                run_start_index += 1

            run_start = words[run_start_index][1]
            surface = cls._POSSESSIVE_RE.sub('', text[run_start:words[run_end_index][2]])
            preceding = text[:run_start]
            if not preceding.strip():
                sentence_initial = True
            elif run_start_index > index or not cls._SENTENCE_END_RE.search(preceding):
                # Something of the run was dropped in front of this word, so
                # whatever else it is, it does not open the sentence.
                sentence_initial = False
            else:
                previous_word, _, previous_end = words[index - 1] if index else ('', 0, 0)
                abbreviated = (
                    0 < len(previous_word) <= cls._ABBREVIATION_MAX_LENGTH
                    and previous_word[:1].isupper()
                    and text[previous_end:run_start].startswith('.')
                )
                sentence_initial = not abbreviated

            record = candidates.setdefault(surface, {'surface': surface, 'count': 0, 'mid_sentence': 0})
            record['count'] += 1
            if not sentence_initial:
                record['mid_sentence'] += 1

            index = run_end_index + 1

        harvested = []
        for record in candidates.values():
            surface = record['surface']
            if len(surface) < 2:
                continue
            if surface.casefold().rstrip('.') in cls._HONORIFICS:
                continue
            # Appears mid-sentence at least once → a name, not a capitalised
            # sentence opener.
            if record['mid_sentence'] > 0:
                harvested.append(record)
                continue
            # Never mid-sentence: the lowercase form being absent from the
            # whole document is the evidence, not the number of mentions.
            # Requiring three cost the essay "Persuasion" and "Walt Whitman",
            # each named once and each at the head of its sentence, while
            # letting through nothing the lowercase test would have caught —
            # a sentence-opening "Also" or "Yet" is written in lowercase
            # elsewhere in any text of length.
            if lowercase_counts.get(surface.casefold(), 0) == 0:
                harvested.append(record)

        # A run that only ever opens a sentence, and whose tail is a candidate
        # in its own right, is that candidate with a sentence-opening word
        # stuck to the front: "Although Miss Austen" beside "Miss Austen".
        # Deciding it by asking the document, rather than by a list of
        # function words, keeps the harvest language-neutral — which is also
        # why "Walt Whitman" and "Mansfield Park" survive: nothing in the text
        # uses "Whitman" or "Park" on its own.
        surfaces = {record['surface'] for record in harvested}
        harvested = [
            record for record in harvested
            if record['mid_sentence'] > 0
            or ' ' not in record['surface']
            or record['surface'].split(maxsplit=1)[1] not in surfaces
        ]
        harvested.sort(key=lambda record: (-record['count'], record['surface']))
        return harvested[:limit] if limit else harvested

    @staticmethod
    def _parse_json_array(raw: Optional[str]) -> List[Dict]:
        """Pull a JSON array of objects out of a model answer.

        Local instruct models wrap JSON in prose or fences more often than
        not, so the outermost bracket pair is extracted rather than trusting
        the whole response to parse. Anything unparseable yields [] — every
        caller here treats "no structured answer" as "nothing to apply",
        never as an error worth failing the pass over.
        """
        if not raw:
            return []
        start, end = raw.find('['), raw.rfind(']')
        if start == -1 or end <= start:
            return []
        try:
            parsed = json.loads(raw[start:end + 1])
        except (ValueError, TypeError):
            return []
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]

    @classmethod
    def build_glossary_candidates(cls, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Run the document-wide glossary builder used by Stage 0 Prepare.

        The builder uses GLiNER to find recurring entity mentions and a
        multilingual embedding model to merge source variants before the LLM
        assigns target-language renderings.  Keeping that work in
        ``build_glossary.py`` also makes the same algorithm available from
        the command line without duplicating its heuristics here.

        The neural clusters are then unioned with the capitalisation harvest,
        because the two miss different things and Stage 0 only gets one pass.
        """
        try:
            from build_glossary import build_document_glossary
        except ImportError as exc:
            raise RuntimeError(
                'The glossary builder dependencies are missing. Install them with '
                'pip install -r requirements.txt.'
            ) from exc

        entries, review_queue = build_document_glossary(text)
        kind_map = {
            'location': 'place',
            'organization': 'organisation',
            'object': 'term',
            'title': 'term',
        }
        candidates = []
        for entry in entries:
            source = str(entry.get('canonical') or '').strip()
            if not source:
                continue
            candidates.append({
                'surface': source,
                'count': int(entry.get('count') or 0),
                'kind': kind_map.get(str(entry.get('type') or '').lower(), str(entry.get('type') or 'other')),
                'evidence': list(entry.get('contexts') or [])[:3],
                'variants': entry.get('variants') or [],
            })
        return cls.merge_harvested_candidates(text, candidates), review_queue

    @classmethod
    def merge_harvested_candidates(cls, text: str, candidates: List[Dict]) -> List[Dict]:
        """Union the neural candidate list with the capitalisation harvest.

        The builder only reports a cluster once it clears its document-wide
        frequency floor, so a street, a firm, or a character mentioned twice
        in a chapter never reaches the rendering call at all — and because
        that call may only render sources the candidate list established,
        Stage 0 came back with two or three entries for a text full of names.

        The harvest is a frequency count over capitalised runs: no model, no
        cost, and grounded in the source the same way, so it is unioned in
        rather than kept as a fallback for when the builder is unavailable.
        Non-names it lets through are the rendering call's job to skip, which
        is what its prompt already asks for.
        """
        merged: Dict[str, Dict] = {}
        # Harvest without a limit and cut once, at the end. Truncating here
        # first threw away most of what the union exists to recover: the
        # harvest is ordered by frequency, everything named once ties, and
        # the tie broke alphabetically — so on one essay the surviving forty
        # ran out at "L" and Mansfield Park, Northanger Abbey, Scott,
        # Smollett and Thackeray were gone for no reason but their initials.
        harvested = [
            {'surface': record['surface'], 'count': record['count'], 'kind': 'other', 'evidence': []}
            for record in cls.harvest_proper_noun_candidates(text, limit=0)
        ]
        for record in list(candidates) + harvested:
            surface = str(record.get('surface') or '').strip()
            if not surface or not cls.is_glossary_source(text, surface):
                continue
            # GLiNER labels the pronoun "she" a person, 13 mentions and all,
            # and a single-word candidate the document also writes in
            # lowercase is a common word wearing a capital at the start of a
            # sentence. This is the harvest's own discriminator, applied to
            # the neural list, which had none.
            if ' ' not in surface and re.search(
                rf'(?<![^\W\d_]){re.escape(surface.lower())}(?![^\W\d_])', text,
            ):
                continue
            existing = merged.get(surface.casefold())
            if existing is None:
                merged[surface.casefold()] = {
                    'surface': surface,
                    'canonical_source': surface,
                    'count': int(record.get('count') or 0),
                    'kind': record.get('kind') or 'other',
                    'evidence': list(record.get('evidence') or []),
                    'variants': list(record.get('variants') or []),
                }
                continue
            existing['count'] = max(existing['count'], int(record.get('count') or 0))
            if existing['kind'] == 'other' and record.get('kind'):
                existing['kind'] = record['kind']
            existing['evidence'] = (existing['evidence'] + list(record.get('evidence') or []))[:3]
        # The harvest reports bare surnames and the builder reports the
        # title-bearing mention, so without this the same person arrives as
        # two entries whose renderings can then be flagged as colliding.
        collapsed = []
        for record in cls.collapse_honorific_aliases(text, list(merged.values())):
            record['surface'] = record.pop('canonical_source')
            # Evidence is what makes a candidate judgeable. The harvest counts
            # occurrences and keeps none, so "Grunnings" reached the model as a
            # bare token appearing once — indistinguishable from a fragment of
            # a chapter heading, and skipped along with it. Quote the source.
            if not record.get('evidence'):
                record['evidence'] = cls.evidence_for(text, record['surface'])
            collapsed.append(record)
        collapsed = cls.drop_ambiguous_given_names(collapsed)
        # Frequency first, then where the document introduces the term. Any
        # tie-break has to be a fact about the text; the alphabet is not one,
        # and it decided which names a reader got to see.
        collapsed.sort(key=lambda record: (-record.get('count', 0), text.find(record['surface'])))
        return collapsed[:cls.PNR_MAX_TERMS]

    @staticmethod
    def drop_ambiguous_given_names(candidates: List[Dict]) -> List[Dict]:
        """Remove a bare first name that several full names in the list share.

        One glossary line is one rendering for every occurrence of its source
        string, so ``Jane`` cannot serve Jane Austen, Jane Bennet and Jane
        Fairfax at once. Keeping the full names and dropping the bare form
        agrees what can actually be agreed, instead of quietly pinning three
        people to one rendering.
        """
        full_names: Dict[str, set] = {}
        for record in candidates:
            parts = record['surface'].split()
            if len(parts) > 1:
                full_names.setdefault(parts[0].casefold(), set()).add(record['surface'])
        return [
            record for record in candidates
            if ' ' in record['surface']
            or len(full_names.get(record['surface'].casefold(), ())) < 2
        ]

    # How much of the sentence around a mention to quote on each side.
    EVIDENCE_WINDOW_CHARS = 140

    @classmethod
    def evidence_for(cls, text: str, term: str, limit: int = 2) -> List[str]:
        """Quote the document around its first few mentions of a term."""
        quotes, start = [], 0
        while len(quotes) < limit:
            found = text.find(term, start)
            if found == -1:
                break
            quotes.append(text[
                max(0, found - cls.EVIDENCE_WINDOW_CHARS):
                found + len(term) + cls.EVIDENCE_WINDOW_CHARS
            ])
            start = found + len(term)
        return quotes

    @classmethod
    def is_glossary_source(cls, text: str, term: str) -> bool:
        """Whether a term may become a glossary source at all.

        The one grounding rule in Stage 0: a source term has to be a literal
        span of the document. A model that renders names is allowed to notice
        one the extractors missed, and is not allowed to invent a character —
        the difference between the two is exactly this check.
        """
        term = term.strip()
        return bool(
            term
            and len(term) <= TerminologyManager.MAX_TERM_LENGTH
            and term.casefold().rstrip('.') not in cls._HONORIFICS
            and term in text
        )

    @staticmethod
    def _quote(text: str, limit: int) -> str:
        """One line of evidence, collapsed and clipped for a prompt."""
        collapsed = re.sub(r'\s+', ' ', text).strip()
        return collapsed[:limit] + ('…' if len(collapsed) > limit else '')

    @classmethod
    def cluster_review_groups(
        cls, text: str, candidates: List[Dict], review_queue: List[Dict],
    ) -> List[Dict]:
        """Collect the clustering decisions worth a model's opinion.

        Two kinds arrive. The embedding model auto-merged variant forms into
        one candidate, which is a claim that they are one entity; and it left
        borderline pairs in a review queue, which is a claim that it could not
        tell. Both are guesses about identity from string and context
        similarity, and identity is precisely what a language model can judge
        and cosine distance cannot.

        Only groups whose forms the document literally contains are put up for
        a ruling, since a ruling on any other form could not be applied to it.
        """
        groups: List[Dict] = []
        seen: set = set()

        def add(forms: List[str], merged_by: str, evidence: List[str]) -> None:
            grounded = [
                form for form in dict.fromkeys(forms)
                if cls.is_glossary_source(text, form)
            ]
            key = tuple(sorted(form.casefold() for form in grounded))
            if len(grounded) < 2 or key in seen:
                return
            seen.add(key)
            groups.append({'forms': grounded, 'merged_by': merged_by, 'evidence': evidence})

        for record in candidates:
            add(
                [record['surface']] + [
                    str(variant.get('form') or '').strip()
                    for variant in record.get('variants') or []
                ],
                'embeddings',
                list(record.get('evidence') or [])[:2],
            )
        for pair in review_queue:
            add(
                [str(pair.get('a') or '').strip(), str(pair.get('b') or '').strip()],
                'unresolved',
                [quote for quote in (pair.get('context_a'), pair.get('context_b')) if quote],
            )
        return groups[:cls.CLUSTER_REVIEW_MAX_GROUPS]

    def adjudicate_entity_clusters(
        self, text: str, source_lang: str, candidates: List[Dict],
        review_queue: Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Have the model rule on what the embedding model merged and split.

        Returns the candidate list with its rulings applied, plus the rulings
        themselves so the interface can show what was decided rather than
        reporting a count of pairs nobody ever looked at.

        Costs nothing when there is nothing ambiguous: with no multi-form
        cluster and an empty review queue there is no call to make.
        """
        groups = self.cluster_review_groups(text, candidates, review_queue or [])
        if not groups:
            return candidates, []

        source_name = LANG_NAMES.get(source_lang, source_lang)
        listing = []
        for index, group in enumerate(groups, 1):
            evidence = ' / '.join(
                repr(self._quote(quote, self.PREPARE_EVIDENCE_CHARS))
                for quote in group['evidence']
            )
            listing.append(
                f"{index}. {' | '.join(group['forms'])}"
                + (f"\n   context: {evidence}" if evidence else '')
            )
        prompt = f"""These groups of expressions come from one {source_name} document. A clustering step guessed that the forms inside each group name the same entity, or could not decide. Rule on each group.

GROUPS:
{chr(10).join(listing)}

Respond with ONLY a JSON array, no prose, no code fence. One element per numbered group:
{{"group": <number>, "same_entity": true|false, "canonical": "<the form to use as the entry, copied exactly from the group>"}}

Rules:
- same_entity is true only when every form in the group refers to one and the same entity.
- A singular name and its plural or family form are DIFFERENT entities: "Dursley" and "Dursleys" must stay separate, and so must a person and the place named after them.
- A name with a title and the bare name are the same entity ("Mrs. Fenwick" and "Fenwick"); prefer the bare name as the canonical form.
- "canonical" must be one of the forms shown in that group, copied character for character. When same_entity is false, give the form that is most usable on its own.
- Rule on every group. Do not add groups."""

        decisions, ruled = [], set()
        for item in self._parse_json_array(self._call_model(
                prompt, temperature=0.0, read_timeout=self.PREPARE_READ_TIMEOUT,
            )):
            try:
                index = int(item.get('group'))
            except (TypeError, ValueError):
                continue
            # One ruling per group. A model that answers the same group twice
            # would otherwise have both applied, in whichever order they came.
            if not 1 <= index <= len(groups) or index in ruled:
                continue
            ruled.add(index)
            group = groups[index - 1]
            canonical = str(item.get('canonical') or '').strip()
            allowed = {form.casefold(): form for form in group['forms']}
            # The model chooses among forms the document already contains; a
            # canonical it invented would create an unenforceable entry.
            if canonical.casefold() not in allowed or not self.is_glossary_source(text, canonical):
                canonical = max(group['forms'], key=lambda form: text.count(form))
                if not self.is_glossary_source(text, canonical):
                    continue
            decisions.append({
                'forms': group['forms'],
                'same_entity': bool(item.get('same_entity')),
                'canonical': canonical,
                'proposed_by': group['merged_by'],
            })
        return self.apply_cluster_decisions(text, candidates, decisions), decisions

    @classmethod
    def apply_cluster_decisions(
        cls, text: str, candidates: List[Dict], decisions: List[Dict],
    ) -> List[Dict]:
        """Rewrite the candidate list to match the adjudicated identities.

        A confirmed group becomes one candidate under the agreed canonical
        form. A rejected group becomes one candidate per form that the
        document actually contains — which is the half no purely automatic
        pipeline could do, because a wrong merge is invisible once made.
        """
        by_surface = {record['surface'].casefold(): record for record in candidates}
        consumed: set = set()
        rebuilt: List[Dict] = []
        for decision in decisions:
            members = [
                by_surface[form.casefold()]
                for form in decision['forms'] if form.casefold() in by_surface
            ]
            if not members:
                continue
            evidence, variants = [], []
            for member in members:
                consumed.add(member['surface'].casefold())
                evidence.extend(member.get('evidence') or [])
                variants.extend(member.get('variants') or [])
            kind = next(
                (member['kind'] for member in members if member.get('kind') not in (None, 'other')),
                'other',
            )
            if decision['same_entity']:
                rebuilt.append({
                    # A confirmed group of "Dursley" and "Mrs. Dursley" is one
                    # person, and the entry has to be the bare name whichever
                    # form the model named as canonical: a constraint on the
                    # titled mention matches almost none of its occurrences.
                    'surface': cls.strip_honorific(text, decision['canonical']),
                    'count': sum(member.get('count', 0) for member in members),
                    'kind': kind,
                    'evidence': evidence[:3],
                    'variants': variants,
                })
                continue
            for form in decision['forms']:
                if not cls.is_glossary_source(text, form):
                    continue
                member = by_surface.get(form.casefold())
                rebuilt.append({
                    'surface': form,
                    'count': (member or {}).get('count') or text.count(form),
                    'kind': (member or {}).get('kind') or kind,
                    'evidence': (member or {}).get('evidence') or evidence[:2],
                    'variants': [],
                })
        untouched = [
            record for record in candidates
            if record['surface'].casefold() not in consumed
        ]
        merged: Dict[str, Dict] = {}
        for record in untouched + rebuilt:
            merged.setdefault(record['surface'].casefold(), record)
        return sorted(
            merged.values(), key=lambda record: (-record.get('count', 0), record['surface']),
        )[:cls.PNR_MAX_TERMS]

    _HONORIFIC_PREFIX_RE = re.compile(
        r'^(?:(?:(?P<t1>mr|mrs|ms|miss|dr|prof|sir|dame|lord|lady)\.?)\s+and\s+)?'
        r'(?:(?:mr|mrs|ms|miss|dr|prof|sir|dame|lord|lady)\.?\s+)?(?P<name>.+)$',
        re.IGNORECASE,
    )

    @classmethod
    def strip_honorific(cls, text: str, form: str) -> str:
        """``Mrs. Dursley`` → ``Dursley``, as long as that is a source span.

        The bare name is the form worth agreeing a rendering for: it is what
        the text mostly uses, and a constraint pinned to the titled mention
        would not apply to any of those occurrences.
        """
        form = form.strip()
        match = cls._HONORIFIC_PREFIX_RE.match(form)
        bare = match.group('name').strip() if match else ''
        return bare if bare and bare != form and bare in text else form

    @classmethod
    def collapse_honorific_aliases(cls, text: str, entities: List[Dict]) -> List[Dict]:
        """Collapse title-based mentions onto a bare source name when proven.

        NER should preserve ``Mr. Dursley`` as evidence, but it must not turn
        it into a separate glossary constraint when ``Dursley`` appears in
        the document. Singular and plural names remain distinct because their
        bare forms are different strings.
        """
        collapsed: Dict[str, Dict] = {}
        for entity in entities:
            record = dict(entity)
            bare = cls.strip_honorific(text, record['canonical_source'])
            if bare != record['canonical_source']:
                record['canonical_source'] = bare
                record['surface'] = bare
            key = record['canonical_source'].casefold()
            existing = collapsed.get(key)
            if existing is None:
                collapsed[key] = record
                continue
            existing_evidence = existing.get('evidence', [])
            existing['evidence'] = (existing_evidence + record.get('evidence', []))[:3]
            existing['count'] = max(existing.get('count', 0), record.get('count', 0))
            if existing.get('kind') == 'other' and record.get('kind') != 'other':
                existing['kind'] = record['kind']
        return sorted(collapsed.values(), key=lambda entity: (-entity.get('count', 0), entity['canonical_source']))

    def propose_proper_noun_records(
        self, text: str, source_lang: str, target_lang: str, genre: str = 'unknown',
        candidates: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """One agreed target rendering per recurring name in the source.

        The candidates are handed over in batches, each name carrying its own
        quoted context from wherever in the book it occurs. A single prompt
        holding the opening pages could not do that: on a novel it showed the
        model an excerpt that mentions almost none of the names it was being
        asked to render, so it rendered them blind.

        The model decides the enforcement mode as well as the rendering. An
        invented brand may have to appear letter for letter while a surname
        must be free to inflect, and that distinction is a fact about the term
        rather than a default worth hard-coding.
        """
        candidates = candidates if candidates is not None else self.harvest_proper_noun_candidates(text)
        counts = {record['surface'].casefold(): record.get('count', 0) for record in candidates}

        records: List[Dict] = []
        seen: set = set()
        batches = [
            candidates[start:start + self.PREPARE_BATCH_TERMS]
            for start in range(0, len(candidates), self.PREPARE_BATCH_TERMS)
        ] or [[]]
        for batch in batches:
            prompt = self._rendering_prompt(text, batch, source_lang, target_lang, genre)
            raw = self._call_model(
                prompt, temperature=0.2, read_timeout=self.PREPARE_READ_TIMEOUT,
            )
            for item in self._parse_json_array(raw):
                record = self._rendering_record(text, item, counts)
                if record is None or record['source'].casefold() in seen:
                    continue
                seen.add(record['source'].casefold())
                records.append(record)
            if len(records) >= self.PNR_MAX_TERMS:
                break

        records.sort(key=lambda record: (-record['count'], record['source']))
        return records[:self.PNR_MAX_TERMS]

    def _rendering_prompt(
        self, text: str, batch: List[Dict], source_lang: str, target_lang: str, genre: str,
    ) -> str:
        """The Stage 0 rendering prompt for one batch of candidates."""
        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)
        genre_line = f"\nThe document is: {genre}." if genre and genre != 'unknown' else ""
        mode_rules = """
Choose "mode" per term:
- "exact": the target must appear letter for letter every time — codes, invented brands, titles of works, anything a reader would notice being altered.
- "inflectable": the lexical choice is fixed but the target language may inflect it — the normal choice for names of people and places.
- "preferred": use this rendering where it fits, and allow a freer translation where it does not — descriptive names and common-noun terms."""

        if batch:
            listing = []
            for record in batch:
                variants = [
                    str(variant.get('form') or '').strip()
                    for variant in record.get('variants') or []
                    if str(variant.get('form') or '').strip() and str(variant.get('form')).strip() != record['surface']
                ]
                entry = (
                    f"- {record['surface']} (type: {record.get('kind', 'candidate')};"
                    f" appears {record.get('count', 0)}x)"
                )
                if variants:
                    entry += f"\n    also written: {', '.join(dict.fromkeys(variants))}"
                # The builder's contexts come from overlapping chunks, so the
                # same passage often arrives twice; a repeat is prompt budget
                # spent saying nothing.
                quotes = dict.fromkeys(
                    self._quote(quote, self.PREPARE_EVIDENCE_CHARS)
                    for quote in record.get('evidence') or []
                )
                for quote in list(quotes)[:2]:
                    entry += f"\n    context: {quote!r}"
                listing.append(entry)
            task = f"""Below are entity candidates from a {source_name} document, each with the context it appears in.

For each expression that is a PROPER NOUN (person, family, place, street, company, brand, invented term), give the single {target_name} rendering that must be used for it everywhere in the document. Skip anything that is not a proper noun — ordinary words, sentence openers, common nouns, dates and weekdays.{genre_line}

CANDIDATES:
{chr(10).join(listing)}"""
        else:
            # No candidate survived extraction, which is the normal case for a
            # script that carries no capitalisation signal. The excerpt is all
            # there is to go on.
            task = f"""Below is an excerpt from a {source_name} document.

List the PROPER NOUNS in it (person, family, place, street, company, brand, invented term) and give the single {target_name} rendering that must be used for each one everywhere in the document.{genre_line}

EXCERPT:
{text[:self.PREPARE_SOURCE_BUDGET]}"""

        return f"""{task}

Respond with ONLY a JSON array, no prose, no code fence. Each element:
{{"source": "<expression exactly as written in the {source_name} text>", "target": "<the {target_name} rendering>", "kind": "person|place|organisation|work|term|other", "mode": "exact|inflectable|preferred"}}

Rules:
- One element per distinct proper noun. Do not repeat.
- "source" must be copied character for character from the document. You may add a proper noun you can see in the context quotes that is missing from the candidate list, as long as you copy it exactly; never write a form the document does not contain.
- "target" is the base form only. Do not add grammatical endings, articles, or explanations.
- Leave titles and honorifics out of "source": write "Fenwick", never "Mrs. Fenwick" or "Mr. and Mrs. Fenwick".
- Distinct source forms need distinct renderings: a singular name and its plural or family form must not both become the same target string.
- Names of people and places are normally transcribed, not translated by meaning, unless the document's tradition clearly demands otherwise.{mode_rules}
- If there are no proper nouns at all, respond with []."""

    @classmethod
    def _rendering_record(cls, text: str, item: Dict, counts: Dict[str, int]) -> Optional[Dict]:
        """Validate one rendering the model proposed, or reject it."""
        source = str(item.get('source') or '').strip()
        target = str(item.get('target') or '').strip()
        if not target or len(target) > TerminologyManager.MAX_TERM_LENGTH:
            return None
        # A source the document does not literally contain cannot be enforced
        # against it, whether the model invented the entity or just normalised
        # its spelling. Everything else about the answer is the model's call.
        if not cls.is_glossary_source(text, source):
            return None
        # "Mr. and Mrs. Dursley" is a real span of the document and still the
        # wrong entry: the rendering agreed for it would match none of the
        # plain mentions. The bare name is the entry, and it comes from the
        # candidate list, so reject the titled form rather than rewrite it and
        # keep a target that was written for the longer phrase.
        if cls.strip_honorific(text, source) != source:
            return None
        kind = str(item.get('kind') or 'other').strip().lower()
        mode = str(item.get('mode') or '').strip().lower()
        return {
            'source': source,
            'target': target,
            'kind': kind if kind in cls.ENTITY_TYPES else 'other',
            'count': counts.get(source.casefold()) or text.count(source),
            'mode': mode if mode in TerminologyManager.VALID_MODES else 'inflectable',
        }

    @staticmethod
    def find_rendering_conflicts(records: List[Dict]) -> List[Dict]:
        """Find term-rendering collisions that make a glossary unsafe.

        A small model can assign the same Russian surface form to distinct
        source forms such as ``Dursley`` and ``Dursleys``. That is not a
        stylistic preference: it destroys the number/entity distinction that
        Stage 0 was created to preserve. Do not silently accept it as a valid
        glossary merely because its syntax parses.
        """
        by_target: Dict[str, List[str]] = {}
        for record in records:
            key = re.sub(r'\s+', ' ', record['target'].casefold()).strip()
            by_target.setdefault(key, []).append(record['source'])
        conflicts = []
        for target, sources in by_target.items():
            unique_sources = sorted(set(sources))
            if len(unique_sources) < 2:
                continue
            conflicts.append({
                'target': target,
                'sources': unique_sources,
                'reason': 'distinct source entities have the same target rendering',
            })
        return conflicts

    def translate_stage1(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        translation_id: int,
        genre: str = 'unknown',
        terminology: Optional[TerminologyManager] = None,
        chapters: Optional[List[str]] = None,
    ):
        """STAGE 1 only: primary draft translation. Persists the draft chunks
        so a later, independent translate_stage2() call can refine them
        without re-translating from scratch."""
        start_time = time.time()
        success = False

        try:
            chunk_chapter_map = []
            if chapters:
                # Split each chapter on its own, so no chunk ever straddles a
                # chapter boundary — that's what lets us reassemble a translated
                # EPUB with the same chapter breaks as the original afterwards.
                chunks = []
                for chapter_index, chapter_text in enumerate(chapters):
                    chapter_chunks = self.split_into_chunks(chapter_text) if chapter_text.strip() else ['']
                    chunks.extend(chapter_chunks)
                    chunk_chapter_map.extend([chapter_index] * len(chapter_chunks))
            else:
                chunks = self.split_into_chunks(text)
                chunk_chapter_map = [0] * len(chunks)
            total_chunks = len(chunks)
            draft_translations = []
            self.terminology = terminology or TerminologyManager()
            glossary_fingerprint = self.terminology.fingerprint()
            used_terms = set()
            stage1_violation_count = 0

            logger.translation_logger.info(f"Starting stage 1 for translation {translation_id} with {total_chunks} chunks (genre: {genre})")

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations
                    SET total_chunks = ?, status = 'in_progress', genre = ?
                    WHERE id = ?
                ''', (total_chunks, genre, translation_id))
            claim_run(translation_id)

            # STAGE 1: Primary translation with context
            logger.translation_logger.info("Stage 1: Primary LLM translation")
            for i, chunk in enumerate(chunks, 1):
                try:
                    if not chunk.strip():
                        # Empty chapter (e.g. a title page) — nothing to send to the model.
                        draft_translations.append('')
                        progress = (i / total_chunks) * 100
                        yield {
                            'progress': progress,
                            'stage': 'primary_translation',
                            'batch_index': i,
                            'original_chunk': chunk,
                            'machine_translation_chunk': '',
                            'current_chunk': i,
                            'total_chunks': total_chunks,
                            'terminology': {
                                'total': len(self.terminology.terms),
                                'used': len(used_terms),
                                'violations': stage1_violation_count,
                            },
                        }
                        continue

                    relevant_terms = self.terminology.relevant_terms(chunk)
                    used_terms.update(term.source.casefold() for term in relevant_terms)
                    terminology_context = self.terminology.prompt_context(chunk)
                    stage1_cache_model = f"{self.model_name}_stage1_glossary_{glossary_fingerprint}"
                    # Check cache
                    cached_result = cache.get_cached_translation(
                        chunk, source_lang, target_lang, stage1_cache_model
                    )
                    stage1_warning = None
                    if cached_result:
                        draft_translation = cached_result['machine_translation']
                        logger.translation_logger.info(f"Cache hit for stage 1 chunk {i}")
                    else:
                        # Get previous context
                        previous_chunk = draft_translations[-1] if draft_translations else ""

                        logger.translation_logger.info(f"Stage 1 translating chunk {i}/{total_chunks}")
                        draft_translation, stage1_warning = self.stage1_primary_translation(
                            text=chunk,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            previous_chunk=previous_chunk,
                            genre=genre,
                            terminology_context=terminology_context,
                        )

                        # Don't cache a fallback result — an untranslated chunk
                        # cached as if it were real would keep being reused on
                        # every future run instead of retrying the model.
                        if stage1_warning is None:
                            cache.cache_translation(
                                chunk, draft_translation, draft_translation,
                                source_lang, target_lang, stage1_cache_model
                            )
                        time.sleep(0.5)

                    # An exact glossary rule is not merely a suggestion when
                    # the source spelling leaked into an otherwise translated
                    # chunk. Apply the non-ambiguous correction after cache
                    # lookup too, so stale cached output cannot bypass it.
                    draft_translation, exact_replacements = self.terminology.enforce_exact_source_forms(
                        draft_translation
                    )
                    if exact_replacements:
                        logger.translation_logger.info(
                            "Stage 1 enforced %s exact glossary source-form replacement(s) in chunk %s",
                            sum(item['count'] for item in exact_replacements), i,
                        )

                    draft_translations.append(draft_translation)
                    terminology_violations = self.terminology.exact_violations(
                        chunk, draft_translation
                    )
                    stage1_violation_count += len(terminology_violations)

                    progress = (i / total_chunks) * 100
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute('''
                            UPDATE translations
                            SET progress = ?,
                                machine_translation = ?,
                                current_chunk = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        ''', (
                            progress,
                            '\n\n'.join(draft_translations),
                            i,
                            translation_id
                        ))
                    yield {
                        'progress': progress,
                        'stage': 'primary_translation',
                        'batch_index': i,
                        'original_chunk': chunk,
                        'machine_translation_chunk': draft_translation,
                        'current_chunk': i,
                        'total_chunks': total_chunks,
                        'warning': stage1_warning,
                        'terminology': {
                            'total': len(self.terminology.terms),
                            'used': len(used_terms),
                            'violations': stage1_violation_count,
                        },
                    }

                except Exception as e:
                    error_msg = f"Error in stage 1 chunk {i}: {str(e)}"
                    logger.translation_logger.error(error_msg)
                    logger.translation_logger.error(traceback.format_exc())
                    raise Exception(error_msg)

            # Persist the draft so translate_stage2() can pick it up later,
            # independently of this request/generator.
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations
                    SET status = 'stage1_completed',
                        progress = 100,
                        original_chunks = ?,
                        draft_chunks = ?,
                        chunk_chapter_map = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (
                    json.dumps(chunks, ensure_ascii=False),
                    json.dumps(draft_translations, ensure_ascii=False),
                    json.dumps(chunk_chapter_map),
                    translation_id,
                ))

            success = True
            yield {
                'progress': 100,
                'status': 'stage1_completed',
                'terminology': {
                    'total': len(self.terminology.terms),
                    'used': len(used_terms),
                    'violations': stage1_violation_count,
                },
            }

        except GeneratorExit:
            # The client stopped reading the stream — a closed tab, a dropped
            # connection. The draft is incomplete and no longer being written,
            # so the row must not stay 'in_progress' claiming otherwise.
            self._abandon_run(
                translation_id, 'error',
                'Interrupted before the draft was finished — press Start again.',
            )
            raise
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            logger.translation_logger.error(error_msg)
            logger.translation_logger.error(traceback.format_exc())

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations
                    SET status = 'error',
                        error_message = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (str(e), translation_id))
            raise
        finally:
            release_run(translation_id)
            translation_time = time.time() - start_time
            monitor.record_translation_attempt(success, translation_time)

    @staticmethod
    def _abandon_run(translation_id: int, status: str, message: Optional[str] = None):
        """Take a row out of 'in_progress' after the stream was cut.

        Only ever touches a row that still says 'in_progress': if the stage had
        already written its own final status, that status is the truth.
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    '''UPDATE translations
                       SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                       WHERE id = ? AND status = 'in_progress' ''',
                    (status, message, translation_id),
                )
        except Exception as e:  # A failure here must not mask the interruption.
            logger.translation_logger.error(
                "Could not reset the status of interrupted translation %s: %s",
                translation_id, e,
            )
        logger.translation_logger.warning(
            "Translation %s was interrupted by the client — status reset to '%s' "
            "so it can be run again", translation_id, status,
        )

    def translate_stage2(
        self,
        translation_id: int,
        source_lang: str,
        target_lang: str,
        genre: str = 'unknown',
        terminology: Optional[TerminologyManager] = None,
    ):
        """STAGE 2 only: reflection/refinement over an already-drafted
        translation, loaded from the DB row saved by translate_stage1()."""
        start_time = time.time()
        success = False

        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    '''SELECT original_chunks, draft_chunks, chunk_chapter_map,
                              source_format
                       FROM translations WHERE id = ?''',
                    (translation_id,)
                ).fetchone()

            if not row or not row['draft_chunks']:
                raise ValueError('No draft translation found to refine — run Start first.')

            chunks = json.loads(row['original_chunks'])
            draft_translations = json.loads(row['draft_chunks'])
            chunk_chapter_map = json.loads(row['chunk_chapter_map']) if row['chunk_chapter_map'] else [0] * len(chunks)
            is_epub = row['source_format'] == 'epub'

            total_chunks = len(chunks)
            final_translations = []
            self.terminology = terminology or TerminologyManager()
            glossary_fingerprint = self.terminology.fingerprint()
            used_terms = set()
            final_violation_count = 0
            errors_found = errors_applied = patches_rejected = 0
            # "Nothing changed" has three different causes — a clean draft, a
            # vetoed patch, a review call that never answered — and they used
            # to look identical from outside. Counted separately, and reported
            # per chunk in the log and per stream update to the UI, because
            # every future decision about this pass depends on knowing which
            # of the three is happening.
            chunks_reviewed = chunks_changed = review_failures = 0

            logger.translation_logger.info(
                "Starting stage 2 for translation %s with %s chunks (genre: %s, "
                "reviewer: %s, verifier: %s)",
                translation_id, total_chunks, genre, self.model_name, self.verifier_model,
            )
            if self.verifier_model == self.model_name:
                logger.translation_logger.warning(
                    "Stage 2 verifier is the reviewing model (%s) — it will be "
                    "grading its own edits, and its A/B verdict tends to follow "
                    "the order the versions are shown in rather than their "
                    "content, which rejects most patches. Pick a larger Verifier "
                    "model in Settings.",
                    self.model_name,
                )

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations
                    SET status = 'in_progress', genre = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (genre, translation_id))
            claim_run(translation_id)

            # STAGE 2: Reflection and improvement
            logger.translation_logger.info("Stage 2: Reflection and improvement")
            for i, (original_chunk, draft_chunk) in enumerate(zip(chunks, draft_translations), 1):
                try:
                    if not original_chunk.strip():
                        final_translations.append('')
                        progress = (i / total_chunks) * 100
                        yield {
                            'progress': progress,
                            'stage': 'reflection_improvement',
                            'batch_index': i,
                            'refined_translation_chunk': '',
                            'current_chunk': i,
                            'total_chunks': total_chunks,
                            'terminology': {
                                'total': len(self.terminology.terms),
                                'used': len(used_terms),
                                'violations': final_violation_count,
                            },
                        }
                        continue

                    terminology_context = self.terminology.prompt_context(original_chunk)
                    # Match Stage 1's progress contract: "used" means an
                    # approved term is relevant to a source chunk processed
                    # by this stage, not that the model happened to emit a
                    # particular surface form.  Without this update every
                    # refinement stream incorrectly remained at zero.
                    relevant_terms = self.terminology.relevant_terms(original_chunk)
                    used_terms.update(term.source.casefold() for term in relevant_terms)
                    draft_violations = self.terminology.exact_violations(
                        original_chunk, draft_chunk
                    )
                    stage2_cache_model = (
                        f"{self.model_name}_stage2{STAGE2_PIPELINE_VERSION}"
                        f"_glossary_{glossary_fingerprint}"
                    )
                    # Check cache
                    cached_result = cache.get_cached_translation(
                        original_chunk, source_lang, target_lang, stage2_cache_model
                    )
                    stage2_warning = None
                    stage2_details = {}
                    if cached_result:
                        final_translation = cached_result['translated_text']
                        logger.translation_logger.info(f"Cache hit for stage 2 chunk {i}")
                    else:
                        logger.translation_logger.info(f"Stage 2 reviewing chunk {i}/{total_chunks}")
                        final_translation, stage2_warning, stage2_details = self.stage2_reflection_improvement(
                            original_text=original_chunk,
                            draft_translation=draft_chunk,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            genre=genre,
                            terminology_context=terminology_context,
                            terminology_violations=draft_violations,
                        )
                        errors_found += stage2_details.get('errors_found', 0)
                        errors_applied += stage2_details.get('errors_applied', 0)
                        verified = stage2_details.get('verified')
                        if isinstance(verified, dict) and not verified.get('accepted'):
                            patches_rejected += 1
                        chunks_reviewed += 1
                        if stage2_warning:
                            review_failures += 1
                        if final_translation != draft_chunk:
                            chunks_changed += 1
                        logger.translation_logger.info(
                            "Stage 2 chunk %s/%s: %s",
                            i, total_chunks, self._describe_stage2_chunk(
                                stage2_details, stage2_warning,
                                changed=final_translation != draft_chunk,
                            ),
                        )

                        # Don't cache a fallback result — a draft cached as if
                        # it were a real refinement would keep being reused on
                        # every future run instead of retrying the model.
                        if stage2_warning is None:
                            cache.cache_translation(
                                original_chunk, final_translation, draft_chunk,
                                source_lang, target_lang, stage2_cache_model
                            )
                        time.sleep(0.5)

                    # Preserve the exact-term guarantee when a refinement
                    # result comes from cache or the review model lets a
                    # literal source form through.
                    final_translation, exact_replacements = self.terminology.enforce_exact_source_forms(
                        final_translation
                    )
                    if exact_replacements:
                        logger.translation_logger.info(
                            "Stage 2 enforced %s exact glossary source-form replacement(s) in chunk %s",
                            sum(item['count'] for item in exact_replacements), i,
                        )

                    final_translations.append(final_translation)
                    terminology_violations = self.terminology.exact_violations(
                        original_chunk, final_translation
                    )
                    final_violation_count += len(terminology_violations)

                    progress = (i / total_chunks) * 100
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute('''
                            UPDATE translations
                            SET progress = ?,
                                translated_text = ?,
                                final_chunks = ?,
                                current_chunk = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        ''', (
                            progress,
                            '\n\n'.join(final_translations),
                            json.dumps(final_translations, ensure_ascii=False),
                            i,
                            translation_id
                        ))

                    yield {
                        'progress': progress,
                        'stage': 'reflection_improvement',
                        'batch_index': i,
                        'refined_translation_chunk': final_translation,
                        'current_chunk': i,
                        'total_chunks': total_chunks,
                        'warning': stage2_warning,
                        'terminology': {
                            'total': len(self.terminology.terms),
                            'used': len(used_terms),
                            'violations': final_violation_count,
                        },
                        'refinement': {
                            'errors_found': errors_found,
                            'errors_applied': errors_applied,
                            'patches_rejected': patches_rejected,
                            'chunks_reviewed': chunks_reviewed,
                            'chunks_changed': chunks_changed,
                            'review_failures': review_failures,
                            'verifier_model': self.verifier_model,
                            'review_model': self.model_name,
                        },
                    }

                except Exception as e:
                    error_msg = f"Error in stage 2 chunk {i}: {str(e)}"
                    logger.translation_logger.error(error_msg)
                    logger.translation_logger.error(traceback.format_exc())
                    # Fallback to draft
                    final_translations.append(draft_chunk)
                    raise Exception(error_msg)

            # Regroup the flat, chapter-tagged chunk list back into per-chapter
            # text so an EPUB source can be rebuilt with its original chapter breaks.
            if is_epub:
                final_chapters = []
                buffer = []
                current_chapter = chunk_chapter_map[0] if chunk_chapter_map else 0
                for chunk_index, chapter_index in enumerate(chunk_chapter_map):
                    if chapter_index != current_chapter:
                        final_chapters.append('\n\n'.join(buffer))
                        buffer = []
                        current_chapter = chapter_index
                    buffer.append(final_translations[chunk_index])
                final_chapters.append('\n\n'.join(buffer))
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute(
                        'UPDATE translations SET translated_chapters = ? WHERE id = ?',
                        (json.dumps(final_chapters, ensure_ascii=False), translation_id),
                    )

            # Mark translation as completed. final_chunks is rewritten here as
            # well as per chunk, because an empty trailing chunk (a title page
            # in an EPUB) skips the per-chunk write and would otherwise leave
            # the stored list one entry short of the final text.
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations
                    SET status = 'completed',
                        progress = 100,
                        final_chunks = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (json.dumps(final_translations, ensure_ascii=False), translation_id))

            success = True
            yield {
                'progress': 100,
                'status': 'completed',
                'terminology': {
                    'total': len(self.terminology.terms),
                    'used': len(used_terms),
                    'violations': final_violation_count,
                },
                'refinement': {
                    'errors_found': errors_found,
                    'errors_applied': errors_applied,
                    'patches_rejected': patches_rejected,
                    'chunks_reviewed': chunks_reviewed,
                    'chunks_changed': chunks_changed,
                    'review_failures': review_failures,
                    'verifier_model': self.verifier_model,
                    'review_model': self.model_name,
                },
            }
            logger.translation_logger.info(
                "Stage 2 finished for translation %s: %s of %s reviewed chunk(s) "
                "changed, %s error(s) found, %s patched, %s patch(es) vetoed by "
                "verifier %s, %s review call(s) gave no answer",
                translation_id, chunks_changed, chunks_reviewed, errors_found,
                errors_applied, patches_rejected, self.verifier_model, review_failures,
            )

        except GeneratorExit:
            # Unlike Stage 1, an interrupted refinement leaves a perfectly good
            # draft behind: back to 'stage1_completed', which is Continue
            # enabled and the chunks already refined kept in the cache.
            self._abandon_run(translation_id, 'stage1_completed')
            raise
        except Exception as e:
            error_msg = f"Refinement failed: {str(e)}"
            logger.translation_logger.error(error_msg)
            logger.translation_logger.error(traceback.format_exc())

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations
                    SET status = 'error',
                        error_message = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (str(e), translation_id))
            raise
        finally:
            release_run(translation_id)
            translation_time = time.time() - start_time
            monitor.record_translation_attempt(success, translation_time)

    # ------------------------------------------------------------------
    # Ollama request/response plumbing.
    #
    # Reasoning ("thinking") is never wanted here: for translation it only
    # burns tokens, and the chain-of-thought leaks straight into the book
    # text. Every single request must therefore go through _ollama_payload,
    # and every response through _strip_reasoning:
    #
    #   * "think": false is Ollama's official switch (docs.ollama.com/
    #     capabilities/thinking) and must be sent explicitly: since Ollama
    #     0.12 a thinking-capable model turns reasoning ON by default when
    #     the field is omitted. Ollama accepts the field for non-thinking
    #     models too, so it is safe to send unconditionally.
    #   * gpt-oss is the documented exception: it ignores the boolean and
    #     its reasoning cannot be switched off at all — only its length,
    #     via the "low"/"medium"/"high" levels. So it gets "low".
    #   * Some reasoning models (deepseek-r1 among them) ignore the switch
    #     and still emit <think>…</think> inline in "response", so the text
    #     is stripped as well.
    #
    # There is no server-side, Modelfile or env-var way to turn reasoning
    # off globally — it is per-request only, hence the single choke point.
    # ------------------------------------------------------------------

    _REASONING_BLOCK_RE = re.compile(
        r'<\s*(think|thinking|reason|reasoning|thought)\s*>.*?<\s*/\s*\1\s*>',
        re.DOTALL | re.IGNORECASE,
    )
    # An unterminated opening tag means the model ran out of tokens mid-
    # thought: everything after it is reasoning, not translation.
    _REASONING_OPEN_RE = re.compile(
        r'<\s*(think|thinking|reason|reasoning|thought)\s*>.*\Z',
        re.DOTALL | re.IGNORECASE,
    )
    # A stray closing tag with no opener (the template swallowed the opener)
    # means everything before it is reasoning.
    _REASONING_CLOSE_RE = re.compile(
        r'\A.*?<\s*/\s*(think|thinking|reason|reasoning|thought)\s*>',
        re.DOTALL | re.IGNORECASE,
    )

    @classmethod
    def _strip_reasoning(cls, text: str) -> str:
        """Remove any reasoning the model emitted inline despite think=false."""
        if not text:
            return text
        cleaned = cls._REASONING_BLOCK_RE.sub('', text)
        cleaned = cls._REASONING_CLOSE_RE.sub('', cleaned)
        stripped_tail = cls._REASONING_OPEN_RE.sub('', cleaned)
        # Only drop the unterminated tail if something is left; otherwise the
        # whole answer was inside the block and the raw text is the best guess.
        if stripped_tail.strip():
            cleaned = stripped_tail
        return cleaned.strip()

    def _think_setting(self):
        """The value of Ollama's "think" field for the active model: always
        the least reasoning the model can be asked for."""
        if 'gpt-oss' in self.model_name.lower():
            # Booleans are ignored by gpt-oss and it always reasons; "low" is
            # the shortest trace available.
            return 'low'
        return False

    def _ollama_payload(
        self,
        prompt: str,
        temperature: float,
        num_ctx: int = 8192,
    ) -> Dict:
        """Build an /api/generate body. The single place where Ollama request
        options are set, so reasoning stays off for every model everywhere."""
        options = {'temperature': temperature, 'num_ctx': num_ctx}
        if is_translategemma(self.model_name):
            # The sampling settings TranslateGemma ships in its own Modelfile.
            # Ollama already applies them, but setting them here keeps decoding
            # identical no matter which tag or quantization was pulled.
            options.update({'top_k': 64, 'top_p': 0.95, 'stop': ['<end_of_turn>']})
        return {
            'model': self.model_name,
            'prompt': prompt,
            'stream': False,
            'think': self._think_setting(),
            'options': options,
        }

    @classmethod
    def _ollama_response_text(cls, result: Dict) -> Optional[str]:
        """Pull the answer out of an /api/generate result, dropping the
        separate 'thinking' field and any inline reasoning."""
        text = cls._strip_reasoning(result.get('response') or '')
        return text or None

    @staticmethod
    def _stage1_prompt_default(
        text: str,
        source_name: str,
        target_name: str,
        previous_chunk: str,
        genre: str,
        terminology_context: str,
    ) -> str:
        """The Stage 1 prompt for general instruct models."""
        context_section = f"\n\nPrevious translated paragraph:\n{previous_chunk}" if previous_chunk else ""

        return f"""You are a professional translator. Translate from {source_name} to {target_name}.

CONTEXT:
- Document type: {genre}
- Preserve formatting (paragraphs, line breaks)
- Adapt idioms and cultural references for target audience
- Maintain tone and emotional coloring of original
{context_section}
{terminology_context}

TEXT TO TRANSLATE:
{text}

Return ONLY the translation without comments."""

    @staticmethod
    def _stage1_prompt_translategemma(
        text: str,
        source_lang: str,
        target_lang: str,
        source_name: str,
        target_name: str,
        previous_chunk: str,
        genre: str,
        terminology_context: str,
    ) -> str:
        """The Stage 1 prompt for TranslateGemma, in the shape the model card
        documents: the persona sentence, then the instruction sentence, then
        exactly two blank lines, then the text — and nothing after it.

        Whatever else this pipeline knows about the chunk (genre, the previous
        translated paragraph, verified terminology) is
        slotted *between* the instruction sentences, so the opening and the
        tail the model was trained on stay untouched. With nothing extra to
        add, the result is byte-identical to the documented prompt.
        """
        opening = (
            f"You are a professional {source_name} ({source_lang}) to "
            f"{target_name} ({target_lang}) translator. Your goal is to accurately "
            f"convey the meaning and nuances of the original {source_name} text "
            f"while adhering to {target_name} grammar, vocabulary, and cultural "
            "sensitivities."
        )
        produce_only = (
            f"Produce only the {target_name} translation, without any additional "
            "explanations or commentary."
        )
        instruction = (
            f"Please translate the following {source_name} text into {target_name}:"
        )

        extras = []
        if previous_chunk:
            extras.append(
                "Previous translated paragraph (context only — do not repeat it "
                f"in your answer):\n{previous_chunk}"
            )
        terminology_context = terminology_context.strip()
        if terminology_context:
            extras.append(terminology_context)

        if extras or (genre and genre != 'unknown'):
            context_lines = [
                "- Preserve formatting (paragraphs, line breaks)",
                "- Adapt idioms and cultural references for target audience",
                "- Maintain tone and emotional coloring of original",
            ]
            if genre and genre != 'unknown':
                context_lines.insert(0, f"- Document type: {genre}")
            extras.insert(0, "CONTEXT:\n" + "\n".join(context_lines))

        if not extras:
            return f"{opening}\n{produce_only} {instruction}\n\n\n{text}"

        middle = "\n\n".join(extras)
        return f"{opening}\n{produce_only}\n\n{middle}\n\n{instruction}\n\n\n{text}"

    def stage1_primary_translation(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        previous_chunk: str = "",
        genre: str = "unknown",
        terminology_context: str = "",
    ) -> Tuple[str, Optional[str]]:
        """
        STAGE 1: Primary translation with maximum context.
        LLM translates with understanding of topic, audience, and style.

        Returns (text, warning) — warning is set (and text falls back to the
        original, untranslated chunk) whenever Ollama didn't actually produce
        a translation, so the caller can surface this instead of silently
        passing off unrefined/untranslated text as a real result.
        """
        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)

        if is_translategemma(self.model_name):
            prompt = self._stage1_prompt_translategemma(
                text, source_lang, target_lang, source_name, target_name,
                previous_chunk, genre, terminology_context,
            )
            temperature = TRANSLATEGEMMA_TEMPERATURE
        else:
            prompt = self._stage1_prompt_default(
                text, source_name, target_name,
                previous_chunk, genre, terminology_context,
            )
            temperature = 0.6

        logger.api_logger.debug(f"Stage 1 prompt ({self.model_name}):\n{prompt}")
        payload = self._ollama_payload(prompt, temperature=temperature)

        try:
            # Do this immediately before the request, not in __init__:
            # /models creates a short-lived helper with the default model and
            # must never overwrite the model a translation is actually using.
            monitor.set_active_model(self.model_name)
            response = self.session.post(self.api_url, json=payload, timeout=(30, 300))  # 5 min timeout
            response.raise_for_status()
            result = json.loads(response.text)

            translated = self._ollama_response_text(result)
            if translated:
                return translated, None
            logger.api_logger.warning("No response field in Stage 1 result")
            return text, 'Model returned no output — kept the original text untranslated for this chunk.'
        except requests.exceptions.Timeout:
            logger.api_logger.error(f"Stage 1 timeout after 300s - text too long or model too slow")
            return text, 'Model timed out after 5 minutes (likely still loading/swapping) — kept the original text untranslated for this chunk.'
        except Exception as e:
            logger.api_logger.error(f"Stage 1 error: {e}")
            return text, f'Model request failed ({e}) — kept the original text untranslated for this chunk.'

    # ------------------------------------------------------------------
    # Stage 2: refinement, as estimate -> patch -> verify.
    #
    # This used to be a single prompt that asked the model to "review and
    # improve" the draft against five criteria and return the improved text.
    # That shape has a structural problem: rewriting the whole chunk is a
    # free hand, and a model given a free hand on a literary text spends it
    # on fluency. The draft/final diff came out around 20% while adequacy
    # went *down* and fluency went up — the pass was trading meaning for
    # polish, chunk after chunk, with nothing able to stop it.
    #
    # So the model no longer writes the final text. It only says what is
    # wrong (estimate), the fix is applied to those spans and nothing else
    # by Python (patch), and the result has to beat the draft on accuracy
    # before it is kept (verify). Anything the model returns that cannot be
    # located in the draft verbatim is discarded rather than guessed at.
    # ------------------------------------------------------------------

    # MQM-style error categories. The estimate pass is asked for these, and
    # the aliases catch the near-misses local models produce instead.
    ERROR_TYPES = {
        'mistranslation', 'omission', 'addition',
        'terminology', 'consistency', 'grammar', 'style', 'other',
    }
    ERROR_TYPE_ALIASES = {
        'accuracy': 'mistranslation', 'mistranslated': 'mistranslation',
        'missing': 'omission', 'omitted': 'omission', 'untranslated': 'mistranslation',
        'added': 'addition', 'hallucination': 'addition',
        'term': 'terminology', 'glossary': 'terminology',
        'inconsistency': 'consistency', 'inconsistent': 'consistency',
        'fluency': 'grammar', 'syntax': 'grammar', 'punctuation': 'grammar',
        'register': 'style', 'tone': 'style',
    }
    SEVERITIES = ('critical', 'major', 'minor')
    # Categories a glossary or a proper-noun record settles on its own: if
    # the required rendering is absent, that is a fact, not an opinion. These
    # are also the categories whose severity does not matter — a missing
    # required rendering is worth fixing at any label the reviewer put on it.
    OBJECTIVE_ERROR_TYPES = {'terminology', 'consistency'}
    # What the verifier is not asked about. Whether a sentence of the source
    # is missing from the translation, or a clause appears that the source
    # never said, is settled by reading the two texts — the A/B "which reads
    # more faithfully" vote adds nothing but a chance to veto a real fix. The
    # judge is kept for mistranslation and grammar, where the reported error
    # is a claim about meaning and an opinion is what is actually needed.
    JUDGE_EXEMPT_ERROR_TYPES = OBJECTIVE_ERROR_TYPES | {'omission', 'addition'}
    # Categories worth editing the text for. 'style' is not among them: this
    # pass exists because rewriting for style is what cost the pipeline its
    # accuracy in the first place, and an edit made on taste has no way to be
    # verified. Observed on a real run — the review pass replaced a perfectly
    # good "испарилось" with "полностью покинуло его" and the verifier waved
    # it through, because on that axis there is nothing to be wrong about.
    ACTIONABLE_ERROR_TYPES = {
        'mistranslation', 'omission', 'addition', 'terminology', 'consistency', 'grammar',
    }
    # And below major severity, only the objectively checkable categories are
    # worth touching the text for. A "minor mistranslation" on a local judge
    # is mostly the judge preferring a synonym.
    ACTIONABLE_SEVERITIES = {'critical', 'major'}
    MAX_ESTIMATE_SPANS = 12
    # The verifier is meant to be a larger model than the reviewer, and it is
    # shown the source plus two full versions of the chunk. The shared 180s is
    # a timeout for per-chunk translation, not for that; a verifier that runs
    # out of patience silently keeps the draft, which is exactly the failure
    # this pass is hardest to notice.
    VERIFY_READ_TIMEOUT = 600

    @classmethod
    def is_actionable_error(cls, error: Dict) -> bool:
        """Whether a reported error justifies changing the text.

        Everything else is still reported to the caller — the count of what
        was found versus what was applied is worth seeing — it just does not
        get to edit the book.
        """
        if error['type'] not in cls.ACTIONABLE_ERROR_TYPES:
            return False
        if error['type'] in cls.OBJECTIVE_ERROR_TYPES:
            return True
        return error['severity'] in cls.ACTIONABLE_SEVERITIES

    def _estimate_prompt(
        self, original_text: str, draft_translation: str,
        source_name: str, target_name: str,
        terminology_context: str, violation_section: str,
    ) -> str:
        return f"""You are a translation quality reviewer. You do not rewrite translations — you report errors in them.

SOURCE ({source_name}):
{original_text}

TRANSLATION TO REVIEW ({target_name}):
{draft_translation}
{terminology_context}
{violation_section}

Find places where the {target_name} translation is WRONG about the source. Report only real errors:
- mistranslation — the {target_name} says something the source does not
- omission — something in the source is missing from the translation
- addition — the translation invents something not in the source
- terminology — a required rendering from the list above was not used
- consistency — a name or term is rendered differently here than the required form
- grammar — ungrammatical or broken {target_name}
- style — register or tone clearly wrong for the source

Do NOT report anything that is merely a matter of taste: a synonym you prefer, a smoother rhythm, a more literary word choice. If the translation is accurate, return an empty list.

Respond with ONLY a JSON array, no prose, no code fence. Each element:
{{"span": "<the exact substring of the {target_name} translation that is wrong, copied character for character>", "type": "<one of the categories above>", "severity": "critical|major|minor", "replacement": "<what that span should say instead>"}}

Rules:
- "span" MUST appear in the {target_name} translation above exactly as you write it. Copy it, do not paraphrase or re-type it from memory.
- Keep spans short — a few words, not whole paragraphs.
- "replacement" fixes only that span and must fit grammatically where the span sat.
- For an omission, let "span" be the words the missing content belongs next to, and "replacement" those same words with the content restored.
- At most {self.MAX_ESTIMATE_SPANS} elements. If there is nothing wrong, respond with []."""

    @classmethod
    def validate_estimate_spans(cls, items: List[Dict], draft_translation: str) -> List[Dict]:
        """Keep only the reported errors that can actually be acted on.

        A span is usable only if it occurs in the draft verbatim: everything
        downstream is a literal substring replacement, so a span the model
        re-typed from memory, translated back, or invented has no position to
        patch and is dropped rather than fuzzy-matched. On a local quantised
        model this discards a fair share of the answer, which is the point —
        a dropped error leaves the draft alone, and leaving the draft alone
        is the safe outcome.
        """
        validated, seen = [], set()
        for item in items:
            span = item.get('span')
            replacement = item.get('replacement')
            if not isinstance(span, str) or not isinstance(replacement, str):
                continue
            span, replacement = span.strip(), replacement.strip()
            if len(span) < 2 or not replacement or span == replacement:
                continue
            if span not in draft_translation or span in seen:
                continue

            error_type = str(item.get('type') or '').strip().lower()
            error_type = cls.ERROR_TYPE_ALIASES.get(error_type, error_type)
            if error_type not in cls.ERROR_TYPES:
                error_type = 'other'
            severity = str(item.get('severity') or '').strip().lower()
            if severity not in cls.SEVERITIES:
                severity = 'minor'

            seen.add(span)
            validated.append({
                'span': span,
                'replacement': replacement,
                'type': error_type,
                'severity': severity,
            })

        # Worst first, so that if the cap bites it bites the trivia.
        validated.sort(key=lambda error: cls.SEVERITIES.index(error['severity']))
        return validated[:cls.MAX_ESTIMATE_SPANS]

    def stage2_estimate(
        self,
        original_text: str,
        draft_translation: str,
        source_lang: str,
        target_lang: str,
        terminology_context: str = "",
        terminology_violations: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[List[Dict], Optional[str]]:
        """STAGE 2a: what is wrong with this draft, as located spans.

        Returns (errors, warning). A warning means the model produced no
        answer at all; an empty list with no warning means it answered that
        the draft is fine, which is a legitimate result and not a failure.
        """
        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)

        violation_section = ""
        if terminology_violations:
            missing = ", ".join(
                f'"{item["source"]}" => "{item["required_target"]}"'
                for item in terminology_violations
            )
            violation_section = (
                "\n\nThese required renderings are missing from the translation "
                f"and must be reported as terminology errors: {missing}"
            )

        prompt = self._estimate_prompt(
            original_text, draft_translation, source_name, target_name,
            terminology_context, violation_section,
        )
        raw = self._call_model(prompt, temperature=0.2)
        if raw is None:
            return [], 'The review pass returned no output — kept the draft for this chunk.'
        return self.validate_estimate_spans(self._parse_json_array(raw), draft_translation), None

    @staticmethod
    def stage2_patch(draft_translation: str, errors: List[Dict]) -> Tuple[str, List[Dict]]:
        """STAGE 2b: apply the reported fixes, and only those.

        Pure string surgery, no model involved: each span is replaced at its
        first free occurrence and every other character of the draft is
        carried over untouched. Overlapping spans are resolved by dropping
        the later one — two edits fighting over the same characters would
        corrupt the text, and the estimate pass is cheap to re-run.

        Returns (patched_text, applied_errors).
        """
        edits, applied = [], []
        claimed: List[Tuple[int, int]] = []
        for error in errors:
            span = error['span']
            search_from = 0
            while True:
                start = draft_translation.find(span, search_from)
                if start == -1:
                    break
                end = start + len(span)
                if any(start < claimed_end and claimed_start < end for claimed_start, claimed_end in claimed):
                    search_from = start + 1
                    continue
                claimed.append((start, end))
                edits.append((start, end, error['replacement']))
                applied.append(error)
                break

        if not edits:
            return draft_translation, []

        edits.sort()
        pieces, cursor = [], 0
        for start, end, replacement in edits:
            pieces.append(draft_translation[cursor:start])
            pieces.append(replacement)
            cursor = end
        pieces.append(draft_translation[cursor:])
        return ''.join(pieces), applied

    def stage2_verify(
        self, original_text: str, before: str, after: str,
        source_lang: str, target_lang: str,
    ) -> Tuple[bool, Dict]:
        """STAGE 2c: did the patch actually improve the translation?

        The judge sees the source and is asked about accuracy — not about
        which version reads better, which is the question that let the old
        pass congratulate itself while drifting from the original. Asked
        twice with the two versions swapped, because a single ordering
        measures position bias as much as quality; the patch is kept only if
        it wins both times.

        Runs on ``self.verifier``, which is a separate model whenever one was
        chosen. Both halves of the vote on the model that produced the edit is
        self-assessment, and on a quantised 12B it answers by position rather
        than by content, so the two orderings disagree and every patch is
        vetoed. The strict rule stays: it is the only thing standing between
        this pass and its old habit of trading meaning for polish. What
        changes is that a model big enough to be consistent is the one
        applying it.
        """
        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)
        verifier = self.verifier
        verdicts = []

        for patched_is_a in (True, False):
            version_a, version_b = (after, before) if patched_is_a else (before, after)
            prompt = f"""You are comparing two {target_name} translations of the same {source_name} source. You wrote neither of them.

SOURCE ({source_name}):
{original_text}

VERSION A:
{version_a}

VERSION B:
{version_b}

Which version conveys the source more faithfully — no meaning changed, nothing left out, nothing invented? Ignore which one sounds more elegant; accuracy is the only question.

Respond with EXACTLY one word: A, B, or TIE."""
            raw = verifier._call_model(
                prompt, temperature=0.0, read_timeout=self.VERIFY_READ_TIMEOUT,
            )
            if raw is None:
                # Distinct from a tie: the model never answered. Recorded
                # under its own name so a run of these reads as "the verifier
                # is too slow for this chunk size" rather than as the verifier
                # disagreeing with the reviewer.
                verdicts.append('unavailable')
                continue
            verdict = raw.strip().upper()
            if verdict.startswith('A'):
                verdicts.append('patched' if patched_is_a else 'draft')
            elif verdict.startswith('B'):
                verdicts.append('draft' if patched_is_a else 'patched')
            else:
                verdicts.append('tie')

        accepted = verdicts == ['patched', 'patched']
        return accepted, {
            'verdicts': verdicts,
            'accepted': accepted,
            'model': verifier.model_name,
        }

    @staticmethod
    def _describe_stage2_chunk(
        details: Dict, warning: Optional[str], changed: bool,
    ) -> str:
        """One line saying what this pass did to one chunk, for the log.

        Written out in full — how many errors, how many were actionable, how
        many were patched, what the verifier said and which model said it —
        because "0 applied" on its own cannot distinguish a clean draft from a
        vetoed fix from a review call that timed out, and those three want
        three different responses from whoever is reading the log.
        """
        parts = [
            f"{details.get('errors_found', 0)} found",
            f"{details.get('errors_actionable', 0)} actionable",
            f"{details.get('errors_applied', 0)} patched",
        ]
        verified = details.get('verified')
        if warning:
            parts.append(f'review pass gave no answer ({warning})')
        elif verified == 'skipped_objective':
            parts.append('verifier skipped — every fix checkable against the source')
        elif isinstance(verified, dict):
            parts.append('verifier {} voted {} → {}'.format(
                verified.get('model') or 'unknown',
                '/'.join(verified.get('verdicts') or ['no verdict']),
                'accepted' if verified.get('accepted') else 'rejected',
            ))
        else:
            parts.append('verifier not needed')
        parts.append('text changed' if changed else 'draft kept')
        return ' · '.join(parts)

    def stage2_reflection_improvement(
        self,
        original_text: str,
        draft_translation: str,
        source_lang: str,
        target_lang: str,
        genre: str = "unknown",
        terminology_context: str = "",
        terminology_violations: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, Optional[str], Dict]:
        """STAGE 2: estimate, patch, verify — the whole refinement of one
        chunk.

        Returns (text, warning, details). The text is either the patched
        draft or the draft itself; it is never freshly generated prose, so a
        chunk can only ever change in the places an error was reported for.
        `details` is what the UI shows about the pass: how many errors were
        found, how many survived validation, whether the verifier kept them.
        """
        errors, warning = self.stage2_estimate(
            original_text=original_text,
            draft_translation=draft_translation,
            source_lang=source_lang,
            target_lang=target_lang,
            terminology_context=terminology_context,
            terminology_violations=terminology_violations,
        )
        actionable = [error for error in errors if self.is_actionable_error(error)]
        details: Dict = {
            'errors_found': len(errors),
            'errors_actionable': len(actionable),
            'errors_applied': 0,
            'verified': None,
            'by_severity': dict(Counter(error['severity'] for error in errors)),
            'by_type': dict(Counter(error['type'] for error in errors)),
            'review_model': self.model_name,
            'verifier_model': self.verifier_model,
        }
        if warning or not actionable:
            return draft_translation, warning, details

        patched, applied = self.stage2_patch(draft_translation, actionable)
        details['errors_applied'] = len(applied)
        if not applied or patched == draft_translation:
            return draft_translation, None, details

        # A patch made only of fixes that can be checked against the source
        # skips the judge: a required rendering is either present or not, and
        # a sentence of the source is either translated or missing. Asking a
        # "which reads more faithfully" vote about those buys nothing and can
        # only veto a fix the pipeline is already confident about.
        if all(error['type'] in self.JUDGE_EXEMPT_ERROR_TYPES for error in applied):
            details['verified'] = 'skipped_objective'
            return patched, None, details

        accepted, verdict = self.stage2_verify(
            original_text, draft_translation, patched, source_lang, target_lang,
        )
        details['verified'] = verdict
        if not accepted:
            return draft_translation, None, details
        return patched, None, details

    # ------------------------------------------------------------------
    # Stage 3: standalone quality tests, run on demand from the UI after
    # Stage 1 and/or Stage 2 — never automatically. Several of these call
    # the model again (backtranslation, LLM-judge) or a QE model
    # (COMET-Kiwi), which is too slow to run on every chunk of a full
    # book, so they sample a handful of chunks instead of the whole text.
    # ------------------------------------------------------------------

    EVAL_SAMPLE_SIZE = 5

    @staticmethod
    def _sample_indices(length: int, sample_size: int) -> List[int]:
        """Evenly spaced indices across [0, length), for sampling chunks
        without re-processing an entire book on every test run."""
        if length <= 0:
            return []
        k = min(sample_size, length)
        if k <= 1:
            return [0]
        return sorted({round(i * (length - 1) / (k - 1)) for i in range(k)})

    # Dialogue openers across the conventions this app's language list uses.
    _DIALOGUE_RE = re.compile(r'["“”«»„]|(?:^|\n)\s*[—–-]\s')

    @classmethod
    def _risk_ranked_indices(cls, chunks: List[str], sample_size: int) -> List[int]:
        """Indices of the riskiest chunks, rather than evenly spaced ones.

        Evenly spaced sampling weights every chunk equally, but translation
        errors are not evenly spaced: they cluster where there are names to
        render consistently, dialogue to keep in register, and numbers to
        carry over exactly. Spending the same five model calls on those
        chunks finds strictly more than spending them on scenery.

        Falls back to even spacing when nothing scores — a chunk list with no
        names, no dialogue and no numbers has no risk profile to rank by.
        """
        scored = []
        for index, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            names = len(cls.harvest_proper_noun_candidates(chunk, limit=20))
            dialogue = len(cls._DIALOGUE_RE.findall(chunk))
            digits = sum(character.isdigit() for character in chunk)
            score = names * 3 + min(dialogue, 10) + min(digits, 10)
            if score:
                scored.append((score, index))

        if not scored:
            return cls._sample_indices(len(chunks), sample_size)
        # Highest score first, index as the tie-break so a rerun samples the
        # same chunks and its numbers stay comparable to the previous run's.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return sorted(index for _, index in scored[:sample_size])

    def _call_model(
        self, prompt: str, temperature: float = 0.2, read_timeout: int = 180,
    ) -> Optional[str]:
        try:
            # This path covers Prepare, refinement, and model-based quality
            # checks.  Mark only a model that is about to reach Ollama active.
            monitor.set_active_model(self.model_name)
            response = self.session.post(
                self.api_url,
                json=self._ollama_payload(prompt, temperature=temperature),
                timeout=(30, read_timeout),
            )
            response.raise_for_status()
            result = json.loads(response.text)
            return self._ollama_response_text(result)
        except Exception as e:
            logger.api_logger.error(f"Evaluation model call failed: {e}")
            return None

    # -- Stage 2 tests: is refinement actually helping, without breaking anything? --

    def eval_length_ratio(self, source_text: str, draft_text: str, final_text: str) -> Dict:
        """Final length against the SOURCE, with final-against-draft as a
        secondary number.

        Measuring the final against the draft answers a question nobody is
        asking: both were produced by the same pipeline from the same text, so
        the ratio sits near 1.00 whatever happened to the meaning. Measuring
        against the source is what shows compression — a target text that
        comes out the same length as an English source is suspicious, because
        most target languages expand.
        """
        source_len, draft_len, final_len = len(source_text), len(draft_text), len(final_text)
        ratio = (final_len / source_len) if source_len else 1.0
        draft_ratio = (final_len / draft_len) if draft_len else 1.0
        # A deliberately wide band: expansion factors are language-pair
        # specific (EN→RU runs 1.10–1.20, EN→ZH well under 1), so this can
        # only catch the gross cases — a target half the size of its source,
        # or twice it — without a per-pair table to compare against.
        flagged = not (0.55 <= ratio <= 1.6)
        return {
            'test': 'length_ratio',
            'label': 'Length ratio (final / source)',
            'value': round(ratio, 3),
            'details': {
                'source_chars': source_len,
                'draft_chars': draft_len,
                'final_chars': final_len,
                'final_over_draft': round(draft_ratio, 3),
            },
            'flagged': flagged,
            'note': (
                f"Final is {ratio:.2f}x the source's length ({source_len} → {final_len} chars) "
                f"— outside 0.55–1.6x, so text was probably lost or duplicated. "
                f"Final/draft {draft_ratio:.2f}x."
                if flagged else
                f"Final is {ratio:.2f}x the source's length ({source_len} → {final_len} chars); "
                f"final/draft {draft_ratio:.2f}x. Compare against what your language pair "
                f"normally does — a ratio near 1.00 into a language that usually expands "
                f"means the translation is compressing."
            ),
        }

    def eval_diff_ratio(self, draft_text: str, final_text: str) -> Dict:
        # autojunk MUST stay off. It treats any character occurring in more
        # than 1% of a long sequence as junk to be ignored — which in prose is
        # most of the alphabet, so it reports two nearly identical texts as
        # wildly different. On a measured example the same pair scored 0.82
        # with autojunk and 0.98 without; 0.98 was the truth.
        ratio = difflib.SequenceMatcher(None, draft_text, final_text, autojunk=False).ratio()
        # Refinement is now a span patcher, not a rewriter, so a high
        # similarity is the expected outcome and no longer a complaint. A LOW
        # one is the alarm: it means something rewrote the text wholesale.
        flagged = ratio < 0.75
        if ratio < 0.75:
            note = f"Similarity {ratio:.2f} — over a quarter of the text changed. Refinement patches reported spans, so this much movement means something rewrote the text wholesale; check for hallucination or duplication."
        elif ratio > 0.995:
            note = f"Similarity {ratio:.2f} — refinement changed almost nothing. Either the draft was already clean or the review pass found nothing it could locate; check how many errors it reported."
        else:
            note = f"Similarity {ratio:.2f} — {(1 - ratio) * 100:.1f}% of the text changed, which is the range a span-level patch should land in."
        return {
            'test': 'diff_ratio',
            'label': 'Draft/final similarity',
            'value': round(ratio, 3),
            'flagged': flagged,
            'note': note,
        }

    def eval_ngram_repetition(self, text: str, n: int = 4) -> Dict:
        words = text.split()
        if len(words) < n * 2:
            return {
                'test': 'ngram_repetition',
                'label': 'Repeated phrase ratio',
                'value': 0.0,
                'flagged': False,
                'note': 'Text too short to evaluate.',
            }
        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(count for count in counts.values() if count > 1)
        ratio = repeated / len(ngrams)
        flagged = ratio > 0.15
        return {
            'test': 'ngram_repetition',
            'label': 'Repeated phrase ratio',
            'value': round(ratio, 3),
            'flagged': flagged,
            'note': (
                f"{ratio:.0%} of {n}-word phrases repeat — likely duplicated passages."
                if flagged else
                f"{ratio:.0%} of {n}-word phrases repeat — normal for natural prose."
            ),
        }

    def eval_terminology_delta(
        self, original_text: str, draft_text: str, final_text: str,
        terminology: 'TerminologyManager',
    ) -> Dict:
        draft_violations = terminology.exact_violations(original_text, draft_text)
        final_violations = terminology.exact_violations(original_text, final_text)
        delta = len(draft_violations) - len(final_violations)
        if len(draft_violations) == 0 and len(final_violations) == 0:
            note = 'No verified terms were violated in either pass.'
        elif delta > 0:
            note = f'Refinement fixed {delta} glossary violation(s) ({len(draft_violations)} → {len(final_violations)}).'
        elif delta < 0:
            note = f'Refinement introduced {-delta} new glossary violation(s) ({len(draft_violations)} → {len(final_violations)}).'
        else:
            note = f'Refinement left {len(final_violations)} glossary violation(s) unfixed.'
        return {
            'test': 'terminology_delta',
            'label': 'Glossary violations (draft vs final)',
            'value': delta,
            'details': {
                'draft_violations': len(draft_violations),
                'final_violations': len(final_violations),
            },
            'flagged': len(final_violations) > 0,
            'note': note,
        }

    # -- Deterministic document-level checks. No model, no sampling: these
    # read the whole final text, which is what the LLM-judge tests can never
    # do, and they are the only tests here whose answer is a fact rather
    # than an opinion. --

    @staticmethod
    def _script_of(character: str) -> Optional[str]:
        """The script a letter belongs to ('LATIN', 'CYRILLIC', 'CJK', …),
        or None for anything that isn't a letter.

        Read out of the Unicode character name rather than a table of code
        point ranges, so it covers every script without this file having to
        know which languages exist."""
        if not character.isalpha():
            return None
        try:
            return unicodedata.name(character).split()[0]
        except ValueError:
            return None

    @classmethod
    def dominant_script(cls, text: str) -> Optional[str]:
        """The script most of a text's letters are written in."""
        counts = Counter(
            script for script in (cls._script_of(character) for character in text) if script
        )
        return counts.most_common(1)[0][0] if counts else None

    def eval_script_leakage(self, source_text: str, final_text: str) -> Dict:
        """Words in the final translation still written in the source's
        script.

        This is the cheapest possible check for a name that was never
        translated at all — "Grunnings" sitting in the middle of a Cyrillic
        page — and no LLM judge is needed to see it. Some leakage is
        legitimate (a brand kept in Latin on purpose), so the words are
        listed rather than just counted, and the reader decides.
        """
        target_script = self.dominant_script(final_text)
        source_script = self.dominant_script(source_text)
        if not target_script or not source_script or target_script == source_script:
            return {
                'test': 'script_leakage',
                'label': 'Untranslated source-script words',
                'value': 0,
                'flagged': False,
                'note': (
                    'Source and target use the same script, so untranslated words '
                    'cannot be told apart this way.'
                    if target_script and source_script else
                    'Not enough text to determine the scripts involved.'
                ),
            }

        leaked = Counter()
        for match in self._WORD_RE.finditer(final_text):
            word = match.group(0)
            if self.dominant_script(word) == source_script:
                leaked[word] += 1

        total = sum(leaked.values())
        examples = ', '.join(
            f'{word} ({count}x)' if count > 1 else word
            for word, count in leaked.most_common(8)
        )
        return {
            'test': 'script_leakage',
            'label': 'Untranslated source-script words',
            'value': total,
            'details': {
                'distinct_words': len(leaked),
                'target_script': target_script.title(),
                'source_script': source_script.title(),
                'words': dict(leaked.most_common(20)),
            },
            'flagged': total > 0,
            'note': (
                f'{total} {source_script.title()}-script word(s) left in the '
                f'{target_script.title()} translation, {len(leaked)} distinct: {examples}. '
                'Check each one — a name left in the source script is a missed '
                'translation, a brand kept on purpose is not.'
                if total else
                f'No {source_script.title()}-script words left in the '
                f'{target_script.title()} translation.'
            ),
        }

    # How much of a target term has to match for two surface forms to count
    # as the same name inflected, rather than two different renderings. Names
    # inflect at the end, so the comparison is on the leading characters.
    ENTITY_STEM_MIN = 4

    @classmethod
    def _entity_stem(cls, term: str) -> str:
        """The leading part of a target term that inflection leaves alone."""
        head = term.split()[-1] if term.split() else term
        if len(head) <= cls.ENTITY_STEM_MIN:
            return head.casefold()
        return head[:max(cls.ENTITY_STEM_MIN, len(head) - 2)].casefold()

    @classmethod
    def _matches_entity_rendering(cls, candidate: str, target: str) -> bool:
        """Whether one target-language word can be an inflected rendering.

        A stem alone allows case endings, but must never let a shorter word
        stand in for the agreed name: ``Фенвик`` is not an inflection of
        ``Фенвикс``. This is deliberately conservative rather than pretending
        to provide morphology for every supported target language.
        """
        head = target.split()[-1] if target.split() else target
        folded = candidate.casefold()
        return len(folded) >= len(head) and folded.startswith(cls._entity_stem(target))

    def eval_entity_consistency(
        self, original_chunks: List[str], final_chunks: List[str],
        terminology: 'TerminologyManager',
    ) -> Dict:
        """Is every agreed rendering actually used everywhere its name occurs?

        Runs per chunk over the whole document, which is the point: a chunk
        can be internally perfect and still call the family something
        different from what the previous chunk called it. Any inflected form
        counts as a use — matching is on the stem, since a name that cannot
        take target-language endings would make the sentence around it
        ungrammatical.

        Reports the chunks where a name's source appears but no form of its
        agreed rendering does. That is the signature of a name being dropped,
        translated by meaning in one place and transcribed in another, or
        left in the source script.
        """
        if not terminology.terms:
            return {
                'test': 'entity_consistency',
                'label': 'Named-entity consistency',
                'value': None,
                'flagged': True,
                'note': (
                    'No terms to check. Run Prepare before Start (or fill the glossary '
                    'by hand) — with an empty glossary this test can only ever pass, '
                    'which for a text whose main risk is proper nouns is the most '
                    'expensive check to skip.'
                ),
            }

        # Two names whose stems are prefixes of each other cannot be told
        # apart by stem matching: "Дурсль" and "Дурсли" both reduce to
        # "Дурс", so a chunk that says the singular where the source has the
        # plural still counts as satisfied. Stem matching is what makes
        # inflection acceptable, and without morphology for the target
        # language the two requirements genuinely conflict — so the pairs are
        # named as needing a human eye rather than quietly passed.
        stems = {term.source: self._entity_stem(term.target) for term in terminology.terms}
        ambiguous = sorted({
            tuple(sorted((left, right)))
            for left, left_stem in stems.items()
            for right, right_stem in stems.items()
            if left != right and (left_stem.startswith(right_stem) or right_stem.startswith(left_stem))
        })

        pairs = list(zip(original_chunks, final_chunks))
        findings, checked = [], 0
        for term in terminology.terms:
            stem = stems[term.source]
            folded_source = term.source.casefold()
            occurrences, satisfied, forms = 0, 0, Counter()
            for original_chunk, final_chunk in pairs:
                if folded_source not in original_chunk.casefold():
                    continue
                occurrences += 1
                chunk_forms = [
                    match.group(0) for match in self._WORD_RE.finditer(final_chunk)
                    if self._matches_entity_rendering(match.group(0), term.target)
                ]
                if chunk_forms:
                    satisfied += 1
                    forms.update(chunk_forms)
            if not occurrences:
                continue
            checked += 1
            if satisfied < occurrences:
                findings.append({
                    'source': term.source,
                    'target': term.target,
                    'chunks_with_source': occurrences,
                    'chunks_with_rendering': satisfied,
                    'forms_used': [form for form, _ in forms.most_common(8)],
                })

        findings.sort(key=lambda finding: finding['chunks_with_source'] - finding['chunks_with_rendering'], reverse=True)
        if not checked:
            note = 'None of the glossary terms occur in the source text, so nothing was checked.'
        elif not findings:
            note = f'All {checked} term(s) that occur in the source are rendered in every chunk they appear in.'
        else:
            worst = '; '.join(
                f'"{finding["source"]}" → "{finding["target"]}" missing from '
                f'{finding["chunks_with_source"] - finding["chunks_with_rendering"]} of '
                f'{finding["chunks_with_source"]} chunk(s)'
                for finding in findings[:4]
            )
            note = f'{len(findings)} of {checked} term(s) are not rendered everywhere: {worst}.'

        if ambiguous:
            listed = '; '.join(f'{left} / {right}' for left, right in ambiguous[:4])
            note += (
                f' Cannot distinguish these renderings automatically, so check them by '
                f'hand: {listed}. They differ only in the endings that inflection is '
                f'allowed to change.'
            )

        return {
            'test': 'entity_consistency',
            'label': 'Named-entity consistency',
            'value': len(findings),
            'details': {
                'terms_checked': checked,
                'findings': findings[:20],
                'ambiguous_pairs': [list(pair) for pair in ambiguous[:20]],
            },
            'flagged': bool(findings),
            'note': note,
        }

    _NUMBER_RE = re.compile(
        r'(?<![\w.,])(?:\d{1,3}(?:[\s,\u00a0]\d{3})+|\d+)(?:[.,]\d+)?(?![\w.,])'
    )

    @classmethod
    def _numeric_tokens(cls, text: str) -> Counter:
        """Numbers in a comparison-safe spelling.

        This deliberately does not guess that ``one`` and ``один`` are the
        same number. Digit-bearing facts (years, dates, prices, quantities,
        section numbers) are factual and can be checked across almost every
        language pair; spelled-out numbers need language-specific parsing.
        """
        values = []
        for match in cls._NUMBER_RE.finditer(text):
            token = match.group(0).replace('\u00a0', '').replace(' ', '')
            # 1,000 is a thousands separator; 1,5 is a decimal comma. The
            # former has exactly three digits after its last separator.
            if ',' in token and token.rsplit(',', 1)[1].isdigit() and len(token.rsplit(',', 1)[1]) == 3:
                token = token.replace(',', '')
            elif ',' in token:
                token = token.replace(',', '.')
            values.append(token)
        return Counter(values)

    def eval_numeric_preservation(self, source_text: str, final_text: str) -> Dict:
        """Deterministic gate for digit-written numbers and dates."""
        source_values = self._numeric_tokens(source_text)
        final_values = self._numeric_tokens(final_text)
        missing = list((source_values - final_values).elements())
        unexpected = list((final_values - source_values).elements())
        if not source_values:
            return {
                'test': 'numeric_preservation',
                'label': 'Numbers and dates',
                'value': 0,
                'flagged': False,
                'details': {'source_values': [], 'missing': [], 'unexpected': []},
                'note': 'No digit-written numbers or dates in the source to check.',
            }
        flagged = bool(missing)
        missing_preview = ', '.join(missing[:12])
        unexpected_preview = ', '.join(unexpected[:12])
        note = (
            f'Missing or changed source value(s): {missing_preview}. '
            'Numbers and dates are facts; inspect the corresponding chunks before shipping.'
            if missing else
            f'All {sum(source_values.values())} digit-written source number(s)/date component(s) survive in the final.'
        )
        if unexpected:
            note += f' Extra target value(s) to inspect: {unexpected_preview}.'
        return {
            'test': 'numeric_preservation',
            'label': 'Numbers and dates',
            'value': len(missing),
            'flagged': flagged,
            'details': {
                'source_values': sorted(source_values.elements()),
                'missing': missing[:50],
                'unexpected': unexpected[:50],
            },
            'note': note,
        }

    def eval_chunk_coverage(self, original_chunks: List[str], final_chunks: List[str]) -> Dict:
        """Detect a missing or blank translation segment without a model."""
        source_count, final_count = len(original_chunks), len(final_chunks)
        empty_final = [
            index + 1 for index, source in enumerate(original_chunks)
            if source.strip() and (index >= final_count or not final_chunks[index].strip())
        ]
        count_mismatch = source_count != final_count
        flagged = count_mismatch or bool(empty_final)
        if flagged:
            parts = []
            if count_mismatch:
                parts.append(f'{source_count} source chunk(s), {final_count} final chunk(s)')
            if empty_final:
                parts.append('empty final chunk(s): ' + ', '.join(map(str, empty_final[:20])))
            note = 'Chunk coverage failed — ' + '; '.join(parts) + '.'
        else:
            note = f'All {source_count} source chunks have a non-empty aligned final chunk.'
        return {
            'test': 'chunk_coverage',
            'label': 'Chunk coverage',
            'value': len(empty_final) + abs(source_count - final_count),
            'flagged': flagged,
            'details': {
                'source_chunks': source_count,
                'final_chunks': final_count,
                'empty_final_chunks': empty_final[:50],
            },
            'note': note,
        }

    @classmethod
    def _get_labse_model(cls):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                'LaBSE is not installed. Run: ./venv/bin/python -m pip install -r requirements-quality.txt'
            ) from exc
        if cls._labse_model is None:
            with cls._labse_lock:
                if cls._labse_model is None:
                    cls._labse_model = SentenceTransformer(cls.LABSE_MODEL_ID)
        return cls._labse_model

    def eval_labse_alignment(self, original_chunks: List[str], final_chunks: List[str]) -> Dict:
        """Document-wide source/final alignment and semantic-drift outliers.

        Chunks are already source-aligned by the translation pipeline. LaBSE
        measures every pair in one shared multilingual embedding space; it is
        a drift signal, not a fabricated claim that it can prove correctness.
        """
        length = min(len(original_chunks), len(final_chunks))
        if not length:
            return {
                'test': 'labse_alignment', 'label': 'LaBSE document alignment',
                'value': None, 'flagged': True,
                'note': 'No aligned source/final chunks available for LaBSE.',
            }
        model = self._get_labse_model()
        source = original_chunks[:length]
        final = final_chunks[:length]
        embeddings = model.encode(source + final, normalize_embeddings=True, show_progress_bar=False)
        scores = [
            float(sum(left * right for left, right in zip(embeddings[index], embeddings[length + index])))
            for index in range(length)
        ]
        baseline = median(scores)
        # A document-relative threshold catches the one paragraph that lost
        # meaning without imposing an English-centric absolute score.
        threshold = max(0.20, baseline - 0.20)
        flags = [
            {'chunk': index + 1, 'similarity': round(score, 3)}
            for index, score in enumerate(scores) if score < threshold
        ]
        return {
            'test': 'labse_alignment',
            'label': 'LaBSE document alignment',
            'value': round(mean(scores), 3),
            'flagged': bool(flags),
            'details': {
                'chunks_compared': length,
                'median_similarity': round(baseline, 3),
                'drift_threshold': round(threshold, 3),
                'drift_flags': flags[:50],
                'lowest_chunks': [
                    {'chunk': index + 1, 'similarity': round(score, 3)}
                    for index, score in sorted(enumerate(scores), key=lambda pair: pair[1])[:10]
                ],
            },
            'note': (
                f'{len(flags)} semantic-drift outlier(s) below the document-relative '
                f'threshold {threshold:.2f}; inspect those source/final chunks.'
                if flags else
                f'{length} aligned chunk pair(s), mean similarity {mean(scores):.2f}; '
                'no document-relative drift outlier.'
            ),
        }

    @classmethod
    def _get_language_id_pipeline(cls):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                'Language ID dependencies are not installed. Run: ./venv/bin/python -m pip install -r requirements-quality.txt'
            ) from exc
        if cls._language_id_pipeline is None:
            with cls._language_id_lock:
                if cls._language_id_pipeline is None:
                    cls._language_id_pipeline = pipeline(
                        'text-classification', model=cls.LANGUAGE_ID_MODEL_ID, tokenizer=cls.LANGUAGE_ID_MODEL_ID,
                    )
        return cls._language_id_pipeline

    def eval_language_id(self, final_chunks: List[str], target_lang: str) -> Dict:
        """Flag non-target-language final segments, including untranslated text."""
        expected = (target_lang or '').casefold()
        classifier = self._get_language_id_pipeline()
        findings = []
        checked = 0
        for index, chunk in enumerate(final_chunks):
            if len(chunk.strip()) < 20:
                continue
            result = classifier(chunk[:2000], truncation=True)[0]
            label = str(result.get('label', '')).removeprefix('__label__').casefold()
            score = float(result.get('score', 0))
            checked += 1
            if label != expected and score >= 0.70:
                findings.append({'chunk': index + 1, 'detected': label, 'confidence': round(score, 3)})
        return {
            'test': 'language_id',
            'label': 'Target-language segments',
            'value': len(findings),
            'flagged': bool(findings),
            'details': {
                'expected_language': expected,
                'chunks_checked': checked,
                'wrong_language_segments': findings[:50],
            },
            'note': (
                f'{len(findings)} segment(s) confidently detected as a language other than {LANG_NAMES.get(expected, expected)}: '
                + '; '.join(f"chunk {f['chunk']} → {f['detected']} ({f['confidence']:.0%})" for f in findings[:10])
                if findings else
                f'All {checked} sufficiently long final chunk(s) were classified as {LANG_NAMES.get(expected, expected)} or were low-confidence.'
            ),
        }

    def eval_llm_judge_stage2(
        self, original_chunks: List[str], draft_chunks: List[str], final_chunks: List[str],
        source_lang: str, target_lang: str,
    ) -> Dict:
        """Pairwise draft vs final, on two separate questions, with the
        source in front of the judge.

        This test used to show the judge two target-language passages and ask
        which read better — no source, "purely on naturalness, style and
        tone". That measures exactly the axis a rewriting refinement pass
        optimises, so it reported the pass as harmless while adequacy fell.
        Accuracy is now asked first and separately, and a final that reads
        better but says less shows up as a split verdict instead of a win.

        Chunk-aligned, not paragraph-aligned: Stage 2 stores final_chunks, so
        draft and final can be compared at the same granularity the Stage 1
        judge uses.
        """
        length = min(len(original_chunks), len(draft_chunks), len(final_chunks))
        if length == 0:
            return {
                'test': 'llm_judge_stage2',
                'label': 'LLM judge — draft vs final',
                'value': None,
                'flagged': True,
                'note': (
                    'Nothing to compare. This test needs per-chunk draft and final text; '
                    're-run Continue if this translation was refined by an older version.'
                ),
            }

        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)
        indices = self._risk_ranked_indices(original_chunks[:length], self.EVAL_SAMPLE_SIZE)
        tally = {
            'accuracy': Counter(),
            'readability': Counter(),
        }
        samples_used = 0

        for idx in indices:
            original, draft, final = original_chunks[idx], draft_chunks[idx], final_chunks[idx]
            if not draft.strip() or not final.strip() or not original.strip():
                continue
            if draft == final:
                # Nothing was changed here, so there is nothing to judge —
                # counting it as a tie would pad the result with agreement
                # the judge never actually expressed.
                continue

            swap = random.random() < 0.5
            version_a, version_b = (final, draft) if swap else (draft, final)
            prompt = f"""You are an independent editor comparing two {target_name} translations of the same {source_name} source. You wrote neither of them.

SOURCE ({source_name}):
{original}

VERSION A:
{version_a}

VERSION B:
{version_b}

Answer two separate questions. They can have different answers, and often do.

1. ACCURACY: which version conveys the source more faithfully — nothing changed, nothing left out, nothing invented?
2. READABILITY: which version reads better as {target_name} prose?

Respond with EXACTLY two lines and nothing else:
ACCURACY: A|B|TIE
READABILITY: A|B|TIE"""
            raw = self._call_model(prompt)
            samples_used += 1
            if not raw:
                continue

            for axis, pattern in (('accuracy', r'ACCURACY:\s*(A|B|TIE)'), ('readability', r'READABILITY:\s*(A|B|TIE)')):
                match = re.search(pattern, raw.upper())
                if not match:
                    continue
                letter = match.group(1)
                if letter == 'A':
                    tally[axis]['final' if swap else 'draft'] += 1
                elif letter == 'B':
                    tally[axis]['draft' if swap else 'final'] += 1
                else:
                    tally[axis]['tie'] += 1

        if not samples_used:
            return {
                'test': 'llm_judge_stage2',
                'label': 'LLM judge — draft vs final',
                'value': 0,
                'details': {'samples': 0},
                'flagged': False,
                'note': (
                    'Refinement left every sampled chunk byte-identical, so there was '
                    'nothing to compare.'
                ),
            }

        accuracy, readability = tally['accuracy'], tally['readability']
        details = {
            'samples': samples_used,
            'accuracy': {'final_wins': accuracy['final'], 'draft_wins': accuracy['draft'], 'ties': accuracy['tie']},
            'readability': {'final_wins': readability['final'], 'draft_wins': readability['draft'], 'ties': readability['tie']},
        }
        # The failure this is here to catch: the final wins on readability
        # while losing on accuracy. That is a refinement pass buying polish
        # with meaning, and it must not read as a pass.
        traded_meaning_for_polish = (
            accuracy['draft'] > accuracy['final'] and readability['final'] >= readability['draft']
        )
        note = (
            f"Accuracy: final {accuracy['final']}, draft {accuracy['draft']}, tie {accuracy['tie']}. "
            f"Readability: final {readability['final']}, draft {readability['draft']}, tie {readability['tie']}. "
            f"({samples_used} changed chunk(s) sampled.)"
        )
        if traded_meaning_for_polish:
            note += (
                ' The draft is more accurate while the final reads better — refinement '
                'is trading meaning for polish. Ship the draft unless you can see why not.'
            )
        return {
            'test': 'llm_judge_stage2',
            'label': 'LLM judge — draft vs final',
            'value': accuracy['final'],
            'details': details,
            'flagged': traded_meaning_for_polish or accuracy['draft'] > accuracy['final'],
            'note': note,
        }

    # -- Stage 1 tests: is the draft itself an adequate, fluent translation? --

    @staticmethod
    def _adequacy_fluency_prompt(source_name: str, target_name: str, original: str, candidate: str) -> str:
        return f"""You are a strict, expert translation quality judge. You did not produce this translation — evaluate it objectively and critically. Most real translations, even good ones, are NOT flawless — reserve top scores for output you would defend against expert scrutiny.

SOURCE ({source_name}):
{original}

TRANSLATION ({target_name}):
{candidate}

Rate the translation on two scales, using the anchors below. Pick the anchor that best matches, even if imperfectly.

ADEQUACY (how completely the source's meaning, including nuance and implication, is preserved):
5 = Every detail and nuance preserved; nothing added, omitted, or distorted.
4 = Meaning fully preserved; at most one minor, inconsequential nuance softened.
3 = Core meaning preserved, but some secondary details, connotations, or tone are lost or altered.
2 = Meaning is noticeably distorted or incomplete: omissions, mistranslations, or added content that changes meaning.
1 = Meaning is largely lost, contradicted, or unrelated to the source.

FLUENCY (how natural the translation reads to a native {target_name} speaker):
5 = Reads as if originally written by a skilled native speaker; no awkward phrasing anywhere.
4 = Natural throughout; at most one minor phrase an editor might tweak but wouldn't flag as wrong.
3 = Understandable and mostly natural, but has noticeable non-native phrasing, odd word order, or clunky sentences.
2 = Grammatically odd or stilted in multiple places; reads as machine-translated.
1 = Broken grammar, garbled syntax, or unreadable.

Respond with EXACTLY two lines and nothing else:
ADEQUACY: <1-5>
FLUENCY: <1-5>"""

    def _score_adequacy_fluency(
        self, pairs: List[Tuple[str, str]], source_name: str, target_name: str,
    ) -> Tuple[List[int], List[int], int]:
        """Runs the adequacy/fluency judge prompt over (original, candidate)
        pairs. Shared by the Stage 1 draft judge and the final-vs-original
        judge — same rubric, different candidate text."""
        adequacy_scores, fluency_scores, samples_used = [], [], 0
        for original, candidate in pairs:
            if not original.strip() or not candidate.strip():
                continue
            prompt = self._adequacy_fluency_prompt(source_name, target_name, original, candidate)
            raw = self._call_model(prompt)
            samples_used += 1
            if not raw:
                continue
            adequacy = re.search(r'ADEQUACY:\s*(\d)', raw)
            fluency = re.search(r'FLUENCY:\s*(\d)', raw)
            if adequacy:
                adequacy_scores.append(int(adequacy.group(1)))
            if fluency:
                fluency_scores.append(int(fluency.group(1)))
        return adequacy_scores, fluency_scores, samples_used

    def eval_llm_judge_stage1(
        self, original_chunks: List[str], draft_chunks: List[str],
        source_lang: str, target_lang: str,
    ) -> Dict:
        return self._judge_adequacy_fluency(
            'llm_judge_stage1', 'LLM judge — adequacy & fluency (draft)',
            original_chunks, draft_chunks, source_lang, target_lang, 'draft',
        )

    def eval_llm_judge_final(
        self, original_chunks: List[str], final_chunks: List[str],
        source_lang: str, target_lang: str,
    ) -> Dict:
        """The same rubric as the Stage 1 judge, scored on the FINAL text.

        Deliberately identical in every respect except which translation is
        being scored — same prompt, same sampling, same chunk boundaries — so
        that the difference between the two numbers means something. Scoring
        the draft by chunk and the final by paragraph position, as this did
        before, produced two numbers on two different texts and an
        "adequacy regression" that was partly an artifact of the alignment.
        """
        return self._judge_adequacy_fluency(
            'llm_judge_final', 'LLM judge — adequacy & fluency (final)',
            original_chunks, final_chunks, source_lang, target_lang, 'final',
        )

    def _judge_adequacy_fluency(
        self, test_name: str, label: str,
        original_chunks: List[str], candidate_chunks: List[str],
        source_lang: str, target_lang: str, candidate_name: str,
    ) -> Dict:
        length = min(len(original_chunks), len(candidate_chunks))
        if length == 0:
            return {
                'test': test_name,
                'label': label,
                'value': None,
                'flagged': True,
                'note': (
                    f'No per-chunk {candidate_name} text to score. Re-run the pass that '
                    'produces it if this translation predates chunk-level storage.'
                ),
            }

        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)
        # Deterministic, and the same for the draft judge and the final judge,
        # which is what makes their two scores subtractable.
        indices = self._risk_ranked_indices(original_chunks[:length], self.EVAL_SAMPLE_SIZE)
        pairs = [(original_chunks[idx], candidate_chunks[idx]) for idx in indices]
        adequacy_scores, fluency_scores, samples_used = self._score_adequacy_fluency(pairs, source_name, target_name)

        if not adequacy_scores and not fluency_scores:
            return {
                'test': test_name,
                'label': label,
                'value': None,
                'flagged': True,
                'note': 'The judge model did not return a usable score for any sampled chunk.',
            }

        avg_adequacy = round(mean(adequacy_scores), 2) if adequacy_scores else None
        avg_fluency = round(mean(fluency_scores), 2) if fluency_scores else None
        return {
            'test': test_name,
            'label': label,
            'value': avg_adequacy,
            'details': {
                'avg_adequacy': avg_adequacy,
                'avg_fluency': avg_fluency,
                'samples': samples_used,
                'sampled_chunks': indices,
                'scored': candidate_name,
            },
            'flagged': (avg_adequacy is not None and avg_adequacy < 3) or (avg_fluency is not None and avg_fluency < 3),
            'note': (
                f'{candidate_name.title()}: adequacy {avg_adequacy}/5, fluency {avg_fluency}/5 '
                f'over {samples_used} sampled chunk(s). Both numbers are averages of five '
                f'1–5 ratings, so treat differences under about 0.5 as noise.'
            ),
        }

    def eval_backtranslation_chrf(
        self, original_chunks: List[str], draft_chunks: List[str],
        source_lang: str, target_lang: str,
    ) -> Dict:
        if sacrebleu is None:
            return {
                'test': 'backtranslation_chrf',
                'label': 'Backtranslation chrF',
                'value': None,
                'flagged': True,
                'note': 'sacrebleu is not installed — run: pip install -r requirements.txt',
            }

        indices = self._sample_indices(len(original_chunks), self.EVAL_SAMPLE_SIZE)
        scores = []
        for idx in indices:
            original = original_chunks[idx]
            draft = draft_chunks[idx]
            if not original.strip() or not draft.strip():
                continue
            back_translation, warning = self.stage1_primary_translation(
                draft, source_lang=target_lang, target_lang=source_lang,
            )
            if warning:
                continue
            scores.append(sacrebleu.sentence_chrf(back_translation, [original]).score)

        if not scores:
            return {
                'test': 'backtranslation_chrf',
                'label': 'Backtranslation chrF',
                'value': None,
                'flagged': True,
                'note': 'Backtranslation did not produce a usable result for any sampled chunk.',
            }

        avg = round(mean(scores), 1)
        return {
            'test': 'backtranslation_chrf',
            'label': 'Backtranslation chrF (diagnostic only)',
            'value': avg,
            'details': {'samples': len(scores), 'per_sample': [round(s, 1) for s in scores], 'diagnostic_only': True},
            'flagged': avg < 40,
            'note': (
                f'chrF {avg}/100 over {len(scores)} sampled chunk(s). Diagnostic only, and '
                'not a quality score: chrF measures character overlap with a reference, and '
                'a back-translation is not a reference. The number is the combined error of '
                'the forward and reverse translations, so it cannot be read as the quality '
                'of either. Useful for spotting a chunk that came back as something '
                'completely different; useless for comparing runs.'
            ),
        }

    COMET_KIWI_MODEL = 'Unbabel/wmt22-cometkiwi-da'

    def eval_comet_kiwi(
        self, original_chunks: List[str], candidate_chunks: List[str],
        candidate_name: str = 'draft',
    ) -> Dict:
        """Reference-free neural QE, over whichever translation is handed to
        it — the draft after Start, the final after Continue. Which one was
        scored is reported, because a QE number with no stated subject is how
        a panel ends up showing the draft's score next to a shipped final.

        Optional — depends on unbabel-comet (pulls in torch and a multi-GB
        checkpoint download), which is kept out of the base requirements.txt
        on purpose.

        Downloads the checkpoint via huggingface_hub directly (rather than
        through comet.models.download_model) because that helper swallows
        the original error on a gated/access-denied repo and re-raises a
        generic "not supported by COMET" KeyError with no way to tell that
        apart from a real problem — this repo IS gated on Hugging Face, so
        that failure mode is the common case, not an edge case.
        """
        try:
            import torch
            from comet import load_from_checkpoint
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import GatedRepoError
        except ImportError:
            return {
                'test': 'comet_kiwi',
                'label': 'COMET-Kiwi (reference-free QE)',
                'value': None,
                'flagged': True,
                'note': 'Optional dependency not installed — run: pip install -r requirements-eval.txt',
            }

        try:
            length = min(len(original_chunks), len(candidate_chunks))
            indices = self._risk_ranked_indices(original_chunks[:length], self.EVAL_SAMPLE_SIZE)
            data = [
                {'src': original_chunks[idx], 'mt': candidate_chunks[idx]}
                for idx in indices
                if original_chunks[idx].strip() and candidate_chunks[idx].strip()
            ]
            if not data:
                raise ValueError('No chunks available to score')

            try:
                model_dir = snapshot_download(repo_id=self.COMET_KIWI_MODEL)
            except GatedRepoError:
                return {
                    'test': 'comet_kiwi',
                    'label': 'COMET-Kiwi (reference-free QE)',
                    'value': None,
                    'flagged': True,
                    'note': (
                        f'COMET-Kiwi ({self.COMET_KIWI_MODEL}) is a gated Hugging Face model — '
                        f'request access at https://huggingface.co/{self.COMET_KIWI_MODEL}, then run '
                        '`huggingface-cli login` (or set the HF_TOKEN env var) and try again.'
                    ),
                }

            checkpoint_path = os.path.join(model_dir, 'checkpoints', 'model.ckpt')
            model = load_from_checkpoint(checkpoint_path)
            # unbabel-comet's DataLoader setup (comet/models/base.py) always
            # passes multiprocessing_context="fork" when MPS is available,
            # but defaults num_workers to 0 for gpus=0 — a combination
            # PyTorch's DataLoader rejects outright, so predict() throws on
            # every Apple Silicon Mac regardless of sample count. Forcing
            # num_workers>0 to satisfy that isn't a fix either: forking a
            # process that already has a loaded HF fast tokenizer breaks the
            # tokenizer in the child (AttributeError inside the worker). The
            # only combination that actually works — no multiprocessing at
            # all — is what comet already does on every other platform, so
            # make it think MPS isn't available for the scope of this one
            # CPU-only (gpus=0) call.
            mps_is_available = torch.backends.mps.is_available
            torch.backends.mps.is_available = lambda: False
            try:
                output = model.predict(data, batch_size=4, gpus=0)
            finally:
                torch.backends.mps.is_available = mps_is_available
            avg = round(float(output.system_score) * 100, 1)  # type: ignore[attr-defined]
            return {
                'test': 'comet_kiwi',
                'label': 'COMET-Kiwi (reference-free QE)',
                'value': avg,
                'details': {'samples': len(data), 'scored': candidate_name, 'sampled_chunks': indices},
                'flagged': avg < 60,
                'note': (
                    f'COMET-Kiwi {avg}/100 for the {candidate_name} over {len(data)} sampled '
                    f'chunk(s) (0–100, higher is better). Sentence-level by design, so it is '
                    f'blind to anything that spans chunks.'
                ),
            }
        except Exception as e:
            logger.api_logger.error(f"COMET-Kiwi evaluation failed: {e}")
            return {
                'test': 'comet_kiwi',
                'label': 'COMET-Kiwi (reference-free QE)',
                'value': None,
                'flagged': True,
                'note': f'COMET-Kiwi failed: {e}',
            }

    def get_installed_models(self) -> List[Dict]:
        """Everything Ollama has pulled, with the details the UI needs.

        The parameter count is carried through because the roles that reason
        rather than translate — Prepare, refinement, the judge — are the ones
        where model size shows, and without it the interface can only default
        to whatever Ollama happens to list first.
        """
        response = self.session.get(
            "http://localhost:11434/api/tags",
            timeout=(5, 5)
        )
        response.raise_for_status()
        return response.json().get('models') or []

    def get_available_models(self) -> List[str]:
        return [model['name'] for model in self.get_installed_models()]
    
# Translation Recovery
class TranslationRecovery:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        
    def get_failed_translations(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute('''
                SELECT * FROM translations 
                WHERE status = 'error'
                ORDER BY created_at DESC
            ''')
            return [dict(row) for row in cur.fetchall()]
        
    def retry_translation(self, translation_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE translations
                SET status = 'pending', progress = 0, error_message = NULL,
                    current_chunk = 0, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (translation_id,))
            
            conn.execute('''
                UPDATE chunks
                SET status = 'pending', error_message = NULL
                WHERE translation_id = ? AND status = 'error'
            ''', (translation_id,))
            
    def cleanup_failed_translations(self, days: int = 7):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"DELETE FROM translations WHERE status = 'error' AND created_at < datetime('now', '-{days} days')"
            )
            
recovery = TranslationRecovery()

# Health checking middleware
@app.before_request
def check_ollama():
    # Managing locally saved tasks must remain possible even when Ollama is
    # stopped, so a user can clear old or failed translations.
    exempt_endpoints = {
        'health_check', 'serve_frontend', 'serve_static', 'delete_translation',
        # The log console is most wanted precisely when the pipeline is
        # failing, and "Ollama is down" is one of the things it exists to show.
        'stream_logs', 'reset_logs', 'rotate_logs',
    }
    if request.endpoint not in exempt_endpoints:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.app_logger.error(f"Ollama health check failed: {str(e)}")
            return jsonify({
                'error': 'Translation service is not available'
            }), 503
        
# Flask routes
@app.route('/')
def serve_frontend():
    response = send_from_directory(STATIC_FOLDER, 'index.html')
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(STATIC_FOLDER, path)

@app.route('/models', methods=['GET'])
@with_error_handling
def get_models():
    translator = BookTranslator()
    models = []
    for model in translator.get_installed_models():
        details = model.get('details') or {}
        models.append({
            'name': model['name'],
            'size': model.get('size') or 0,
            'parameter_size': details.get('parameter_size') or '',
            'modified': model.get('modified_at') or 'Unknown',
        })
    return jsonify({'models': models})

@app.route('/translations', methods=['GET'])
@with_error_handling
def get_translations():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute('''
            SELECT t.id, t.filename, t.source_lang, t.target_lang, t.model,
                   t.status, t.progress, t.detected_language, t.created_at,
                   t.updated_at, t.error_message,
                   COUNT(tt.id) AS glossary_terms
            FROM translations AS t
            LEFT JOIN translation_terms AS tt ON tt.translation_id = t.id
            GROUP BY t.id
            ORDER BY t.created_at DESC
        ''')
        translations = [dict(row) for row in cur.fetchall()]
    return jsonify({'translations': translations})

@app.route('/translations/<int:translation_id>', methods=['GET'])
@with_error_handling
def get_translation(translation_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute('SELECT * FROM translations WHERE id = ?', (translation_id,))
        translation = cur.fetchone()
        if not translation:
            return jsonify({'error': 'Translation not found'}), 404

        eval_rows = conn.execute(
            'SELECT test_name, judge_model, value, flagged, note, details FROM evaluation_results WHERE translation_id = ?',
            (translation_id,)
        ).fetchall()

        data = dict(translation)
        data['evaluation_results'] = {
            r['test_name']: {
                'judge_model': r['judge_model'],
                'value': r['value'],
                'flagged': bool(r['flagged']),
                'note': r['note'],
                'details': json.loads(r['details']) if r['details'] else None,
            }
            for r in eval_rows
        }
        return jsonify(data)

@app.route('/translations/<int:translation_id>', methods=['DELETE'])
@with_error_handling
def delete_translation(translation_id):
    """Remove a saved translation task and its chunk-level records."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            'SELECT id FROM translations WHERE id = ?', (translation_id,)
        ).fetchone()
        if not row:
            return jsonify({'error': 'Translation not found'}), 404

        # Delete children explicitly: SQLite foreign keys are not enabled for
        # every connection, so relying on the schema alone leaves orphan rows.
        conn.execute('DELETE FROM chunks WHERE translation_id = ?', (translation_id,))
        conn.execute('DELETE FROM translation_terms WHERE translation_id = ?', (translation_id,))
        conn.execute('DELETE FROM evaluation_results WHERE translation_id = ?', (translation_id,))
        conn.execute('DELETE FROM translations WHERE id = ?', (translation_id,))

    logger.app_logger.info('Deleted translation task %s', translation_id)
    return jsonify({'status': 'success', 'id': translation_id})
    
class UploadError(Exception):
    """An upload the caller should reject with a 400."""


def read_uploaded_book(file):
    """Save an uploaded book into UPLOAD_FOLDER and decode it.

    Returns (text, chapters, book_title, book_author, source_format, filepath).
    `chapters` is None for plain text; for EPUB it is what drives chunking, so
    that no chunk ever straddles a chapter boundary. Deleting filepath is the
    caller's job.
    """
    if not file or file.filename == '':
        raise UploadError('No selected file')

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    if is_epub_filename(filename):
        chapters, book_title, book_author = extract_epub_book(filepath)
        if not chapters:
            raise UploadError('Could not find any chapters in this EPUB')
        # Readable in the UI's "Original text" preview; chapter splitting for
        # translation itself is driven by the `chapters` list, not this string.
        text = '\n\n'.join(
            f'=== Chapter {i} ===\n\n{chapter}' for i, chapter in enumerate(chapters, 1)
        )
        return text, chapters, book_title, book_author, 'epub', filepath

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='cp1251') as f:
            text = f.read()
    return text, None, None, None, 'txt', filepath


@app.route('/prepare', methods=['POST'])
@with_error_handling
def prepare():
    """STAGE 0: read the book and propose the contract the translation will
    run under — one agreed rendering per recurring proper noun, plus a short
    brief about the document as a whole.

    Deliberately does not create a translation row or translate anything.
    Both artifacts come back as editable text and are then submitted with
    Start like any hand-written glossary, so what actually reaches the model
    is always what the user saw and approved.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    filepath = None
    try:
        source_lang = request.form.get('sourceLanguage')
        target_lang = request.form.get('targetLanguage')
        model_name = request.form.get('model')
        genre = request.form.get('genre', 'unknown')
        # Stage 0 makes two different demands. Ruling on whether two source
        # forms name one entity is reasoning about the document; rendering a
        # name into the target language is translation knowledge. They get
        # their own model choices, and the entity role falls back to the
        # rendering one when the browser has no separate preference.
        entity_model_name = request.form.get('entityModel') or model_name

        if not all([source_lang, target_lang, model_name]):
            return jsonify({'error': 'Missing required parameters'}), 400
        for name in (model_name, entity_model_name):
            if is_translategemma(name):
                return jsonify({'error': (
                    'TranslateGemma is translation-only and cannot extract names or '
                    'write a brief. Pick a general instruct model to prepare, then '
                    'switch back to TranslateGemma for Start if you like.'
                )}), 400

        try:
            text, _, _, _, _, filepath = read_uploaded_book(request.files['file'])
        except UploadError as e:
            return jsonify({'error': str(e)}), 400

        translator = BookTranslator(model_name=model_name)
        try:
            candidates, review_queue = translator.build_glossary_candidates(text)
        except RuntimeError as e:
            return jsonify({'error': str(e)}), 503
        extracted = len(candidates)
        # The clustering step guessed which source forms are one entity. Have
        # the model rule on those guesses before anything is rendered, because
        # a wrong merge silently agrees one rendering for two entities and no
        # later stage can tell that happened.
        resolver = (
            translator if entity_model_name == model_name
            else BookTranslator(model_name=entity_model_name)
        )
        candidates, cluster_decisions = resolver.adjudicate_entity_clusters(
            text, source_lang, candidates, review_queue,
        )
        records = translator.propose_proper_noun_records(
            text, source_lang, target_lang, genre, candidates=candidates,
        )
        rendering_conflicts = translator.find_rendering_conflicts(records)
        # Serialised in the glossary's own text format, so the proposal lands
        # in the existing textarea and goes through the same parser and the
        # same validation as anything typed by hand.
        glossary = '\n'.join(
            f"{record['source']} => {record['target']} | {record['mode']}"
            for record in records
        )
        proposed = {record['source'].casefold() for record in records}
        return jsonify({
            'glossary': glossary,
            'entities': records,
            'entity_resolution': {
                'clustered_candidates': len(candidates),
                'extracted_candidates': extracted,
                'review_pairs': len(review_queue),
                # What the model ruled on the clustering, and what it found
                # that extraction did not — both were previously invisible.
                'cluster_decisions': cluster_decisions,
                'clusters_confirmed': sum(1 for d in cluster_decisions if d['same_entity']),
                'clusters_split': sum(1 for d in cluster_decisions if not d['same_entity']),
                'added_by_model': sorted(
                    proposed - {record['surface'].casefold() for record in candidates}
                ),
            },
            'rendering_conflicts': rendering_conflicts,
            'source_chars': len(text),
        })
    finally:
        # Prepare and Start are given the same upload under the same name, so
        # the second one to finish finds the temp file already gone. That is
        # the normal path, not an error worth a line in the log.
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except OSError as e:
            logger.app_logger.error(f"Failed to cleanup uploaded file: {str(e)}")


@app.route('/translate', methods=['POST'])
@with_error_handling
def translate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    filepath = None
    try:
        file = request.files['file']
        source_lang = request.form.get('sourceLanguage')
        target_lang = request.form.get('targetLanguage')
        model_name = request.form.get('model')
        genre = request.form.get('genre', 'unknown')  # Get genre from request
        try:
            terminology = TerminologyManager.from_text(
                request.form.get('glossary', '')
            )
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        if not all([file, source_lang, target_lang, model_name]):
            return jsonify({'error': 'Missing required parameters'}), 400

        try:
            text, chapters, book_title, book_author, source_format, filepath = read_uploaded_book(file)
        except UploadError as e:
            return jsonify({'error': str(e)}), 400
        filename = secure_filename(file.filename)

        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute('''
                INSERT INTO translations (
                    filename, source_lang, target_lang, model,
                    status, original_text, genre, source_format, book_title, book_author
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (filename, source_lang, target_lang, model_name,
                  'in_progress', text, genre, source_format, book_title, book_author))
            translation_id = cur.lastrowid
            conn.executemany(
                '''
                INSERT INTO translation_terms (
                    translation_id, source_term, target_term,
                    enforcement_mode, status
                ) VALUES (?, ?, ?, ?, ?)
                ''',
                [
                    (
                        translation_id,
                        term.source,
                        term.target,
                        term.mode,
                        # Clicking Start is the explicit approval boundary:
                        # the editable glossary that reaches this endpoint is
                        # the user's accepted contract for this job.
                        'verified',
                    )
                    for term in terminology.terms
                ],
            )
            
        translator = BookTranslator(model_name=model_name)
        
        def generate():
            try:
                starting = {
                    'progress': 1,
                    'stage': 'starting',
                    'translation_id': translation_id,
                    'message': 'Book uploaded. Loading the model and starting the first batch…',
                    'terminology': {
                        'total': len(terminology.terms),
                        'used': 0,
                        'violations': 0,
                    },
                }
                yield f"data: {json.dumps(starting, ensure_ascii=False)}\n\n"
                for update in translator.translate_stage1(
                    text,
                    source_lang,
                    target_lang,
                    translation_id,
                    genre=genre,
                    terminology=terminology,
                    chapters=chapters,
                ):
                    update['translation_id'] = translation_id
                    yield f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
            except Exception as e:
                error_message = str(e)
                logger.translation_logger.error(f"Translation error: {error_message}")
                logger.translation_logger.error(traceback.format_exc())
                yield f"data: {json.dumps({'error': error_message})}\n\n"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
            },
        )

    except Exception as e:
        logger.app_logger.error(f"Translation request error: {str(e)}")
        logger.app_logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        # Prepare and Start are given the same upload under the same name, so
        # the second one to finish finds the temp file already gone. That is
        # the normal path, not an error worth a line in the log.
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except OSError as e:
            logger.app_logger.error(f"Failed to cleanup uploaded file: {str(e)}")


@app.route('/refine/<int:translation_id>', methods=['POST'])
@with_error_handling
def refine(translation_id):
    """Run STAGE 2 (refinement) only, over the draft a prior /translate call
    already produced and saved for this translation_id.

    By default this reuses the model Stage 1 was translated with. The
    caller may instead pass {"model": "..."} to refine with a different
    model than the draft was produced with — e.g. the user changed the
    model selector after Start but before Continue.

    {"verifier_model": "..."} chooses who rules on the patches. It is a
    separate role because the reviewing model grading its own edits is not a
    check: its A/B verdict follows the order the versions are shown in, the
    two orderings disagree, and every patch is then vetoed.
    """
    payload = request.get_json(silent=True) or {}
    override_model = (payload.get('model') or '').strip()
    verifier_model = (payload.get('verifier_model') or '').strip()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT * FROM translations WHERE id = ?', (translation_id,)
        ).fetchone()

        if not row:
            return jsonify({'error': 'Translation not found'}), 404
        if not row['draft_chunks']:
            return jsonify({'error': 'No draft translation to refine yet — run Start first'}), 400
        if row['status'] == 'in_progress':
            # Only this process's own live streams count as running. A row left
            # 'in_progress' by a closed tab or by a server restart is a leftover,
            # and refusing to run it again locked the draft out permanently.
            if is_run_active(translation_id):
                return jsonify({'error': 'This translation is already running'}), 409
            logger.translation_logger.warning(
                "Translation %s was left 'in_progress' by an earlier run that is "
                "no longer streaming — starting refinement over the saved draft",
                translation_id,
            )

        term_rows = conn.execute(
            '''SELECT source_term, target_term, enforcement_mode
               FROM translation_terms WHERE translation_id = ?''',
            (translation_id,)
        ).fetchall()

    terminology = TerminologyManager([
        GlossaryTerm(source=r['source_term'], target=r['target_term'], mode=r['enforcement_mode'])
        for r in term_rows
    ])

    refine_model = override_model or row['model']
    for name in (refine_model, verifier_model):
        if name and is_translategemma(name):
            return jsonify({'error': (
                'TranslateGemma is translation-only and cannot run the refinement '
                'pass or rule on its patches. Pick a general instruct model in the '
                'Model selector, then press Continue.'
            )}), 400

    translator = BookTranslator(
        model_name=refine_model, verifier_model=verifier_model or None,
    )

    def generate():
        try:
            starting = {
                'progress': 1,
                'stage': 'starting',
                'translation_id': translation_id,
                'message': 'Starting the refinement pass…',
                'terminology': {
                    'total': len(terminology.terms),
                    'used': 0,
                    'violations': 0,
                },
                'refinement': {
                    'review_model': translator.model_name,
                    'verifier_model': translator.verifier_model,
                },
            }
            yield f"data: {json.dumps(starting, ensure_ascii=False)}\n\n"
            for update in translator.translate_stage2(
                translation_id,
                row['source_lang'],
                row['target_lang'],
                genre=row['genre'],
                terminology=terminology,
            ):
                update['translation_id'] = translation_id
                yield f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_message = str(e)
            logger.translation_logger.error(f"Refinement error: {error_message}")
            logger.translation_logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'error': error_message})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


# ------------------------------------------------------------------
# Live log console.
#
# Everything this pipeline decides is already written to logs/, but a file you
# have to remember to tail is a file nobody reads while a book is running. The
# three logs are followed and merged into one stream so the console can show
# what the run is doing as it happens.
# ------------------------------------------------------------------

LOG_STREAM_SOURCES = {
    'translations': 'translations.log',
    'api': 'api.log',
    'app': 'app.log',
}
# How often a followed file is checked for new lines. Log lines arrive seconds
# apart at best — a model call is the fast case at ~2s — so polling faster than
# this only burns CPU on an already busy machine.
LOG_STREAM_POLL_SECONDS = 0.5
# How much history a newly opened console shows before it starts following.
LOG_STREAM_BACKLOG_LINES = 300
LOG_STREAM_BACKLOG_BYTES = 256 * 1024
# A comment frame often enough that an idle stream is not mistaken for a dead
# one, by the browser or by the person watching it.
LOG_STREAM_KEEPALIVE_SECONDS = 15
# "2026-07-27 07:42:27,388 - translation_logger - INFO - Stage 2 …". Lines that
# do not match are continuations — a logged prompt is many lines long — and are
# passed through attached to whatever came before them.
LOG_LINE_RE = re.compile(
    r'^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d[,.]\d+) - (?P<logger>\S+) - '
    r'(?P<level>[A-Z]+) - (?P<message>.*)$'
)


class LogTail:
    """One log file, followed like ``tail -f``.

    Opened in binary and decoded per read, because the files are being written
    to while they are read and a multi-byte character can be split across two
    reads. Reopens itself when the file is rotated or truncated — these logs
    are on a RotatingFileHandler, so that happens on its own schedule.
    """

    def __init__(self, source: str, path: str):
        self.source = source
        self.path = path
        self.handle = None
        self.inode = None
        self.pending = b''

    def _open(self, at_end: bool = True) -> bool:
        try:
            self.handle = open(self.path, 'rb')
        except OSError:
            self.handle = None
            return False
        self.inode = os.fstat(self.handle.fileno()).st_ino
        self.pending = b''
        if at_end:
            self.handle.seek(0, os.SEEK_END)
        return True

    def backlog(self, lines: int) -> List[Dict]:
        """The tail of the file as it stands, then leave the handle at its end."""
        if not self._open(at_end=False):
            return []
        size = os.fstat(self.handle.fileno()).st_size
        window = min(size, LOG_STREAM_BACKLOG_BYTES)
        self.handle.seek(size - window)
        chunk = self.handle.read().decode('utf-8', errors='replace').splitlines()
        # A window that starts mid-file almost certainly starts mid-line.
        if window < size and chunk:
            chunk = chunk[1:]
        return [self._entry(line) for line in chunk[-lines:] if line.strip()]

    def read_new(self) -> List[Dict]:
        if self.handle is None and not self._open():
            return []
        try:
            stat = os.stat(self.path)
        except OSError:
            return []
        if stat.st_ino != self.inode or stat.st_size < self.handle.tell():
            # Rotated or truncated: the lines we have not read are gone with
            # the old file, so pick the new one up from its beginning.
            self.handle.close()
            if not self._open(at_end=False):
                return []
        data = self.pending + self.handle.read()
        if not data:
            return []
        *complete, self.pending = data.split(b'\n')
        return [
            self._entry(line.decode('utf-8', errors='replace').rstrip('\r'))
            for line in complete if line.strip()
        ]

    def _entry(self, line: str) -> Dict:
        match = LOG_LINE_RE.match(line)
        if not match:
            return {'source': self.source, 'level': 'CONT', 'time': '', 'message': line}
        return {
            'source': self.source,
            'level': match.group('level'),
            'time': match.group('time'),
            'message': match.group('message'),
        }

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


@app.route('/logs/rotate', methods=['POST'])
def rotate_logs():
    """Begin new log files, keeping the previous ones as ``*.log.1``.

    Called when a new document is loaded, so the console shows one book at a
    time. Refused while a run is streaming: cutting a book's log in half
    midway is exactly what makes a log useless afterwards.
    """
    if ACTIVE_RUNS:
        return jsonify({
            'error': 'A translation is running — its log is still being written.',
            'rotated': [],
        }), 409

    payload = request.get_json(silent=True) or {}
    # The document name comes from the browser and is about to become a log
    # line, so it is flattened first: a name containing a newline could
    # otherwise forge an entry that looks like the pipeline's own.
    document = re.sub(r'\s+', ' ', str(payload.get('document') or '')).strip()[:120]

    rotated = logger.rotate()
    logger.app_logger.info(
        "New document loaded%s — previous log kept as *.log.1",
        f": {document}" if document else '',
    )
    return jsonify({'rotated': rotated, 'document': document})


@app.route('/logs/reset', methods=['POST'])
def reset_logs():
    """Empty the three log files, so the next run starts on a clean console.

    Truncates in place rather than deleting: the handlers hold these files
    open, and a deleted file would leave them writing to nothing until the
    server was restarted. Only the three known names under logs/ are touched —
    nothing about the target comes from the request.

    Truncating a file another handle has open is only safe because logging
    opens its files in append mode: every write goes to the current end of the
    file rather than to a remembered offset. Measured — 244 KB of log, then a
    truncate, then one line: 27 bytes and no padding. A handle opened without
    O_APPEND (a shell's ``> file``) behaves the opposite way and leaves the gap
    filled with NUL bytes.
    """
    emptied, failed = [], {}
    for filename in LOG_STREAM_SOURCES.values():
        path = os.path.join(LOG_FOLDER, filename)
        try:
            with open(path, 'w', encoding='utf-8'):
                pass
            emptied.append(filename)
        except OSError as e:
            failed[filename] = str(e)

    # First line of the new file, so the console is never blank and the reset
    # itself is on the record.
    logger.app_logger.info("Log files emptied from the log console: %s", ', '.join(emptied) or 'none')
    if failed:
        logger.app_logger.error(f"Could not empty log file(s): {failed}")
        return jsonify({'emptied': emptied, 'failed': failed}), 500
    return jsonify({'emptied': emptied})


@app.route('/logs/stream')
def stream_logs():
    """Server-sent events: the three log files, merged, as they are written.

    ``?tail=N`` sets how many past lines to send first (0 for none).
    ``?since=<timestamp>`` sends only lines newer than one already seen, which
    is what a reconnecting console passes. Without it, an EventSource that
    reconnects — and it reconnects on its own, after any hiccup — is served the
    whole backlog again, and the console jumps back in time as if it had been
    reset.
    """
    try:
        backlog_lines = max(0, min(2000, int(request.args.get('tail', LOG_STREAM_BACKLOG_LINES))))
    except (TypeError, ValueError):
        backlog_lines = LOG_STREAM_BACKLOG_LINES
    since = (request.args.get('since') or '').strip()

    def generate():
        tails = [
            LogTail(source, os.path.join(LOG_FOLDER, filename))
            for source, filename in LOG_STREAM_SOURCES.items()
        ]
        try:
            history = [entry for tail in tails for entry in tail.backlog(backlog_lines)]
            # The timestamp format sorts chronologically as text, which is what
            # lets three separately written files be interleaved correctly.
            history.sort(key=lambda entry: entry['time'] or '')
            if since:
                # Strictly newer, so the line the console already ends with is
                # not sent twice. Continuations carry no timestamp of their own
                # and ride along with whatever preceded them.
                history = [entry for entry in history if (entry['time'] or since) > since]
            for entry in history[-backlog_lines:] if backlog_lines else []:
                yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
            if not since:
                yield f"data: {json.dumps({'source': 'console', 'level': 'MARK', 'time': '', 'message': f'— following {len(tails)} log file(s) —'})}\n\n"

            last_keepalive = time.time()
            while True:
                new_entries = [entry for tail in tails for entry in tail.read_new()]
                if new_entries:
                    new_entries.sort(key=lambda entry: entry['time'] or '')
                    for entry in new_entries:
                        yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
                    last_keepalive = time.time()
                elif time.time() - last_keepalive > LOG_STREAM_KEEPALIVE_SECONDS:
                    yield ': keepalive\n\n'
                    last_keepalive = time.time()
                time.sleep(LOG_STREAM_POLL_SECONDS)
        except GeneratorExit:
            raise
        finally:
            # The console is opened and closed freely; a followed handle per
            # visit that is never released is a file descriptor leak.
            for tail in tails:
                tail.close()

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


# Stage 1 tests need only the draft (original_chunks/draft_chunks); Stage 2
# tests need a completed refinement (translated_text) too.
STAGE1_TESTS = {'backtranslation_chrf', 'llm_judge_stage1', 'comet_kiwi'}
STAGE2_TESTS = {
    'length_ratio', 'diff_ratio', 'ngram_repetition', 'terminology_delta',
    'script_leakage', 'entity_consistency', 'numeric_preservation', 'chunk_coverage',
    'labse_alignment', 'language_id',
    'llm_judge_stage2', 'llm_judge_final', 'comet_kiwi_final',
}
ALL_TESTS = STAGE1_TESTS | STAGE2_TESTS
# Tests whose answer is a fact about the text rather than a model's opinion:
# no sampling, no judge, and they read the whole document instead of five
# chunks of it. These are the only tests that can see across chunk
# boundaries, which is where this pipeline's real errors live.
DETERMINISTIC_TESTS = {
    'length_ratio', 'diff_ratio', 'ngram_repetition',
    'terminology_delta', 'script_leakage', 'entity_consistency',
    'numeric_preservation', 'chunk_coverage',
}


@app.route('/evaluate/<int:translation_id>/<test_name>', methods=['POST'])
@with_error_handling
def evaluate(translation_id, test_name):
    """Run a single, on-demand quality test over an existing translation.
    Never runs automatically — some of these re-invoke the model
    (backtranslation, LLM-judge) or a QE model (COMET-Kiwi), so they're
    triggered one at a time from the UI rather than baked into the
    translate/refine pipeline."""
    if test_name not in ALL_TESTS:
        return jsonify({'error': f'Unknown test "{test_name}"'}), 400

    payload = request.get_json(silent=True) or {}
    judge_model = (payload.get('judge_model') or '').strip()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT * FROM translations WHERE id = ?', (translation_id,)
        ).fetchone()

        if not row:
            return jsonify({'error': 'Translation not found'}), 404
        if test_name in STAGE1_TESTS and not row['draft_chunks']:
            return jsonify({'error': 'Run Start first — no draft translation to evaluate yet'}), 400
        if test_name in STAGE2_TESTS and not row['translated_text']:
            return jsonify({'error': 'Run Continue (refinement) first — no final translation to evaluate yet'}), 400

        term_rows = conn.execute(
            '''SELECT source_term, target_term, enforcement_mode
               FROM translation_terms WHERE translation_id = ?''',
            (translation_id,)
        ).fetchall()

    terminology = TerminologyManager([
        GlossaryTerm(source=r['source_term'], target=r['target_term'], mode=r['enforcement_mode'])
        for r in term_rows
    ])
    # LLM judge tests and backtranslation re-invoke a model rather than just
    # computing a metric, so the caller may point them at a different model
    # than the one that produced the translation itself. For LLM judge this
    # avoids self-judging; for backtranslation_chrf it avoids conflating
    # "translates well forward" with "round-trips well with itself" — a
    # model reused for its own reverse leg gets credit for being literal and
    # self-consistent, not for translation quality. Every other test is
    # metric-only, no model call.
    uses_evaluator_model = test_name in ('llm_judge_stage1', 'llm_judge_stage2', 'llm_judge_final', 'backtranslation_chrf')
    evaluator_model = judge_model if uses_evaluator_model and judge_model else row['model']
    # Backtranslation just runs Stage 1 in reverse, which TranslateGemma is
    # good at. Judging is a different job: asked for a score it would translate
    # the rubric instead of answering it.
    if test_name.startswith('llm_judge') and is_translategemma(evaluator_model):
        return jsonify({'error': (
            'TranslateGemma is translation-only and cannot score a translation. '
            'Pick a general instruct model as the judge model for this test.'
        )}), 400
    translator = BookTranslator(model_name=evaluator_model)
    original_chunks = json.loads(row['original_chunks']) if row['original_chunks'] else []
    draft_chunks = json.loads(row['draft_chunks']) if row['draft_chunks'] else []
    # Refinements produced before final_chunks existed have only the joined
    # text, so the chunk-aligned tests fall back to splitting it on the same
    # boundary the joins used. Not exact if a chunk contained a blank line,
    # but it keeps old rows evaluable instead of erroring.
    final_chunks = (
        json.loads(row['final_chunks']) if row['final_chunks']
        else (row['translated_text'] or '').split('\n\n')
    )

    try:
        if test_name == 'length_ratio':
            result = translator.eval_length_ratio(
                row['original_text'] or '', row['machine_translation'] or '', row['translated_text'] or '',
            )
        elif test_name == 'diff_ratio':
            result = translator.eval_diff_ratio(row['machine_translation'] or '', row['translated_text'] or '')
        elif test_name == 'ngram_repetition':
            result = translator.eval_ngram_repetition(row['translated_text'] or '')
        elif test_name == 'terminology_delta':
            result = translator.eval_terminology_delta(
                row['original_text'] or '', row['machine_translation'] or '', row['translated_text'] or '', terminology,
            )
        elif test_name == 'script_leakage':
            result = translator.eval_script_leakage(
                row['original_text'] or '', row['translated_text'] or '',
            )
        elif test_name == 'entity_consistency':
            result = translator.eval_entity_consistency(original_chunks, final_chunks, terminology)
        elif test_name == 'numeric_preservation':
            result = translator.eval_numeric_preservation(
                row['original_text'] or '', row['translated_text'] or '',
            )
        elif test_name == 'chunk_coverage':
            result = translator.eval_chunk_coverage(original_chunks, final_chunks)
        elif test_name == 'labse_alignment':
            result = translator.eval_labse_alignment(original_chunks, final_chunks)
        elif test_name == 'language_id':
            result = translator.eval_language_id(final_chunks, row['target_lang'])
        elif test_name == 'llm_judge_stage2':
            result = translator.eval_llm_judge_stage2(
                original_chunks, draft_chunks, final_chunks, row['source_lang'], row['target_lang'],
            )
        elif test_name == 'backtranslation_chrf':
            result = translator.eval_backtranslation_chrf(
                original_chunks, draft_chunks, row['source_lang'], row['target_lang'],
            )
        elif test_name == 'llm_judge_stage1':
            result = translator.eval_llm_judge_stage1(
                original_chunks, draft_chunks, row['source_lang'], row['target_lang'],
            )
        elif test_name == 'llm_judge_final':
            result = translator.eval_llm_judge_final(
                original_chunks, final_chunks, row['source_lang'], row['target_lang'],
            )
        elif test_name == 'comet_kiwi_final':
            result = translator.eval_comet_kiwi(original_chunks, final_chunks, candidate_name='final')
            result['test'] = 'comet_kiwi_final'
        else:  # comet_kiwi
            result = translator.eval_comet_kiwi(original_chunks, draft_chunks, candidate_name='draft')
    except Exception as e:
        logger.app_logger.error(f"Evaluation '{test_name}' failed: {e}")
        logger.app_logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

    result['translation_id'] = translation_id

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            INSERT INTO evaluation_results (translation_id, test_name, judge_model, value, flagged, note, details, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (translation_id, test_name) DO UPDATE SET
                judge_model = excluded.judge_model,
                value = excluded.value,
                flagged = excluded.flagged,
                note = excluded.note,
                details = excluded.details,
                updated_at = CURRENT_TIMESTAMP
        ''', (
            translation_id, test_name, judge_model or None,
            result.get('value'), 1 if result.get('flagged') else 0,
            result.get('note'), json.dumps(result.get('details')) if result.get('details') is not None else None,
        ))

    return jsonify(result)


@app.route('/download/<int:translation_id>', methods=['GET'])
@with_error_handling
def download_translation(translation_id):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute('''
            SELECT filename, translated_text, source_format, translated_chapters, book_title, book_author
            FROM translations
            WHERE id = ? AND status = 'completed'
        ''', (translation_id,))
        result = cur.fetchone()

        if not result:
            return jsonify({'error': 'Translation not found or not completed'}), 404

        filename, translated_text, source_format, translated_chapters, book_title, book_author = result

        if source_format == 'epub' and translated_chapters:
            chapters = json.loads(translated_chapters)
            epub_bytes = build_epub_from_chapters(
                chapters,
                title=book_title or os.path.splitext(filename)[0],
                author=book_author or 'Unknown Author',
            )
            download_name = f"translated_{os.path.splitext(filename)[0]}.epub"
            download_path = os.path.join(TRANSLATIONS_FOLDER, download_name)
            with open(download_path, 'wb') as f:
                f.write(epub_bytes)
            return send_file(
                download_path,
                as_attachment=True,
                download_name=download_name,
                mimetype='application/epub+zip',
            )

        download_path = os.path.join(TRANSLATIONS_FOLDER, f'translated_{filename}')
        with open(download_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)

        return send_file(
            download_path,
            as_attachment=True,
            download_name=f'translated_{filename}'
        )


@app.route('/export/epub', methods=['POST'])
@with_error_handling
def export_epub():
    """Export translation as EPUB file"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        title = data.get('title', 'Translation')
        author = data.get('author', 'Book Translator')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Create unique filename
        epub_id = str(uuid.uuid4())
        epub_filename = f'translation_{epub_id}.epub'
        epub_path = os.path.join(TRANSLATIONS_FOLDER, epub_filename)
        
        # Create EPUB structure
        with zipfile.ZipFile(epub_path, 'w', zipfile.ZIP_DEFLATED) as epub:
            # mimetype (must be first, uncompressed)
            epub.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)
            
            # META-INF/container.xml
            container_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>'''
            epub.writestr('META-INF/container.xml', container_xml)
            
            # OEBPS/content.opf
            content_opf = f'''<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="BookID">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:title>{title}</dc:title>
        <dc:creator>{author}</dc:creator>
        <dc:language>en</dc:language>
        <dc:identifier id="BookID">{epub_id}</dc:identifier>
        <meta property="dcterms:modified">{dt.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}</meta>
    </metadata>
    <manifest>
        <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
        <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    </manifest>
    <spine toc="ncx">
        <itemref idref="chapter1"/>
    </spine>
</package>'''
            epub.writestr('OEBPS/content.opf', content_opf)
            
            # OEBPS/toc.ncx
            toc_ncx = f'''<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
    <head>
        <meta name="dtb:uid" content="{epub_id}"/>
        <meta name="dtb:depth" content="1"/>
    </head>
    <docTitle>
        <text>{title}</text>
    </docTitle>
    <navMap>
        <navPoint id="chapter1" playOrder="1">
            <navLabel>
                <text>Chapter 1</text>
            </navLabel>
            <content src="chapter1.xhtml"/>
        </navPoint>
    </navMap>
</ncx>'''
            epub.writestr('OEBPS/toc.ncx', toc_ncx)
            
            # OEBPS/chapter1.xhtml
            # Convert paragraphs to HTML
            paragraphs = text.split('\n\n')
            html_paragraphs = ''.join([f'<p>{p.strip()}</p>\n' for p in paragraphs if p.strip()])
            
            chapter_xhtml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: serif; line-height: 1.6; margin: 2em; }}
        p {{ margin-bottom: 1em; text-indent: 1.5em; }}
        p:first-of-type {{ text-indent: 0; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {html_paragraphs}
</body>
</html>'''
            epub.writestr('OEBPS/chapter1.xhtml', chapter_xhtml)
        
        logger.app_logger.info(f"EPUB created: {epub_filename}")
        
        return send_file(
            epub_path,
            as_attachment=True,
            download_name=f'{title.replace(" ", "_")}.epub',
            mimetype='application/epub+zip'
        )
        
    except Exception as e:
        logger.app_logger.error(f"EPUB export error: {str(e)}")
        logger.app_logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/failed-translations', methods=['GET'])
@with_error_handling
def get_failed_translations():
    return jsonify(recovery.get_failed_translations())

@app.route('/retry-translation/<int:translation_id>', methods=['POST'])
@with_error_handling
def retry_failed_translation(translation_id):
    recovery.retry_translation(translation_id)
    return jsonify({'status': 'success'})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(monitor.get_metrics())

@app.route('/health', methods=['GET'])
def health_check():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('SELECT 1')
            
        disk_usage = psutil.disk_usage('/')
        if disk_usage.percent > 90:
            logger.app_logger.warning("Low disk space")
            
        return jsonify({
            'status': 'healthy',
            'ollama': 'connected',
            'database': 'connected',
            'disk_usage': f"{disk_usage.percent}%"
        })
    except Exception as e:
        logger.app_logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503
    
def cleanup_old_data():
    while True:
        try:
            # Housekeeping, at DEBUG: it runs once a day and says the same
            # three things every time, which is half of app.log and reads as
            # the only thing happening when the log is watched live.
            logger.app_logger.debug("Running cleanup task")
            try:
                cache.cleanup_old_entries()
                logger.app_logger.debug("Cache cleanup completed")
            except Exception as e:
                logger.app_logger.error(f"Cache cleanup error: {str(e)}")

            try:
                recovery.cleanup_failed_translations()
                logger.app_logger.debug("Failed translations cleanup completed")
            except Exception as e:
                logger.app_logger.error(f"Failed translations cleanup error: {str(e)}")

            time.sleep(24 * 60 * 60)  # Run daily
        except Exception as e:
            logger.app_logger.error(f"Cleanup task error: {str(e)}")
            time.sleep(60 * 60)  # Retry in an hour


if __name__ == "__main__":
    # The cleanup thread belongs to a running server, not to an import: every
    # test that imports this module was starting a daily-cleanup thread and
    # writing its three lines into the real app log.
    threading.Thread(target=cleanup_old_data, daemon=True).start()

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("Shutting down gracefully...")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print_terminal_banner()
    setup_access_log()

    # Start the Flask application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5001)),
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    )

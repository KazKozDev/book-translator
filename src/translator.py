import json
import requests
import time
from typing import List, Dict, Optional, Callable, Set, Tuple, Iterator, Any
import os
# COMET pins SentencePiece below 0.2. Its generated protobuf bindings require
# Python parsing mode; set this before any library can import SentencePiece.
# Setting it lazily during a request can load the same proto descriptor twice.
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import sqlite3
import traceback
import threading
import signal
import re
import sys
from io import BytesIO
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from queue import Empty, Queue

try:
    import sacrebleu
except ImportError:
    sacrebleu = None
from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

import prompts  # noqa: E402
from frontier_glossary import (  # noqa: E402
    FrontierGlossaryError,
    provider_catalog,
    verify_glossary,
)
from frontier_review import decide_review_cases  # noqa: E402
from languages import LANG_NAMES  # noqa: E402

# TranslateGemma (Gemma 3 based, 4B/12B/27B) is a translation-only model: it
# was trained on one fixed prompt shape and cannot follow editor or judge
# instructions. Stage 1 gets its native prompt format; Stage 2 (refinement)
# and the LLM-judge tests have to run on a general instruct model instead.
TRANSLATEGEMMA_TEMPERATURE = 0.3  # A dedicated MT model wants near-greedy decoding.

# The banner lives in its own stdlib-only module so the launcher — which runs
# before the virtual environment exists and cannot import this file — shows the
# same logo from the same source.
# TERMINAL_LOGO is re-exported, not used here: test_the_banner_has_one_source
# asserts both entry points resolve to the same object, so linters calling it
# an unused import are wrong.
from banner import TERMINAL_LOGO, print_terminal_banner  # noqa: E402,F401


def is_translategemma(model_name: Optional[str]) -> bool:
    return 'translategemma' in (model_name or '').lower()

# Folders setup. The runtime ones stay relative to the working directory —
# uploads, exports, logs and the two databases belong to the checkout the app
# was started from, not to the code. The front end is the exception: it ships
# beside this module in src/, so it is found from __file__ and not from
# wherever the process happens to be standing.
UPLOAD_FOLDER = 'uploads'
TRANSLATIONS_FOLDER = 'translations'
STATIC_FOLDER = str(Path(__file__).resolve().parent / 'static')
DB_PATH = 'translations.db'
CACHE_DB_PATH = 'cache.db'

# Logging and process metrics live in monitoring.py, and so do the two live
# instances — quality_tests.py needs the same logger, and a second module
# building its own would split the log in half. Imported rather than
# constructed here, so ``translator.logger`` still names the one object the
# rest of the app and the tests already reach for.
from monitoring import (  # noqa: E402
    LOG_FOLDER,
    logger,
    monitor,
    project_disk_usage,
    setup_access_log,
)

# Create necessary directories
for folder in [UPLOAD_FOLDER, TRANSLATIONS_FOLDER, STATIC_FOLDER, LOG_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# The chunk cache and the glossary rules are each their own module; only
# the one live cache instance stays here.
from translation_cache import TranslationCache  # noqa: E402
from terminology import GlossaryTerm, TerminologyManager  # noqa: E402,F401

cache = TranslationCache(CACHE_DB_PATH)

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

# Progress events for a running job. The HTTP SSE response only *reads* this
# queue — the work itself runs in a daemon thread — so closing the browser
# tab no longer raises GeneratorExit inside Stage 1 / Stage 2 and kills the
# overnight book mid-chunk.
_PROGRESS_QUEUES: Dict[int, Queue] = {}
_PROGRESS_QUEUES_LOCK = threading.Lock()
_PROGRESS_SENTINEL = object()


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


def _progress_queue(translation_id: int) -> Queue:
    with _PROGRESS_QUEUES_LOCK:
        queue = _PROGRESS_QUEUES.get(translation_id)
        if queue is None:
            queue = Queue()
            _PROGRESS_QUEUES[translation_id] = queue
        return queue


def _emit_progress(translation_id: int, event: Any) -> None:
    _progress_queue(translation_id).put(event)


def _clear_progress_queue(translation_id: int) -> None:
    with _PROGRESS_QUEUES_LOCK:
        _PROGRESS_QUEUES.pop(translation_id, None)


def _sse_from_progress_queue(translation_id: int) -> Iterator[str]:
    """Tail a job's progress queue until the worker posts the sentinel."""
    queue = _progress_queue(translation_id)
    while True:
        try:
            item = queue.get(timeout=20)
        except Empty:
            # Keep the connection warm; the worker may still be calling Ollama.
            yield ': heartbeat\n\n'
            continue
        if item is _PROGRESS_SENTINEL:
            break
        yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    if not is_run_active(translation_id):
        _clear_progress_queue(translation_id)


def _start_detached_job(translation_id: int, updates: Iterator[Dict]) -> None:
    """Run a stage generator in a daemon thread; SSE clients only observe it."""

    def runner():
        try:
            for update in updates:
                if isinstance(update, dict):
                    update.setdefault('translation_id', translation_id)
                _emit_progress(translation_id, update)
        except Exception as e:
            logger.translation_logger.error(
                "Detached job %s failed: %s", translation_id, e,
            )
            logger.translation_logger.error(traceback.format_exc())
            _emit_progress(translation_id, {
                'error': str(e),
                'translation_id': translation_id,
            })
        finally:
            _emit_progress(translation_id, _PROGRESS_SENTINEL)

    threading.Thread(
        target=runner,
        name=f'tolmach-job-{translation_id}',
        daemon=True,
    ).start()


def _effective_status(translation_id: int, status: str) -> str:
    """Map abandoned in_progress rows to interrupted for the UI."""
    if status == 'in_progress' and not is_run_active(translation_id):
        return 'interrupted'
    return status


def _heal_orphaned_runs() -> None:
    """Flip leftover in_progress rows to interrupted when nothing is streaming them."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT id FROM translations WHERE status = 'in_progress'"
            ).fetchall()
            for (translation_id,) in rows:
                if not is_run_active(translation_id):
                    conn.execute(
                        '''UPDATE translations
                           SET status = 'interrupted',
                               error_message = COALESCE(
                                   error_message,
                                   'Interrupted — press Resume to continue from the last finished chunk.'
                               ),
                               updated_at = CURRENT_TIMESTAMP
                           WHERE id = ? AND status = 'in_progress' ''',
                        (translation_id,),
                    )
    except Exception as e:
        logger.translation_logger.warning("Could not heal orphaned runs: %s", e)


# Part of the Stage 2 cache key, so that changing what the refinement pass
# DOES invalidates results produced by the old behaviour. The inputs alone are
# not enough: same chunk, same glossary, same model — and a different answer,
# because the pass itself changed. Bump this whenever the estimate/patch/verify
# behaviour changes in a way that would alter output.
#   v2: single "review and improve" rewrite replaced by estimate/patch/verify
#   v3: style-only and minor subjective errors reported but no longer applied
#   v4: omission/addition patches skip the verifier, which now runs on its own
#       model rather than on the one that wrote the draft
#   v5: tie/position-biased votes get a position-free edit check, and the
#       verifier identity is part of the cache key
STAGE2_PIPELINE_VERSION = 'v5'


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
                chunk_chapter_map TEXT,
                document_fingerprint TEXT
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

            -- Stage 0 finishes before a translation row exists.  Keep its
            -- editable draft in SQLite anyway, scoped to the exact document
            -- and language pair, so a new book never inherits the last
            -- book's terminology from browser-wide storage.
            CREATE TABLE IF NOT EXISTS workspace_glossaries (
                document_fingerprint TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                glossary TEXT NOT NULL DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (document_fingerprint, source_lang, target_lang)
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

            CREATE TABLE IF NOT EXISTS chunk_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                translation_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                review_details TEXT,
                warning TEXT,
                alternatives TEXT,
                judge_model TEXT,
                review_status TEXT NOT NULL DEFAULT 'open',
                resolution_kind TEXT,
                selected_candidate TEXT,
                revision INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (translation_id, chunk_index),
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
            # Which book this job was started from, so reopening the job can
            # rebind its editable glossary draft in workspace_glossaries
            # instead of showing an empty editor.
            ('document_fingerprint', 'ALTER TABLE translations ADD COLUMN document_fingerprint TEXT'),
        ):
            if column not in existing_columns:
                conn.execute(ddl)
        # Remove the obsolete per-document context column from existing job
        # history as well as from the new-table schema above.
        if 'doc_summary' in existing_columns:
            conn.execute('ALTER TABLE translations DROP COLUMN doc_summary')
        review_columns = {
            row[1] for row in conn.execute('PRAGMA table_info(chunk_reviews)')
        }
        if 'revision' not in review_columns:
            conn.execute(
                'ALTER TABLE chunk_reviews ADD COLUMN revision INTEGER NOT NULL DEFAULT 0'
            )

init_db()


def save_chunk_review(
    translation_id: int,
    chunk_index: int,
    *,
    details: Optional[Dict] = None,
    warning: Optional[str] = None,
):
    """Persist what Stage 2 found for one chunk.

    The final text already lives in ``translations.final_chunks``. This table
    holds only the editorial evidence around it, so the review desk can show
    the exact reported spans without duplicating the book text.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            '''
            INSERT INTO chunk_reviews (
                translation_id, chunk_index, review_details, warning,
                review_status, updated_at
            ) VALUES (?, ?, ?, ?, 'open', CURRENT_TIMESTAMP)
            ON CONFLICT (translation_id, chunk_index) DO UPDATE SET
                review_details = excluded.review_details,
                warning = excluded.warning,
                review_status = 'open',
                resolution_kind = NULL,
                selected_candidate = NULL,
                revision = chunk_reviews.revision + 1,
                updated_at = CURRENT_TIMESTAMP
            ''',
            (
                translation_id,
                chunk_index,
                json.dumps(details or {}, ensure_ascii=False),
                warning,
            ),
        )


from quality_tests import QualityTests  # noqa: E402

# EPUB in and out lives in its own module: it depends on nothing in here.
from epub_io import (  # noqa: E402
    is_epub_filename,
    extract_epub_book,
    build_epub_from_chapters,
)

# PDF and DOCX are read-only and for the same reason stand apart: they produce
# plain text (and optional chapters) and nothing downstream knows the container.
from pdf_io import is_pdf_filename, extract_pdf_book  # noqa: E402
from docx_io import is_docx_filename, extract_docx_book  # noqa: E402


# The Stage 3 quality tests are the other half of this class, kept in
# quality_tests.py: a thousand lines of optional, on-demand diagnostics that
# were burying the pipeline they diagnose. Inherited rather than delegated to,
# so every call site — inside the class and in the /evaluate route — reads
# exactly as it did when the methods were defined below.
class BookTranslator(QualityTests):
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
        # A/B verdict it gives flips with the order the two versions are shown.
        # That is now detected and sent through a position-free edit check,
        # but an independent, capable verifier is still the reliable setup.
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
    # How many of those calls are in flight at once. Ollama serves concurrent
    # requests from one loaded model when OLLAMA_NUM_PARALLEL allows it, and
    # queues them when it does not, so this is a ceiling rather than a
    # promise. Kept small on purpose: a request that waits its turn is still
    # counting against PREPARE_READ_TIMEOUT, and four slow calls queued behind
    # each other stay well inside it.
    PREPARE_CONCURRENCY = max(1, int(os.getenv('PREPARE_CONCURRENCY', '4')))
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

    @classmethod
    def _parse_json_array(cls, raw: Optional[str]) -> List[Dict]:
        """Pull the objects a model answered with out of its reply.

        Local instruct models wrap JSON in prose or fences more often than
        not, so the outermost bracket pair is extracted rather than trusting
        the whole response to parse. Anything unparseable yields [] — every
        caller here treats "no structured answer" as "nothing to apply",
        never as an error worth failing the pass over.

        Asking for an array is not the same as getting one. A model that
        answers with the objects one after another and no brackets around
        them has answered the question correctly, and reading only the
        bracketed form threw whole batches of good renderings away as "0 of 8
        accepted" — the answers were complete, well-formed and simply not
        wrapped. So a reply with no usable array is read object by object,
        which also rescues an array that was cut off before its closing
        bracket.
        """
        if not raw:
            return []
        start, end = raw.find('['), raw.rfind(']')
        if start != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                items = [item for item in parsed if isinstance(item, dict)]
                if items:
                    return items
        return cls._scan_json_objects(raw)

    @staticmethod
    def _scan_json_objects(raw: str) -> List[Dict]:
        """Every complete JSON object in a string, in the order written.

        Decodes from each opening brace and skips past whatever it decoded,
        so nested objects are consumed with their parent rather than reported
        twice, and prose between objects is stepped over.
        """
        decoder = json.JSONDecoder()
        items: List[Dict] = []
        index = raw.find('{')
        while index != -1:
            try:
                value, offset = decoder.raw_decode(raw, index)
            except ValueError:
                index = raw.find('{', index + 1)
                continue
            if isinstance(value, dict):
                items.append(value)
                index = raw.find('{', offset)
            else:
                index = raw.find('{', index + 1)
        return items

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
            import build_glossary
            from build_glossary import build_document_glossary
        except ImportError as exc:
            raise RuntimeError(
                'The glossary builder dependencies are missing. Install them with '
                'pip install -r requirements.txt.'
            ) from exc

        # Model loading and the per-batch NER sweep are the longest silences in
        # Stage 0. They report to stderr for the command line; running under the
        # server they belong in the log the interface is following.
        build_glossary.report = logger.translation_logger.info

        logger.translation_logger.info(
            f"Stage 0: extracting glossary candidates from {len(text):,} characters"
        )
        entries, review_queue = build_document_glossary(text)
        logger.translation_logger.info(
            f"Stage 0: extraction found {len(entries)} clustered candidate(s), "
            f"{len(review_queue)} pair(s) for review"
        )
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
        # One pass over the document instead of one per candidate: the check
        # below asks the same question of a few thousand candidates, and a
        # regex scan of a whole book each time is most of what Stage 0 spends
        # before it reaches a model at all.
        letter_runs = cls._letter_runs(text)
        for record in list(candidates) + harvested:
            surface = str(record.get('surface') or '').strip()
            if not surface or not cls.is_glossary_source(text, surface):
                continue
            # GLiNER labels the pronoun "she" a person, 13 mentions and all,
            # and a single-word candidate the document also writes in
            # lowercase is a common word wearing a capital at the start of a
            # sentence. This is the harvest's own discriminator, applied to
            # the neural list, which had none.
            if ' ' not in surface and cls._written_lowercase(text, surface, letter_runs):
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

    # A maximal run of letters: what a word looks like to the boundary
    # assertions the lowercase check used to spell out inline.
    _LETTER_RUN_RE = re.compile(r'[^\W\d_]+', re.UNICODE)

    @classmethod
    def _letter_runs(cls, text: str) -> frozenset:
        """Every maximal run of letters in the document, exactly as written."""
        return frozenset(cls._LETTER_RUN_RE.findall(text))

    @classmethod
    def _written_lowercase(cls, text: str, surface: str, letter_runs: frozenset) -> bool:
        """Whether the document also writes this term in lowercase.

        An occurrence with no letter on either side is precisely a maximal
        letter run, so for an all-letter term the answer is a set lookup
        rather than a scan of the book. Terms with an apostrophe or a hyphen
        span more than one run and keep the scan.
        """
        lowered = surface.lower()
        if cls._LETTER_RUN_RE.fullmatch(lowered):
            return lowered in letter_runs
        return bool(re.search(
            rf'(?<![^\W\d_]){re.escape(lowered)}(?![^\W\d_])', text,
        ))

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
            logger.translation_logger.info(
                'Stage 0: nothing ambiguous to adjudicate — no entity-resolution call'
            )
            return candidates, []
        logger.translation_logger.info(
            f"Stage 0: adjudicating {len(groups)} entity group(s) with {self.model_name}"
        )

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
        prompt = prompts.render(
            'stage0_prepare/cluster_adjudication',
            source_name=source_name, listing=chr(10).join(listing),
        )

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
        confirmed = sum(1 for decision in decisions if decision['same_entity'])
        logger.translation_logger.info(
            f"Stage 0: entity resolution ruled on {len(decisions)}/{len(groups)} group(s) — "
            f"{confirmed} confirmed, {len(decisions) - confirmed} split"
        )
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
        workers = min(self.PREPARE_CONCURRENCY, len(batches))
        logger.translation_logger.info(
            f"Stage 0: rendering {len(candidates)} candidate(s) into "
            f"{LANG_NAMES.get(target_lang, target_lang)} in {len(batches)} batch(es)"
            + (f", {workers} at a time" if workers > 1 else "")
        )
        def render(batch: List[Dict]) -> Optional[str]:
            return self._call_model(
                self._rendering_prompt(text, batch, source_lang, target_lang, genre),
                temperature=0.2, read_timeout=self.PREPARE_READ_TIMEOUT,
            )

        # Nothing in a batch depends on another batch's answer: each is its own
        # list of terms with its own quoted context. So the calls overlap —
        # which is essentially all of Stage 0's wall clock on a book, fifteen
        # sequential calls to a large local model. map() submits them all and
        # still yields in batch order, so the merge below sees exactly the
        # sequence it always saw: same prompts, same sampling, same order, and
        # the same line in the log as each batch's turn comes up.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for number, (batch, raw) in enumerate(zip(batches, pool.map(render, batches)), 1):
                accepted = 0
                for item in self._parse_json_array(raw):
                    record = self._rendering_record(text, item, counts)
                    if record is None or record['source'].casefold() in seen:
                        continue
                    seen.add(record['source'].casefold())
                    records.append(record)
                    accepted += 1
                # A source form the document does not literally contain is
                # dropped by _rendering_record, so batch size and accepted
                # count differ on purpose — worth seeing rather than inferring
                # from a final total.
                logger.translation_logger.info(
                    f"Stage 0: rendering batch {number}/{len(batches)} — "
                    f"{accepted} of {len(batch)} term(s) accepted"
                )
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
        genre_line = ""
        if genre and genre != 'unknown':
            genre_line = "\n" + prompts.render(
                'stage0_prepare/rendering_rules', 'genre_line', genre=genre,
            )

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
            task = prompts.render(
                'stage0_prepare/rendering_from_candidates',
                source_name=source_name, target_name=target_name,
                genre_line=genre_line, listing=chr(10).join(listing),
            )
        else:
            # No candidate survived extraction, which is the normal case for a
            # script that carries no capitalisation signal. The excerpt is all
            # there is to go on.
            task = prompts.render(
                'stage0_prepare/rendering_from_excerpt',
                source_name=source_name, target_name=target_name,
                genre_line=genre_line, excerpt=text[:self.PREPARE_SOURCE_BUDGET],
            )

        rules = prompts.render(
            'stage0_prepare/rendering_rules',
            source_name=source_name, target_name=target_name,
        )
        return f"{task}\n\n{rules}"

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
        resume: bool = False,
    ):
        """STAGE 1 only: primary draft translation. Persists the draft chunks
        so a later, independent translate_stage2() call can refine them
        without re-translating from scratch.

        When resume=True, continues from draft_chunks already saved on the row
        (overnight interrupt / server restart) instead of starting over.
        """
        start_time = time.time()
        success = False

        try:
            self.terminology = terminology or TerminologyManager()
            glossary_fingerprint = self.terminology.fingerprint()
            used_terms = set()
            stage1_violation_count = 0

            if resume:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.row_factory = sqlite3.Row
                    row = conn.execute(
                        "SELECT original_chunks, draft_chunks, chunk_chapter_map, "
                        "original_text, genre FROM translations WHERE id = ?",
                        (translation_id,),
                    ).fetchone()
                if not row or not row['original_chunks']:
                    raise ValueError(
                        'Nothing to resume — original chunks were never saved. Press Start again.'
                    )
                chunks = json.loads(row['original_chunks'])
                draft_translations = json.loads(row['draft_chunks'] or '[]')
                chunk_chapter_map = json.loads(row['chunk_chapter_map'] or '[]')
                if len(chunk_chapter_map) != len(chunks):
                    chunk_chapter_map = [0] * len(chunks)
                if not text:
                    text = row['original_text'] or ''
                if row['genre']:
                    genre = row['genre']
                start_index = len(draft_translations)
                if start_index >= len(chunks):
                    raise ValueError('Draft is already complete — press Continue to refine.')
            else:
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
                draft_translations = []
                start_index = 0

            total_chunks = len(chunks)

            logger.translation_logger.info(
                "Starting stage 1 for translation %s with %s chunks (genre: %s, resume=%s, from=%s)",
                translation_id, total_chunks, genre, resume, start_index + 1,
            )

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "UPDATE translations "
                    "SET total_chunks = ?, status = 'in_progress', genre = ?, "
                    "original_chunks = ?, chunk_chapter_map = ?, "
                    "draft_chunks = ?, error_message = NULL, "
                    "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (
                        total_chunks,
                        genre,
                        json.dumps(chunks, ensure_ascii=False),
                        json.dumps(chunk_chapter_map),
                        json.dumps(draft_translations, ensure_ascii=False),
                        translation_id,
                    ),
                )
            claim_run(translation_id)

            # STAGE 1: Primary translation with context
            logger.translation_logger.info("Stage 1: Primary LLM translation")
            for i in range(start_index, total_chunks):
                chunk = chunks[i]
                batch_index = i + 1
                try:
                    if not chunk.strip():
                        # Empty chapter (e.g. a title page) — nothing to send to the model.
                        draft_translations.append('')
                        progress = (batch_index / total_chunks) * 100
                        with sqlite3.connect(DB_PATH) as conn:
                            conn.execute(
                                "UPDATE translations "
                                "SET progress = ?, machine_translation = ?, draft_chunks = ?, "
                                "current_chunk = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                                (
                                    progress,
                                    '\n\n'.join(draft_translations),
                                    json.dumps(draft_translations, ensure_ascii=False),
                                    batch_index,
                                    translation_id,
                                ),
                            )
                        yield {
                            'progress': progress,
                            'stage': 'primary_translation',
                            'batch_index': batch_index,
                            'original_chunk': chunk,
                            'machine_translation_chunk': '',
                            'current_chunk': batch_index,
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
                        logger.translation_logger.info(f"Cache hit for stage 1 chunk {batch_index}")
                    else:
                        # Get previous context
                        previous_chunk = draft_translations[-1] if draft_translations else ""

                        logger.translation_logger.info(f"Stage 1 translating chunk {batch_index}/{total_chunks}")
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
                            sum(item['count'] for item in exact_replacements), batch_index,
                        )

                    draft_translations.append(draft_translation)
                    terminology_violations = self.terminology.exact_violations(
                        chunk, draft_translation
                    )
                    stage1_violation_count += len(terminology_violations)

                    progress = (batch_index / total_chunks) * 100
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute(
                            "UPDATE translations "
                            "SET progress = ?, machine_translation = ?, draft_chunks = ?, "
                            "current_chunk = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                            (
                                progress,
                                '\n\n'.join(draft_translations),
                                json.dumps(draft_translations, ensure_ascii=False),
                                batch_index,
                                translation_id,
                            ),
                        )
                    yield {
                        'progress': progress,
                        'stage': 'primary_translation',
                        'batch_index': batch_index,
                        'original_chunk': chunk,
                        'machine_translation_chunk': draft_translation,
                        'current_chunk': batch_index,
                        'total_chunks': total_chunks,
                        'warning': stage1_warning,
                        'terminology': {
                            'total': len(self.terminology.terms),
                            'used': len(used_terms),
                            'violations': stage1_violation_count,
                        },
                    }

                except Exception as e:
                    error_msg = f"Error in stage 1 chunk {batch_index}: {str(e)}"
                    logger.translation_logger.error(error_msg)
                    logger.translation_logger.error(traceback.format_exc())
                    raise Exception(error_msg)

            # Persist the draft so translate_stage2() can pick it up later,
            # independently of this request/generator.
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "UPDATE translations "
                    "SET status = 'stage1_completed', progress = 100, "
                    "original_chunks = ?, draft_chunks = ?, chunk_chapter_map = ?, "
                    "error_message = NULL, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (
                        json.dumps(chunks, ensure_ascii=False),
                        json.dumps(draft_translations, ensure_ascii=False),
                        json.dumps(chunk_chapter_map),
                        translation_id,
                    ),
                )

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
            # Only reached if a consumer stops iterating the generator early.
            # Detached jobs normally run to completion in their own thread;
            # this path still leaves a resumable partial draft behind.
            self._abandon_run(
                translation_id, 'interrupted',
                'Interrupted — press Resume to continue from the last finished chunk.',
            )
            raise
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            logger.translation_logger.error(error_msg)
            logger.translation_logger.error(traceback.format_exc())

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "UPDATE translations "
                    "SET status = 'interrupted', error_message = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (
                        f'{e} — press Resume to continue from the last finished chunk.',
                        translation_id,
                    ),
                )
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
            position_biases = neutral_checks = 0
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
                    stage2_cache_model = self._stage2_cache_model(
                        glossary_fingerprint
                    )
                    # Check cache
                    cached_result = cache.get_cached_translation(
                        original_chunk, source_lang, target_lang, stage2_cache_model
                    )
                    stage2_warning = None
                    stage2_details = {}
                    if cached_result:
                        final_translation = cached_result['translated_text']
                        stage2_details = {
                            'cache_hit': True,
                            'evidence_available': False,
                            'review_not_replayed': True,
                            'issues': [],
                        }
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
                        if (
                            isinstance(verified, dict)
                            and verified.get('position_bias_detected')
                        ):
                            position_biases += 1
                        if (
                            isinstance(verified, dict)
                            and verified.get('neutral_check')
                        ):
                            neutral_checks += 1
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
                    stage2_details['exact_replacements'] = exact_replacements
                    stage2_details['terminology_violations'] = terminology_violations
                    save_chunk_review(
                        translation_id,
                        i - 1,
                        details=stage2_details,
                        warning=stage2_warning,
                    )

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
                            'position_biases': position_biases,
                            'neutral_checks': neutral_checks,
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
                    'position_biases': position_biases,
                    'neutral_checks': neutral_checks,
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
                "verifier %s, %s position bias event(s), %s neutral edit "
                "check(s), %s review call(s) gave no answer",
                translation_id, chunks_changed, chunks_reviewed, errors_found,
                errors_applied, patches_rejected, self.verifier_model,
                position_biases, neutral_checks, review_failures,
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
        separate 'thinking' field and any inline reasoning.

        Both ways an answer can be spoiled are reported here, because neither
        is visible anywhere else: a caller receives None and cannot tell
        whether the model found nothing or never got to say what it found.
        """
        raw = result.get('response') or ''
        thinking = result.get('thinking') or ''
        if thinking or cls._REASONING_BLOCK_RE.search(raw) or '<think' in raw.lower():
            logger.api_logger.warning(
                f"Model reasoned despite think=false: {len(thinking):,} chars in the "
                f"thinking field, {len(raw):,} in the answer. The trace is dropped, but "
                f"it was generated — and paid for — before the answer."
            )
        # Ollama stops for 'length' when the answer ran into num_ctx. Whatever
        # came back is the first half of a sentence or of a JSON array, and
        # every parser downstream reads that as "nothing".
        if result.get('done_reason') == 'length':
            logger.api_logger.warning(
                f"Model answer was cut off at the context limit after "
                f"{result.get('eval_count') or '?'} tokens — the reply is incomplete."
            )
        text = cls._strip_reasoning(raw)
        if raw and not text:
            logger.api_logger.warning(
                'Model answer was entirely reasoning — nothing left after stripping it.'
            )
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
        context_section = ""
        if previous_chunk:
            context_section = "\n\n" + prompts.render(
                'stage1_translate/default', 'previous_paragraph',
                previous_chunk=previous_chunk,
            )

        return prompts.render(
            'stage1_translate/default',
            source_name=source_name, target_name=target_name, genre=genre,
            context_section=context_section, terminology_context=terminology_context,
            text=text,
        )

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
        opening = prompts.render(
            'stage1_translate/translategemma', 'opening',
            source_name=source_name, source_lang=source_lang,
            target_name=target_name, target_lang=target_lang,
        )
        produce_only = prompts.render(
            'stage1_translate/translategemma', 'produce_only', target_name=target_name,
        )
        instruction = prompts.render(
            'stage1_translate/translategemma', 'instruction',
            source_name=source_name, target_name=target_name,
        )

        extras = []
        if previous_chunk:
            extras.append(prompts.render(
                'stage1_translate/translategemma', 'previous_paragraph',
                previous_chunk=previous_chunk,
            ))
        terminology_context = terminology_context.strip()
        if terminology_context:
            extras.append(terminology_context)

        if extras or (genre and genre != 'unknown'):
            document_type = ""
            if genre and genre != 'unknown':
                document_type = "\n" + prompts.render(
                    'stage1_translate/translategemma', 'document_type', genre=genre,
                )
            extras.insert(0, prompts.render(
                'stage1_translate/translategemma', 'context',
                document_type=document_type,
            ))

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
            logger.api_logger.error("Stage 1 timeout after 300s - text too long or model too slow")
            return text, 'Model timed out after 5 minutes (likely still loading/swapping) — kept the original text untranslated for this chunk.'
        except Exception as e:
            logger.api_logger.error(f"Stage 1 error: {e}")
            return text, f'Model request failed ({e}) — kept the original text untranslated for this chunk.'

    def generate_translation_candidate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        *,
        previous_chunk: str = "",
        genre: str = "unknown",
        terminology_context: str = "",
        temperature: float = 0.6,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate one deliberately non-cached candidate for the review desk.

        This is opt-in and chunk-scoped. It reuses the exact Stage 1 prompt
        contract, but accepts a caller-selected sampling temperature so two
        requested alternatives are capable of being genuinely different.
        """
        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)
        if is_translategemma(self.model_name):
            prompt = self._stage1_prompt_translategemma(
                text, source_lang, target_lang, source_name, target_name,
                previous_chunk, genre, terminology_context,
            )
        else:
            prompt = self._stage1_prompt_default(
                text, source_name, target_name, previous_chunk, genre,
                terminology_context,
            )
        translated = self._call_model(
            prompt, temperature=temperature, read_timeout=300,
        )
        if translated:
            return translated, None
        return None, 'The translation model returned no candidate.'

    @staticmethod
    def _parse_json_object(raw: Optional[str]) -> Dict:
        """Read the first JSON object from a possibly prose-wrapped reply."""
        if not raw:
            return {}
        raw = raw.strip()
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            pass
        decoder = json.JSONDecoder()
        for start, character in enumerate(raw):
            if character != '{':
                continue
            try:
                parsed, _ = decoder.raw_decode(raw[start:])
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}

    def judge_translation_candidates(
        self,
        original_text: str,
        candidates: List[str],
        source_lang: str,
        target_lang: str,
    ) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """Ask an independent model to rank candidates for one chunk.

        Returns a zero-based candidate index, its short reason, and an error.
        An unusable verdict is exposed as an error; it never becomes a made-up
        recommendation.
        """
        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)
        candidate_text = '\n\n'.join(
            f'CANDIDATE {index}\n{text}'
            for index, text in enumerate(candidates, 1)
        )
        prompt = prompts.render(
            'quality/candidate_judge',
            source_name=source_name,
            target_name=target_name,
            original_text=original_text,
            candidates=candidate_text,
            candidate_count=len(candidates),
        )
        raw = self._call_model(prompt, temperature=0.0, read_timeout=600)
        verdict = self._parse_json_object(raw)
        try:
            best = int(verdict.get('best')) - 1
        except (TypeError, ValueError):
            best = -1
        if best not in range(len(candidates)):
            return None, None, 'The judge returned no usable ranking.'
        reason = str(verdict.get('reason') or '').strip() or None
        return best, reason, None

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
        return prompts.render(
            'stage2_refine/estimate',
            source_name=source_name, target_name=target_name,
            original_text=original_text, draft_translation=draft_translation,
            terminology_context=terminology_context, violation_section=violation_section,
            max_spans=self.MAX_ESTIMATE_SPANS,
        )

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
            violation_section = "\n\n" + prompts.render(
                'stage2_refine/estimate', 'terminology_violations', missing=missing,
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
        applied_errors: Optional[List[Dict]] = None,
    ) -> Tuple[bool, Dict]:
        """STAGE 2c: did the patch actually improve the translation?

        The judge sees the source and is asked about accuracy — not about
        which version reads better, which is the question that let the old
        pass congratulate itself while drifting from the original. Asked
        twice with the two versions swapped, because a single ordering
        measures position bias as much as quality. A patch that wins both is
        kept; a tie or a vote that follows A/B position gets one final check
        phrased only as concrete replacements, with no ordered versions.

        Runs on ``self.verifier``, which is a separate model whenever one was
        chosen. Both halves of the vote on the model that produced the edit is
        self-assessment, and on a quantised 12B it answers by position rather
        than by content, so the two orderings disagree. That disagreement is
        evidence of position bias, not evidence that the draft is better, and
        now routes to the position-free edit check instead of an automatic
        veto.
        """
        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)
        verifier = self.verifier
        verdicts = []

        for patched_is_a in (True, False):
            version_a, version_b = (after, before) if patched_is_a else (before, after)
            prompt = prompts.render(
                'stage2_refine/verify',
                source_name=source_name, target_name=target_name,
                original_text=original_text, version_a=version_a, version_b=version_b,
            )
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
        position_bias_detected = verdicts in (
            ['patched', 'draft'],
            ['draft', 'patched'],
        )
        tie_detected = 'tie' in verdicts
        neutral_check = None
        if position_bias_detected or tie_detected:
            neutral_check = self.stage2_verify_edits(
                original_text=original_text,
                draft_translation=before,
                applied_errors=applied_errors or [],
                source_lang=source_lang,
                target_lang=target_lang,
            )
            accepted = neutral_check == 'accepted'

        details = {
            'verdicts': verdicts,
            'accepted': accepted,
            'model': verifier.model_name,
        }
        if position_bias_detected:
            details['position_bias_detected'] = True
        if tie_detected:
            details['tie_detected'] = True
        if neutral_check:
            details['neutral_check'] = neutral_check
        return accepted, details

    def stage2_verify_edits(
        self,
        original_text: str,
        draft_translation: str,
        applied_errors: List[Dict],
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Resolve an inconclusive A/B vote without showing ordered versions.

        A model that always chooses VERSION A produces opposite content
        verdicts when the versions are swapped. Asking about the concrete
        replacements instead removes that position from the decision. The
        whole patch still has to pass: one rejected or uncertain edit keeps
        the draft.
        """
        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)
        edits = '\n'.join(
            '{}. [{}] {} => {}'.format(
                index,
                error.get('type') or 'other',
                json.dumps(error.get('span') or '', ensure_ascii=False),
                json.dumps(error.get('replacement') or '', ensure_ascii=False),
            )
            for index, error in enumerate(applied_errors, 1)
        )
        if not edits:
            return 'unavailable'

        prompt = prompts.render(
            'stage2_refine/verify_edits',
            source_name=source_name,
            target_name=target_name,
            original_text=original_text,
            draft_translation=draft_translation,
            edits=edits,
        )
        raw = self.verifier._call_model(
            prompt, temperature=0.0, read_timeout=self.VERIFY_READ_TIMEOUT,
        )
        if raw is None:
            return 'unavailable'
        verdict = raw.strip().upper()
        if verdict == 'ACCEPT':
            return 'accepted'
        if verdict == 'REJECT':
            return 'rejected'
        return 'unavailable'

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
            vote = 'verifier {} voted {}'.format(
                verified.get('model') or 'unknown',
                '/'.join(verified.get('verdicts') or ['no verdict']),
            )
            if verified.get('position_bias_detected'):
                parts.extend([vote, 'position bias detected'])
            elif verified.get('tie_detected'):
                parts.extend([vote, 'tie detected'])
            else:
                parts.append('{} → {}'.format(
                    vote,
                    'accepted' if verified.get('accepted') else 'rejected',
                ))
            neutral_check = verified.get('neutral_check')
            if neutral_check:
                parts.append('neutral edit check {} → {}'.format(
                    neutral_check,
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
            'evidence_available': True,
            'errors_found': len(errors),
            'errors_actionable': len(actionable),
            'errors_applied': 0,
            # Keep the actual located spans, not just aggregate counts. The
            # review desk needs to explain why a chunk was flagged and let a
            # human decide whether the proposed replacement is right.
            'issues': errors,
            'actionable_issues': actionable,
            'applied_issues': [],
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
        details['applied_issues'] = applied
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
            applied_errors=applied,
        )
        details['verified'] = verdict
        if not accepted:
            return draft_translation, None, details
        return patched, None, details

    def _stage2_cache_model(self, glossary_fingerprint: str) -> str:
        """Every model-dependent input that can change a Stage 2 result."""
        return (
            f"{self.model_name}_verifier_{self.verifier_model}"
            f"_stage2{STAGE2_PIPELINE_VERSION}"
            f"_glossary_{glossary_fingerprint}"
        )


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


# Translation Recovery
def _load_failed_translations() -> List[Dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute('''
            SELECT * FROM translations
            WHERE status = 'error'
            ORDER BY created_at DESC
        ''')
        return [dict(row) for row in cur.fetchall()]


def _retry_failed_translation(translation_id: int):
    with sqlite3.connect(DB_PATH) as conn:
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


def _cleanup_failed_translations(days: int = 7):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM translations WHERE status = 'error' "
            "AND created_at < datetime('now', ?)",
            (f'-{int(days)} days',),
        )

# Health checking middleware
@app.before_request
def check_ollama():
    # Managing locally saved tasks must remain possible even when Ollama is
    # stopped, so a user can clear old or failed translations.
    exempt_endpoints = {
        'health_check', 'serve_frontend', 'serve_static', 'delete_translation',
        'get_translation', 'get_translations', 'get_failed_translations',
        'get_review_chunks',
        'update_review_chunk', 'download_translation',
        'stream_translation_progress',
        'source_preview', 'get_workspace_glossary', 'save_workspace_glossary',
        'get_glossary_verification_prompt', 'get_frontier_providers',
        'verify_glossary_with_frontier',
        'decide_review_chunk_with_frontier',
        'decide_all_review_chunks_with_frontier',
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


@app.route('/glossary-verification-prompt', methods=['POST'])
@with_error_handling
def get_glossary_verification_prompt():
    """Build the manual frontier-model prompt from the visible glossary.

    This route deliberately does not call a model. The user gets the complete
    prompt in the clipboard and can choose any frontier model outside Tolmach.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get('glossary'), str):
        return jsonify({'error': 'Glossary text is required'}), 400

    source_code = data.get('sourceLanguage')
    target_code = data.get('targetLanguage')
    if not isinstance(source_code, str) or not isinstance(target_code, str):
        return jsonify({'error': 'Source and target languages are required'}), 400
    source_language = LANG_NAMES.get(source_code, source_code).strip()
    target_language = LANG_NAMES.get(target_code, target_code).strip()
    if not source_language or not target_language:
        return jsonify({'error': 'Source and target languages are required'}), 400

    entities = '\n'.join(
        line.strip()
        for line in data['glossary'].splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    )
    if not entities:
        return jsonify({'error': 'Add at least one glossary entry first'}), 400

    return jsonify({
        'prompt': _render_glossary_verification_prompt(
            source_language,
            target_language,
            entities,
        ),
    })


def _render_glossary_verification_prompt(
    source_language: str,
    target_language: str,
    entities: str,
) -> str:
    return prompts.render(
        'manual/glossary_verification',
        source_language=source_language,
        target_language=target_language,
        entities=entities,
    )


@app.route('/frontier-providers', methods=['GET'])
def get_frontier_providers():
    response = jsonify({'providers': provider_catalog()})
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/verify-glossary-frontier', methods=['POST'])
def verify_glossary_with_frontier():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Verification request is required'}), 400

    glossary = data.get('glossary')
    source_code = data.get('sourceLanguage')
    target_code = data.get('targetLanguage')
    provider = data.get('provider')
    submitted_key = data.get('apiKey')
    submitted_model = data.get('model')
    if not isinstance(glossary, str) or not glossary.strip():
        return jsonify({'error': 'Add at least one glossary entry first'}), 400
    if not isinstance(source_code, str) or not isinstance(target_code, str):
        return jsonify({'error': 'Source and target languages are required'}), 400
    if not isinstance(provider, str):
        return jsonify({'error': 'Choose a frontier provider in Settings'}), 400
    if submitted_key is not None and not isinstance(submitted_key, str):
        return jsonify({'error': 'API key must be text'}), 400
    if submitted_model is not None and not isinstance(submitted_model, str):
        return jsonify({'error': 'Model name must be text'}), 400

    source_language = LANG_NAMES.get(source_code, source_code).strip()
    target_language = LANG_NAMES.get(target_code, target_code).strip()
    if not source_language or not target_language:
        return jsonify({'error': 'Source and target languages are required'}), 400
    entities = '\n'.join(
        line.strip()
        for line in glossary.splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    )
    prompt = _render_glossary_verification_prompt(
        source_language,
        target_language,
        entities,
    )
    try:
        result = verify_glossary(
            provider,
            prompt,
            glossary,
            submitted_key,
            submitted_model,
        )
    except FrontierGlossaryError as error:
        return jsonify({'error': str(error)}), error.status_code
    except Exception:
        logger.app_logger.error(
            'Unexpected frontier glossary verification error\n'
            + traceback.format_exc()
        )
        return jsonify({'error': 'Frontier verification failed unexpectedly'}), 502

    response = jsonify({
        'glossary': result.glossary,
        'provider': result.provider,
        'provider_label': result.provider_label,
        'model': result.model,
        'changes': result.changes,
        'searched': result.searched,
    })
    response.headers['Cache-Control'] = 'no-store'
    return response


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


WORKSPACE_GLOSSARY_FINGERPRINT = re.compile(r'^[a-f0-9]{64}$')
MAX_WORKSPACE_GLOSSARY_LENGTH = 200_000


def _workspace_glossary_context(document_fingerprint: str, source_lang: Optional[str], target_lang: Optional[str]):
    """Validate the per-document identity used for an editable Stage 0 draft."""
    if not WORKSPACE_GLOSSARY_FINGERPRINT.fullmatch(document_fingerprint or ''):
        return None, ('Invalid document fingerprint', 400)
    if not isinstance(source_lang, str) or not source_lang.strip():
        return None, ('Source language is required', 400)
    if not isinstance(target_lang, str) or not target_lang.strip():
        return None, ('Target language is required', 400)
    return (
        document_fingerprint,
        source_lang.strip(),
        target_lang.strip(),
    ), None


def _store_workspace_glossary(conn, context, glossary: str) -> None:
    """Upsert one document + language pair's editable glossary draft."""
    conn.execute(
        '''
        INSERT INTO workspace_glossaries (
            document_fingerprint, source_lang, target_lang, glossary, updated_at
        ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(document_fingerprint, source_lang, target_lang)
        DO UPDATE SET glossary = excluded.glossary, updated_at = CURRENT_TIMESTAMP
        ''',
        (*context, glossary),
    )


@app.route('/workspace-glossary/<document_fingerprint>', methods=['GET'])
@with_error_handling
def get_workspace_glossary(document_fingerprint):
    """Return the editable Stage 0 glossary for one book and language pair."""
    context, error = _workspace_glossary_context(
        document_fingerprint,
        request.args.get('sourceLanguage'),
        request.args.get('targetLanguage'),
    )
    if error:
        return jsonify({'error': error[0]}), error[1]

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            '''
            SELECT glossary FROM workspace_glossaries
            WHERE document_fingerprint = ? AND source_lang = ? AND target_lang = ?
            ''',
            context,
        ).fetchone()
    return jsonify({'glossary': row[0] if row else '', 'found': row is not None})


@app.route('/workspace-glossary/<document_fingerprint>', methods=['PUT'])
@with_error_handling
def save_workspace_glossary(document_fingerprint):
    """Save an editable Stage 0 glossary without creating a translation job."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Expected a JSON glossary draft'}), 400
    context, error = _workspace_glossary_context(
        document_fingerprint,
        data.get('sourceLanguage'),
        data.get('targetLanguage'),
    )
    if error:
        return jsonify({'error': error[0]}), error[1]
    glossary = data.get('glossary')
    if not isinstance(glossary, str):
        return jsonify({'error': 'Glossary must be text'}), 400
    if len(glossary) > MAX_WORKSPACE_GLOSSARY_LENGTH:
        return jsonify({'error': 'Glossary draft is too large'}), 400

    with sqlite3.connect(DB_PATH) as conn:
        _store_workspace_glossary(conn, context, glossary)
    return jsonify({'status': 'saved'})


@app.route('/translations', methods=['GET'])
@with_error_handling
def get_translations():
    _heal_orphaned_runs()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute('''
            SELECT t.id, t.filename, t.source_lang, t.target_lang, t.model,
                   t.status, t.progress, t.detected_language, t.created_at,
                   t.updated_at, t.error_message, t.current_chunk, t.total_chunks,
                   COUNT(tt.id) AS glossary_terms
            FROM translations AS t
            LEFT JOIN translation_terms AS tt ON tt.translation_id = t.id
            GROUP BY t.id
            ORDER BY t.created_at DESC
        ''')
        translations = []
        for row in cur.fetchall():
            item = dict(row)
            item['status'] = _effective_status(item['id'], item['status'])
            item['running'] = is_run_active(item['id'])
            translations.append(item)
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

        term_rows = conn.execute(
            '''
            SELECT source_term, target_term, enforcement_mode
            FROM translation_terms WHERE translation_id = ?
            ORDER BY id
            ''',
            (translation_id,)
        ).fetchall()

        data = dict(translation)
        data['status'] = _effective_status(translation_id, data['status'])
        data['running'] = is_run_active(translation_id)
        # The glossary this job actually ran under, in the textarea's own
        # format, so reopening a translation can show its terminology instead
        # of an empty editor. Same serialisation as /prepare.
        data['glossary'] = '\n'.join(
            f"{r['source_term']} => {r['target_term']} | {r['enforcement_mode']}"
            for r in term_rows
        )
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


def _json_list(value) -> List:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _translated_chapters_from_chunks(
    final_chunks: List[str], chunk_chapter_map: List[int],
) -> List[str]:
    """Rebuild chapter text from the same aligned map Stage 1 created."""
    if not final_chunks:
        return []
    chapter_map = (
        chunk_chapter_map
        if len(chunk_chapter_map) == len(final_chunks)
        else [0] * len(final_chunks)
    )
    chapters: List[str] = []
    buffer: List[str] = []
    current_chapter = chapter_map[0] if chapter_map else 0
    for chunk, chapter_index in zip(final_chunks, chapter_map):
        if chapter_index != current_chapter:
            chapters.append('\n\n'.join(buffer))
            buffer = []
            current_chapter = chapter_index
        buffer.append(chunk)
    chapters.append('\n\n'.join(buffer))
    return chapters


def _evaluation_signals_by_chunk(conn, translation_id: int) -> Dict[int, List[Dict]]:
    """Extract only quality findings that identify an exact aligned chunk."""
    signals: Dict[int, List[Dict]] = {}

    def add(chunk_number, test, label, detail=None):
        try:
            index = int(chunk_number) - 1
        except (TypeError, ValueError):
            return
        if index < 0:
            return
        item = {'test': test, 'label': label}
        if detail is not None:
            item['detail'] = detail
        signals.setdefault(index, []).append(item)

    rows = conn.execute(
        '''SELECT test_name, details FROM evaluation_results
           WHERE translation_id = ? AND flagged = 1''',
        (translation_id,),
    ).fetchall()
    for row in rows:
        try:
            details = json.loads(row['details']) if row['details'] else {}
        except (TypeError, ValueError):
            details = {}
        test = row['test_name']
        if test == 'chunk_coverage':
            for chunk_number in details.get('empty_final_chunks') or []:
                add(chunk_number, test, 'Final translation is empty')
        elif test == 'labse_alignment':
            for finding in details.get('drift_flags') or []:
                add(
                    finding.get('chunk'), test, 'Semantic-alignment outlier',
                    {'similarity': finding.get('similarity')},
                )
        elif test == 'language_id':
            for finding in details.get('wrong_language_segments') or []:
                add(
                    finding.get('chunk'), test, 'Wrong target language detected',
                    {
                        'detected': finding.get('detected'),
                        'confidence': finding.get('confidence'),
                    },
                )
    return signals


def _review_chunks_payload(conn, translation_row) -> Dict:
    original_chunks = _json_list(translation_row['original_chunks'])
    draft_chunks = _json_list(translation_row['draft_chunks'])
    final_chunks = _json_list(translation_row['final_chunks'])
    if not final_chunks and translation_row['translated_text']:
        final_chunks = (translation_row['translated_text'] or '').split('\n\n')

    review_rows = {
        row['chunk_index']: row
        for row in conn.execute(
            'SELECT * FROM chunk_reviews WHERE translation_id = ?',
            (translation_row['id'],),
        ).fetchall()
    }
    evaluation_signals = _evaluation_signals_by_chunk(
        conn, translation_row['id'],
    )
    term_rows = conn.execute(
        '''SELECT source_term, target_term, enforcement_mode
           FROM translation_terms WHERE translation_id = ?''',
        (translation_row['id'],),
    ).fetchall()
    terminology = TerminologyManager([
        GlossaryTerm(
            source=row['source_term'],
            target=row['target_term'],
            mode=row['enforcement_mode'],
        )
        for row in term_rows
    ])

    # The refinement and glossary counters the Progress rail shows while a run
    # streams. They were live-only, so reopening a finished translation — or
    # just reloading the page — redrew a completed refinement as "Not started"
    # and its glossary as unused. Every number below is recovered from what
    # the run already stored per chunk, and is counted the same way the stream
    # counts it, so the rail reads the same before and after a reload.
    refinement = {
        'errors_found': 0,
        'errors_applied': 0,
        'patches_rejected': 0,
        'position_biases': 0,
        'neutral_checks': 0,
        'chunks_reviewed': 0,
        'chunks_changed': 0,
        'review_failures': 0,
        'verifier_model': None,
        'review_model': None,
    }
    used_terms = set()
    violation_count = 0

    chunks = []
    for index, source in enumerate(original_chunks):
        draft = draft_chunks[index] if index < len(draft_chunks) else ''
        final = final_chunks[index] if index < len(final_chunks) else ''
        stored = review_rows.get(index)
        try:
            details = (
                json.loads(stored['review_details'])
                if stored and stored['review_details'] else {}
            )
        except (TypeError, ValueError):
            details = {}
        try:
            alternatives = (
                json.loads(stored['alternatives'])
                if stored and stored['alternatives'] else None
            )
        except (TypeError, ValueError):
            alternatives = None

        issues = details.get('issues') if isinstance(details.get('issues'), list) else []
        signals = list(evaluation_signals.get(index, []))
        warning = stored['warning'] if stored else None
        evidence_available = details.get('evidence_available')
        if evidence_available is False:
            signals.append({
                'test': 'stage2_review',
                'label': 'Review evidence is unavailable for this cached result',
            })
        if source.strip() and not final.strip():
            signals.append({
                'test': 'chunk_coverage',
                'label': 'Final translation is empty',
            })
        if (
            source.strip()
            and translation_row['source_lang'] != translation_row['target_lang']
            and final.strip() == source.strip()
        ):
            signals.append({
                'test': 'untranslated',
                'label': 'Final text is identical to the source',
            })
        chunk_violations = terminology.exact_violations(source, final)
        for violation in chunk_violations:
            signals.append({
                'test': 'terminology',
                'label': 'Required glossary rendering is missing',
                'detail': violation,
            })
        verified = details.get('verified')
        if isinstance(verified, dict) and not verified.get('accepted'):
            signals.append({
                'test': 'stage2_verifier',
                'label': 'Refinement patch was rejected by the verifier',
                'detail': verified,
            })

        used_terms.update(
            term.source.casefold() for term in terminology.relevant_terms(source)
        )
        violation_count += len(chunk_violations)
        if details and not details.get('cache_hit'):
            # A cache hit is not a review — the stream does not count it as one
            # either, and it has no findings of its own to add.
            refinement['chunks_reviewed'] += 1
            refinement['errors_found'] += int(details.get('errors_found') or 0)
            refinement['errors_applied'] += int(details.get('errors_applied') or 0)
            if isinstance(verified, dict):
                if not verified.get('accepted'):
                    refinement['patches_rejected'] += 1
                if verified.get('position_bias_detected'):
                    refinement['position_biases'] += 1
                if verified.get('neutral_check'):
                    refinement['neutral_checks'] += 1
            if warning:
                refinement['review_failures'] += 1
            if final != draft:
                refinement['chunks_changed'] += 1
            refinement['verifier_model'] = details.get('verifier_model') or refinement['verifier_model']
            refinement['review_model'] = details.get('review_model') or refinement['review_model']

        problematic = bool(issues or signals or warning)
        review_status = (
            stored['review_status'] if stored
            else ('open' if problematic else 'not_needed')
        )
        chunks.append({
            'index': index,
            'number': index + 1,
            'source': source,
            'draft': draft,
            'final': final,
            'issues': issues,
            'signals': signals,
            'warning': warning,
            'evidence_available': evidence_available,
            'problematic': problematic,
            'review_status': review_status,
            'resolution_kind': stored['resolution_kind'] if stored else None,
            'selected_candidate': stored['selected_candidate'] if stored else None,
            'revision': stored['revision'] if stored else 0,
            'alternatives': alternatives,
        })

    return {
        'translation_id': translation_row['id'],
        'status': translation_row['status'],
        'refinement': refinement,
        'terminology': {
            'total': len(terminology.terms),
            'used': len(used_terms),
            'violations': violation_count,
        },
        'total_chunks': len(chunks),
        'problematic_count': sum(1 for chunk in chunks if chunk['problematic']),
        'open_count': sum(
            1 for chunk in chunks
            if chunk['problematic'] and chunk['review_status'] != 'resolved'
        ),
        'chunks': chunks,
    }


@app.route('/translations/<int:translation_id>/review-chunks', methods=['GET'])
@with_error_handling
def get_review_chunks(translation_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT * FROM translations WHERE id = ?', (translation_id,),
        ).fetchone()
        if not row:
            return jsonify({'error': 'Translation not found'}), 404
        if not row['original_chunks'] or not row['draft_chunks']:
            return jsonify({
                'error': 'No aligned chunks yet — run Start first',
            }), 400
        return jsonify(_review_chunks_payload(conn, row))


def _frontier_review_case(chunk: Dict, translation_row) -> Optional[Dict]:
    """The exact manual Apply choices a provider is allowed to decide."""
    final = chunk.get('final') or ''
    issues = []
    for issue_index, issue in enumerate(chunk.get('issues') or []):
        if not isinstance(issue, dict):
            continue
        span = issue.get('span')
        replacement = issue.get('replacement')
        if (
            not isinstance(span, str)
            or not isinstance(replacement, str)
            or not span.strip()
            or not replacement.strip()
            or span == replacement
            or span not in final
        ):
            continue
        issues.append({
            'issue_index': issue_index,
            'span': span,
            'replacement': replacement,
            'type': str(issue.get('type') or 'issue'),
            'severity': str(issue.get('severity') or 'unspecified'),
        })
    if not issues:
        return None
    return {
        'chunk_index': int(chunk['index']),
        'revision': int(chunk.get('revision') or 0),
        'source_language': LANG_NAMES.get(
            translation_row['source_lang'],
            translation_row['source_lang'],
        ),
        'target_language': LANG_NAMES.get(
            translation_row['target_lang'],
            translation_row['target_lang'],
        ),
        'source': chunk.get('source') or '',
        'draft': chunk.get('draft') or '',
        'final': final,
        'issues': issues,
    }


def _frontier_review_payload(
    translation_id: int,
    chunk_index: Optional[int] = None,
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT * FROM translations WHERE id = ?',
            (translation_id,),
        ).fetchone()
        if not row:
            return None, None, (jsonify({'error': 'Translation not found'}), 404)
        if is_run_active(translation_id):
            return None, None, (jsonify({
                'error': 'This translation is still running. Wait for it to finish before reviewing.',
            }), 409)
        if not row['final_chunks']:
            return None, None, (jsonify({
                'error': 'Run Continue first — there is no final translation to review',
            }), 400)
        review_payload = _review_chunks_payload(conn, row)

    chunks = review_payload['chunks']
    if chunk_index is not None:
        selected = next(
            (chunk for chunk in chunks if int(chunk['index']) == chunk_index),
            None,
        )
        if not selected:
            return None, None, (
                jsonify({'error': 'Chunk index is out of range'}),
                404,
            )
        chunks = [selected]
    else:
        chunks = [
            chunk for chunk in chunks
            if chunk.get('review_status') != 'resolved'
        ]

    cases = [
        case
        for case in (
            _frontier_review_case(chunk, row)
            for chunk in chunks
        )
        if case is not None
    ]
    if not cases:
        message = (
            'This chunk has no applicable proposed fixes'
            if chunk_index is not None
            else 'No open Review Desk chunks have applicable proposed fixes'
        )
        return None, None, (jsonify({'error': message}), 400)
    return row, cases, None


def _run_frontier_review_decision(
    translation_id: int,
    chunk_index: Optional[int] = None,
):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': 'Review Desk decision request is required'}), 400
    provider = payload.get('provider')
    submitted_key = payload.get('apiKey')
    submitted_model = payload.get('model')
    if not isinstance(provider, str):
        return jsonify({'error': 'Choose a frontier provider in Settings'}), 400
    if submitted_key is not None and not isinstance(submitted_key, str):
        return jsonify({'error': 'API key must be text'}), 400
    if submitted_model is not None and not isinstance(submitted_model, str):
        return jsonify({'error': 'Model name must be text'}), 400

    _, cases, error_response = _frontier_review_payload(
        translation_id,
        chunk_index,
    )
    if error_response:
        return error_response
    try:
        result = decide_review_cases(
            provider,
            cases,
            submitted_key,
            submitted_model,
        )
    except FrontierGlossaryError as error:
        return jsonify({'error': str(error)}), error.status_code
    except Exception:
        logger.app_logger.error(
            'Unexpected frontier Review Desk decision error\n'
            + traceback.format_exc()
        )
        return jsonify({
            'error': 'Frontier Review Desk decision failed unexpectedly',
        }), 502

    response = jsonify({
        'translation_id': translation_id,
        'provider': result.provider,
        'provider_label': result.provider_label,
        'model': result.model,
        'decisions': result.decisions,
    })
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.route(
    '/translations/<int:translation_id>/review-chunks/<int:chunk_index>/frontier-decision',
    methods=['POST'],
)
def decide_review_chunk_with_frontier(translation_id, chunk_index):
    """Ask the selected cloud provider about one chunk's manual Apply choices."""
    return _run_frontier_review_decision(translation_id, chunk_index)


@app.route(
    '/translations/<int:translation_id>/review-chunks/frontier-decisions',
    methods=['POST'],
)
def decide_all_review_chunks_with_frontier(translation_id):
    """Decide every open, applicable Review Desk fix in one provider pass."""
    return _run_frontier_review_decision(translation_id)


@app.route(
    '/translations/<int:translation_id>/review-chunks/<int:chunk_index>',
    methods=['PATCH'],
)
@with_error_handling
def update_review_chunk(translation_id, chunk_index):
    """Make one human- or candidate-selected chunk canonical for export."""
    payload = request.get_json(silent=True) or {}
    requested_text = payload.get('text')
    candidate_id = str(payload.get('candidate_id') or '').strip() or None
    expected_revision = payload.get('expected_revision')
    if expected_revision is not None:
        try:
            expected_revision = int(expected_revision)
        except (TypeError, ValueError):
            return jsonify({'error': 'expected_revision must be an integer'}), 400
    if requested_text is not None and not isinstance(requested_text, str):
        return jsonify({'error': 'text must be a string'}), 400

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT * FROM translations WHERE id = ?', (translation_id,),
        ).fetchone()
        if not row:
            return jsonify({'error': 'Translation not found'}), 404
        if is_run_active(translation_id):
            return jsonify({
                'error': 'This translation is still running. Wait for it to finish before editing.',
            }), 409
        original_chunks = _json_list(row['original_chunks'])
        final_chunks = _json_list(row['final_chunks'])
        if not final_chunks:
            return jsonify({
                'error': 'Run Continue first — there is no final translation to edit',
            }), 400
        if chunk_index < 0 or chunk_index >= len(original_chunks):
            return jsonify({'error': 'Chunk index is out of range'}), 404
        if len(final_chunks) < len(original_chunks):
            final_chunks.extend([''] * (len(original_chunks) - len(final_chunks)))
        review_state = conn.execute(
            '''SELECT revision FROM chunk_reviews
               WHERE translation_id = ? AND chunk_index = ?''',
            (translation_id, chunk_index),
        ).fetchone()
        current_revision = review_state['revision'] if review_state else 0
        if (
            expected_revision is not None
            and expected_revision != current_revision
        ):
            return jsonify({
                'error': 'This chunk changed in another window. Reload it before saving.',
                'current_revision': current_revision,
            }), 409

        resolution_kind = 'manual'
        if candidate_id:
            review = conn.execute(
                '''SELECT alternatives FROM chunk_reviews
                   WHERE translation_id = ? AND chunk_index = ?''',
                (translation_id, chunk_index),
            ).fetchone()
            alternatives = {}
            if review and review['alternatives']:
                try:
                    alternatives = json.loads(review['alternatives'])
                except (TypeError, ValueError):
                    alternatives = {}
            selected = next(
                (
                    option for option in alternatives.get('options') or []
                    if option.get('id') == candidate_id
                ),
                None,
            )
            if not selected:
                return jsonify({'error': 'Candidate not found for this chunk'}), 404
            requested_text = selected.get('text')
            resolution_kind = (
                'kept_current' if selected.get('kind') == 'current'
                else 'candidate'
            )

        if requested_text is None:
            return jsonify({'error': 'Provide text or candidate_id'}), 400
        requested_text = requested_text.strip()
        if original_chunks[chunk_index].strip() and not requested_text:
            return jsonify({'error': 'A non-empty source chunk cannot have an empty final translation'}), 400

        final_chunks[chunk_index] = requested_text
        translated_text = '\n\n'.join(final_chunks)
        translated_chapters = row['translated_chapters']
        if row['source_format'] == 'epub':
            chapter_map = _json_list(row['chunk_chapter_map'])
            translated_chapters = json.dumps(
                _translated_chapters_from_chunks(final_chunks, chapter_map),
                ensure_ascii=False,
            )

        stage2_tests = sorted(STAGE2_TESTS)
        placeholders = ','.join('?' for _ in stage2_tests)
        invalidated = conn.execute(
            f'''SELECT COUNT(*) FROM evaluation_results
                WHERE translation_id = ? AND test_name IN ({placeholders})''',
            (translation_id, *stage2_tests),
        ).fetchone()[0]
        conn.execute(
            '''UPDATE translations
               SET final_chunks = ?, translated_text = ?,
                   translated_chapters = ?, updated_at = CURRENT_TIMESTAMP
               WHERE id = ?''',
            (
                json.dumps(final_chunks, ensure_ascii=False),
                translated_text,
                translated_chapters,
                translation_id,
            ),
        )
        conn.execute(
            '''
            INSERT INTO chunk_reviews (
                translation_id, chunk_index, review_status, resolution_kind,
                selected_candidate, revision, updated_at
            ) VALUES (?, ?, 'resolved', ?, ?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT (translation_id, chunk_index) DO UPDATE SET
                review_status = 'resolved',
                resolution_kind = excluded.resolution_kind,
                selected_candidate = excluded.selected_candidate,
                revision = chunk_reviews.revision + 1,
                updated_at = CURRENT_TIMESTAMP
            ''',
            (translation_id, chunk_index, resolution_kind, candidate_id),
        )
        # Stage 2 quality results were computed over the old canonical final.
        # Draft-only Stage 1 checks remain valid.
        conn.execute(
            f'''DELETE FROM evaluation_results
                WHERE translation_id = ? AND test_name IN ({placeholders})''',
            (translation_id, *stage2_tests),
        )

        term_rows = conn.execute(
            '''SELECT source_term, target_term, enforcement_mode
               FROM translation_terms WHERE translation_id = ?''',
            (translation_id,),
        ).fetchall()
        terminology = TerminologyManager([
            GlossaryTerm(
                source=term['source_term'],
                target=term['target_term'],
                mode=term['enforcement_mode'],
            )
            for term in term_rows
        ])
        violations = terminology.exact_violations(
            original_chunks[chunk_index], requested_text,
        )

    logger.translation_logger.info(
        'Review desk saved chunk %s for translation %s (%s); invalidated %s quality result(s)',
        chunk_index + 1, translation_id, resolution_kind, invalidated,
    )
    return jsonify({
        'status': 'saved',
        'translation_id': translation_id,
        'chunk_index': chunk_index,
        'text': requested_text,
        'resolution_kind': resolution_kind,
        'invalidated_quality_results': invalidated,
        'terminology_violations': violations,
        'revision': current_revision + 1,
    })


@app.route(
    '/translations/<int:translation_id>/review-chunks/<int:chunk_index>/alternatives',
    methods=['POST'],
)
@with_error_handling
def generate_review_chunk_alternatives(translation_id, chunk_index):
    """Generate alternatives for this chunk only, never for the whole book."""
    payload = request.get_json(silent=True) or {}
    try:
        count = int(payload.get('count', 2))
    except (TypeError, ValueError):
        return jsonify({'error': 'count must be 2 or 3'}), 400
    if count not in (2, 3):
        return jsonify({'error': 'count must be 2 or 3'}), 400

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT * FROM translations WHERE id = ?', (translation_id,),
        ).fetchone()
        if not row:
            return jsonify({'error': 'Translation not found'}), 404
        original_chunks = _json_list(row['original_chunks'])
        draft_chunks = _json_list(row['draft_chunks'])
        final_chunks = _json_list(row['final_chunks'])
        if not final_chunks:
            return jsonify({
                'error': 'Run Continue first — alternatives compare against the final translation',
            }), 400
        if chunk_index < 0 or chunk_index >= len(original_chunks):
            return jsonify({'error': 'Chunk index is out of range'}), 404
        term_rows = conn.execute(
            '''SELECT source_term, target_term, enforcement_mode
               FROM translation_terms WHERE translation_id = ?''',
            (translation_id,),
        ).fetchall()

    candidate_model = str(payload.get('model') or row['model']).strip()
    judge_model = str(payload.get('judge_model') or '').strip()
    if not candidate_model:
        return jsonify({'error': 'A translation model is required'}), 400
    if judge_model and judge_model.casefold() == candidate_model.casefold():
        return jsonify({
            'error': 'The judge must be independent from the model generating the candidates',
        }), 400
    if judge_model and is_translategemma(judge_model):
        return jsonify({
            'error': 'TranslateGemma is translation-only and cannot judge candidates',
        }), 400

    terminology = TerminologyManager([
        GlossaryTerm(
            source=term['source_term'],
            target=term['target_term'],
            mode=term['enforcement_mode'],
        )
        for term in term_rows
    ])
    source = original_chunks[chunk_index]
    current = final_chunks[chunk_index] if chunk_index < len(final_chunks) else ''
    draft = draft_chunks[chunk_index] if chunk_index < len(draft_chunks) else ''
    previous = final_chunks[chunk_index - 1] if chunk_index > 0 else ''
    translator = BookTranslator(model_name=candidate_model)
    translator.terminology = terminology
    context = terminology.prompt_context(source)
    temperatures = (
        [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
        if is_translategemma(candidate_model)
        else [0.45, 0.60, 0.75, 0.90, 1.00, 1.10]
    )
    generated: List[str] = []
    seen = {text.strip() for text in (current, draft) if text and text.strip()}
    call_errors: List[str] = []
    for temperature in temperatures:
        if len(generated) >= count:
            break
        candidate, warning = translator.generate_translation_candidate(
            source,
            row['source_lang'],
            row['target_lang'],
            previous_chunk=previous,
            genre=row['genre'],
            terminology_context=context,
            temperature=temperature,
        )
        if warning:
            call_errors.append(warning)
            continue
        candidate, _ = terminology.enforce_exact_source_forms(candidate or '')
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        generated.append(candidate)

    if not generated:
        return jsonify({
            'error': 'The translation model did not produce a distinct candidate',
            'details': call_errors[:3],
        }), 502

    generated_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    options = [{
        'id': 'current',
        'kind': 'current',
        'label': 'Current final',
        'text': current,
        'generation_model': row['model'],
        'created_at': generated_at,
    }]
    options.extend({
        'id': f'candidate-{index}',
        'kind': 'generated',
        'label': f'Candidate {index}',
        'text': text,
        'generation_model': candidate_model,
        'created_at': generated_at,
    } for index, text in enumerate(generated, 1))

    recommended_id = judge_reason = judge_error = None
    if judge_model:
        judge = BookTranslator(model_name=judge_model)
        best, judge_reason, judge_error = judge.judge_translation_candidates(
            source,
            [option['text'] for option in options],
            row['source_lang'],
            row['target_lang'],
        )
        if best is not None:
            recommended_id = options[best]['id']

    result = {
        'translation_id': translation_id,
        'chunk_index': chunk_index,
        'requested_count': count,
        'generated_count': len(generated),
        'generated_model': candidate_model,
        'judge_model': judge_model or None,
        'recommended_id': recommended_id,
        'judge_reason': judge_reason,
        'judge_error': judge_error,
        'warning': (
            f'Only {len(generated)} distinct candidate(s) were produced.'
            if len(generated) < count else None
        ),
        'options': options,
    }
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            '''
            INSERT INTO chunk_reviews (
                translation_id, chunk_index, alternatives, judge_model,
                review_status, revision, updated_at
            ) VALUES (?, ?, ?, ?, 'open', 1, CURRENT_TIMESTAMP)
            ON CONFLICT (translation_id, chunk_index) DO UPDATE SET
                alternatives = excluded.alternatives,
                judge_model = excluded.judge_model,
                revision = chunk_reviews.revision + 1,
                updated_at = CURRENT_TIMESTAMP
            ''',
            (
                translation_id,
                chunk_index,
                json.dumps(result, ensure_ascii=False),
                judge_model or None,
            ),
        )
        result['revision'] = conn.execute(
            '''SELECT revision FROM chunk_reviews
               WHERE translation_id = ? AND chunk_index = ?''',
            (translation_id, chunk_index),
        ).fetchone()[0]
    logger.translation_logger.info(
        'Review desk generated %s/%s alternative(s) for translation %s chunk %s with %s; judge: %s',
        len(generated), count, translation_id, chunk_index + 1,
        candidate_model, judge_model or 'manual choice',
    )
    return jsonify(result)


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
        conn.execute('DELETE FROM chunk_reviews WHERE translation_id = ?', (translation_id,))
        conn.execute('DELETE FROM translation_terms WHERE translation_id = ?', (translation_id,))
        conn.execute('DELETE FROM evaluation_results WHERE translation_id = ?', (translation_id,))
        conn.execute('DELETE FROM translations WHERE id = ?', (translation_id,))

    logger.app_logger.info('Deleted translation task %s', translation_id)
    return jsonify({'status': 'success', 'id': translation_id})
    
class UploadError(Exception):
    """An upload the caller should reject with a 400."""


# Languages whose books arrive in cp1251 rather than cp1252 when they are not
# UTF-8. Kept wider than this app's own language list: the source file's
# encoding does not care which languages the interface offers.
CYRILLIC_SOURCE_LANGUAGES = frozenset({'ru', 'uk', 'be', 'bg', 'sr', 'mk', 'kk'})


def decode_text_file(filepath: str, source_lang: Optional[str] = None) -> str:
    """Read a .txt book, trying the encodings a real book arrives in.

    A single-byte codepage cannot fail to decode — every byte maps to some
    character — so nothing raises when the guess is wrong: it just returns
    mojibake. Measured on a real cp1251 Russian sample, strict cp1252 decoded
    all of it happily into "Ìèñòåð Äóðñëü". So the order is not guessed, it is
    taken from the source language the user already selected, and only then
    falls back. This code used to try cp1251 alone, which did the same damage in
    the other direction to any Portuguese or French book.

    utf-8-sig is tried first and settles most files: real UTF-8 is
    self-validating, and the -sig form also strips the BOM that Windows Notepad
    writes and that plain utf-8 would leave as an invisible character at the
    head of the first chunk.
    """
    raw = open(filepath, 'rb').read()
    try:
        return raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        pass

    cyrillic = (source_lang or '').strip().lower()[:2] in CYRILLIC_SOURCE_LANGUAGES
    fallbacks = ('cp1251', 'cp1252', 'latin-1') if cyrillic else ('cp1252', 'cp1251', 'latin-1')
    for encoding in fallbacks:
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        logger.app_logger.warning(
            "%s is not UTF-8 — decoded as %s (source language %s). If the text "
            "looks like nonsense, re-save the file as UTF-8.",
            os.path.basename(filepath), encoding, source_lang or 'not given',
        )
        return text
    # latin-1 cannot fail, so this is unreachable; kept so a future edit to the
    # list cannot silently return None.
    raise UploadError('Could not decode this file — re-save it as UTF-8.')


def read_uploaded_book(file, source_lang: Optional[str] = None):
    """Save an uploaded book into UPLOAD_FOLDER and decode it.

    Returns (text, chapters, book_title, book_author, source_format, filepath).
    `chapters` is None for plain text; for EPUB (and for DOCX/PDF that carry
    clear chapter structure) it is what drives chunking, so that no chunk ever
    straddles a chapter boundary. Deleting filepath on the way out is the
    caller's job — but a file rejected here never reaches a caller that has a
    path to delete, so it is removed here instead.
    """
    if not file or file.filename == '':
        raise UploadError('No selected file')

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        if is_pdf_filename(filename):
            try:
                text, chapters, book_title, book_author = extract_pdf_book(filepath)
            except Exception as e:
                # pypdf's own errors name internal objects, so the message the
                # user gets is the one thing worth adding to it.
                logger.app_logger.error('Failed to read PDF %s: %s', filename, e)
                raise UploadError(f'Could not read this PDF: {e}')
            if not text.strip():
                raise UploadError(
                    'No text could be extracted from this PDF. If it is a scan, run '
                    'OCR on it first, or upload the book as TXT, EPUB, or DOCX.'
                )
            if chapters:
                text = '\n\n'.join(
                    f'=== Chapter {i} ===\n\n{chapter}'
                    for i, chapter in enumerate(chapters, 1)
                )
            return text, chapters, book_title, book_author, 'pdf', filepath

        if is_docx_filename(filename):
            try:
                text, chapters, book_title, book_author = extract_docx_book(filepath)
            except Exception as e:
                logger.app_logger.error('Failed to read DOCX %s: %s', filename, e)
                raise UploadError(f'Could not read this DOCX: {e}')
            if not text.strip():
                raise UploadError('No text could be extracted from this DOCX.')
            if chapters:
                text = '\n\n'.join(
                    f'=== Chapter {i} ===\n\n{chapter}'
                    for i, chapter in enumerate(chapters, 1)
                )
            return text, chapters, book_title, book_author, 'docx', filepath

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

        text = decode_text_file(filepath, source_lang)
        return text, None, None, None, 'txt', filepath
    except UploadError:
        try:
            os.remove(filepath)
        except OSError as e:
            logger.app_logger.error('Failed to clean up a rejected upload: %s', e)
        raise


@app.route('/source-preview', methods=['POST'])
@with_error_handling
def source_preview():
    """Extract a readable, bounded preview without starting a translation."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    filepath = None
    try:
        text, _, title, author, source_format, filepath = read_uploaded_book(
            request.files['file'], request.form.get('sourceLanguage')
        )
        preview_limit = 100_000
        return jsonify({
            'preview': text[:preview_limit],
            'truncated': len(text) > preview_limit,
            'source_chars': len(text),
            'source_format': source_format,
            'title': title,
            'author': author,
        })
    except UploadError as e:
        return jsonify({'error': str(e)}), 400
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as e:
                logger.app_logger.error('Failed to clean up source preview upload: %s', e)


@app.route('/prepare', methods=['POST'])
@with_error_handling
def prepare():
    """STAGE 0: read the book and propose the contract the translation will
    run under — one agreed rendering per recurring proper noun.

    Deliberately does not create a translation row or translate anything.
    The glossary comes back as editable text and is then submitted with Start
    like any hand-written one, so what actually reaches the model is always
    what the user saw and approved.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    filepath = None
    try:
        source_lang = request.form.get('sourceLanguage')
        target_lang = request.form.get('targetLanguage')
        model_name = request.form.get('model')
        genre = request.form.get('genre', 'unknown')
        # Stage 0 makes two different demands — ruling on whether two source
        # forms name one entity, and rendering a name into the target language
        # — but on the same-prompt comparison the two roles were run on, a
        # small model matched a much larger one at both, and splitting them
        # only made a 32 GB machine unload one model and load the other
        # halfway through Prepare. So the interface offers one choice, and a
        # separate entity model stays available to anything calling /prepare
        # directly.
        entity_model_name = request.form.get('entityModel') or model_name

        if not all([source_lang, target_lang, model_name]):
            return jsonify({'error': 'Missing required parameters'}), 400
        for name in (model_name, entity_model_name):
            if is_translategemma(name):
                return jsonify({'error': (
                    'TranslateGemma is translation-only and cannot extract or '
                    'render names. Pick a general instruct model to prepare, then '
                    'switch back to TranslateGemma for Start if you like.'
                )}), 400

        try:
            text, _, _, _, _, filepath = read_uploaded_book(request.files['file'], source_lang)
        except UploadError as e:
            return jsonify({'error': str(e)}), 400

        logger.translation_logger.info(
            f"Stage 0 started: {source_lang} → {target_lang} (genre: {genre}), "
            + (f"model: {model_name}" if entity_model_name == model_name
               else f"entities: {entity_model_name}, rendering: {model_name}")
        )
        translator = BookTranslator(model_name=model_name)
        try:
            candidates, review_queue = translator.build_glossary_candidates(text)
        except RuntimeError as e:
            logger.translation_logger.error(f"Stage 0 failed: {e}")
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
        logger.translation_logger.info(
            f"Stage 0 finished: {len(records)} glossary record(s) proposed, "
            f"{len(rendering_conflicts)} rendering conflict(s) to review"
        )
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

        # The browser already fingerprints the book to scope its editable
        # glossary draft. Keeping that identity on the job is what lets a
        # reopened translation find the same draft again.
        document_fingerprint = request.form.get('documentFingerprint') or ''
        if not WORKSPACE_GLOSSARY_FINGERPRINT.fullmatch(document_fingerprint):
            document_fingerprint = None

        try:
            text, chapters, book_title, book_author, source_format, filepath = read_uploaded_book(file, source_lang)
        except UploadError as e:
            return jsonify({'error': str(e)}), 400
        filename = secure_filename(file.filename)

        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute('''
                INSERT INTO translations (
                    filename, source_lang, target_lang, model,
                    status, original_text, genre, source_format, book_title, book_author,
                    document_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (filename, source_lang, target_lang, model_name,
                  'in_progress', text, genre, source_format, book_title, book_author,
                  document_fingerprint))
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
            # Start is also a save point for the editable draft: the browser
            # debounces its autosave, so the very last keystroke before Start
            # would otherwise never reach SQLite.
            if document_fingerprint:
                _store_workspace_glossary(
                    conn,
                    (document_fingerprint, source_lang, target_lang),
                    request.form.get('glossary', '')[:MAX_WORKSPACE_GLOSSARY_LENGTH],
                )

        translator = BookTranslator(model_name=model_name)

        # Work runs in a daemon thread. The SSE response only observes the
        # progress queue — closing the browser no longer kills the overnight run.
        _emit_progress(translation_id, {
            'progress': 1,
            'stage': 'starting',
            'translation_id': translation_id,
            'message': 'Book uploaded. Loading the model and starting the first batch…',
            'terminology': {
                'total': len(terminology.terms),
                'used': 0,
                'violations': 0,
            },
        })
        _start_detached_job(
            translation_id,
            translator.translate_stage1(
                text,
                source_lang,
                target_lang,
                translation_id,
                genre=genre,
                terminology=terminology,
                chapters=chapters,
            ),
        )

        return Response(
            _sse_from_progress_queue(translation_id),
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


@app.route('/resume-translation/<int:translation_id>', methods=['POST'])
@with_error_handling
def resume_translation(translation_id):
    """Continue Stage 1 from the last finished chunk after an interrupt."""
    if is_run_active(translation_id):
        return jsonify({'error': 'This translation is already running'}), 409

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT * FROM translations WHERE id = ?', (translation_id,)
        ).fetchone()
        if not row:
            return jsonify({'error': 'Translation not found'}), 404

        original_chunks = _json_list(row['original_chunks'])
        draft_chunks = _json_list(row['draft_chunks'])
        if not original_chunks:
            return jsonify({
                'error': 'Nothing to resume — press Start to begin this book again.',
            }), 400
        if len(draft_chunks) >= len(original_chunks):
            return jsonify({
                'error': 'Draft is already complete — press Continue to refine.',
            }), 400
        if row['status'] not in (
            'interrupted', 'error', 'in_progress', 'pending',
        ):
            return jsonify({
                'error': f"Cannot resume a translation with status '{row['status']}'.",
            }), 400

        term_rows = conn.execute(
            '''SELECT source_term, target_term, enforcement_mode
               FROM translation_terms WHERE translation_id = ?''',
            (translation_id,),
        ).fetchall()

    terminology = TerminologyManager([
        GlossaryTerm(
            source=r['source_term'], target=r['target_term'], mode=r['enforcement_mode'],
        )
        for r in term_rows
    ])
    translator = BookTranslator(model_name=row['model'])

    _clear_progress_queue(translation_id)
    _emit_progress(translation_id, {
        'progress': max(1, int(100 * len(draft_chunks) / max(len(original_chunks), 1))),
        'stage': 'starting',
        'translation_id': translation_id,
        'message': (
            f'Resuming draft from chunk {len(draft_chunks) + 1}/'
            f'{len(original_chunks)}…'
        ),
        'terminology': {
            'total': len(terminology.terms),
            'used': 0,
            'violations': 0,
        },
    })
    _start_detached_job(
        translation_id,
        translator.translate_stage1(
            row['original_text'] or '',
            row['source_lang'],
            row['target_lang'],
            translation_id,
            genre=row['genre'] or 'unknown',
            terminology=terminology,
            resume=True,
        ),
    )

    return Response(
        _sse_from_progress_queue(translation_id),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@app.route('/translations/<int:translation_id>/stream', methods=['GET'])
@with_error_handling
def stream_translation_progress(translation_id):
    """Re-attach to a live job's progress after a page reload."""
    if not is_run_active(translation_id):
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                'SELECT status, progress, error_message FROM translations WHERE id = ?',
                (translation_id,),
            ).fetchone()
        if not row:
            return jsonify({'error': 'Translation not found'}), 404
        payload = {
            'translation_id': translation_id,
            'status': _effective_status(translation_id, row['status']),
            'progress': row['progress'] or 0,
        }
        if row['error_message']:
            payload['message'] = row['error_message']
        return Response(
            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n",
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
        )
    return Response(
        _sse_from_progress_queue(translation_id),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


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
    check: its A/B verdict often follows the order the versions are shown in.
    Stage 2 detects that disagreement and retries without ordered versions,
    but an independent verifier remains the supported setup.
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
        original_chunks = _json_list(row['original_chunks'])
        draft_chunks = _json_list(row['draft_chunks'])
        if original_chunks and len(draft_chunks) < len(original_chunks):
            return jsonify({
                'error': 'Draft is incomplete — press Resume to finish Stage 1 before Continue.',
            }), 400
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

    _emit_progress(translation_id, {
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
    })
    _start_detached_job(
        translation_id,
        translator.translate_stage2(
            translation_id,
            row['source_lang'],
            row['target_lang'],
            genre=row['genre'],
            terminology=terminology,
        ),
    )

    return Response(
        _sse_from_progress_queue(translation_id),
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
            return send_file(
                BytesIO(epub_bytes),
                as_attachment=True,
                download_name=download_name,
                mimetype='application/epub+zip',
            )

        return send_file(
            BytesIO((translated_text or '').encode('utf-8')),
            as_attachment=True,
            download_name=f'translated_{filename}',
            mimetype='text/plain; charset=utf-8',
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
        
        # One chapter, but built by the same writer the multi-chapter path
        # uses. The hand-rolled copy that used to live here interpolated the
        # title, author and body text into XML unescaped, so an ampersand
        # anywhere in the book produced an EPUB no reader would open.
        epub_bytes = build_epub_from_chapters([text], title=title, author=author)

        download_name = f'{title.replace(" ", "_")}.epub'
        logger.app_logger.info("EPUB created for download: %s", download_name)

        return send_file(
            BytesIO(epub_bytes),
            as_attachment=True,
            download_name=download_name,
            mimetype='application/epub+zip'
        )
        
    except Exception as e:
        logger.app_logger.error(f"EPUB export error: {str(e)}")
        logger.app_logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/failed-translations', methods=['GET'])
@with_error_handling
def get_failed_translations():
    return jsonify(_load_failed_translations())

@app.route('/retry-translation/<int:translation_id>', methods=['POST'])
@with_error_handling
def retry_failed_translation(translation_id):
    _retry_failed_translation(translation_id)
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
            
        disk_usage = project_disk_usage()
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
                _cleanup_failed_translations()
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

"""Logging, request-log formatting and process metrics.

Everything here is about watching the app run rather than about translating:
the rotating file loggers behind the Log window, the plain-text access log, and
the counters /metrics reports.

The two live instances are built here too, at the bottom. They have to be: the
pipeline and the Stage 3 quality tests both write to the logger, and a second
module constructing its own AppLogger would open the same files twice and split
the Log window's output in half. translator.py imports them, so
``translator.logger`` and ``translator.monitor`` still name the same objects
the rest of the app — and the tests, which patch them — already reach for.
"""

import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

import psutil

# Overridable so that a test run does not write into the log a person is
# watching: three warnings about "translation 1" from a fixture, landing in the
# live console between two real chunks, is worse than no log at all.
LOG_FOLDER = os.environ.get('TOLMACH_LOG_DIR', 'logs')


class AppLogger:
    def __init__(self, log_dir):
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

# psutil.disk_usage('/') is a Unix assumption: on Windows the project's drive
# is what matters, and '/' there is not a path at all. Reported by a contributor
# running this on Windows.
def project_disk_usage():
    return psutil.disk_usage(os.path.abspath(os.sep if os.name != 'nt' else os.getcwd()))


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
            'disk_usage': project_disk_usage().percent,
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


# The one logger and the one monitor for the process. Built here rather than in
# translator.py so that every module needing them gets the same object.
logger = AppLogger(log_dir=LOG_FOLDER)
monitor = AppMonitor()

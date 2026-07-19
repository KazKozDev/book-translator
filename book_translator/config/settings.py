"""
Centralized Configuration for Book Translator
==============================================
All configuration values in one place, configurable via environment variables.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


def _get_bool_env(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(key, "").lower()
    if val in ("true", "1", "yes", "on"):
        return True
    elif val in ("false", "0", "no", "off"):
        return False
    return default


def _get_int_env(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _get_float_env(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _get_optional_int_env(key: str, default: Optional[int]) -> Optional[int]:
    """
    Get an optional integer from an environment variable. Unset falls back
    to `default`; "0"/"none"/"" explicitly means "no limit" (None).
    """
    val = os.environ.get(key)
    if val is None:
        return default
    val = val.strip().lower()
    if val in ("", "0", "none", "null"):
        return None
    try:
        return int(val)
    except ValueError:
        return default


def _generate_secret_key() -> str:
    """Generate a secure random secret key."""
    import secrets

    return secrets.token_hex(32)


def get_app_paths() -> Tuple[str, str]:
    """Get the correct paths based on execution environment."""
    if getattr(sys, "frozen", False):
        # Running as packaged .exe with PyInstaller
        app_dir = os.environ.get(
            "BOOK_TRANSLATOR_APP_DIR", os.path.dirname(sys.executable)
        )
        bundle_dir = os.environ.get(
            "BOOK_TRANSLATOR_BUNDLE_DIR", getattr(sys, "_MEIPASS", app_dir)
        )
    else:
        # Running as normal Python script
        app_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        bundle_dir = app_dir
    return app_dir, bundle_dir


APP_DIR, BUNDLE_DIR = get_app_paths()


@dataclass
class ServerConfig:
    """Flask server configuration."""

    host: str = field(
        default_factory=lambda: os.environ.get("BOOK_TRANSLATOR_HOST", "127.0.0.1")
    )
    port: int = field(
        default_factory=lambda: _get_int_env("BOOK_TRANSLATOR_PORT", 5001)
    )
    debug: bool = field(
        default_factory=lambda: _get_bool_env("BOOK_TRANSLATOR_DEBUG", False)
    )
    secret_key: str = field(
        default_factory=lambda: os.environ.get("SECRET_KEY") or _generate_secret_key()
    )

    # CORS settings
    cors_origins: List[str] = field(
        default_factory=lambda: ["http://localhost:5001", "http://127.0.0.1:5001"]
    )


@dataclass
class OllamaConfig:
    """Ollama API configuration."""

    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
    )
    default_model: str = field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_DEFAULT_MODEL", "llama3.3:70b-instruct-q2_K"
        )
    )

    # Timeouts
    connect_timeout: int = field(
        default_factory=lambda: _get_int_env("OLLAMA_CONNECT_TIMEOUT", 30)
    )
    # How long to wait for Ollama to finish generating a response. Once the
    # request has been sent and Ollama accepted it, there is no point
    # cutting it off - a slow machine or a big/reasoning model can
    # legitimately take many minutes, and Ollama WILL eventually respond.
    # Defaults to no limit (waits indefinitely); set OLLAMA_READ_TIMEOUT to
    # a number of seconds if you want a hard cap instead.
    read_timeout: Optional[int] = field(
        default_factory=lambda: _get_optional_int_env("OLLAMA_READ_TIMEOUT", None)
    )
    health_check_timeout: int = field(
        default_factory=lambda: _get_int_env("OLLAMA_HEALTH_TIMEOUT", 5)
    )

    # Generation parameters
    temperature: float = field(
        default_factory=lambda: _get_float_env("OLLAMA_TEMPERATURE", 0.3)
    )
    top_p: float = field(default_factory=lambda: _get_float_env("OLLAMA_TOP_P", 0.9))

    # Context window (tokens). Ollama defaults to a mere 4096 tokens for most
    # models, which reasoning models like gpt-oss can exhaust on hidden
    # "thinking" tokens before ever writing the actual translation, coming
    # back with a success response but an empty "response" field. Set a much
    # larger window so the prompt + reasoning + output all fit.
    num_ctx: int = field(default_factory=lambda: _get_int_env("OLLAMA_NUM_CTX", 8192))

    # Reasoning effort override for thinking-capable models ("", "false",
    # "true", "low", "medium", "high"). Empty string = auto-detect from the
    # model name (see ollama_client._resolve_think_option).
    think: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_THINK", "").strip().lower()
    )

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/api/generate"

    @property
    def model_list_url(self) -> str:
        return f"{self.base_url}/api/tags"


@dataclass
class TranslationConfig:
    """Translation processing configuration."""

    # Chunk settings
    chunk_size: int = field(default_factory=lambda: _get_int_env("CHUNK_SIZE", 1000))
    max_prompt_length: int = field(
        default_factory=lambda: _get_int_env("MAX_PROMPT_LENGTH", 4000)
    )

    # Retry settings
    max_retries: int = field(default_factory=lambda: _get_int_env("MAX_RETRIES", 3))
    retry_delay: float = field(
        default_factory=lambda: _get_float_env("RETRY_DELAY", 2.0)
    )

    # Sleep between chunks (0 to disable). Gives Ollama a breather between
    # requests instead of firing the next one immediately.
    chunk_delay: float = field(
        default_factory=lambda: _get_float_env("CHUNK_DELAY", 1.5)
    )

    # Parallel processing
    max_workers: int = field(default_factory=lambda: _get_int_env("MAX_WORKERS", 3))
    enable_parallel: bool = field(
        default_factory=lambda: _get_bool_env("ENABLE_PARALLEL", True)
    )

    # Validation thresholds
    similarity_threshold: float = field(
        default_factory=lambda: _get_float_env("SIMILARITY_THRESHOLD", 0.65)
    )
    min_translation_length: int = field(
        default_factory=lambda: _get_int_env("MIN_TRANSLATION_LENGTH", 50)
    )


@dataclass
class CacheConfig:
    """Cache configuration."""

    enabled: bool = field(default_factory=lambda: _get_bool_env("CACHE_ENABLED", True))
    context_hash_length: int = field(
        default_factory=lambda: _get_int_env("CACHE_CONTEXT_HASH_LENGTH", 32)
    )
    cleanup_interval_hours: int = field(
        default_factory=lambda: _get_int_env("CACHE_CLEANUP_HOURS", 24)
    )
    max_age_days: int = field(
        default_factory=lambda: _get_int_env("CACHE_MAX_AGE_DAYS", 30)
    )


@dataclass
class FileConfig:
    """File handling configuration."""

    max_file_size_mb: int = field(
        default_factory=lambda: _get_int_env("MAX_FILE_SIZE_MB", 10)
    )
    allowed_extensions: Tuple[str, ...] = field(default_factory=lambda: (".txt",))

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@dataclass
class LoggingConfig:
    """Logging configuration."""

    verbose_debug: bool = field(
        default_factory=lambda: _get_bool_env("VERBOSE_DEBUG", True)
    )
    log_buffer_size: int = field(
        default_factory=lambda: _get_int_env("LOG_BUFFER_SIZE", 500)
    )
    log_file_max_bytes: int = field(
        default_factory=lambda: _get_int_env("LOG_FILE_MAX_BYTES", 10 * 1024 * 1024)
    )
    log_file_backup_count: int = field(
        default_factory=lambda: _get_int_env("LOG_FILE_BACKUP_COUNT", 5)
    )
    strip_ansi_in_files: bool = field(
        default_factory=lambda: _get_bool_env("STRIP_ANSI_LOGS", True)
    )


@dataclass
class SecurityConfig:
    """Security configuration."""

    enable_auth: bool = field(
        default_factory=lambda: _get_bool_env("ENABLE_AUTH", False)
    )
    api_key: str = field(default_factory=lambda: os.environ.get("API_KEY", ""))
    rate_limit_per_minute: int = field(
        default_factory=lambda: _get_int_env("RATE_LIMIT_PER_MINUTE", 60)
    )
    rate_limit_per_hour: int = field(
        default_factory=lambda: _get_int_env("RATE_LIMIT_PER_HOUR", 500)
    )
    db_timeout: int = field(default_factory=lambda: _get_int_env("DB_TIMEOUT", 30))


@dataclass
class PathConfig:
    """Path configuration."""

    app_dir: str = field(default_factory=lambda: APP_DIR)
    bundle_dir: str = field(default_factory=lambda: BUNDLE_DIR)

    @property
    def upload_folder(self) -> Path:
        return Path(self.app_dir) / "uploads"

    @property
    def translations_folder(self) -> Path:
        return Path(self.app_dir) / "translations"

    @property
    def static_folder(self) -> str:
        return os.path.join(self.bundle_dir, "static")

    @property
    def log_folder(self) -> Path:
        return Path(self.app_dir) / "logs"

    @property
    def db_path(self) -> str:
        return os.path.join(self.app_dir, "translations.db")

    @property
    def cache_db_path(self) -> str:
        return os.path.join(self.app_dir, "cache.db")


@dataclass
class Config:
    """Main application configuration container."""

    server: ServerConfig = field(default_factory=ServerConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    file: FileConfig = field(default_factory=FileConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def __post_init__(self):
        """Create necessary directories after initialization."""
        self._create_directories()
        self._validate()

    def _create_directories(self):
        """Create necessary directories."""
        for folder in [
            self.paths.upload_folder,
            self.paths.translations_folder,
            self.paths.log_folder,
        ]:
            os.makedirs(folder, exist_ok=True)

    def _validate(self):
        """Validate configuration values."""
        if self.translation.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if (
            self.translation.similarity_threshold < 0
            or self.translation.similarity_threshold > 1
        ):
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.file.max_file_size_mb < 1:
            raise ValueError("max_file_size_mb must be at least 1")


# Global configuration instance
config = Config()

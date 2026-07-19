"""
Unit Tests for Configuration System
====================================
Tests for the refactored book_translator configuration.
"""

import os
import sys

import pytest

# Setup test environment
os.environ.setdefault("BOOK_TRANSLATOR_ENV", "testing")
os.environ.setdefault("VERBOSE_DEBUG", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from book_translator.config.settings import (
    CacheConfig,
    Config,
    FileConfig,
    LoggingConfig,
    OllamaConfig,
    ServerConfig,
    TranslationConfig,
    _get_bool_env,
    _get_float_env,
    _get_int_env,
)


class TestEnvHelpers:
    """Test environment variable helper functions."""

    def test_get_bool_env_true_values(self, monkeypatch):
        for val in ["true", "1", "yes", "on", "TRUE", "True"]:
            monkeypatch.setenv("TEST_BOOL", val)
            assert _get_bool_env("TEST_BOOL", False) == True

    def test_get_bool_env_false_values(self, monkeypatch):
        for val in ["false", "0", "no", "off", "FALSE", "False"]:
            monkeypatch.setenv("TEST_BOOL", val)
            assert _get_bool_env("TEST_BOOL", True) == False

    def test_get_bool_env_default(self, monkeypatch):
        monkeypatch.delenv("TEST_BOOL", raising=False)
        assert _get_bool_env("TEST_BOOL", True) == True
        assert _get_bool_env("TEST_BOOL", False) == False

    def test_get_int_env(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "42")
        assert _get_int_env("TEST_INT", 0) == 42

    def test_get_int_env_invalid(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "not_a_number")
        assert _get_int_env("TEST_INT", 99) == 99

    def test_get_float_env(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        assert _get_float_env("TEST_FLOAT", 0.0) == pytest.approx(3.14)

    def test_get_float_env_invalid(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "not_a_float")
        assert _get_float_env("TEST_FLOAT", 2.5) == 2.5


class TestServerConfig:
    """Test server configuration."""

    def test_default_values(self):
        config = ServerConfig()
        assert config.host in ["0.0.0.0", "127.0.0.1", "localhost"]
        assert config.port > 0
        assert isinstance(config.debug, bool)
        assert len(config.cors_origins) > 0

    def test_secret_key_generated(self):
        """Test that secret key is generated if not set."""
        config = ServerConfig()
        assert config.secret_key is not None
        assert len(config.secret_key) >= 32  # At least 32 hex chars


class TestOllamaConfig:
    """Test Ollama API configuration."""

    def test_default_values(self):
        config = OllamaConfig()
        assert "localhost:11434" in config.base_url or "11434" in config.base_url
        assert config.connect_timeout > 0
        # read_timeout defaults to None (wait indefinitely for Ollama to
        # respond, since a slow machine/model is not a failure)
        assert config.read_timeout is None
        assert 0 <= config.temperature <= 1
        assert 0 <= config.top_p <= 1

    def test_api_url_property(self):
        config = OllamaConfig()
        assert "/api/generate" in config.api_url


class TestTranslationConfig:
    """Test translation processing configuration."""

    def test_default_values(self):
        config = TranslationConfig()
        assert config.chunk_size >= 100
        assert config.max_prompt_length >= 1000
        assert config.max_retries >= 1
        assert 0 <= config.similarity_threshold <= 1

    def test_max_workers(self):
        config = TranslationConfig()
        assert config.max_workers >= 1


class TestCacheConfig:
    """Test cache configuration."""

    def test_default_values(self):
        config = CacheConfig()
        assert isinstance(config.enabled, bool)
        assert config.max_age_days > 0
        assert config.context_hash_length > 0


class TestFileConfig:
    """Test file handling configuration."""

    def test_max_file_size_bytes(self):
        config = FileConfig()
        assert config.max_file_size_bytes == config.max_file_size_mb * 1024 * 1024

    def test_allowed_extensions(self):
        config = FileConfig()
        assert ".txt" in config.allowed_extensions


class TestLoggingConfig:
    """Test logging configuration."""

    def test_default_values(self):
        config = LoggingConfig()
        assert isinstance(config.verbose_debug, bool)
        assert config.log_buffer_size > 0
        assert config.log_file_max_bytes > 0


class TestMainConfig:
    """Test main application configuration."""

    def test_all_sections_present(self):
        config = Config()
        assert hasattr(config, "server")
        assert hasattr(config, "ollama")
        assert hasattr(config, "translation")
        assert hasattr(config, "cache")
        assert hasattr(config, "file")
        assert hasattr(config, "logging")
        assert hasattr(config, "security")
        assert hasattr(config, "paths")

    def test_paths_are_valid(self):
        config = Config()
        assert config.paths.upload_folder is not None
        assert config.paths.translations_folder is not None
        assert config.paths.static_folder is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

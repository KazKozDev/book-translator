"""
Ollama API Client
=================
Client for interacting with the Ollama API.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import requests

from book_translator.config import config
from book_translator.models.schemas import ModelInfo
from book_translator.utils.logging import debug_print, get_logger


@dataclass
class OllamaResponse:
    """Response from Ollama API."""

    success: bool
    text: Optional[str] = None
    error: Optional[str] = None
    model: Optional[str] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


# Reasoning-capable model families and how their "think" field behaves.
# gpt-oss ONLY accepts a level string ("low"/"medium"/"high") - passing a
# boolean is silently ignored (per Ollama's docs), while Qwen3/DeepSeek
# reasoning models take a plain boolean.
_THINK_LEVEL_MODELS = ("gpt-oss",)
_THINK_BOOL_MODELS = (
    "qwen3",
    "qwen-3",
    "deepseek-r1",
    "deepseek-v3.1",
    "deepseek-v3",
    "magistral",
)


def _resolve_think_option(model: str, override: str = ""):
    """
    Work out what to send as the "think" option for a given model.

    Returns None if the field should be omitted entirely (model isn't a
    known reasoning model, so we don't want to send it an option it might
    not understand).
    """
    override = (override or "").strip().lower()
    if override:
        if override in ("true", "false"):
            return override == "true"
        if override in ("low", "medium", "high"):
            return override
        return None

    model_lower = (model or "").lower()
    if any(name in model_lower for name in _THINK_LEVEL_MODELS):
        return "low"
    if any(name in model_lower for name in _THINK_BOOL_MODELS):
        return False
    return None


class OllamaClient:
    """Client for Ollama API interactions."""

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or config.ollama.base_url
        self.model = model or config.ollama.default_model
        self.logger = get_logger().app_logger

        # Set up session with connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=config.translation.max_retries,
            pool_connections=10,
            pool_maxsize=10,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/api/generate"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/api/tags"

    def is_healthy(self) -> bool:
        """Check if Ollama is accessible."""
        try:
            response = self.session.get(
                self.models_url, timeout=config.ollama.health_check_timeout
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Ollama health check failed: {e}")
            return False

    def list_models(self) -> List[ModelInfo]:
        """List available models."""
        try:
            response = self.session.get(
                self.models_url, timeout=config.ollama.connect_timeout
            )
            response.raise_for_status()
            data = response.json()

            models = []
            for model_data in data.get("models", []):
                models.append(
                    ModelInfo(
                        name=model_data.get("name", ""),
                        size=model_data.get("size"),
                        modified_at=model_data.get("modified_at"),
                        digest=model_data.get("digest"),
                    )
                )
            return models
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []

    def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        top_p: float = None,
        stream: bool = False,
    ) -> OllamaResponse:
        """
        Generate text using Ollama.

        Args:
            prompt: The prompt to send
            model: Model to use (defaults to configured model)
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            stream: Whether to stream the response

        Returns:
            OllamaResponse with the result
        """
        model = model or self.model
        temperature = (
            temperature if temperature is not None else config.ollama.temperature
        )
        top_p = top_p if top_p is not None else config.ollama.top_p

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_ctx": config.ollama.num_ctx,
            },
        }

        think_option = _resolve_think_option(model, config.ollama.think)
        if think_option is not None:
            payload["think"] = think_option

        try:
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=(config.ollama.connect_timeout, config.ollama.read_timeout),
            )
            response.raise_for_status()

            if stream:
                # For streaming, return the response object
                return OllamaResponse(success=True, text="", model=model)

            result = response.json()
            text = result.get("response", "")

            if not text:
                # The model returned HTTP 200 but produced no final answer.
                # This happens with reasoning models (e.g. gpt-oss) when the
                # context window fills up with hidden "thinking" tokens
                # before the real answer is written - it is NOT a network
                # error, so treat it as an explicit failure rather than
                # silently succeeding with empty text.
                thinking = result.get("thinking", "")
                self.logger.warning(
                    f"Empty response from {model} "
                    f"(thinking={len(thinking)} chars, done_reason={result.get('done_reason')})"
                )
                return OllamaResponse(
                    success=False,
                    error=(
                        "Empty response from model (likely ran out of context/"
                        "reasoning budget - try a larger OLLAMA_NUM_CTX or a "
                        "lower think level)"
                    ),
                    model=model,
                )

            return OllamaResponse(
                success=True,
                text=text,
                model=model,
                eval_count=result.get("eval_count"),
                eval_duration=result.get("eval_duration"),
            )

        except requests.Timeout:
            return OllamaResponse(success=False, error="Request timed out")
        except requests.RequestException as e:
            return OllamaResponse(success=False, error=str(e))
        except json.JSONDecodeError as e:
            return OllamaResponse(success=False, error=f"Invalid JSON response: {e}")

    def generate_stream(
        self, prompt: str, model: str = None, temperature: float = None
    ) -> Generator[str, None, None]:
        """
        Generate text with streaming response.

        Args:
            prompt: The prompt to send
            model: Model to use
            temperature: Temperature for generation

        Yields:
            Text chunks as they are generated
        """
        model = model or self.model
        temperature = (
            temperature if temperature is not None else config.ollama.temperature
        )

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": config.ollama.top_p,
                "num_ctx": config.ollama.num_ctx,
            },
        }

        think_option = _resolve_think_option(model, config.ollama.think)
        if think_option is not None:
            payload["think"] = think_option

        try:
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=(config.ollama.connect_timeout, config.ollama.read_timeout),
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}")
            yield f"[ERROR: {e}]"

    def close(self):
        """Close the session."""
        self.session.close()


# Global client instance
_client_instance: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get or create the global Ollama client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = OllamaClient()
    return _client_instance

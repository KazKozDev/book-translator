"""Frontier-provider calls for web-grounded glossary verification.

The browser sends a provider choice and, optionally, a session-only API key.
Provider secrets are never logged or persisted here. Every response is checked
before it can reach the Apply button: web search must actually have run, every
source term must be preserved in order, and every output line must contain a
translation plus one explicit glossary mode.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


PROVIDERS: Dict[str, Dict[str, str]] = {
    'openai': {
        'label': 'OpenAI',
        'model': 'gpt-5.6-luna',
        'env_key': 'OPENAI_API_KEY',
    },
    'anthropic': {
        'label': 'Anthropic',
        'model': 'claude-haiku-4-5-20251001',
        'env_key': 'ANTHROPIC_API_KEY',
    },
    'google': {
        'label': 'Google',
        'model': 'gemini-3.5-flash-lite',
        'env_key': 'GEMINI_API_KEY',
    },
}

FRONTIER_READ_TIMEOUT = 600
LOCAL_ENV_FILE = Path(__file__).resolve().parents[1] / '.env.local'
LOCAL_ENV_NAMES = {
    'OPENAI_API_KEY',
    'ANTHROPIC_API_KEY',
    'GEMINI_API_KEY',
    'GOOGLE_API_KEY',
}
VALID_MODES = {'exact', 'inflectable', 'preferred'}
MODEL_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')
OUTPUT_LINE = re.compile(
    r'^(?P<source>.+?)\s*=>\s*(?P<target>.+?)\s*\|\s*'
    r'(?P<mode>exact|inflectable|preferred)\s*$',
    re.IGNORECASE,
)


class FrontierGlossaryError(RuntimeError):
    """A safe error whose text may be shown in the browser."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def _local_env_values() -> Dict[str, str]:
    """Read only supported keys from the ignored, owner-only local env file."""
    if not LOCAL_ENV_FILE.exists():
        return {}
    if LOCAL_ENV_FILE.is_symlink():
        raise FrontierGlossaryError(
            '.env.local must be a regular file, not a symbolic link',
            400,
        )
    if os.name != 'nt':
        file_mode = stat.S_IMODE(LOCAL_ENV_FILE.stat().st_mode)
        if file_mode & 0o077:
            raise FrontierGlossaryError(
                '.env.local permissions are too open; run chmod 600 .env.local',
                400,
            )

    values: Dict[str, str] = {}
    try:
        lines = LOCAL_ENV_FILE.read_text(encoding='utf-8').splitlines()
    except OSError as exc:
        raise FrontierGlossaryError(
            '.env.local could not be read safely',
            400,
        ) from exc

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        name, value = line.split('=', 1)
        name = name.strip()
        if name not in LOCAL_ENV_NAMES:
            continue
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        if value:
            values[name] = value
    return values


@dataclass(frozen=True)
class FrontierResult:
    glossary: str
    provider: str
    provider_label: str
    model: str
    changes: List[Dict[str, str]]
    searched: bool


def provider_catalog() -> List[Dict[str, Any]]:
    """Public provider metadata, with secret availability but never values."""
    try:
        local_values = _local_env_values()
    except FrontierGlossaryError:
        local_values = {}
    catalog = []
    for provider, details in PROVIDERS.items():
        env_names = [details['env_key']]
        if provider == 'google':
            env_names.append('GOOGLE_API_KEY')
        catalog.append({
            'id': provider,
            'label': details['label'],
            'model': details['model'],
            'environment_key_available': any(
                bool(os.environ.get(name, '').strip() or local_values.get(name))
                for name in env_names
            ),
        })
    return catalog


def _api_key(provider: str, submitted_key: Optional[str]) -> str:
    if provider not in PROVIDERS:
        raise FrontierGlossaryError('Choose a supported frontier provider', 400)
    if isinstance(submitted_key, str) and submitted_key.strip():
        return submitted_key.strip()

    local_values = _local_env_values()
    env_names = [PROVIDERS[provider]['env_key']]
    if provider == 'google':
        env_names.append('GOOGLE_API_KEY')
    for name in env_names:
        value = os.environ.get(name, '').strip() or local_values.get(name, '')
        if value:
            return value
    raise FrontierGlossaryError(
        f"Add a {PROVIDERS[provider]['label']} API key in Settings first",
        400,
    )


def _safe_http_error(provider: str, response: requests.Response) -> FrontierGlossaryError:
    label = PROVIDERS[provider]['label']
    if response.status_code in {401, 403}:
        message = f'{label} rejected the API key'
    elif response.status_code == 429:
        message = f'{label} rate limit or account quota was reached'
    else:
        message = f'{label} request failed (HTTP {response.status_code})'
    return FrontierGlossaryError(message)


def _model_id(provider: str, submitted_model: Optional[str]) -> str:
    if submitted_model is None or not submitted_model.strip():
        return PROVIDERS[provider]['model']
    model = submitted_model.strip()
    if not MODEL_ID.fullmatch(model):
        raise FrontierGlossaryError(
            'Model name may contain only letters, numbers, dots, colons, '
            'underscores, and hyphens',
            400,
        )
    return model


def _post_json(
    provider: str,
    client: Any,
    url: str,
    *,
    headers: Dict[str, str],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        response = client.post(
            url,
            headers=headers,
            json=payload,
            timeout=(10, FRONTIER_READ_TIMEOUT),
        )
    except requests.Timeout as exc:
        raise FrontierGlossaryError(
            f"{PROVIDERS[provider]['label']} verification timed out"
        ) from exc
    except requests.RequestException as exc:
        raise FrontierGlossaryError(
            f"{PROVIDERS[provider]['label']} could not be reached"
        ) from exc

    if not response.ok:
        raise _safe_http_error(provider, response)
    try:
        data = response.json()
    except ValueError as exc:
        raise FrontierGlossaryError(
            f"{PROVIDERS[provider]['label']} returned an unreadable response"
        ) from exc
    if not isinstance(data, dict):
        raise FrontierGlossaryError(
            f"{PROVIDERS[provider]['label']} returned an unreadable response"
        )
    return data


def _openai(
    prompt: str,
    api_key: str,
    model: str,
    client: Any,
) -> Tuple[str, bool]:
    data = _post_json(
        'openai',
        client,
        'https://api.openai.com/v1/responses',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        payload={
            'model': model,
            'input': prompt,
            'reasoning': {'effort': 'medium'},
            'tools': [{'type': 'web_search', 'search_context_size': 'high'}],
            'tool_choice': 'auto',
            'store': False,
        },
    )
    output = data.get('output') or []
    searched = any(
        isinstance(item, dict) and item.get('type') == 'web_search_call'
        for item in output
    )
    text_parts = []
    for item in output:
        if not isinstance(item, dict) or item.get('type') != 'message':
            continue
        for part in item.get('content') or []:
            if isinstance(part, dict) and part.get('type') == 'output_text':
                text_parts.append(part.get('text') or '')
    return '\n'.join(part for part in text_parts if part).strip(), searched


def _anthropic(
    prompt: str,
    api_key: str,
    model: str,
    client: Any,
) -> Tuple[str, bool]:
    messages: List[Dict[str, Any]] = [{'role': 'user', 'content': prompt}]
    searched = False
    final_data: Dict[str, Any] = {}
    for _ in range(2):
        final_data = _post_json(
            'anthropic',
            client,
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json',
            },
            payload={
                'model': model,
                'max_tokens': 12000,
                'messages': messages,
                'tools': [{
                    'type': 'web_search_20250305',
                    'name': 'web_search',
                    'max_uses': 12,
                }],
            },
        )
        content = final_data.get('content') or []
        usage = final_data.get('usage') or {}
        server_usage = usage.get('server_tool_use') or {}
        searched = searched or bool(server_usage.get('web_search_requests'))
        searched = searched or any(
            isinstance(block, dict)
            and (
                block.get('type') == 'web_search_tool_result'
                or (
                    block.get('type') == 'server_tool_use'
                    and block.get('name') == 'web_search'
                )
            )
            for block in content
        )
        if final_data.get('stop_reason') != 'pause_turn':
            break
        messages.append({'role': 'assistant', 'content': content})

    if final_data.get('stop_reason') == 'pause_turn':
        raise FrontierGlossaryError(
            'Anthropic paused the verification before producing a glossary'
        )
    text = '\n'.join(
        block.get('text') or ''
        for block in final_data.get('content') or []
        if isinstance(block, dict) and block.get('type') == 'text'
    ).strip()
    return text, searched


def _google(
    prompt: str,
    api_key: str,
    model: str,
    client: Any,
) -> Tuple[str, bool]:
    data = _post_json(
        'google',
        client,
        f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
        headers={
            'x-goog-api-key': api_key,
            'Content-Type': 'application/json',
        },
        payload={
            'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
            'tools': [{'google_search': {}}],
            'generationConfig': {'maxOutputTokens': 12000},
        },
    )
    candidates = data.get('candidates') or []
    if not candidates or not isinstance(candidates[0], dict):
        return '', False
    candidate = candidates[0]
    metadata = candidate.get('groundingMetadata') or {}
    searched = bool(
        metadata.get('webSearchQueries') or metadata.get('groundingChunks')
    )
    parts = (candidate.get('content') or {}).get('parts') or []
    text = '\n'.join(
        part.get('text') or ''
        for part in parts
        if isinstance(part, dict) and part.get('text')
    ).strip()
    return text, searched


def _source_from_input_line(line: str) -> str:
    if '\t' in line:
        return line.split('\t', 1)[0].strip()
    if '=>' in line:
        return line.split('=>', 1)[0].strip()
    if '=' in line:
        return line.split('=', 1)[0].strip()
    if '|' in line:
        return line.split('|', 1)[0].strip()
    return line.strip()


def _input_entries(glossary: str) -> List[Tuple[str, str]]:
    entries = []
    for raw_line in glossary.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        source = _source_from_input_line(line)
        if not source:
            raise FrontierGlossaryError(
                'A damaged glossary line has no recoverable source term',
                400,
            )
        entries.append((source, line))
    if not entries:
        raise FrontierGlossaryError('Add at least one glossary entry first', 400)
    return entries


def validate_frontier_output(
    original_glossary: str,
    candidate_glossary: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """Validate and normalize a provider response without inventing fallbacks."""
    original_entries = _input_entries(original_glossary)
    candidate_lines = [
        line.strip() for line in candidate_glossary.splitlines() if line.strip()
    ]
    if len(candidate_lines) != len(original_entries):
        raise FrontierGlossaryError(
            'The frontier model changed the number of glossary entries'
        )

    normalized = []
    changes = []
    for index, ((expected_source, before), line) in enumerate(
        zip(original_entries, candidate_lines),
        start=1,
    ):
        match = OUTPUT_LINE.fullmatch(line)
        if not match:
            raise FrontierGlossaryError(
                f'The frontier model returned a malformed glossary line {index}'
            )
        source = match.group('source').strip()
        target = match.group('target').strip()
        mode = match.group('mode').lower()
        if source != expected_source:
            raise FrontierGlossaryError(
                f'The frontier model changed source term on line {index}'
            )
        if not target or mode not in VALID_MODES:
            raise FrontierGlossaryError(
                f'The frontier model returned a malformed glossary line {index}'
            )
        after = f'{source} => {target} | {mode}'
        normalized.append(after)
        if before != after:
            changes.append({
                'source': source,
                'before': before,
                'after': after,
            })
    return '\n'.join(normalized), changes


def verify_glossary(
    provider: str,
    prompt: str,
    original_glossary: str,
    submitted_key: Optional[str] = None,
    submitted_model: Optional[str] = None,
    *,
    client: Any = requests,
) -> FrontierResult:
    """Call one selected provider and validate its web-grounded answer."""
    if provider not in PROVIDERS:
        raise FrontierGlossaryError('Choose a supported frontier provider', 400)
    model = _model_id(provider, submitted_model)
    api_key = _api_key(provider, submitted_key)
    if provider == 'openai':
        candidate, searched = _openai(prompt, api_key, model, client)
    elif provider == 'anthropic':
        candidate, searched = _anthropic(prompt, api_key, model, client)
    elif provider == 'google':
        candidate, searched = _google(prompt, api_key, model, client)
    else:  # _api_key already guards this; keep the dispatch total.
        raise FrontierGlossaryError('Choose a supported frontier provider', 400)

    if not searched:
        raise FrontierGlossaryError(
            f"{PROVIDERS[provider]['label']} did not use web search; "
            'the glossary was not treated as verified'
        )
    if not candidate:
        raise FrontierGlossaryError(
            f"{PROVIDERS[provider]['label']} returned no corrected glossary"
        )
    glossary, changes = validate_frontier_output(original_glossary, candidate)
    details = PROVIDERS[provider]
    return FrontierResult(
        glossary=glossary,
        provider=provider,
        provider_label=details['label'],
        model=model,
        changes=changes,
        searched=True,
    )

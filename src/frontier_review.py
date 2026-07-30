"""Cloud-provider decisions for Review Desk proposed fixes.

This module owns the remote call and the strict decision contract. It never
writes translation data: the browser applies every returned Final through the
Review Desk's existing revision-checked PATCH route, so cloud decisions cannot
bypass canonical-text, EPUB regrouping, or quality-result invalidation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

import prompts
from frontier_glossary import (
    PROVIDERS,
    FrontierGlossaryError,
    _api_key,
    _model_id,
    _post_json,
)


@dataclass(frozen=True)
class FrontierReviewResult:
    decisions: List[Dict[str, Any]]
    provider: str
    provider_label: str
    model: str


def build_review_prompt(cases: List[Dict[str, Any]]) -> str:
    """Render the complete provider prompt from review cases."""
    return prompts.render(
        'review/frontier_decision',
        cases_json=json.dumps(cases, ensure_ascii=False, indent=2),
    )


def _openai(prompt: str, api_key: str, model: str, client: Any) -> str:
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
            'store': False,
        },
    )
    parts = []
    for item in data.get('output') or []:
        if not isinstance(item, dict) or item.get('type') != 'message':
            continue
        for part in item.get('content') or []:
            if isinstance(part, dict) and part.get('type') == 'output_text':
                parts.append(part.get('text') or '')
    return '\n'.join(part for part in parts if part).strip()


def _anthropic(prompt: str, api_key: str, model: str, client: Any) -> str:
    data = _post_json(
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
            'messages': [{'role': 'user', 'content': prompt}],
        },
    )
    return '\n'.join(
        block.get('text') or ''
        for block in data.get('content') or []
        if isinstance(block, dict) and block.get('type') == 'text'
    ).strip()


def _google(prompt: str, api_key: str, model: str, client: Any) -> str:
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
            'generationConfig': {
                'maxOutputTokens': 12000,
                'responseMimeType': 'application/json',
            },
        },
    )
    candidates = data.get('candidates') or []
    if not candidates or not isinstance(candidates[0], dict):
        return ''
    return '\n'.join(
        part.get('text') or ''
        for part in (candidates[0].get('content') or {}).get('parts') or []
        if isinstance(part, dict) and part.get('text')
    ).strip()


def _json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith('```') and candidate.endswith('```'):
        lines = candidate.splitlines()
        candidate = '\n'.join(lines[1:-1]).strip()
    try:
        payload = json.loads(candidate)
    except (TypeError, ValueError) as exc:
        raise FrontierGlossaryError(
            'The frontier model returned an unreadable Review Desk decision'
        ) from exc
    if not isinstance(payload, dict):
        raise FrontierGlossaryError(
            'The frontier model returned an unreadable Review Desk decision'
        )
    return payload


def _validated_decisions(
    cases: List[Dict[str, Any]],
    response_text: str,
) -> List[Dict[str, Any]]:
    payload = _json_object(response_text)
    returned = payload.get('decisions')
    if not isinstance(returned, list):
        raise FrontierGlossaryError(
            'The frontier model omitted Review Desk decisions'
        )

    expected_cases = {
        int(case['chunk_index']): case
        for case in cases
    }
    returned_cases: Dict[int, Dict[str, Any]] = {}
    for decision in returned:
        if not isinstance(decision, dict):
            raise FrontierGlossaryError(
                'The frontier model returned a malformed chunk decision'
            )
        chunk_index = decision.get('chunk_index')
        if type(chunk_index) is not int or chunk_index not in expected_cases:
            raise FrontierGlossaryError(
                'The frontier model changed a Review Desk chunk index'
            )
        if chunk_index in returned_cases:
            raise FrontierGlossaryError(
                'The frontier model duplicated a Review Desk chunk decision'
            )
        returned_cases[chunk_index] = decision

    if set(returned_cases) != set(expected_cases):
        raise FrontierGlossaryError(
            'The frontier model did not decide every Review Desk chunk'
        )

    validated = []
    for chunk_index, case in expected_cases.items():
        expected_issues = {
            int(issue['issue_index']): issue
            for issue in case['issues']
        }
        raw_choices = returned_cases[chunk_index].get('choices')
        if not isinstance(raw_choices, list):
            raise FrontierGlossaryError(
                f'The frontier model omitted choices for chunk {chunk_index + 1}'
            )
        choices: Dict[int, Dict[str, Any]] = {}
        for choice in raw_choices:
            if not isinstance(choice, dict):
                raise FrontierGlossaryError(
                    f'The frontier model returned a malformed choice for chunk {chunk_index + 1}'
                )
            issue_index = choice.get('issue_index')
            apply = choice.get('apply')
            reason = choice.get('reason')
            if type(issue_index) is not int or issue_index not in expected_issues:
                raise FrontierGlossaryError(
                    f'The frontier model changed an issue index for chunk {chunk_index + 1}'
                )
            if issue_index in choices:
                raise FrontierGlossaryError(
                    f'The frontier model duplicated an issue choice for chunk {chunk_index + 1}'
                )
            if type(apply) is not bool or not isinstance(reason, str) or not reason.strip():
                raise FrontierGlossaryError(
                    f'The frontier model returned a malformed choice for chunk {chunk_index + 1}'
                )
            choices[issue_index] = {
                'issue_index': issue_index,
                'apply': apply,
                'reason': reason.strip(),
            }
        if set(choices) != set(expected_issues):
            raise FrontierGlossaryError(
                f'The frontier model did not decide every issue for chunk {chunk_index + 1}'
            )

        final_text = str(case['final'])
        applied = []
        kept = []
        ordered_choices = []
        for issue_index in sorted(expected_issues):
            choice = choices[issue_index]
            ordered_choices.append(choice)
            if not choice['apply']:
                kept.append(issue_index)
                continue
            issue = expected_issues[issue_index]
            span = issue['span']
            replacement = issue['replacement']
            position = final_text.find(span)
            if position < 0:
                raise FrontierGlossaryError(
                    f'Chosen fixes conflict in chunk {chunk_index + 1}; nothing was applied'
                )
            final_text = (
                final_text[:position]
                + replacement
                + final_text[position + len(span):]
            )
            applied.append(issue_index)

        validated.append({
            'chunk_index': chunk_index,
            'revision': int(case['revision']),
            'text': final_text,
            'applied_issue_indexes': applied,
            'kept_issue_indexes': kept,
            'choices': ordered_choices,
        })
    return validated


def decide_review_cases(
    provider: str,
    cases: List[Dict[str, Any]],
    submitted_key: Optional[str] = None,
    submitted_model: Optional[str] = None,
    *,
    client: Any = requests,
) -> FrontierReviewResult:
    """Ask one configured cloud provider to apply or keep every proposed fix."""
    if provider not in PROVIDERS:
        raise FrontierGlossaryError('Choose a supported frontier provider', 400)
    if not cases:
        raise FrontierGlossaryError(
            'No open Review Desk chunks have applicable proposed fixes',
            400,
        )
    model = _model_id(provider, submitted_model)
    api_key = _api_key(provider, submitted_key)
    prompt = build_review_prompt(cases)
    if provider == 'openai':
        response_text = _openai(prompt, api_key, model, client)
    elif provider == 'anthropic':
        response_text = _anthropic(prompt, api_key, model, client)
    else:
        response_text = _google(prompt, api_key, model, client)
    if not response_text:
        raise FrontierGlossaryError(
            f"{PROVIDERS[provider]['label']} returned no Review Desk decision"
        )
    return FrontierReviewResult(
        decisions=_validated_decisions(cases, response_text),
        provider=provider,
        provider_label=PROVIDERS[provider]['label'],
        model=model,
    )

import json

import pytest

import frontier_review as review
from frontier_glossary import FrontierGlossaryError


CASES = [
    {
        'chunk_index': 1,
        'revision': 2,
        'source_language': 'English',
        'target_language': 'Russian',
        'source': 'The firm made drills.',
        'draft': 'Фирма делала свёрла.',
        'final': 'Фирма выпускала дрели.',
        'issues': [
            {
                'issue_index': 0,
                'span': 'выпускала',
                'replacement': 'производила',
                'type': 'mistranslation',
                'severity': 'major',
            },
            {
                'issue_index': 1,
                'span': 'дрели',
                'replacement': 'сверла',
                'type': 'style',
                'severity': 'minor',
            },
        ],
    },
]


class FakeResponse:
    ok = True
    status_code = 200

    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


class FakeClient:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.data)


def _openai_response(decisions):
    return {
        'output': [{
            'type': 'message',
            'content': [{
                'type': 'output_text',
                'text': json.dumps({'decisions': decisions}, ensure_ascii=False),
            }],
        }],
    }


def test_openai_decides_every_issue_and_returns_a_deterministic_final():
    client = FakeClient(_openai_response([{
        'chunk_index': 1,
        'choices': [
            {
                'issue_index': 0,
                'apply': True,
                'reason': 'The source describes production.',
            },
            {
                'issue_index': 1,
                'apply': False,
                'reason': 'The current noun is the established rendering.',
            },
        ],
    }]))

    result = review.decide_review_cases(
        'openai',
        CASES,
        'session-key',
        'gpt-5.4-mini',
        client=client,
    )

    assert result.model == 'gpt-5.4-mini'
    assert result.decisions == [{
        'chunk_index': 1,
        'revision': 2,
        'text': 'Фирма производила дрели.',
        'applied_issue_indexes': [0],
        'kept_issue_indexes': [1],
        'choices': [
            {
                'issue_index': 0,
                'apply': True,
                'reason': 'The source describes production.',
            },
            {
                'issue_index': 1,
                'apply': False,
                'reason': 'The current noun is the established rendering.',
            },
        ],
    }]
    url, request = client.calls[0]
    assert url == 'https://api.openai.com/v1/responses'
    assert request['json']['store'] is False
    assert 'tools' not in request['json']
    assert request['timeout'] == (10, 600)


@pytest.mark.parametrize('decisions, message', [
    ([], 'did not decide every'),
    (
        [{
            'chunk_index': 1,
            'choices': [{
                'issue_index': 0,
                'apply': True,
                'reason': 'Supported.',
            }],
        }],
        'did not decide every issue',
    ),
    (
        [{
            'chunk_index': 1,
            'choices': [
                {
                    'issue_index': 0,
                    'apply': 'yes',
                    'reason': 'Supported.',
                },
                {
                    'issue_index': 1,
                    'apply': False,
                    'reason': 'Keep.',
                },
            ],
        }],
        'malformed choice',
    ),
])
def test_incomplete_or_malformed_cloud_decisions_are_rejected(decisions, message):
    client = FakeClient(_openai_response(decisions))

    with pytest.raises(FrontierGlossaryError, match=message):
        review.decide_review_cases(
            'openai',
            CASES,
            'session-key',
            client=client,
        )


def test_conflicting_applied_fixes_fail_without_returning_partial_text():
    cases = [{**CASES[0], 'issues': [
        {
            'issue_index': 0,
            'span': 'выпускала дрели',
            'replacement': 'производила сверла',
            'type': 'mistranslation',
            'severity': 'major',
        },
        {
            'issue_index': 1,
            'span': 'дрели',
            'replacement': 'инструменты',
            'type': 'style',
            'severity': 'minor',
        },
    ]}]
    client = FakeClient(_openai_response([{
        'chunk_index': 1,
        'choices': [
            {'issue_index': 0, 'apply': True, 'reason': 'First.'},
            {'issue_index': 1, 'apply': True, 'reason': 'Second.'},
        ],
    }]))

    with pytest.raises(FrontierGlossaryError, match='conflict'):
        review.decide_review_cases(
            'openai',
            cases,
            'session-key',
            client=client,
        )

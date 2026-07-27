"""Stage 2: the estimate/patch half that runs without a model.

The model's answer is untrusted input here — it arrives as prose-wrapped
JSON from a quantised local model and may name spans that do not exist. What
these tests pin down is that nothing unverifiable ever reaches the text.
"""

import pytest

from translator import BookTranslator


DRAFT = 'Мистер Дурсли был директором фирмы Grunnings, которая делала свёрла.'


def test_parses_json_wrapped_in_prose_and_fences():
    raw = 'Sure! Here are the errors I found:\n```json\n[{"span": "x"}]\n```\nHope that helps.'

    assert BookTranslator._parse_json_array(raw) == [{'span': 'x'}]


def test_unparseable_answers_yield_no_errors():
    for raw in (None, '', 'I could not find any problems.', '[not json]', '{"span": "x"}'):
        assert BookTranslator._parse_json_array(raw) == []


def test_only_spans_present_in_the_draft_survive_validation():
    errors = BookTranslator.validate_estimate_spans(
        [
            {'span': 'Grunnings', 'type': 'terminology', 'severity': 'major', 'replacement': 'Граннингс'},
            # Re-typed from memory rather than copied — no position to patch.
            {'span': 'Мистер Дурсль', 'type': 'consistency', 'severity': 'major', 'replacement': 'Мистер Дурсль'},
            # No replacement to apply.
            {'span': 'свёрла', 'type': 'mistranslation', 'severity': 'minor', 'replacement': ''},
        ],
        DRAFT,
    )

    assert [error['span'] for error in errors] == ['Grunnings']


def test_severity_orders_errors_and_unknown_categories_are_normalised():
    errors = BookTranslator.validate_estimate_spans(
        [
            {'span': 'свёрла', 'type': 'nonsense-category', 'severity': 'minor', 'replacement': 'дрели'},
            {'span': 'Grunnings', 'type': 'accuracy', 'severity': 'critical', 'replacement': 'Граннингс'},
        ],
        DRAFT,
    )

    assert [error['span'] for error in errors] == ['Grunnings', 'свёрла']
    assert errors[0]['type'] == 'mistranslation'  # aliased
    assert errors[1]['type'] == 'other'  # unrecognised, kept but not objective


def test_a_no_op_replacement_is_dropped():
    assert BookTranslator.validate_estimate_spans(
        [{'span': 'свёрла', 'type': 'style', 'severity': 'minor', 'replacement': 'свёрла'}],
        DRAFT,
    ) == []


def test_patch_touches_only_the_reported_spans():
    patched, applied = BookTranslator.stage2_patch(
        DRAFT,
        [
            {'span': 'Grunnings', 'replacement': 'Граннингс', 'type': 'terminology', 'severity': 'major'},
            {'span': 'свёрла', 'replacement': 'дрели', 'type': 'mistranslation', 'severity': 'major'},
        ],
    )

    assert patched == 'Мистер Дурсли был директором фирмы Граннингс, которая делала дрели.'
    assert len(applied) == 2
    # Everything outside the two spans is carried over character for
    # character — this is what keeps the draft/final diff small.
    assert patched.startswith('Мистер Дурсли был директором фирмы ')


def test_overlapping_spans_do_not_corrupt_the_text():
    draft = 'один два три'
    patched, applied = BookTranslator.stage2_patch(
        draft,
        [
            {'span': 'один два', 'replacement': 'ONE TWO', 'type': 'style', 'severity': 'major'},
            {'span': 'два три', 'replacement': 'TWO THREE', 'type': 'style', 'severity': 'major'},
        ],
    )

    assert patched == 'ONE TWO три'
    assert len(applied) == 1


def test_a_repeated_span_is_patched_once_per_reported_error():
    patched, applied = BookTranslator.stage2_patch(
        'кот и кот',
        [{'span': 'кот', 'replacement': 'пёс', 'type': 'style', 'severity': 'minor'}],
    )

    assert patched == 'пёс и кот'
    assert len(applied) == 1


def test_no_errors_leaves_the_draft_identical():
    patched, applied = BookTranslator.stage2_patch(DRAFT, [])

    assert patched == DRAFT
    assert applied == []


def test_style_edits_never_reach_the_text():
    """The observed failure: the review pass swapped a perfectly good verb
    for a longer synonym and the verifier waved it through, because on the
    style axis there is nothing to be wrong about."""
    assert not BookTranslator.is_actionable_error(
        {'type': 'style', 'severity': 'major', 'span': 'x', 'replacement': 'y'}
    )
    assert not BookTranslator.is_actionable_error(
        {'type': 'other', 'severity': 'critical', 'span': 'x', 'replacement': 'y'}
    )


def test_minor_subjective_errors_are_reported_but_not_applied():
    assert not BookTranslator.is_actionable_error(
        {'type': 'mistranslation', 'severity': 'minor', 'span': 'x', 'replacement': 'y'}
    )
    assert BookTranslator.is_actionable_error(
        {'type': 'mistranslation', 'severity': 'major', 'span': 'x', 'replacement': 'y'}
    )


def test_glossary_fixes_are_applied_at_any_severity():
    """A required rendering being absent is a fact, not a matter of degree."""
    for severity in ('critical', 'major', 'minor'):
        assert BookTranslator.is_actionable_error(
            {'type': 'terminology', 'severity': severity, 'span': 'x', 'replacement': 'y'}
        )
        assert BookTranslator.is_actionable_error(
            {'type': 'consistency', 'severity': severity, 'span': 'x', 'replacement': 'y'}
        )


@pytest.mark.parametrize('span', ['', 'x', '   '])
def test_spans_too_short_to_locate_are_rejected(span):
    assert BookTranslator.validate_estimate_spans(
        [{'span': span, 'type': 'style', 'severity': 'minor', 'replacement': 'что-то'}],
        DRAFT,
    ) == []


# -- The verify half. Still no Ollama: the model answers are scripted, which
# is the only way to pin down who was asked and what was done with the reply.

SOURCE = 'Mr Dursley was the director of a firm called Grunnings, which made drills.'


def _script_model_calls(monkeypatch, answers):
    """Replace every model call with a scripted answer, recording the model."""
    calls = []

    def fake_call(self, prompt, temperature=0.2, read_timeout=180):
        calls.append({'model': self.model_name, 'prompt': prompt})
        return answers.pop(0) if answers else None

    monkeypatch.setattr(BookTranslator, '_call_model', fake_call)
    return calls


def _refine(translator, monkeypatch, answers):
    calls = _script_model_calls(monkeypatch, answers)
    text, warning, details = translator.stage2_reflection_improvement(
        original_text=SOURCE,
        draft_translation=DRAFT,
        source_lang='english',
        target_lang='russian',
    )
    return text, warning, details, calls


def _reported(error_type, span='свёрла', replacement='дрели', severity='major'):
    return (
        '[{"span": "%s", "type": "%s", "severity": "%s", "replacement": "%s"}]'
        % (span, error_type, severity, replacement)
    )


def test_the_verdict_is_asked_of_the_verifier_model_not_the_reviewer(monkeypatch):
    """The reviewer grading its own edit is the arrangement that made this
    pass a no-op, so the two calls must reach the other model."""
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')

    # 'A' then 'B' is the patched version winning both orderings.
    text, warning, details, calls = _refine(
        translator, monkeypatch, [_reported('mistranslation'), 'A', 'B'],
    )

    assert [call['model'] for call in calls] == ['reviewer:12b', 'verifier:27b', 'verifier:27b']
    assert text == 'Мистер Дурсли был директором фирмы Grunnings, которая делала дрели.'
    assert warning is None
    assert details['verified'] == {
        'verdicts': ['patched', 'patched'],
        'accepted': True,
        'model': 'verifier:27b',
    }


def test_a_verdict_that_flips_with_the_order_rejects_the_patch(monkeypatch):
    """Answering 'A' both times is position bias, not agreement — and it is
    what a model asked to grade its own edit does. The draft is kept, and the
    details say who said so."""
    translator = BookTranslator(model_name='reviewer:12b')

    text, _, details, calls = _refine(
        translator, monkeypatch, [_reported('mistranslation'), 'A', 'A'],
    )

    assert text == DRAFT
    assert details['verified']['verdicts'] == ['patched', 'draft']
    assert details['verified']['accepted'] is False
    # No separate verifier chosen: every call went to the reviewing model.
    assert {call['model'] for call in calls} == {'reviewer:12b'}


def test_a_verifier_that_never_answers_is_recorded_as_unavailable(monkeypatch):
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')

    text, _, details, _ = _refine(
        translator, monkeypatch, [_reported('mistranslation')],  # verifier: no answer
    )

    assert text == DRAFT
    assert details['verified']['verdicts'] == ['unavailable', 'unavailable']


@pytest.mark.parametrize('error_type', ['omission', 'addition', 'terminology', 'consistency'])
def test_fixes_checkable_against_the_source_skip_the_verifier(monkeypatch, error_type):
    """Whether a clause is missing, invented, or rendered against the
    glossary is settled by reading the two texts. The A/B vote adds nothing
    there and can only veto a real fix."""
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')

    text, _, details, calls = _refine(
        translator, monkeypatch, [_reported(error_type)],
    )

    assert text != DRAFT
    assert details['verified'] == 'skipped_objective'
    assert len(calls) == 1  # estimate only — the verifier was never asked


def test_a_mixed_patch_still_faces_the_verifier(monkeypatch):
    """One mistranslation among the objective fixes puts the whole patch back
    under the vote — the spans are applied together and cannot be split."""
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')

    reported = (
        '[{"span": "свёрла", "type": "omission", "severity": "major", "replacement": "дрели"},'
        ' {"span": "Grunnings", "type": "mistranslation", "severity": "major",'
        ' "replacement": "Граннингс"}]'
    )
    _, _, details, calls = _refine(translator, monkeypatch, [reported, 'A', 'B'])

    assert details['verified']['accepted'] is True
    assert [call['model'] for call in calls] == ['reviewer:12b', 'verifier:27b', 'verifier:27b']


def test_no_separate_verifier_builds_no_second_translator():
    translator = BookTranslator(model_name='reviewer:12b')

    assert translator.verifier is translator
    assert translator.verifier_model == 'reviewer:12b'


def test_details_name_both_models_so_a_run_can_be_read_back(monkeypatch):
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')

    _, _, details, _ = _refine(translator, monkeypatch, ['[]'])

    assert details['review_model'] == 'reviewer:12b'
    assert details['verifier_model'] == 'verifier:27b'


def test_the_chunk_log_line_distinguishes_the_three_ways_nothing_changes():
    clean = BookTranslator._describe_stage2_chunk(
        {'errors_found': 0, 'errors_actionable': 0, 'errors_applied': 0, 'verified': None},
        warning=None, changed=False,
    )
    vetoed = BookTranslator._describe_stage2_chunk(
        {
            'errors_found': 2, 'errors_actionable': 2, 'errors_applied': 2,
            'verified': {'verdicts': ['patched', 'draft'], 'accepted': False, 'model': 'verifier:27b'},
        },
        warning=None, changed=False,
    )
    timed_out = BookTranslator._describe_stage2_chunk(
        {'errors_found': 0, 'errors_actionable': 0, 'errors_applied': 0, 'verified': None},
        warning='The review pass returned no output — kept the draft for this chunk.',
        changed=False,
    )

    assert 'verifier not needed' in clean
    assert 'verifier verifier:27b voted patched/draft → rejected' in vetoed
    assert 'review pass gave no answer' in timed_out
    assert clean != timed_out

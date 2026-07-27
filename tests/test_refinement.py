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

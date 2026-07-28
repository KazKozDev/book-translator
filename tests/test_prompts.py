"""The prompts in ``prompts/`` are the prompts the pipeline used to inline.

Every case here renders one role's prompt and compares it, character for
character, with a golden captured from the code as it stood before the texts
moved into files. That comparison is the whole warrant for the move: a prompt
is the program here, and a stray space or a lost line in one is a change in
translation quality that no other test in this suite would notice.

Editing a prompt on purpose means updating its golden in the same commit —
which is the point. The diff then says what the model will now be told.
"""

import random
from pathlib import Path

import pytest

import prompts
from quality_tests import QualityTests
from terminology import GlossaryTerm, TerminologyManager
from translator import BookTranslator

GOLDEN_DIR = Path(__file__).parent / 'fixtures' / 'prompts'

SOURCE = (
    'Mr Dursley was the director of a firm called Grunnings, which made drills. '
    'The Dursleys had everything they wanted, but they also had a secret. '
    "Mrs Potter was Mrs Dursley's sister, and they had not met for several years."
)
DRAFT = 'Мистер Дурсли был директором фирмы Grunnings, которая делала свёрла.'
FINAL = 'Мистер Дурсль был директором фирмы «Граннингс», которая выпускала дрели.'
PREVIOUS = 'Они были последними людьми, от которых можно было ожидать чего-то странного.'

GLOSSARY = TerminologyManager([
    GlossaryTerm(source='Grunnings', target='Граннингс', mode='exact'),
    GlossaryTerm(source='Dursley', target='Дурсль', mode='inflectable'),
    GlossaryTerm(source='drill', target='дрель', mode='preferred'),
])
TERMINOLOGY_CONTEXT = GLOSSARY.prompt_context(SOURCE)

# Long enough to be clipped at PREPARE_EVIDENCE_CHARS, so the golden pins the
# clipping too and not only the wording around it.
LONG_EVIDENCE = SOURCE + ' ' + SOURCE


def _translator() -> BookTranslator:
    """A translator built for its prompt builders alone — nothing here opens a
    socket, and no case is allowed to reach a model."""
    translator = BookTranslator(model_name='gemma3:27b')
    translator.terminology = GLOSSARY
    return translator


class _Recorder:
    """Stands in for ``_call_model``: keeps the prompt, answers nothing.

    Returning ``None`` is the pipeline's "the model did not answer" path, which
    every caller already handles, so a recorded case runs to completion without
    a model and without a branch of its own.
    """

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return None


def _recorded(translator: BookTranslator, call) -> list:
    recorder = _Recorder()
    translator._call_model = recorder
    call(translator)
    return recorder.prompts


# --------------------------------------------------------------------------
# Stage 0: prepare
# --------------------------------------------------------------------------

def stage0_rendering_from_candidates() -> str:
    return _translator()._rendering_prompt(
        SOURCE,
        [
            {
                'surface': 'Dursley',
                'kind': 'person',
                'count': 4,
                'variants': [{'form': 'Dursleys'}, {'form': 'Dursley'}],
                'evidence': [LONG_EVIDENCE, LONG_EVIDENCE, 'Mrs Potter was Mrs Dursley’s sister'],
            },
            {'surface': 'Grunnings', 'kind': 'organisation', 'count': 1, 'evidence': []},
        ],
        'en', 'ru', "a children's novel",
    )


def stage0_rendering_from_excerpt() -> str:
    """No candidate survived extraction, and no genre was detected — the two
    optional pieces of the prompt are both absent."""
    return _translator()._rendering_prompt(SOURCE, [], 'en', 'ru', 'unknown')


def stage0_cluster_adjudication() -> str:
    candidates = [{
        'surface': 'Dursley',
        'kind': 'person',
        'count': 4,
        'variants': [{'form': 'Dursleys'}],
        'evidence': [LONG_EVIDENCE, 'The Dursleys had everything they wanted'],
    }]
    review_queue = [{
        'a': 'Potter',
        'b': 'Dursley',
        'context_a': 'Mrs Potter was Mrs Dursley’s sister',
        'context_b': 'Mr Dursley was the director of a firm',
    }]
    return _recorded(
        _translator(),
        lambda translator: translator.adjudicate_entity_clusters(
            SOURCE, 'en', candidates, review_queue,
        ),
    )[0]


# --------------------------------------------------------------------------
# Stage 1: draft translation
# --------------------------------------------------------------------------

def stage1_default() -> str:
    return BookTranslator._stage1_prompt_default(
        SOURCE, 'English', 'Russian', PREVIOUS, "a children's novel", TERMINOLOGY_CONTEXT,
    )


def stage1_default_bare() -> str:
    """First chunk of a book with no glossary: no previous paragraph, no terms."""
    return BookTranslator._stage1_prompt_default(
        SOURCE, 'English', 'Russian', '', 'unknown', '',
    )


def stage1_translategemma() -> str:
    return BookTranslator._stage1_prompt_translategemma(
        SOURCE, 'en', 'ru', 'English', 'Russian',
        PREVIOUS, "a children's novel", TERMINOLOGY_CONTEXT,
    )


def stage1_translategemma_bare() -> str:
    """Nothing to add — the prompt the model card documents, byte for byte."""
    return BookTranslator._stage1_prompt_translategemma(
        SOURCE, 'en', 'ru', 'English', 'Russian', '', 'unknown', '',
    )


# --------------------------------------------------------------------------
# Stage 2: refinement
# --------------------------------------------------------------------------

def stage2_estimate() -> str:
    return _recorded(
        _translator(),
        lambda translator: translator.stage2_estimate(
            SOURCE, DRAFT, 'en', 'ru',
            terminology_context=TERMINOLOGY_CONTEXT,
            terminology_violations=[
                {'source': 'Grunnings', 'required_target': 'Граннингс'},
                {'source': 'drill', 'required_target': 'дрель'},
            ],
        ),
    )[0]


def stage2_estimate_bare() -> str:
    return _recorded(
        _translator(),
        lambda translator: translator.stage2_estimate(SOURCE, DRAFT, 'en', 'ru'),
    )[0]


def _verify_prompts() -> list:
    return _recorded(
        _translator(),
        lambda translator: translator.stage2_verify(SOURCE, DRAFT, FINAL, 'en', 'ru'),
    )


def stage2_verify_patched_first() -> str:
    return _verify_prompts()[0]


def stage2_verify_draft_first() -> str:
    """The same question with the two versions swapped — the second half of the
    double-blind vote, and the reason the ordering is pinned here."""
    return _verify_prompts()[1]


# --------------------------------------------------------------------------
# Stage 3: quality tests
# --------------------------------------------------------------------------

def quality_pairwise_editor() -> str:
    original_random = random.random
    random.random = lambda: 0.9  # No swap: version A is the draft.
    try:
        return _recorded(
            _translator(),
            lambda translator: translator.eval_llm_judge_stage2(
                [SOURCE], [DRAFT], [FINAL], 'en', 'ru',
            ),
        )[0]
    finally:
        random.random = original_random


def quality_adequacy_fluency() -> str:
    return QualityTests._adequacy_fluency_prompt('English', 'Russian', SOURCE, DRAFT)


# --------------------------------------------------------------------------
# Shared blocks
# --------------------------------------------------------------------------

def shared_terminology_context() -> str:
    return TERMINOLOGY_CONTEXT


CASES = {
    'stage0_rendering_from_candidates': stage0_rendering_from_candidates,
    'stage0_rendering_from_excerpt': stage0_rendering_from_excerpt,
    'stage0_cluster_adjudication': stage0_cluster_adjudication,
    'stage1_default': stage1_default,
    'stage1_default_bare': stage1_default_bare,
    'stage1_translategemma': stage1_translategemma,
    'stage1_translategemma_bare': stage1_translategemma_bare,
    'stage2_estimate': stage2_estimate,
    'stage2_estimate_bare': stage2_estimate_bare,
    'stage2_verify_patched_first': stage2_verify_patched_first,
    'stage2_verify_draft_first': stage2_verify_draft_first,
    'quality_pairwise_editor': quality_pairwise_editor,
    'quality_adequacy_fluency': quality_adequacy_fluency,
    'shared_terminology_context': shared_terminology_context,
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_prompt_matches_the_golden_it_was_captured_from(name):
    golden = (GOLDEN_DIR / f'{name}.txt').read_text(encoding='utf-8')
    assert CASES[name]() == golden


def test_every_prompt_file_is_reachable_from_the_pipeline():
    """A prompt file nobody loads is a prompt somebody is editing in vain."""
    for case in CASES.values():
        case()

    assert prompts.loaded() == set(prompts.names())


def test_a_missing_value_fails_loudly_rather_than_rendering_empty():
    with pytest.raises(KeyError):
        prompts.render('stage2_refine/verify', target_name='Russian')

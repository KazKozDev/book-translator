"""Unit tests for the reusable Stage 0 glossary-clustering module."""

import sys
import types

import pytest

import build_glossary


def test_builder_source_is_english_and_reusable(monkeypatch):
    """The web route and command line share an English-only builder API."""
    source = open(build_glossary.__file__, encoding='utf-8').read()
    assert not any('\u0400' <= char <= '\u04ff' for char in source)

    raw_records = [{
        'norm': 'mr fenwick',
        'label': 'person',
        'variants': build_glossary.Counter({'Mr. Fenwick': 3}),
        'chunks': [0, 1, 2],
        'contexts': ['Mr. Fenwick arrived.'],
    }]
    monkeypatch.setattr(build_glossary, 'extract', lambda *args, **kwargs: raw_records)
    monkeypatch.setattr(build_glossary, 'embed', lambda *args, **kwargs: None)
    monkeypatch.setattr(build_glossary, 'candidate_pairs', lambda *args, **kwargs: ([], []))

    glossary, review_queue = build_glossary.build_document_glossary('Mr. Fenwick arrived three times.')

    assert review_queue == []
    assert glossary == [{
        'id': 1,
        'canonical': 'Mr. Fenwick',
        'type': 'person',
        'count': 3,
        'variants': [{'form': 'Mr. Fenwick', 'count': 3}],
        'contexts': ['Mr. Fenwick arrived.'],
        'first_seen_chunk': 0,
        'status': 'auto',
    }]


def test_builder_progress_reaches_the_application_log(monkeypatch):
    """Stage 0's long waits belong in the log the interface follows.

    Model loading and the NER sweep are the slowest part of Prepare, and while
    they printed straight to stderr the Log window showed nothing at all
    between pressing Prepare and the glossary appearing.
    """
    import translator

    logged = []
    monkeypatch.setattr(translator.logger.translation_logger, 'info', logged.append)
    monkeypatch.setattr(
        build_glossary, 'build_document_glossary',
        lambda text: (build_glossary.report('NER: 2/2 (100.0%)'), ([], []))[1],
    )
    monkeypatch.setitem(sys.modules, 'build_glossary', build_glossary)
    original_report = build_glossary.report
    try:
        translator.BookTranslator.build_glossary_candidates('Mr. Fenwick arrived.')
    finally:
        build_glossary.report = original_report

    assert 'NER: 2/2 (100.0%)' in logged, 'builder progress never reached the log'
    assert any('Stage 0: extracting glossary candidates' in line for line in logged)
    assert any('Stage 0: extraction found' in line for line in logged)


def test_extract_uses_gliner_batch_inference(monkeypatch):
    """The integrated builder calls the same batched GLiNER API as its CLI."""
    seen = {}

    class FakeNer:
        def to(self, device):
            seen['device'] = device

        def inference(self, texts, labels, *, threshold, batch_size):
            seen.update(texts=texts, labels=labels, threshold=threshold, batch_size=batch_size)
            return [[{'text': 'Mr. Fenwick', 'label': 'person'}] for _ in texts]

    class FakeGLiNER:
        @staticmethod
        def from_pretrained(model_name):
            seen['model_name'] = model_name
            return FakeNer()

    monkeypatch.setitem(sys.modules, 'gliner', types.SimpleNamespace(GLiNER=FakeGLiNER))

    records = build_glossary.extract(
        ['Mr. Fenwick arrived.', 'Mr. Fenwick left.'], ['person'], 0.4,
        'local-gliner', 'cpu', 2,
    )

    assert seen == {
        'model_name': 'local-gliner',
        'device': 'cpu',
        'texts': ['Mr. Fenwick arrived.', 'Mr. Fenwick left.'],
        'labels': ['person'],
        'threshold': 0.4,
        'batch_size': 2,
    }
    assert records[0]['variants'] == build_glossary.Counter({'Mr. Fenwick': 2})


def test_pair_scoring_survives_being_done_in_blocks():
    """The pair sweep is quadratic, so it is computed a block of rows at a
    time in C rather than pair by pair in Python. It has to come back with
    the same pairs, in the same order, including across a block boundary."""
    from collections import Counter

    import numpy as np
    from rapidfuzz import fuzz

    names = ['harry potter', 'harry', 'potter', 'hogwarts', 'privet drive',
             'drive', 'dursley', 'dursleys', 'mrs dursley', 'surrey']
    records = [
        {
            'norm': names[i % len(names)] + ('' if i % 3 else ' ii'),
            'label': ['person', 'location', 'organization'][i % 3],
            'variants': Counter({names[i % len(names)].title(): 1 + i % 9}),
            'chunks': [], 'contexts': ['context %d' % i], 'count': 3 + i % 40,
            'buckets': {(i * 7) % 20, (i * 3) % 20},
        }
        for i in range(140)
    ]
    vectors = np.random.default_rng(11).random((len(records), 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    def reference():
        sims = vectors @ vectors.T
        merges, review = [], []
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                a, b = records[i], records[j]
                if a['label'] != b['label']:
                    continue
                cos = float(sims[i, j])
                if fuzz.token_set_ratio(a['norm'], b['norm']) < 60 and cos < 0.75:
                    continue
                score, features = build_glossary.score_pair(a, b, cos)
                if score >= 0.80:
                    merges.append((i, j, score))
                elif score >= 0.55:
                    review.append(round(score, 3))
        return merges, review

    original_block, build_glossary.PAIR_BLOCK = build_glossary.PAIR_BLOCK, 32
    try:
        merges, review = build_glossary.candidate_pairs(records, vectors, 0.80, 0.55)
    finally:
        build_glossary.PAIR_BLOCK = original_block
    expected_merges, expected_review = reference()

    # Which pairs, and in what order — this is the property blocking could
    # plausibly break, and it holds exactly.
    assert [(i, j) for i, j, _ in merges] == [(i, j) for i, j, _ in expected_merges]

    # The scores themselves are only equal to float32 precision, and asking for
    # more would be asking BLAS for a guarantee it does not give. A block of
    # rows against the whole set is a different shape of matmul than the full
    # square, so it accumulates in a different order; the vectors are float32,
    # and 0.45 * cos carries that ~1e-7 through into the score. The fuzzy half
    # of the score *is* bit-identical — candidate_pairs asks cdist for float64
    # precisely so no borderline pair changes side because of it.
    assert [score for *_, score in merges] == pytest.approx(
        [score for *_, score in expected_merges], abs=1e-6
    )
    assert [entry['score'] for entry in review] == sorted(expected_review, reverse=True)

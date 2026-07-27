"""Unit tests for the reusable Stage 0 glossary-clustering module."""

import sys
import types

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

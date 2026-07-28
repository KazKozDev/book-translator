#!/usr/bin/env python3
"""Build a reviewable glossary from a book.

Input:  book.epub or book.txt (any language)
Output: glossary.json + review_queue.json

Install:
    pip install gliner sentence-transformers rapidfuzz ebooklib beautifulsoup4

Run:
    python build_glossary.py book.epub -o out/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

DEFAULT_LABELS = ["person", "location", "organization", "object", "title"]
BUCKETS = 20  # Document segments used as a co-occurrence signal.

# Both checkpoints take tens of seconds to load and hundreds of megabytes of
# memory, and neither depends on the document. Loading them per call meant
# every press of Prepare paid the whole startup cost again — the dominant
# term in Stage 0's wall clock, well ahead of the model calls it exists for.
_MODEL_LOCK = threading.Lock()
_LOADED: dict[tuple[str, str, str], object] = {}
# One lock per checkpoint, not one for all of them: the two models here are
# loaded at the same time on purpose, and a single lock would put the second
# request in line behind the first instead of beside it.
_BUILD_LOCKS: dict[tuple[str, str, str], threading.Lock] = {}


def _print_to_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# Stage 0's long silences happen inside this module: a checkpoint that loads
# for tens of seconds, then one NER pass per batch over the whole book. On the
# command line stderr is the right place for that. An embedder has a log of its
# own, and translator.py points this at the one the interface follows — where
# the glossary run was previously doing all this work invisibly.
report = _print_to_stderr


def _load_once(kind: str, model_name: str, device: str, build):
    """Return a cached model, building it once however many callers ask."""
    key = (kind, model_name, device)
    with _MODEL_LOCK:
        model = _LOADED.get(key)
        if model is not None:
            return model
        build_lock = _BUILD_LOCKS.setdefault(key, threading.Lock())
    with build_lock:
        model = _LOADED.get(key)
        if model is None:
            report(f"Loading {model_name} …")
            model = build()
            with _MODEL_LOCK:
                _LOADED[key] = model
        return model


def _ignore_errors(function, *args) -> None:
    """Run a warm-up call for its cache entry, never for its outcome."""
    try:
        function(*args)
    except Exception:  # noqa: BLE001 — the real call reports this properly.
        pass


def _missing(package: str) -> RuntimeError:
    """Explain a missing model package the way the rest of the app does.

    Deferring these imports into the builders put them outside the guard the
    caller wraps around ``import build_glossary``, so an environment without
    them reached the log as a bare ModuleNotFoundError. The usual cause is not
    an incomplete venv but a server started with the system interpreter, which
    has enough of the app's dependencies to import this module and none of the
    model stack; say so, because "pip install" alone does not fix that.
    """
    return RuntimeError(
        f"{package} is missing from the interpreter running the app. Start it with "
        "./Launch Book-Translator.command (or ./venv/bin/python src/translator.py) so it "
        "runs inside the venv; if it already does, the venv is incomplete — run: "
        "./venv/bin/python -m pip install -r requirements.txt"
    )


def load_ner(model_name: str, device: str):
    def build():
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise _missing("gliner") from exc

        model = GLiNER.from_pretrained(model_name)
        model.to(device)
        return model

    return _load_once("ner", model_name, device, build)


def load_embedder(model_name: str, device: str):
    def build():
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise _missing("sentence-transformers") from exc

        return SentenceTransformer(model_name, device=device)

    return _load_once("embed", model_name, device, build)


# ---------------------------------------------------------------- reading

def read_book(path: Path) -> str:
    if path.suffix.lower() == ".epub":
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(str(path))
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            parts.append(soup.get_text("\n"))
        return "\n\n".join(parts)
    return path.read_text(encoding="utf-8", errors="ignore")


_SENT = re.compile(r"(?<=[.!?…。！？])\s+")


def split_chunks(text: str, max_chars: int = 600) -> list[str]:
    """Create two-to-three-sentence chunks with a one-sentence overlap."""
    sentences: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for s in _SENT.split(para):
            s = s.strip()
            if not s:
                continue
            # Split a very long sentence by length for scripts without spaces.
            while len(s) > max_chars:
                sentences.append(s[:max_chars])
                s = s[max_chars:]
            if s:
                sentences.append(s)

    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for s in sentences:
        if cur and size + len(s) > max_chars:
            chunks.append(" ".join(cur))
            cur = cur[-1:]  # overlap
            size = len(cur[0])
        cur.append(s)
        size += len(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ---------------------------------------------------------------- extraction

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize(s: str) -> str:
    return _SPACE.sub(" ", _PUNCT.sub(" ", s.casefold())).strip()


def extract(chunks: list[str], labels: list[str], threshold: float, model_name: str,
            device: str, batch_size: int):
    ner = load_ner(model_name, device)

    records: dict[tuple[str, str], dict] = {}
    started = time.monotonic()
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for start in range(0, len(chunks), batch_size):
        batch_number = start // batch_size + 1
        if batch_number == 1 or batch_number % 10 == 0 or start + batch_size >= len(chunks):
            done = min(start, len(chunks))
            elapsed = time.monotonic() - started
            rate = done / elapsed if elapsed else 0
            eta = (len(chunks) - done) / rate if rate else 0
            report(
                f"NER: {done:,}/{len(chunks):,} ({done / len(chunks):.1%}) | "
                f"{rate:.1f} chunks/s | ~{eta / 60:.1f} min remaining "
                f"(batch {batch_number}/{total_batches})"
            )
        batch = chunks[start:start + batch_size]
        predictions = ner.inference(batch, labels, threshold=threshold,
                                    batch_size=batch_size)
        for i, (chunk, entities) in enumerate(zip(batch, predictions), start):
            for ent in entities:
                surface = ent["text"].strip()
                norm = normalize(surface)
                if len(norm) < 2:
                    continue
                key = (norm, ent["label"])
                rec = records.setdefault(key, {
                    "norm": norm,
                    "label": ent["label"],
                    "variants": Counter(),
                    "chunks": [],
                    "contexts": [],
                })
                rec["variants"][surface] += 1
                rec["chunks"].append(i)
                if len(rec["contexts"]) < 3:
                    rec["contexts"].append(chunk)
    return list(records.values())


def keep_meaningful(records: list[dict], min_count: int, n_chunks: int) -> list[dict]:
    kept = []
    for r in records:
        count = sum(r["variants"].values())
        if count < min_count:
            continue
        # A low share of capitalised mentions is usually a common noun.
        caps = sum(n for v, n in r["variants"].items() if v[:1].isupper())
        if caps / count < 0.5 and not any(c.isupper() for v in r["variants"] for c in v):
            continue
        r["count"] = count
        r["buckets"] = {min(BUCKETS - 1, c * BUCKETS // max(1, n_chunks)) for c in r["chunks"]}
        kept.append(r)
    return kept


# ---------------------------------------------------------------- matching

def embed(records: list[dict], model_name: str, device: str) -> np.ndarray:
    model = load_embedder(model_name, device)
    texts = [
        r["variants"].most_common(1)[0][0] + ". " + " ".join(r["contexts"])[:800]
        for r in records
    ]
    vecs = model.encode(texts, batch_size=8, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True)
    return vecs


def score_pair(a: dict, b: dict, cos: float, f: float | None = None) -> tuple[float, dict]:
    # The caller normally has the token_set_ratio already, from the matrix it
    # built to decide this pair was worth scoring at all.
    if f is None:
        f = fuzz.token_set_ratio(a["norm"], b["norm"]) / 100.0
    ta, tb = set(a["norm"].split()), set(b["norm"].split())
    contained = ta <= tb or tb <= ta
    jac = len(a["buckets"] & b["buckets"]) / max(1, len(a["buckets"] | b["buckets"]))

    s = 0.40 * f + 0.45 * cos + 0.15 * jac
    if contained:
        s = min(1.0, s + 0.12)
    return s, {"fuzzy": round(f, 3), "cosine": round(cos, 3),
               "overlap": round(jac, 3), "contained": contained}


# How many records are compared against the whole set at a time. Bounds the
# similarity matrices to one block instead of the full n x n, which on a novel
# is the difference between tens of megabytes and hundreds.
PAIR_BLOCK = 512


def candidate_pairs(records: list[dict], vecs: np.ndarray, high: float, low: float):
    """Every same-label pair worth scoring, as (merges, review queue).

    The comparison is quadratic in the number of records, so on a full book
    this ran tens of millions of Python-level rapidfuzz calls — two per pair,
    since the cheap prefilter and the score recomputed the same ratio. Both
    matrices are now built a block of rows at a time in C, and the Python loop
    only visits pairs that survive the prefilter. The arithmetic and the
    thresholds are unchanged, so the pairs it produces are the same ones.
    """
    norms = [r["norm"] for r in records]
    labels = np.array([r["label"] for r in records])
    positions = np.arange(len(records))
    merges, review = [], []
    for start in range(0, len(records), PAIR_BLOCK):
        stop = min(start + PAIR_BLOCK, len(records))
        # token_set_ratio over one block of rows against every record, in C
        # and across cores. float64 keeps each value bit-identical to the
        # scalar call it replaces, so no borderline pair changes side.
        fuzzy = process.cdist(
            norms[start:stop], norms, scorer=fuzz.token_set_ratio,
            dtype=np.float64, workers=-1,
        )
        sims = vecs[start:stop] @ vecs.T
        for i in range(start, stop):
            row_fuzzy, row_cos = fuzzy[i - start], sims[i - start]
            # Never merge different entity types; only look forward, so each
            # pair is considered once; and skip what is too dissimilar to
            # score at all.
            interesting = np.flatnonzero(
                (positions > i)
                & (labels == labels[i])
                & ((row_fuzzy >= 60) | (row_cos >= 0.75))
            )
            a = records[i]
            for j in interesting:
                b = records[j]
                s, feats = score_pair(a, b, float(row_cos[j]), row_fuzzy[j] / 100.0)
                if s >= high:
                    merges.append((i, int(j), s))
                    continue
                if s < low:
                    continue
                review.append({
                    "a": a["variants"].most_common(1)[0][0],
                    "b": b["variants"].most_common(1)[0][0],
                    "type": a["label"],
                    "score": round(s, 3),
                    "features": feats,
                    "context_a": a["contexts"][0][:200],
                    "context_b": b["contexts"][0][:200],
                    "decision": None,   # "merge" | "separate" — set during review
                })
    review.sort(key=lambda r: -r["score"])
    return merges, review


# ---------------------------------------------------------------- clusters

class Union:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def join(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_glossary(records: list[dict], merges) -> list[dict]:
    uf = Union(len(records))
    for i, j, _ in merges:
        uf.join(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        groups[uf.find(i)].append(i)

    out = []
    for gid, members in enumerate(sorted(groups.values(), key=lambda m: -sum(
            records[i]["count"] for i in m)), start=1):
        variants: Counter = Counter()
        chunks: list[int] = []
        contexts: list[str] = []
        for i in members:
            variants.update(records[i]["variants"])
            chunks.extend(records[i]["chunks"])
            contexts.extend(records[i]["contexts"])
        canonical = max(variants.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        out.append({
            "id": gid,
            "canonical": canonical,
            "type": records[members[0]]["label"],
            "count": sum(variants.values()),
            "variants": [{"form": v, "count": n} for v, n in variants.most_common()],
            "contexts": contexts[:3],
            "first_seen_chunk": min(chunks),
            "status": "auto" if len(members) == 1 else "auto_merged",
        })
    return out


def select_device() -> str:
    """Choose the best available PyTorch device for local inference."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_document_glossary(
    text: str,
    *,
    labels: list[str] | None = None,
    min_count: int = 3,
    ner_threshold: float = 0.5,
    high: float = 0.80,
    low: float = 0.55,
    ner_model: str = "urchade/gliner_multi-v2.1",
    embed_model: str = "BAAI/bge-m3",
    device: str | None = None,
    batch_size: int = 16,
) -> tuple[list[dict], list[dict]]:
    """Return clustered glossary candidates and ambiguous pairs for review.

    This is the reusable API used by the web application's Prepare stage as
    well as the command-line tool below.  It never writes files or translates
    a term; callers remain responsible for rendering and approving entries.
    """
    if not text.strip():
        return [], []
    device = device or select_device()
    chunks = split_chunks(text)
    # The embedder is not needed until extraction is over, and loading it is
    # tens of seconds of disk and CPU that has nothing to do with the NER
    # sweep. Start it now so the wait happens behind the sweep instead of
    # after it; embed() then finds it in the cache. A failure here is not
    # raised — embed() makes the same call and reports it properly.
    warm = threading.Thread(
        target=lambda: _ignore_errors(load_embedder, embed_model, device),
        daemon=True,
    )
    warm.start()
    records = extract(
        chunks, labels or DEFAULT_LABELS, ner_threshold, ner_model, device, batch_size,
    )
    records = keep_meaningful(records, min_count, len(chunks))
    if not records:
        return [], []
    vectors = embed(records, embed_model, device)
    merges, review = candidate_pairs(records, vectors, high, low)
    return build_glossary(records, merges), review


# ---------------------------------------------------------------- command line

def main() -> None:
    ap = argparse.ArgumentParser(description="Build a glossary from a book")
    ap.add_argument("book", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("."))
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--ner-threshold", type=float, default=0.5)
    ap.add_argument("--high", type=float, default=0.80, help="merge automatically at or above this score")
    ap.add_argument("--low", type=float, default=0.55, help="ignore pairs below this score")
    ap.add_argument("--ner-model", default="urchade/gliner_multi-v2.1")
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--device", default=None,
                    help="PyTorch device: mps, cuda, or cpu (default: best available)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="number of chunks in one NER pass")
    args = ap.parse_args()

    args.device = args.device or select_device()
    print(f"Device: {args.device}", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)

    text = read_book(args.book)
    print(f"Source characters: {len(text):,}", file=sys.stderr)
    glossary, review = build_document_glossary(
        text, labels=args.labels, min_count=args.min_count,
        ner_threshold=args.ner_threshold, high=args.high, low=args.low,
        ner_model=args.ner_model, embed_model=args.embed_model,
        device=args.device, batch_size=args.batch_size,
    )
    if not glossary:
        print("No recurring entities found — lower --min-count or --ner-threshold.", file=sys.stderr)
        return
    print(f"Automatically clustered: {len(glossary)}, review pairs: {len(review)}", file=sys.stderr)

    (args.out / "glossary.json").write_text(
        json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "review_queue.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nGlossary entries: {len(glossary)}", file=sys.stderr)
    print(f"{args.out / 'glossary.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()

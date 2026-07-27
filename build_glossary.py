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
from rapidfuzz import fuzz

DEFAULT_LABELS = ["person", "location", "organization", "object", "title"]
BUCKETS = 20  # Document segments used as a co-occurrence signal.

# Both checkpoints take tens of seconds to load and hundreds of megabytes of
# memory, and neither depends on the document. Loading them per call meant
# every press of Prepare paid the whole startup cost again — the dominant
# term in Stage 0's wall clock, well ahead of the model calls it exists for.
_MODEL_LOCK = threading.Lock()
_LOADED: dict[tuple[str, str, str], object] = {}


def _load_once(kind: str, model_name: str, device: str, build):
    """Return a cached model, building it under the lock on first use."""
    key = (kind, model_name, device)
    with _MODEL_LOCK:
        model = _LOADED.get(key)
        if model is None:
            print(f"Loading {model_name} …", file=sys.stderr)
            model = build()
            _LOADED[key] = model
        return model


def load_ner(model_name: str, device: str):
    def build():
        from gliner import GLiNER

        model = GLiNER.from_pretrained(model_name)
        model.to(device)
        return model

    return _load_once("ner", model_name, device, build)


def load_embedder(model_name: str, device: str):
    def build():
        from sentence_transformers import SentenceTransformer

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
            print(
                f"NER: {done:,}/{len(chunks):,} ({done / len(chunks):.1%}) | "
                f"{rate:.1f} chunks/s | ~{eta / 60:.1f} min remaining "
                f"(batch {batch_number}/{total_batches})",
                file=sys.stderr, flush=True,
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


def score_pair(a: dict, b: dict, cos: float) -> tuple[float, dict]:
    f = fuzz.token_set_ratio(a["norm"], b["norm"]) / 100.0
    ta, tb = set(a["norm"].split()), set(b["norm"].split())
    contained = ta <= tb or tb <= ta
    jac = len(a["buckets"] & b["buckets"]) / max(1, len(a["buckets"] | b["buckets"]))

    s = 0.40 * f + 0.45 * cos + 0.15 * jac
    if contained:
        s = min(1.0, s + 0.12)
    return s, {"fuzzy": round(f, 3), "cosine": round(cos, 3),
               "overlap": round(jac, 3), "contained": contained}


def candidate_pairs(records: list[dict], vecs: np.ndarray, high: float, low: float):
    sims = vecs @ vecs.T
    merges, review = [], []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            if a["label"] != b["label"]:          # Never merge different entity types.
                continue
            cos = float(sims[i, j])
            quick = fuzz.token_set_ratio(a["norm"], b["norm"])
            if quick < 60 and cos < 0.75:          # Too dissimilar to consider.
                continue
            s, feats = score_pair(a, b, cos)
            if s >= high:
                merges.append((i, j, s))
            elif s >= low:
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

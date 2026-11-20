"""④ Build the two indexes: BM25 (keywords) and dense vectors (meaning).

An Embedder turns a list of texts into a matrix of L2-normalised vectors, so that
a dot product equals cosine similarity.
  * HashEmbedder  - offline, deterministic, no model download; for tests and demos only.
  * STEmbedder    - a real sentence-transformers model (default BAAI/bge-m3).
Vectors are cached on disk, keyed by model name + chunk texts, so changing the
chunking or the model automatically rebuilds them.
"""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from tokenize_zh_de import tokenize


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class HashEmbedder:
    """Bag-of-hashed-tokens vectors. Not a real embedding model."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                col = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim
                out[row, col] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class STEmbedder:
    """sentence-transformers wrapper. The model is downloaded on first use."""

    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 16):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, normalize_embeddings=True, batch_size=self.batch_size,
            show_progress_bar=len(texts) > 64, convert_to_numpy=True,
        ).astype(np.float32)


def make_embedder(name: str):
    if name == "none":
        return None
    if name == "hash":
        return HashEmbedder()
    return STEmbedder(name)


@dataclass
class Index:
    chunks: list[dict]
    bm25: BM25Okapi
    embedder: object | None
    embeddings: np.ndarray | None
    drop_stopwords: bool


def _cache_path(cache_dir: Path, embedder, chunks: list[dict]) -> Path:
    digest = hashlib.sha1(
        (embedder.name + "\x00" + "\x00".join(c["text"] for c in chunks)).encode("utf-8")
    ).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", embedder.name)
    return cache_dir / f"emb_{safe}_{digest}.npy"


def build_index(chunks: list[dict], embedder=None, drop_stopwords: bool = False,
                cache_dir: str | None = None) -> Index:
    token_lists = [tokenize(c["text"], drop_stopwords) for c in chunks]
    bm25 = BM25Okapi(token_lists)
    embeddings = None
    if embedder is not None:
        texts = [c["text"] for c in chunks]
        if cache_dir:
            path = _cache_path(Path(cache_dir), embedder, chunks)
            if path.exists():
                embeddings = np.load(path)
            else:
                embeddings = embedder.encode(texts)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, embeddings)
        else:
            embeddings = embedder.encode(texts)
    return Index(chunks, bm25, embedder, embeddings, drop_stopwords)

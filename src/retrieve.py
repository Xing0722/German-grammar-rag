"""⑤ Retrieval: BM25 / dense / hybrid (RRF), then merge sub-chunks back to sections.

Section ranking rule: a section's rank is the rank of its BEST sub-chunk.
(Equivalent to taking the max score over its sub-chunks.)
"""
import numpy as np

from tokenize_zh_de import tokenize

MODES = ("bm25", "dense", "hybrid")


def rank_bm25(index, query: str, n: int) -> list[int]:
    tokens = tokenize(query, index.drop_stopwords)
    if not tokens:
        return []
    scores = index.bm25.get_scores(tokens)
    order = np.argsort(-scores)[:n]
    return [int(i) for i in order if scores[i] > 0]   # no word overlap = not retrieved


def rank_dense(index, query: str, n: int) -> list[int]:
    if index.embeddings is None:
        raise ValueError("dense/hybrid retrieval needs an embedder; build the index with one")
    q = index.embedder.encode([query])[0]
    scores = index.embeddings @ q
    return [int(i) for i in np.argsort(-scores)[:n]]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: only ranks are used, so BM25 scores and cosine
    similarities (different scales) can be combined."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]


def retrieve(index, query: str, mode: str = "hybrid", top_sections: int = 5,
             n_chunks: int = 30, rrf_k: int = 60, chunks_per_section: int = 2) -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "bm25":
        ranked = rank_bm25(index, query, n_chunks)
    elif mode == "dense":
        ranked = rank_dense(index, query, n_chunks)
    else:
        ranked = rrf([rank_bm25(index, query, n_chunks), rank_dense(index, query, n_chunks)], rrf_k)

    by_section: dict[str, list[int]] = {}
    for ci in ranked:                      # dict keeps first-seen order = best chunk's order
        by_section.setdefault(index.chunks[ci]["section_id"], []).append(ci)

    results = []
    for rank, (sid, chunk_ids) in enumerate(list(by_section.items())[:top_sections], start=1):
        best = chunk_ids[:chunks_per_section]
        first = index.chunks[best[0]]
        results.append({
            "rank": rank,
            "section_id": sid,
            "title": first["title"],
            "page_start": first["page_start"],
            "page_end": first["page_end"],
            "chunk_ids": [index.chunks[c]["chunk_id"] for c in best],
            "texts": [index.chunks[c]["text"] for c in best],
        })
    return results

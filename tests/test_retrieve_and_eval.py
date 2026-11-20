from pathlib import Path

import numpy as np
import pytest

from chunking import build_chunks
from evaluate import (bootstrap_ci, evaluate, gold_rank, load_questions,
                      paired_bootstrap_diff, scores_from_ranks)
from index import HashEmbedder, build_index, load_jsonl
from retrieve import retrieve, rrf

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = load_jsonl(ROOT / "data/demo/sections.jsonl")
QUESTIONS = load_questions(ROOT / "data/eval/demo_questions.jsonl")


@pytest.fixture(scope="module")
def index():
    return build_index(build_chunks(SECTIONS, 500, 80), HashEmbedder())


def test_bm25_finds_the_obvious_section(index):
    assert retrieve(index, "weil 引导的句子里动词放在哪里?", "bm25")[0]["section_id"] == "demo-6"
    assert retrieve(index, "mit 后面要接什么格?", "bm25")[0]["section_id"] == "demo-2"


def test_results_are_unique_sections_with_consecutive_ranks(index):
    for mode in ("bm25", "dense", "hybrid"):
        res = retrieve(index, "介词 与格 宾格", mode, top_sections=5)
        ids = [r["section_id"] for r in res]
        assert len(ids) == len(set(ids))
        assert [r["rank"] for r in res] == list(range(1, len(res) + 1))


def test_bm25_returns_nothing_when_no_word_matches(index):
    assert retrieve(index, "zzzz qqqq", "bm25") == []


def test_dense_without_embedder_raises():
    idx = build_index(build_chunks(SECTIONS, 500, 80), None)
    with pytest.raises(ValueError):
        retrieve(idx, "mit", "dense")


def test_rrf_prefers_items_ranked_high_in_both_lists():
    assert rrf([[1, 2, 3], [1, 3, 2]])[0] == 1
    assert set(rrf([[1, 2], [3]])) == {1, 2, 3}


def test_embedding_cache_is_reused_and_keyed_by_chunking(tmp_path):
    chunks_a = build_chunks(SECTIONS, 500, 80)
    chunks_b = build_chunks(SECTIONS, 120, 20)
    e1 = build_index(chunks_a, HashEmbedder(), cache_dir=str(tmp_path)).embeddings
    e2 = build_index(chunks_a, HashEmbedder(), cache_dir=str(tmp_path)).embeddings
    assert len(list(tmp_path.glob("*.npy"))) == 1 and np.allclose(e1, e2)
    build_index(chunks_b, HashEmbedder(), cache_dir=str(tmp_path))
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_gold_rank_and_scores():
    assert gold_rank(["a", "b", "c"], ["c", "x"]) == 3
    assert gold_rank(["a"], ["z"]) is None
    s = scores_from_ranks([1, 3, None])
    assert s["hit@1"] == [1.0, 0.0, 0.0] and s["hit@3"] == [1.0, 1.0, 0.0]
    assert s["mrr"][1] == pytest.approx(1 / 3) and s["mrr"][2] == 0.0


def test_bootstrap_ci_contains_mean_and_collapses_for_constants():
    mean, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1])
    assert lo <= mean <= hi
    assert bootstrap_ci([1, 1, 1, 1]) == (1.0, 1.0, 1.0)


def test_paired_diff_is_positive_when_a_beats_b_everywhere():
    mean, lo, hi = paired_bootstrap_diff([1, 1, 1, 1], [0, 1, 0, 0])
    assert mean > 0 and lo > 0


def test_load_questions_rejects_empty_gold(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"id": "q1", "query": "x", "gold": []}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(p)


def test_evaluate_runs_end_to_end_on_demo(index):
    res = evaluate(index, QUESTIONS, "hybrid")
    assert res["n"] == len(QUESTIONS)
    for m in res["metrics"].values():
        assert 0.0 <= m["mean"] <= 1.0 and m["ci95"][0] <= m["mean"] <= m["ci95"][1]
    assert all("top" in r and "gold_rank" in r for r in res["per_question"])

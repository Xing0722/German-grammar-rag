"""⑧ Evaluation: does the right section appear in the top-k?   ★ READ AND UNDERSTAND THIS FILE ★

Input : data/eval/questions.jsonl, one JSON object per line:
        {"id": "q01", "query": "dem 什么时候用", "gold": ["1.2"], "style": "natural"}
        gold is a LIST of section ids; a question counts as a hit if ANY of them is retrieved.
Metrics (per question, then averaged):
        hit@k  = 1 if a gold section is in the top k sections, else 0   (k = 1, 3, 5)
        MRR    = 1 / (rank of the first gold section), 0 if not retrieved
Every mean comes with n and a 95% bootstrap confidence interval: with 40-60 questions
a single question moves a mean by ~2 points, so numbers without an interval mean little.
The output file contains questions, section ids and metrics only - never textbook text.

Usage:
    python src/evaluate.py --sections data/demo/sections.jsonl \
        --questions data/eval/demo_questions.jsonl --mode bm25 --out results/demo_bm25.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from index import load_jsonl
from pipeline import add_common_args, index_from_args
from retrieve import retrieve

KS = (1, 3, 5)


def load_questions(path) -> list[dict]:
    rows = load_jsonl(path)
    for r in rows:
        missing = {"id", "query", "gold"} - set(r)
        if missing:
            raise ValueError(f"question {r.get('id')} is missing fields: {sorted(missing)}")
        if not isinstance(r["gold"], list) or not r["gold"]:
            raise ValueError(f"question {r['id']}: gold must be a non-empty list of section ids")
    return rows


def gold_rank(section_ids: list[str], gold: list[str]) -> int | None:
    """1-based rank of the first gold section, or None if none was retrieved."""
    for rank, sid in enumerate(section_ids, start=1):
        if sid in gold:
            return rank
    return None


def scores_from_ranks(ranks: list[int | None], ks=KS) -> dict[str, list[float]]:
    out = {f"hit@{k}": [1.0 if r is not None and r <= k else 0.0 for r in ranks] for k in ks}
    out["mrr"] = [1.0 / r if r else 0.0 for r in ranks]
    return out


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile interval, resampling QUESTIONS with replacement."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_diff(a, b, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean difference a-b over the SAME questions, with a 95% interval.
    If the interval excludes 0, the two systems differ on these questions."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), n_boot, seed)


def evaluate(index, questions: list[dict], mode: str, top_sections: int = 5,
             n_chunks: int = 30, rrf_k: int = 60) -> dict:
    rows, ranks = [], []
    for q in questions:
        hits = retrieve(index, q["query"], mode, max(top_sections, max(KS)), n_chunks, rrf_k)
        ids = [h["section_id"] for h in hits]
        rank = gold_rank(ids, q["gold"])
        ranks.append(rank)
        rows.append({"id": q["id"], "query": q["query"], "style": q.get("style"),
                     "gold": q["gold"], "top": ids[:5], "gold_rank": rank})
    per_q = scores_from_ranks(ranks)
    metrics = {}
    for name, values in per_q.items():
        mean, lo, hi = bootstrap_ci(values)
        metrics[name] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return {"n": len(questions), "metrics": metrics, "per_question": rows, "_scores": per_q}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--questions", default="data/eval/questions.jsonl")
    p.add_argument("--out", default=None, help="write results JSON here (e.g. results/e1_hybrid.json)")
    args = p.parse_args()

    questions = load_questions(args.questions)
    index = index_from_args(args)
    result = evaluate(index, questions, args.mode, args.top_sections, args.n_chunks, args.rrf_k)

    print(f"mode={args.mode} model={args.model if args.mode != 'bm25' else '-'} "
          f"stopwords_dropped={args.drop_stopwords} chunks={len(index.chunks)} n={result['n']}")
    for name, m in result["metrics"].items():
        print(f"  {name:6s} {m['mean']:.3f}  95% CI [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]")
    misses = [r for r in result["per_question"] if r["gold_rank"] is None]
    print(f"  questions with gold not in the retrieved list: {len(misses)}")

    if args.out:
        scores = result.pop("_scores")
        result["config"] = {k: v for k, v in vars(args).items() if k not in ("out",)}
        result["scores_per_question"] = scores        # needed for paired comparisons later
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())

"""Shared glue: command-line options and "sections file -> index" for cli.py and evaluate.py."""
import argparse

from chunking import build_chunks
from index import build_index, load_jsonl, make_embedder

DEFAULTS = {
    "sections": "data/sections.jsonl",
    "index_dir": "data/index",
    "model": "BAAI/bge-m3",
    "max_chars": 500,
    "overlap": 80,
    "top_sections": 5,
    "n_chunks": 30,
    "rrf_k": 60,
}


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sections", default=DEFAULTS["sections"],
                   help="sections.jsonl from parse_pdf.py (or data/demo/sections.jsonl)")
    p.add_argument("--index-dir", default=DEFAULTS["index_dir"], help="where embeddings are cached")
    p.add_argument("--mode", choices=("bm25", "dense", "hybrid"), default="hybrid")
    p.add_argument("--model", default=DEFAULTS["model"],
                   help="embedding model name, or 'hash' (offline test embedder)")
    p.add_argument("--max-chars", type=int, default=DEFAULTS["max_chars"])
    p.add_argument("--overlap", type=int, default=DEFAULTS["overlap"])
    p.add_argument("--top-sections", type=int, default=DEFAULTS["top_sections"])
    p.add_argument("--n-chunks", type=int, default=DEFAULTS["n_chunks"])
    p.add_argument("--rrf-k", type=int, default=DEFAULTS["rrf_k"])
    p.add_argument("--drop-stopwords", action="store_true",
                   help="experiment E2: remove stopwords (incl. der/die/das/dem/den/des)")


def index_from_args(args):
    sections = load_jsonl(args.sections)
    chunks = build_chunks(sections, args.max_chars, args.overlap)
    embedder = None if args.mode == "bm25" else make_embedder(args.model)
    return build_index(chunks, embedder, args.drop_stopwords, args.index_dir)

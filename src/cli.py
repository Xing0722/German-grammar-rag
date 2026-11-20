"""⑦ Ask a question from the terminal.

    python src/cli.py "dem 什么时候用" --sections data/demo/sections.jsonl --model hash
    python src/cli.py "dem 什么时候用"                      # real PDF + bge-m3

Prints the retrieved sections. If LLM_API_KEY is set, also prints an answer with citations.
"""
import argparse
import os

from generate import build_messages, call_llm, check_citations
from pipeline import add_common_args, index_from_args
from retrieve import retrieve


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    add_common_args(p)
    p.add_argument("--no-llm", action="store_true", help="only show retrieval")
    args = p.parse_args()
    args.top_sections = min(args.top_sections, 3)

    index = index_from_args(args)
    sections = retrieve(index, args.question, args.mode, args.top_sections, args.n_chunks, args.rrf_k)
    print("检索到的小节:")
    for s in sections:
        pages = f"p.{s['page_start']}" if s["page_start"] is not None else ""
        print(f"  {s['rank']}. [{s['section_id']}] {s['title']} {pages}")

    if args.no_llm or not os.environ.get("LLM_API_KEY"):
        print("\n(未设置 LLM_API_KEY 或使用了 --no-llm:只显示检索结果)")
        return
    answer = call_llm(build_messages(args.question, sections))
    print("\n" + answer)
    check = check_citations(answer, [s["section_id"] for s in sections])
    if check["invalid"]:
        print(f"\n⚠ 引用了没有被检索到的小节: {check['invalid']}")
    elif not check["has_citation"]:
        print("\n⚠ 答案没有引用任何小节")


if __name__ == "__main__":
    main()

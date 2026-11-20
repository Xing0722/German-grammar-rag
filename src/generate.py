"""⑥ Generation: put the retrieved sections and the question into a prompt, call an LLM.

Uses any OpenAI-compatible /chat/completions endpoint. Configure with environment variables:
    LLM_API_KEY   (required)
    LLM_BASE_URL  (default https://api.deepseek.com)
    LLM_MODEL     (default deepseek-chat)

The model is told to cite sections as [6.3]. check_citations() then verifies that every
cited id was really among the retrieved sections (a cheap guard against invented citations;
it does NOT check that the cited text supports the claim).
"""
import json
import os
import re
import urllib.request

SYSTEM_PROMPT = (
    "你是德语语法学习助手。只能根据【资料】回答,不要使用资料以外的知识。"
    "每个结论后面用方括号标注来源小节号,例如 [6.3]。"
    "如果资料里没有答案,直接回答“资料中没有找到相关内容”,不要猜测。用中文回答,德语例句保持原文。"
)
CITE = re.compile(r"\[([A-Za-z0-9.\-]+)\]")


def build_context(sections: list[dict]) -> str:
    blocks = []
    for s in sections:
        pages = s.get("page_start")
        where = f" · 第{pages}页" if pages is not None else ""
        blocks.append(f"【资料 {s['section_id']} · {s['title']}{where}】\n" + "\n".join(s["texts"]))
    return "\n\n".join(blocks)


def build_messages(question: str, sections: list[dict]) -> list[dict]:
    user = f"{build_context(sections)}\n\n【问题】\n{question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def extract_citations(answer: str) -> list[str]:
    seen: list[str] = []
    for cid in CITE.findall(answer):
        if cid not in seen:
            seen.append(cid)
    return seen


def check_citations(answer: str, retrieved_ids: list[str]) -> dict:
    cited = extract_citations(answer)
    return {
        "cited": cited,
        "invalid": [c for c in cited if c not in retrieved_ids],
        "has_citation": bool(cited),
    }


def call_llm(messages: list[dict], base_url: str | None = None, model: str | None = None,
             api_key: str | None = None, timeout: int = 60) -> str:
    api_key = api_key or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Set the LLM_API_KEY environment variable first.")
    base_url = (base_url or os.environ.get("LLM_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    model = model or os.environ.get("LLM_MODEL") or "deepseek-chat"
    body = json.dumps({"model": model, "messages": messages, "temperature": 0}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]

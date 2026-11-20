import json

import pytest

import generate
from generate import build_messages, call_llm, check_citations, extract_citations

SECTIONS = [
    {"section_id": "6.3", "title": "可分前缀", "page_start": 36, "texts": ["可分前缀放在句末"]},
    {"section_id": "6.4", "title": "两用前缀", "page_start": 43, "texts": ["两用前缀视情况而定"]},
]


def test_prompt_contains_every_section_id_text_and_the_question():
    msgs = build_messages("可分前缀怎么放?", SECTIONS)
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    user = msgs[1]["content"]
    for needle in ("6.3", "6.4", "可分前缀放在句末", "第36页", "可分前缀怎么放?"):
        assert needle in user


def test_citations_are_extracted_once_and_validated():
    answer = "前缀放句末 [6.3]。两用前缀要看情况 [6.4] [6.3],另见 [9.9]。"
    assert extract_citations(answer) == ["6.3", "6.4", "9.9"]
    check = check_citations(answer, ["6.3", "6.4"])
    assert check["invalid"] == ["9.9"] and check["has_citation"]
    assert not check_citations("没有引用", ["6.3"])["has_citation"]


def test_call_llm_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        call_llm([{"role": "user", "content": "hi"}])


def test_call_llm_parses_openai_style_response(monkeypatch):
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "答案 [6.3]"}}]}).encode("utf-8")

    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        return FakeResponse()

    monkeypatch.setattr(generate.urllib.request, "urlopen", fake_urlopen)
    out = call_llm([{"role": "user", "content": "hi"}], base_url="https://example.test/", api_key="k")
    assert out == "答案 [6.3]"
    assert seen["url"] == "https://example.test/chat/completions" and seen["auth"] == "Bearer k"

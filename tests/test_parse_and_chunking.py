import pytest

from chunking import build_chunks, split_section
from parse_pdf import parse_text

SAMPLE = (
    "德语基础语法      第 1 章 · 名词与冠词\n\n@@1.1@@\n\n"
    "   1.1     名词的性与复数        Genus · Plural\n"
    "   看词尾判断性\n"
    "   阳性 der      -er · -ling\n\n"
    "Kapitel 1                              1\n\f"
    "德语基础语法      第 1 章 · 名词与冠词\n\n@@1.2@@\n"
    "   1.2     冠词          Artikel\n"
    "   主格     der      die      das\n\n"
    "Kapitel 1                              2\n"
)


def test_parse_text_cuts_at_markers_and_strips_page_furniture():
    sections = parse_text(SAMPLE)
    assert [s["id"] for s in sections] == ["1.1", "1.2"]
    assert sections[0]["title"].startswith("名词的性与复数")
    assert (sections[0]["page_start"], sections[1]["page_start"]) == (1, 2)
    joined = "\n".join(s["text"] for s in sections)
    assert "Kapitel" not in joined and "德语基础语法" not in joined and "@@" not in joined


def test_parse_text_keeps_table_columns_visible():
    sections = parse_text(SAMPLE)
    assert "主格 | der | die | das" in sections[1]["text"]


def make_section(n_lines=50, line_len=40):
    lines = [f"{i:02d} " + "x" * (line_len - 3) for i in range(n_lines)]
    return {"id": "9.9", "title": "测试", "page_start": 3, "page_end": 4, "text": "\n".join(lines)}


def test_chunks_respect_max_chars_and_keep_section_id():
    chunks = split_section(make_section(), max_chars=200, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        body = c["text"].split("\n", 1)[1]
        assert len(body) <= 200
        assert c["section_id"] == "9.9"
        assert c["text"].startswith("[9.9] 测试")
    assert [c["chunk_id"] for c in chunks][:2] == ["9.9#0", "9.9#1"]


def test_every_line_is_in_some_chunk_and_neighbours_overlap():
    sec = make_section()
    chunks = split_section(sec, max_chars=200, overlap=50)
    all_text = "\n".join(c["text"] for c in chunks)
    for line in sec["text"].splitlines():
        assert line in all_text
    last_line_of_first = chunks[0]["text"].splitlines()[-1]
    assert last_line_of_first in chunks[1]["text"]          # overlap


def test_table_row_is_not_split():
    row = "主格 一格 | der | die | das | die"
    sec = {"id": "1.2", "title": "冠词", "page_start": 2, "page_end": 2,
           "text": "\n".join(["说明" * 30, row, "说明" * 30, "说明" * 30])}
    chunks = split_section(sec, max_chars=150, overlap=20)
    assert any(row in c["text"] for c in chunks)


def test_oversized_single_line_is_hard_split():
    sec = {"id": "1.1", "title": "t", "page_start": 1, "page_end": 1, "text": "y" * 1000}
    chunks = split_section(sec, max_chars=300, overlap=30)
    assert len(chunks) >= 4
    assert all(len(c["text"].split("\n", 1)[1]) <= 300 for c in chunks)


def test_overlap_must_be_smaller_than_max_chars():
    with pytest.raises(AssertionError):
        split_section(make_section(), max_chars=100, overlap=100)


def test_build_chunks_covers_all_sections():
    secs = [make_section(), {**make_section(), "id": "9.8"}]
    chunks = build_chunks(secs, 200, 50)
    assert {c["section_id"] for c in chunks} == {"9.9", "9.8"}

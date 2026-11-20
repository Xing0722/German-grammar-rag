"""② Sections -> sub-chunks.   ★ READ AND UNDERSTAND THIS FILE ★

Evaluation unit  = a section (gold label is a section id such as "6.3").
Retrieval unit   = a sub-chunk, because embedding models truncate long inputs and
                   one section can be 9,000+ characters.
Every sub-chunk remembers its section_id, so results are merged back to sections.

Rules (each one is a decision you should be able to explain; see docs/decisions.md):
  * split on LINES, never inside a line, so a table row stays in one piece
    (a line longer than max_chars is hard-split, the only exception);
  * pack lines into a chunk until max_chars is reached;
  * the next chunk starts with the last few lines of the previous one (overlap),
    so an answer sitting on a boundary is complete in at least one chunk;
  * every chunk text starts with a heading "[id] title" so a chunk alone still
    says which section it is from.
"""


def _hard_split(line: str, max_chars: int) -> list[str]:
    return [line[i:i + max_chars] for i in range(0, len(line), max_chars)]


def split_section(section: dict, max_chars: int = 500, overlap: int = 80) -> list[dict]:
    assert 0 <= overlap < max_chars, "overlap must be smaller than max_chars"
    header = f"[{section['id']}] {section['title']}".strip()

    lines: list[str] = []
    for raw in section["text"].splitlines():
        raw = raw.strip()
        if not raw:
            continue
        lines.extend(_hard_split(raw, max_chars) if len(raw) > max_chars else [raw])

    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > max_chars:
            groups.append(current)
            carry: list[str] = []          # trailing lines of the previous chunk
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) + 1 > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev) + 1
            current, size = carry, carried
        current.append(line)
        size += len(line) + 1
    if current:
        groups.append(current)

    return [
        {
            "chunk_id": f"{section['id']}#{k}",
            "section_id": section["id"],
            "title": section["title"],
            "page_start": section.get("page_start"),
            "page_end": section.get("page_end"),
            "text": header + "\n" + "\n".join(group),
        }
        for k, group in enumerate(groups)
    ]


def build_chunks(sections: list[dict], max_chars: int = 500, overlap: int = 80) -> list[dict]:
    chunks: list[dict] = []
    for section in sections:
        chunks.extend(split_section(section, max_chars, overlap))
    return chunks

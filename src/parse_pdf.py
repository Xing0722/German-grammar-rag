"""① Cut the grammar PDF into one record per numbered section (1.1 ... 9.5).

Usage:
    python src/parse_pdf.py data/raw/grammar.pdf data/sections.jsonl

Why pdftotext: this PDF carries a hidden marker "@@1.2@@" at the start of every
section. pdftotext keeps the marker at the start of its section; other extractors
(PyMuPDF, pdfplumber) put it elsewhere or garble it, and would cut at the wrong
place. Needs poppler:  conda install -c conda-forge poppler   (or brew install poppler)

The HEADER/FOOTER patterns below are specific to this PDF.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

MARKER = re.compile(r"\s*@@(\d+\.\d+)@@\s*")        # hidden section anchor
HEADER = re.compile(r"\s*德语基础语法\s+第 \d+ 章")      # running page header
FOOTER = re.compile(r"\s*Kapitel \d+\s+(\d+)\s*$")    # running footer; group 1 = printed page


def clean_line(line: str) -> str:
    """Keep table columns readable: 3+ spaces become ' | '."""
    return re.sub(r"\s{3,}", " | ", line.strip())


def parse_text(text: str) -> list[dict]:
    """Split pdftotext output (pages separated by form feed) into section records."""
    sections: list[dict] = []
    current = None
    for pdf_page, page in enumerate(text.split("\f"), start=1):
        printed = pdf_page - 2  # fallback: cover + table of contents take 2 pages
        lines = page.splitlines()
        for line in lines:
            m = FOOTER.match(line)
            if m:
                printed = int(m.group(1))
        for line in lines:
            if HEADER.match(line) or FOOTER.match(line):
                continue
            m = MARKER.fullmatch(line)
            if m:
                current = {"id": m.group(1), "page_start": printed, "page_end": printed, "lines": []}
                sections.append(current)
                continue
            line = clean_line(line)
            if current is not None and line:
                current["lines"].append(line)
                current["page_end"] = printed
    out = []
    for s in sections:
        # chapter banners ("02 | 形容词") can come first; the heading starts with the id
        first = next((ln for ln in s["lines"] if ln.startswith(s["id"])), "")
        title = re.sub(rf"^{re.escape(s['id'])}\s*\|?\s*", "", first)
        out.append({
            "id": s["id"],
            "chapter": int(s["id"].split(".")[0]),
            "title": title,
            "page_start": s["page_start"],
            "page_end": s["page_end"],
            "text": "\n".join(s["lines"]),
        })
    return out


def parse(pdf_path: str) -> list[dict]:
    text = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"], capture_output=True, text=True, check=True
    ).stdout
    return parse_text(text)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    rows = parse(src)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} sections -> {dst}")

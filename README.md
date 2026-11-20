# german-grammar-rag

Question answering over a Chinese-language German grammar textbook (70-page PDF, 52 numbered sections), with a
small, honest evaluation harness. Ask in Chinese, e.g. *"dem 什么时候用?"*, and get the relevant sections with
page numbers and a cited answer.

**Status: work in progress.** The pipeline, tests and a demo corpus are done. The evaluation set for the real
textbook and the experiment results are **not done yet** (see [Results](#results)). No number in this README
is a measured result unless it is in the results table.

## Why this is more than "chat with a PDF"

- **Structure-aware parsing.** The PDF has a hidden anchor at the start of each section; sections are cut there,
  not by guessing. Different PDF extractors disagree about this file (one misplaces the anchors, another emits
  look-alike radical characters instead of normal Chinese characters), so the extractor choice is documented
  in [docs/decisions.md](docs/decisions.md).
- **Two units on purpose.** The *evaluation* unit is a section (gold label = section id such as `6.3`), the
  *retrieval* unit is a sub-chunk (embedding models truncate long input; one section is up to ~9,000 characters).
  Sub-chunk scores are merged back to sections.
- **Mixed Chinese/German tokenizer.** Chinese is segmented with jieba; German words keep their umlauts and ß
  (a plain `[A-Za-z]` pattern turns `Abhängigkeiten` into `Abh` + `ngigkeiten`).
- **Grammar words are content.** In a grammar book `der/die/das/dem/den/des` and prepositions like `mit/nach/bei`
  are what users ask about, so stopword removal is off by default and is tested as an experiment (E2).
- **Evaluation with uncertainty.** Hit@k and MRR on a hand-written question set, each with a 95% bootstrap interval
  and the number of questions, plus paired comparisons between systems.

## Pipeline

```
PDF ─① parse_pdf─▶ sections ─② chunking─▶ sub-chunks ─④ index─▶ BM25 + dense vectors
question ─⑤ retrieve (BM25 / dense / hybrid-RRF)─▶ top sections ─⑥ generate (LLM, cited)─▶ answer
questions.jsonl ─⑧ evaluate─▶ hit@k, MRR, 95% CI
```

## Quickstart (no textbook needed)

```bash
pip install -r requirements.txt
pytest                                           # 28 tests
python src/cli.py "mit 后面要接什么格?" --sections data/demo/sections.jsonl --model hash
python src/evaluate.py --sections data/demo/sections.jsonl --questions data/eval/demo_questions.jsonl \
       --mode hybrid --model hash
```

`--model hash` is an offline stand-in for an embedding model so the demo runs without downloads.
`data/demo/` is a six-section corpus written for this repository; it is far too small to say anything
about retrieval quality, it only checks that the pipeline runs.

## Using your own copy of the textbook

The textbook is not distributed with this repository and must not be committed (`.gitignore` blocks
`data/raw/`, `data/sections.jsonl` and `data/index/`).

```bash
conda install -c conda-forge poppler               # provides pdftotext
python src/parse_pdf.py data/raw/<your-file>.pdf data/sections.jsonl
python src/cli.py "dem 什么时候用"                  # downloads BAAI/bge-m3 on first use
```

`parse_pdf.py` contains patterns specific to the PDF it was written for (hidden `@@x.y@@` anchors, running
header/footer). Another document needs its own parser.

Answers need an OpenAI-compatible endpoint: set `LLM_API_KEY` (and optionally `LLM_BASE_URL`, `LLM_MODEL`).
Without a key the CLI prints the retrieved sections only.

## Evaluation protocol

Questions are written by hand, in the way a learner would ask (see [docs/writing_questions.md](docs/writing_questions.md)),
and labelled with the section(s) that answer them. Planned experiments, each changing one variable:

| | Compared | Question |
|---|---|---|
| E1 | BM25 / dense / hybrid | Does hybrid help? |
| E2 | grammar words kept / removed as stopwords | Does standard stopword removal hurt? |
| E3 | questions copying section titles / natural questions | Does keyword search profit from title wording? |
| E4 | sub-chunk size, reranking | Sensitivity |

## Results

Not available yet: the evaluation set for the real textbook has not been written.

| Experiment | n | hit@1 | hit@3 | MRR |
|---|---|---|---|---|
| E1 – BM25 | – | – | – | – |
| E1 – dense | – | – | – | – |
| E1 – hybrid | – | – | – | – |

## Limitations

- Single textbook, 52 sections: retrieval may be too easy to separate methods; distractor documents are planned.
- The question set is small and written by one person; intervals are wide.
- Colour coding in the PDF (e.g. case colours) is not used.
- Citation check only verifies that a cited section was retrieved, not that it supports the claim.
- `BAAI/bge-m3` is the intended embedding model; it has not been run in this repository's tests (tests use the offline `hash` stand-in).

## Layout

```
src/        parse_pdf · chunking · tokenize_zh_de · index · retrieve · generate · evaluate · cli · pipeline
tests/      28 tests (parsing, chunking, tokenizer, retrieval, metrics, generation)
data/demo/  demo corpus written for this repo      data/eval/  question sets
docs/       decisions.md (design log) · writing_questions.md
results/    experiment outputs (section ids and metrics only, no textbook text)
```

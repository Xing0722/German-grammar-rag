"""③ Mixed Chinese / German tokenizer for BM25.   ★ READ AND UNDERSTAND THIS FILE ★

The notes mix Chinese explanations with German examples, so one tokenizer has to
handle both:
  * Latin runs (German words incl. ä ö ü ß) are kept whole and lower-cased.
    A plain [A-Za-z] pattern would cut "Abhängigkeiten" into "Abh" + "ngigkeiten".
  * Chinese runs have no spaces, so they are segmented with jieba.

Stopwords are OFF by default. In a grammar book, der/die/das/dem/den/des and the
prepositions mit/nach/bei/seit/von/zu/aus ARE the content people ask about;
a normal German stopword list would delete exactly those. Experiment E2 measures
what happens with drop_stopwords=True.

The lists below are short approximations of common stopword lists, not copies of
a specific one.
"""
import logging
import re

import jieba

jieba.setLogLevel(logging.WARNING)

LATIN = r"[A-Za-z0-9À-ÖØ-öø-ÿ]+(?:-[A-Za-z0-9À-ÖØ-öø-ÿ]+)*"   # À-ÿ covers ä ö ü ß
CJK = r"[一-鿿]+"
TOKEN = re.compile(f"{LATIN}|{CJK}")
IS_CJK = re.compile(CJK)

STOP_DE = frozenset(
    """der die das den dem des ein eine einen einem einer eines und oder aber ist sind war
    ich du er sie es wir ihr nicht kein keine zu mit von aus bei nach seit in im an am auf
    für um über unter vor zwischen hinter neben als wie wenn dass auch noch nur schon""".split()
)
STOP_ZH = frozenset(
    "的 了 是 在 和 与 或 吗 呢 吧 啊 什么 怎么 如何 怎样 为什么 哪个 哪些 有 就 都 也 要 会 能 可以 这 那 一个".split()
)


def tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    tokens: list[str] = []
    for piece in TOKEN.findall(text):
        if IS_CJK.fullmatch(piece):
            tokens.extend(t.strip() for t in jieba.lcut(piece) if t.strip())
        else:
            tokens.append(piece.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOP_DE and t not in STOP_ZH]
    return tokens

from tokenize_zh_de import tokenize


def test_german_umlauts_and_eszett_stay_inside_words():
    tokens = tokenize("Abhängigkeiten und Fußball über schön")
    assert "abhängigkeiten" in tokens
    assert "fußball" in tokens and "über" in tokens and "schön" in tokens
    assert "abh" not in tokens                       # the bug a plain [A-Za-z] regex has


def test_chinese_is_segmented_without_losing_characters():
    tokens = tokenize("可分动词的前缀")
    assert "".join(tokens) == "可分动词的前缀"
    assert len(tokens) >= 2


def test_mixed_text_gives_both_scripts():
    tokens = tokenize("dem 什么时候用")
    assert "dem" in tokens
    assert any("时候" in t for t in tokens)


def test_hyphenated_german_words_are_kept_whole():
    assert "e-mail" in tokenize("Eine E-Mail schreiben")


def test_stopwords_are_off_by_default_and_remove_grammar_words_when_on():
    text = "dem 什么时候用 mit dem Bus"
    kept = tokenize(text)
    dropped = tokenize(text, drop_stopwords=True)
    assert "dem" in kept and "mit" in kept
    assert "dem" not in dropped and "mit" not in dropped
    assert "bus" in dropped

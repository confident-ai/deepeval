import math

import pytest

from deepeval.scorer import Scorer


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    nltk = pytest.importorskip("nltk")
    # Exercise NLTK's real word tokenizer without downloading Punkt data.
    monkeypatch.setattr(
        nltk.tokenize,
        "word_tokenize",
        nltk.tokenize.TreebankWordTokenizer().tokenize,
    )


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_sentence_bleu_uses_cumulative_ngram_precision(order):
    # Equal lengths mean no brevity penalty. Changing only the last token
    # leaves 7/8 unigrams, 6/7 bigrams, 5/6 trigrams, and 4/5 four-grams.
    expected = math.prod((8 - n) / (9 - n) for n in range(1, order + 1)) ** (
        1 / order
    )
    score = Scorer.sentence_bleu_score(
        "a b c d e f g h", "a b c d e f g x", bleu_type=f"bleu{order}"
    )
    assert score == pytest.approx(expected)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_sentence_bleu_accepts_multiple_references(order):
    score = Scorer.sentence_bleu_score(
        ["a b c d e f g h", "a b c d e f g x"],
        "a b c d e f g x",
        bleu_type=f"bleu{order}",
    )
    assert score == pytest.approx(1.0)

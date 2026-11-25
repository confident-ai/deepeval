"""
Regression tests for `Scorer.rouge_score` on non-Latin text.

`rouge_score`'s default tokenizer keeps only `[a-z0-9]`, so text in other scripts produced no tokens:
identical Hindi/Malayalam/Chinese texts scored 0, and two different texts that shared only an embedded
English word scored 1. Offline: no model, network or API key required.
"""

import pytest

pytest.importorskip("rouge_score")

from rouge_score import rouge_scorer

from deepeval.scorer.scorer import Scorer, _UnicodeRougeTokenizer

SCORE_TYPES = ["rouge1", "rouge2", "rougeL"]

NON_LATIN = [
    "आपका खाता आज बंद हो जाएगा",  # Hindi
    "നിങ്ങളുടെ അക്കൗണ്ട് ഇന്ന് ബ്ലോക്ക് ആകും",  # Malayalam
    "Ваш аккаунт будет заблокирован сегодня",  # Russian
    "سيتم حظر حسابك اليوم",  # Arabic
    "您的账户今天将被冻结",  # Chinese
    "アカウントは今日ロックされます",  # Japanese
]


@pytest.mark.parametrize("score_type", SCORE_TYPES)
@pytest.mark.parametrize("text", NON_LATIN)
def test_identical_non_latin_text_scores_one(text, score_type):
    assert Scorer.rouge_score(text, text, score_type) == pytest.approx(1.0)


@pytest.mark.parametrize("score_type", SCORE_TYPES)
def test_shared_english_word_does_not_make_different_texts_match(score_type):
    # Previously only "kyc" survived tokenization on both sides, so this scored 1.0.
    target = "आपका KYC आज बंद"
    prediction = "मौसम KYC अच्छा रहेगा"
    assert Scorer.rouge_score(target, prediction, score_type) < 0.5


def test_partial_overlap_in_devanagari_is_scored():
    target = "आपका खाता आज बंद हो जाएगा"
    prediction = "आपका खाता कल बंद होगा"
    score = Scorer.rouge_score(target, prediction, "rouge1")
    assert 0.0 < score < 1.0


@pytest.mark.parametrize(
    "text",
    [
        "The company was founded in 2016.",
        "Running runners ran; it's a well-known fact (see e.g. RFC-2119)!",
        "  mixed CASE, punctuation... and_underscores 123abc  ",
    ],
)
def test_ascii_tokens_match_rouge_score_default_tokenizer(text):
    default = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)._tokenizer
    assert _UnicodeRougeTokenizer(use_stemmer=True).tokenize(
        text
    ) == default.tokenize(text)


def test_cjk_is_split_into_characters():
    assert _UnicodeRougeTokenizer().tokenize("您的账户") == [
        "您",
        "的",
        "账",
        "户",
    ]


def test_combining_marks_stay_inside_words():
    # Devanagari vowel signs and virama are combining marks; they must not split words.
    assert _UnicodeRougeTokenizer().tokenize("मुझे क्या") == ["मुझे", "क्या"]

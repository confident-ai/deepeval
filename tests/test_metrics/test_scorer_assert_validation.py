"""Tests for Scorer argument validation using explicit exceptions.

``Scorer.rouge_score``, ``Scorer.sentence_bleu_score`` and
``Scorer.answer_relevancy_score`` used bare ``assert`` statements to validate
their arguments. ``assert`` is an anti-pattern here: it is stripped when the
interpreter runs with ``-O``/``-OO``, and it raises ``AssertionError`` instead
of the ``ValueError``/``TypeError`` callers expect from a scoring API.

These tests verify that:
  * invalid option values (``score_type``, ``bleu_type``, ``model_type``)
    raise ``ValueError``;
  * a non-``str`` prediction with ``model_type="cross_encoder"`` raises
    ``TypeError``;
  * valid inputs keep their previous behaviour (no regression);
  * the validation still fires under ``python -O`` (where asserts vanish).

The tests are fully offline: the optional third-party packages (``rouge-score``,
``nltk``, ``sentence-transformers``) are stubbed out, so no model or network
access is required.
"""

import subprocess
import sys
import types
from unittest.mock import patch

import pytest

from deepeval.scorer.scorer import Scorer


# --------------------------------------------------------------------------- #
# Offline stubs for the optional dependencies these scorers import
# --------------------------------------------------------------------------- #


class _RougeResult:
    def __init__(self, fmeasure=0.5):
        self.fmeasure = fmeasure


class _FakeRougeScorer:
    def __init__(self, score_types, use_stemmer):
        self.score_types = score_types

    def score(self, target, prediction):
        return {st: _RougeResult() for st in self.score_types}


def _fake_rouge_modules():
    rouge_scorer = types.ModuleType("rouge_score.rouge_scorer")
    rouge_scorer.RougeScorer = _FakeRougeScorer
    rouge = types.ModuleType("rouge_score")
    rouge.rouge_scorer = rouge_scorer
    return {
        "rouge_score": rouge,
        "rouge_score.rouge_scorer": rouge_scorer,
    }


def _fake_nltk_modules():
    bleu_score = types.ModuleType("nltk.translate.bleu_score")
    bleu_score.sentence_bleu = lambda refs, pred, weights: 0.75
    translate = types.ModuleType("nltk.translate")
    translate.bleu_score = bleu_score
    tokenize = types.ModuleType("nltk.tokenize")
    tokenize.word_tokenize = lambda s: s.split()
    nltk = types.ModuleType("nltk")
    nltk.tokenize = tokenize
    nltk.translate = translate
    return {
        "nltk": nltk,
        "nltk.tokenize": tokenize,
        "nltk.translate": translate,
        "nltk.translate.bleu_score": bleu_score,
    }


class _FakeTensor:
    def cpu(self):
        return self

    def tolist(self):
        return [0.8]


class _FakeDotScore:
    def __getitem__(self, _):
        return _FakeTensor()


class _FakeCrossEncoderModel:
    def __init__(self, model_name=None):
        self.model_name = model_name

    def __call__(self, predictions, target):
        return 0.85


class _FakeSelfEncoderModel:
    def __init__(self, model_name=None):
        self.model_name = model_name

    def __call__(self, text):
        return text


def _fake_sentence_transformers_modules():
    util = types.ModuleType("sentence_transformers.util")
    util.dot_score = lambda a, b: _FakeDotScore()
    st = types.ModuleType("sentence_transformers")
    st.util = util
    return {
        "sentence_transformers": st,
        "sentence_transformers.util": util,
    }


def _fake_answer_relevancy_model_modules():
    arm = types.ModuleType("deepeval.models.answer_relevancy_model")
    arm.AnswerRelevancyModel = _FakeSelfEncoderModel
    arm.CrossEncoderAnswerRelevancyModel = _FakeCrossEncoderModel
    return {"deepeval.models.answer_relevancy_model": arm}


# --------------------------------------------------------------------------- #
# rouge_score argument validation
# --------------------------------------------------------------------------- #


class TestRougeScoreValidation:
    @pytest.mark.parametrize(
        "score_type", ["invalid", "rouge0", "ROUGE1", "RougeL", ""]
    )
    def test_invalid_score_type_raises_value_error(self, score_type):
        with pytest.raises(ValueError, match="score_type"):
            Scorer.rouge_score("target text", "prediction text", score_type)

    @pytest.mark.parametrize("score_type", ["rouge1", "rouge2", "rougeL"])
    def test_valid_score_type_not_rejected(self, score_type):
        with patch.dict(sys.modules, _fake_rouge_modules()):
            score = Scorer.rouge_score(
                "target text", "prediction text", score_type
            )
        assert score == 0.5


# --------------------------------------------------------------------------- #
# sentence_bleu_score argument validation
# --------------------------------------------------------------------------- #


class TestSentenceBleuScoreValidation:
    @pytest.mark.parametrize(
        "bleu_type", ["invalid", "bleu0", "BLEU1", "Bleu2", ""]
    )
    def test_invalid_bleu_type_raises_value_error(self, bleu_type):
        with pytest.raises(ValueError, match="bleu_type"):
            Scorer.sentence_bleu_score("reference", "prediction", bleu_type)

    @pytest.mark.parametrize("bleu_type", ["bleu1", "bleu2", "bleu3", "bleu4"])
    def test_valid_bleu_type_not_rejected(self, bleu_type):
        with patch.dict(sys.modules, _fake_nltk_modules()):
            score = Scorer.sentence_bleu_score(
                "reference sentence", "prediction sentence", bleu_type
            )
        assert score == 0.75


# --------------------------------------------------------------------------- #
# answer_relevancy_score argument validation
# --------------------------------------------------------------------------- #


class TestAnswerRelevancyScoreValidation:
    @pytest.mark.parametrize(
        "model_type", ["invalid", "encoder", "dual_encoder", ""]
    )
    def test_invalid_model_type_raises_value_error(self, model_type):
        with patch.dict(sys.modules, _fake_sentence_transformers_modules()):
            with pytest.raises(ValueError, match="model_type"):
                Scorer.answer_relevancy_score(
                    predictions="prediction",
                    target="target",
                    model_type=model_type,
                )

    def test_cross_encoder_non_str_predictions_raises_type_error(self):
        with patch.dict(sys.modules, _fake_sentence_transformers_modules()):
            with pytest.raises(TypeError, match="cross_encoder"):
                Scorer.answer_relevancy_score(
                    predictions=["not", "a", "string"],
                    target="target",
                    model_type="cross_encoder",
                )

    def test_cross_encoder_str_predictions_not_rejected(self):
        with patch.dict(
            sys.modules,
            {
                **_fake_sentence_transformers_modules(),
                **_fake_answer_relevancy_model_modules(),
            },
        ):
            score = Scorer.answer_relevancy_score(
                predictions="prediction",
                target="target",
                model_type="cross_encoder",
            )
        assert score == 0.85

    def test_default_model_type_not_rejected(self):
        # model_type defaults to None -> cross_encoder (unchanged behaviour).
        with patch.dict(
            sys.modules,
            {
                **_fake_sentence_transformers_modules(),
                **_fake_answer_relevancy_model_modules(),
            },
        ):
            score = Scorer.answer_relevancy_score(
                predictions="prediction", target="target"
            )
        assert score == 0.85

    def test_self_encoder_list_predictions_not_rejected(self):
        with patch.dict(
            sys.modules,
            {
                **_fake_sentence_transformers_modules(),
                **_fake_answer_relevancy_model_modules(),
            },
        ):
            score = Scorer.answer_relevancy_score(
                predictions=["a", "b"],
                target="target",
                model_type="self_encoder",
            )
        assert score == 0.8


# --------------------------------------------------------------------------- #
# Validation survives `python -O` (where bare asserts are stripped)
# --------------------------------------------------------------------------- #


def test_validation_fires_under_python_optimized():
    """Regression for the core motivation: asserts vanish under -O, explicit
    exceptions do not."""
    code = (
        "import sys\n"
        "sys.path.insert(0, '.')\n"
        "from deepeval.scorer.scorer import Scorer\n"
        "try:\n"
        "    Scorer.rouge_score('target', 'prediction', 'bogus_type')\n"
        "except ValueError:\n"
        "    sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        cwd=__file__.rsplit("/", 2)[0],
    )
    assert result.returncode == 0, result.stderr

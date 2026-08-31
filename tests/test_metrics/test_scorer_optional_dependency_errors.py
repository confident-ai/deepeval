"""
Tests for optional-dependency handling in the deterministic Scorer.

`rouge_score` and `sentence_bleu_score` depend on the optional third-party
packages `rouge-score` and `nltk` respectively. Previously a missing
dependency was swallowed and the function crashed later with a bare
`NameError` (`rouge_score`), or printed a hint and then crashed with a
`NameError` anyway (`sentence_bleu_score`). Both now raise a clear
`ImportError` with install instructions, and the scoring path is unchanged
for users who do have the dependency installed.

These tests are offline: they simulate the presence/absence of the optional
packages via `sys.modules` and need no model, network, or API key.
"""

import sys
import types

import pytest

from deepeval.scorer.scorer import Scorer


class TestRougeScoreMissingDependency:
    def test_raises_import_error_with_install_hint(self, monkeypatch):
        # simulate "rouge-score is not installed"
        monkeypatch.setitem(sys.modules, "rouge_score", None)
        with pytest.raises(ImportError, match="pip install rouge-score"):
            Scorer.rouge_score("a b c", "a b", "rouge1")


class TestRougeScoreDependencyPresent:
    def test_scoring_path_unchanged(self, monkeypatch):
        class _FakeScore:
            fmeasure = 0.75

        class _FakeRougeScorer:
            def __init__(self, score_types, use_stemmer):
                pass

            def score(self, target, prediction):
                return {"rouge1": _FakeScore()}

        fake = types.ModuleType("rouge_score")
        fake.rouge_scorer = types.ModuleType("rouge_score.rouge_scorer")
        fake.rouge_scorer.RougeScorer = _FakeRougeScorer
        monkeypatch.setitem(sys.modules, "rouge_score", fake)

        assert Scorer.rouge_score("a b c", "a b", "rouge1") == 0.75


class TestSentenceBleuScoreMissingDependency:
    def test_raises_import_error_with_install_hint(self, monkeypatch):
        # simulate "nltk is not installed"
        monkeypatch.setitem(sys.modules, "nltk", None)
        with pytest.raises(ImportError, match="pip install nltk"):
            Scorer.sentence_bleu_score(["a reference"], "a prediction")


class TestSentenceBleuScoreDependencyPresent:
    def test_scoring_path_unchanged(self, monkeypatch):
        fake_tokenize = types.ModuleType("nltk.tokenize")
        fake_tokenize.word_tokenize = lambda text: text.split()
        fake_bleu = types.ModuleType("nltk.translate.bleu_score")
        fake_bleu.sentence_bleu = lambda refs, hyp, weights=None: 0.5
        monkeypatch.setitem(sys.modules, "nltk", types.ModuleType("nltk"))
        monkeypatch.setitem(sys.modules, "nltk.tokenize", fake_tokenize)
        monkeypatch.setitem(
            sys.modules, "nltk.translate", types.ModuleType("nltk.translate")
        )
        monkeypatch.setitem(sys.modules, "nltk.translate.bleu_score", fake_bleu)

        assert Scorer.sentence_bleu_score("a b", "a c") == 0.5

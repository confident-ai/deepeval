"""
Tests that Scorer raises a clear ImportError with install instructions when an
optional dependency (rouge-score, nltk) is missing, instead of crashing later
with a confusing NameError.

All tests are offline - no model, network, or API key required.
"""

import sys
import pytest

from deepeval.scorer.scorer import Scorer


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _BlockImport:
    """Context manager that makes specific top-level imports appear missing."""

    def __init__(self, *blocked: str):
        self._blocked = set(blocked)
        self._original = None

    def __enter__(self):
        import builtins

        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name in self._blocked:
                raise ImportError(f"Mocked missing package: {name}")
            return real_import(name, *args, **kwargs)

        self._original = real_import
        builtins.__import__ = _import
        # Remove any already-cached module so the guarded import actually runs.
        for pkg in self._blocked:
            sys.modules.pop(pkg, None)
            sys.modules.pop(pkg.split(".")[0], None)
        return self

    def __exit__(self, *args):
        import builtins

        builtins.__import__ = self._original
        for pkg in self._blocked:
            sys.modules.pop(pkg, None)


# --------------------------------------------------------------------------- #
# rouge_score - clear ImportError when rouge-score is not installed
# --------------------------------------------------------------------------- #


def test_rouge_score_raises_import_error_when_package_missing():
    with _BlockImport("rouge_score"):
        with pytest.raises(ImportError, match="rouge-score"):
            Scorer.rouge_score(
                target="The cat sat on the mat",
                prediction="The cat sat on the mat",
                score_type="rouge1",
            )


def test_rouge_score_error_message_contains_install_hint():
    with _BlockImport("rouge_score"):
        with pytest.raises(ImportError) as exc_info:
            Scorer.rouge_score("a", "b", "rouge1")
        assert "pip install" in str(exc_info.value).lower()


# --------------------------------------------------------------------------- #
# sentence_bleu_score - clear ImportError when nltk is not installed
# --------------------------------------------------------------------------- #


def test_sentence_bleu_score_raises_import_error_when_nltk_missing():
    with _BlockImport("nltk"):
        with pytest.raises(ImportError, match="nltk"):
            Scorer.sentence_bleu_score(
                references="the cat sat on the mat",
                prediction="the cat sat on the mat",
            )


def test_sentence_bleu_score_error_message_contains_install_hint():
    with _BlockImport("nltk"):
        with pytest.raises(ImportError) as exc_info:
            Scorer.sentence_bleu_score("a reference", "a prediction")
        assert "pip install" in str(exc_info.value).lower()


# --------------------------------------------------------------------------- #
# Exact-match and quasi-exact-match - never had optional deps, still work
# --------------------------------------------------------------------------- #


def test_exact_match_score_identical_strings():
    assert Scorer.exact_match_score("hello world", "hello world") == 1


def test_exact_match_score_different_strings():
    assert Scorer.exact_match_score("hello world", "hi world") == 0


def test_exact_match_score_empty_prediction():
    assert Scorer.exact_match_score("hello", "") == 0


def test_quasi_exact_match_score_normalizes_whitespace():
    assert Scorer.quasi_exact_match_score("Hello World", "hello  world") == 1

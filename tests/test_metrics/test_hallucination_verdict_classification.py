"""Regression tests for halluctimation verdict classification (#3098).

The score side (only explicit "no" counts as hallucination) and the
reason side (everything not exactly "yes" was bucketed as contradiction)
used different predicates: untagged verdicts that slip past the lenient
JSON extraction ("yes.", "yes ", trailing non-Latin appendages) scored
clean but were reported as contradictions in the generated reason.

These tests are offline: no model calls, no API key — they exercise the
classification predicate and the reason-side bucketing via _get_prompt
capture.
"""

import pytest

from deepeval.metrics.hallucination.hallucination import (
    _is_contradiction,
    HallucinationMetric,
)
from deepeval.metrics.hallucination.schema import HallucinationVerdict


@pytest.fixture(autouse=True)
def _dummy_openai_key(monkeypatch):
    """HallucinationMetric construction initializes a model client lazily;
    a non-empty key satisfies key-presence checks without any API call."""  # noqa: E501
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-regression-test")


def _verdict(value: str, reason: str = "r") -> HallucinationVerdict:
    # model_construct bypasses the Literal validation of HallucinationVerdict:
    # malformed strings arrive at the classification layer through the lenient
    # JSON-extraction fallbacks, not through pydantic.
    return HallucinationVerdict.model_construct(verdict=value, reason=reason)


VERDICT_MATRIX = [
    ("no", True),
    ("no ", True),
    (" NO", True),
    ("yes", False),
    ("yes.", False),
    (" yes ", False),
    (
        "yes\u0576\u0578",
        False,
    ),  # non-Latin appendage (reporter's observed class)
    ("maybe", False),
]


class TestIsContradiction:
    @pytest.mark.parametrize("value,expected", VERDICT_MATRIX)
    def test_predicate_matrix(self, value: str, expected: bool):
        assert _is_contradiction(value) is expected


class TestScoreReasonAgreement:
    """For every malformed verdict the extraction layer can produce, the
    reason bucket must agree with the score count (pre-fix: "yes." scored
    0.0 hallucinations but the reason listed it as a contradiction)."""

    @pytest.mark.parametrize("value,expected", VERDICT_MATRIX)
    def test_reason_bucket_matches_score(
        self, monkeypatch, value: str, expected: bool
    ):
        metric = HallucinationMetric(async_mode=False)
        metric.verdicts = [_verdict(value)]
        metric.score = metric._calculate_score()

        # Recompute the score's classification the way the fix defines it.
        # Contradiction-bucketed verdicts are exactly the ones the score counts.
        classification_as_bucket = "contradiction" if expected else "aligned"

        captured = {}
        metric._get_prompt = lambda key, **kw: captured.update(kw) or {}
        import types

        def _stub(**kw):
            return types.SimpleNamespace(reason="stub")

        monkeypatch.setattr(
            "deepeval.metrics.hallucination.hallucination."
            "generate_with_schema_and_extract",
            _stub,
        )

        metric._generate_reason()

        if classification_as_bucket == "contradiction":
            assert captured["contradictions"] == ["r"]
            assert captured["factual_alignments"] == []
        else:
            assert captured["contradictions"] == []
            assert captured["factual_alignments"] == ["r"]

    def test_untagged_verdict_does_not_contradict_score(self, monkeypatch):
        """Regression heart: 'yes.' must not be reported as a contradiction
        while scoring 0.0 hallucinations (pre-fix FAIL)."""
        metric = HallucinationMetric(async_mode=False)
        metric.verdicts = [_verdict("yes.")]

        assert metric._calculate_score() == 0.0
        metric.score = 0.0

        captured = {}
        metric._get_prompt = lambda key, **kw: captured.update(kw) or {}
        import types

        def _stub(**kw):
            return types.SimpleNamespace(reason="stub")

        monkeypatch.setattr(
            "deepeval.metrics.hallucination.hallucination."
            "generate_with_schema_and_extract",
            _stub,
        )
        metric._generate_reason()

        assert captured["contradictions"] == []
        assert captured["factual_alignments"] == ["r"]

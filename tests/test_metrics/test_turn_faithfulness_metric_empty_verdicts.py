"""Regression tests for TurnFaithfulnessMetric's empty-verdicts code path.

The score-and-reason helpers must always return a (float, str) tuple so that
callers can unpack the result with `score, reason = ...`. Returning a bare int
on the empty-verdicts branch crashed any conversation containing an assistant
turn with no extractable factual claims (e.g. "Sure, happy to help!").

These tests exercise the helpers directly with a stub model, so they run
without an LLM provider API key.
"""

import asyncio

import pytest

from deepeval.metrics import TurnFaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM


class _StubLLM(DeepEvalBaseLLM):
    """Minimal DeepEvalBaseLLM that never calls out to a provider."""

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, *args, **kwargs) -> str:
        raise AssertionError("generate should not be called in this test")

    async def a_generate(self, *args, **kwargs) -> str:
        raise AssertionError("a_generate should not be called in this test")

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub-llm"


@pytest.fixture
def metric() -> TurnFaithfulnessMetric:
    return TurnFaithfulnessMetric(model=_StubLLM())


def test_sync_empty_verdicts_returns_tuple(metric: TurnFaithfulnessMetric):
    result = metric._get_interaction_score_and_reason([], multimodal=False)

    score, reason = result  # must be unpackable
    assert score == 1.0
    assert isinstance(score, float)
    assert isinstance(reason, str) and reason


def test_async_empty_verdicts_returns_tuple(metric: TurnFaithfulnessMetric):
    result = asyncio.run(
        metric._a_get_interaction_score_and_reason([], multimodal=False)
    )

    score, reason = result
    assert score == 1.0
    assert isinstance(score, float)
    assert isinstance(reason, str) and reason


def test_sync_empty_verdicts_respects_include_reason_false():
    m = TurnFaithfulnessMetric(model=_StubLLM(), include_reason=False)
    score, reason = m._get_interaction_score_and_reason([], multimodal=False)
    assert score == 1.0
    assert reason is None


def test_async_empty_verdicts_respects_include_reason_false():
    m = TurnFaithfulnessMetric(model=_StubLLM(), include_reason=False)
    score, reason = asyncio.run(
        m._a_get_interaction_score_and_reason([], multimodal=False)
    )
    assert score == 1.0
    assert reason is None

"""Regression tests for TopicAdherenceMetric's reason generation.

Two related defects:

- `include_reason` was accepted by the constructor but never consulted, so
  both the sync and async paths always made the reason LLM call the user
  explicitly disabled.
- On the empty case (no question-answer pairs), `_generate_reason` returns a
  canned message without an LLM call, but its async twin was missing that
  guard and still prompted the model, so sync and async produced different
  reasons for the same conversation.

These tests run with a stub model, so they need no LLM provider API key.
"""

import asyncio

import pytest

from deepeval.metrics import TopicAdherenceMetric
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


def _metric(**kwargs) -> TopicAdherenceMetric:
    metric = TopicAdherenceMetric(
        relevant_topics=["refunds"], model=_StubLLM(), **kwargs
    )
    metric.score = 0.5
    metric.success = True
    return metric


_NON_EMPTY = ([1, ["on topic"]], [1, ["off topic"]], [0, []], [0, []])
_EMPTY = ([0, []], [0, []], [0, []], [0, []])


def test_sync_generate_reason_respects_include_reason_false():
    metric = _metric(include_reason=False)
    assert metric._generate_reason(*_NON_EMPTY, multimodal=False) is None


def test_async_generate_reason_respects_include_reason_false():
    metric = _metric(include_reason=False)
    assert (
        asyncio.run(metric._a_generate_reason(*_NON_EMPTY, multimodal=False))
        is None
    )


def test_async_empty_case_matches_sync_without_llm_call():
    metric = _metric()
    sync_reason = metric._generate_reason(*_EMPTY, multimodal=False)
    async_reason = asyncio.run(
        metric._a_generate_reason(*_EMPTY, multimodal=False)
    )

    assert isinstance(sync_reason, str) and sync_reason
    assert async_reason == sync_reason

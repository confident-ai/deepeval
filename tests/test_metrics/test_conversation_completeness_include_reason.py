"""Regression tests for ConversationCompletenessMetric and include_reason.

`_generate_reason` returns None when `include_reason=False`, but its async
twin `_a_generate_reason` was missing the guard, so the default
`async_mode=True` configuration still made the reason LLM call (and attached
a reason) that the user explicitly disabled.

These tests run with a stub model, so they need no LLM provider API key.
"""

import asyncio

import pytest

from deepeval.metrics import ConversationCompletenessMetric
from deepeval.metrics.conversation_completeness.schema import (
    ConversationCompletenessVerdict,
    UserIntentions,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, Turn


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


class _SchemaStubLLM(_StubLLM):
    """Answers intention/verdict prompts; the reason prompt must never come."""

    def generate(self, prompt, schema=None, *args, **kwargs):
        if schema is None:
            raise AssertionError("schema-less generate call not expected")
        if schema.__name__ == "UserIntentions":
            return UserIntentions(intentions=["get a refund"])
        if schema.__name__ == "ConversationCompletenessVerdict":
            return ConversationCompletenessVerdict(verdict="yes")
        raise AssertionError(
            f"unexpected schema requested: {schema.__name__} "
            "(the reason prompt must be skipped when include_reason=False)"
        )

    async def a_generate(self, prompt, schema=None, *args, **kwargs):
        return self.generate(prompt, schema=schema, *args, **kwargs)


def _metric(**kwargs) -> ConversationCompletenessMetric:
    metric = ConversationCompletenessMetric(model=_StubLLM(), **kwargs)
    metric.user_intentions = ["get a refund"]
    metric.verdicts = [ConversationCompletenessVerdict(verdict="yes")]
    metric.score = 1.0
    return metric


def test_sync_generate_reason_respects_include_reason_false():
    metric = _metric(include_reason=False)
    assert metric._generate_reason(multimodal=False) is None


def test_async_generate_reason_respects_include_reason_false():
    metric = _metric(include_reason=False)
    assert asyncio.run(metric._a_generate_reason(multimodal=False)) is None


def test_default_async_measure_skips_reason_when_disabled():
    metric = ConversationCompletenessMetric(
        model=_SchemaStubLLM(), include_reason=False
    )
    test_case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="These shoes don't fit."),
            Turn(role="assistant", content="We offer a 30-day full refund."),
        ]
    )
    metric.measure(test_case, _show_indicator=False)

    assert metric.score is not None
    assert metric.reason is None

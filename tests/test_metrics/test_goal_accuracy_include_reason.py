"""Regression tests for GoalAccuracyMetric and include_reason.

`include_reason` was accepted by the constructor but never consulted, so the
final-reason LLM call was made in both the sync and async paths even when the
user explicitly disabled it.

These tests run with stub models, so they need no LLM provider API key.
"""

import asyncio

import pytest

from deepeval.metrics import GoalAccuracyMetric
from deepeval.metrics.goal_accuracy.schema import GoalScore, PlanScore
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
    """Answers the score prompts; the schema-less reason prompt must never come."""

    def generate(self, prompt, schema=None, *args, **kwargs):
        if schema is None:
            raise AssertionError(
                "the final-reason prompt must be skipped when "
                "include_reason=False"
            )
        if schema.__name__ == "GoalScore":
            return GoalScore(score=1.0, reason="goal met")
        if schema.__name__ == "PlanScore":
            return PlanScore(score=1.0, reason="plan followed")
        raise AssertionError(f"unexpected schema requested: {schema.__name__}")

    async def a_generate(self, prompt, schema=None, *args, **kwargs):
        return self.generate(prompt, schema=schema, *args, **kwargs)


_GOAL_SCORES = [GoalScore(score=1.0, reason="goal met")]
_PLAN_SCORES = [PlanScore(score=1.0, reason="plan followed")]


def _metric(**kwargs) -> GoalAccuracyMetric:
    metric = GoalAccuracyMetric(model=_StubLLM(), **kwargs)
    metric.score = 1.0
    return metric


def test_sync_generate_reason_respects_include_reason_false():
    metric = _metric(include_reason=False)
    assert metric._generate_reason(_GOAL_SCORES, _PLAN_SCORES, False) is None


def test_async_generate_reason_respects_include_reason_false():
    metric = _metric(include_reason=False)
    assert (
        asyncio.run(
            metric._a_generate_reason(_GOAL_SCORES, _PLAN_SCORES, False)
        )
        is None
    )


def test_default_async_measure_skips_reason_when_disabled():
    metric = GoalAccuracyMetric(model=_SchemaStubLLM(), include_reason=False)
    test_case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Book me a table for two tonight."),
            Turn(role="assistant", content="Done, your table is booked."),
        ]
    )
    metric.measure(test_case, _show_indicator=False)

    assert metric.score is not None
    assert metric.reason is None

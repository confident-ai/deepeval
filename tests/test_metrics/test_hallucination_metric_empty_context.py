"""Regression tests for HallucinationMetric's empty-context handling.

`context` is a required param, but the shared param check only rejects `None`.
An empty list used to slip through, produce zero verdicts, and get scored as
0 ("no hallucination") -- a silent passing score on a test case the metric
never actually checked. An empty context is now treated the same as a missing
one and raises `MissingTestCaseParamsError`, which the evaluate pipeline can
surface as an error or skip (via `skip_on_missing_params`) instead of a
misleading pass.

These tests use a stub model, so they run without an LLM provider API key. The
stub's generate methods raise if called, proving the metric short-circuits
before ever reaching the judge model.
"""

import asyncio

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import HallucinationMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


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


def _test_case() -> LLMTestCase:
    return LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        context=[],
    )


def test_sync_empty_context_raises():
    metric = HallucinationMetric(model=_StubLLM(), async_mode=False)
    with pytest.raises(MissingTestCaseParamsError):
        metric.measure(_test_case())


def test_async_empty_context_raises():
    metric = HallucinationMetric(model=_StubLLM())
    with pytest.raises(MissingTestCaseParamsError):
        asyncio.run(metric.a_measure(_test_case()))

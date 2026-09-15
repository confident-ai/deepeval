"""Offline tests for ContextualPrecisionMetric's degenerate-input handling.

Contextual precision ranks how early *relevant* chunks appear in the retrieval
context. When there is no context to rank, or no expected output to judge
relevance against, the answer is trivially 0. Previously the metric still
round-tripped through the eval LLM for these cases — wasting latency/cost and
producing non-deterministic results. We now short-circuit to a fixed,
explainable outcome.

These tests are fully offline: a stub ``DeepEvalBaseLLM`` is used and its
``generate``/``a_generate`` are wired to raise, proving the short-circuit path
never touches the LLM.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deepeval.metrics import ContextualPrecisionMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


def _stub_model() -> DeepEvalBaseLLM:
    m = MagicMock(spec=DeepEvalBaseLLM)
    m.get_model_name.return_value = "mock-llm"
    m.supports_multimodal.return_value = False
    # If any code path reaches the LLM for a degenerate input, fail loudly.
    m.generate.side_effect = AssertionError("LLM called on degenerate input")
    m.a_generate.side_effect = AssertionError("LLM called on degenerate input")
    return m


def _make_metric() -> ContextualPrecisionMetric:
    return ContextualPrecisionMetric(
        model=_stub_model(),
        async_mode=False,
        include_reason=True,
        strict_mode=False,
        threshold=0.5,
    )


def _case(expected_output: str, retrieval_context) -> LLMTestCase:
    return LLMTestCase(
        input="q",
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )


class TestDegenerateShortCircuit:
    @pytest.mark.parametrize(
        "expected_output,retrieval_context",
        [
            pytest.param("Some answer.", [], id="empty-context"),
            pytest.param(
                "", ["A great context sentence."], id="empty-expected"
            ),
            pytest.param("   ", ["A context."], id="blank-expected"),
        ],
    )
    def test_degenerate_inputs_short_circuit_to_zero(
        self, expected_output, retrieval_context
    ):
        metric = _make_metric()
        score = metric.measure(_case(expected_output, retrieval_context))
        assert score == 0.0
        assert metric.score == 0.0
        assert metric.verdicts == []
        assert metric.success is False
        assert metric.reason is not None
        assert "could not be measured" in metric.reason

    def test_sync_and_async_agree(self):
        sync = _make_metric()
        sync.measure(_case("Some answer.", []))
        async_metric = ContextualPrecisionMetric(
            model=_stub_model(),
            async_mode=True,
            include_reason=True,
        )
        score = async_metric.measure(_case("Some answer.", []))
        assert score == sync.score == 0.0

    def test_include_reason_false_still_short_circuits(self):
        metric = ContextualPrecisionMetric(
            model=_stub_model(),
            async_mode=False,
            include_reason=False,
        )
        metric.measure(_case("Some answer.", []))
        assert metric.score == 0.0
        assert metric.reason is None

    def test_non_degenerate_is_not_empty(self):
        metric = _make_metric()
        helper = metric._try_degenerate_short_circuit
        assert (
            helper(
                expected_output="A real answer.",
                retrieval_context=["A real context sentence."],
            )
            is False
        )


class TestLLMNotContacted:
    def test_empty_context_never_calls_generate(self):
        metric = _make_metric()  # generate/a_generate raise if invoked
        assert metric.measure(_case("Some answer.", [])) == 0.0

    def test_empty_expected_never_calls_a_generate(self):
        metric = ContextualPrecisionMetric(model=_stub_model(), async_mode=True)
        assert metric.measure(_case("", ["A context sentence."])) == 0.0

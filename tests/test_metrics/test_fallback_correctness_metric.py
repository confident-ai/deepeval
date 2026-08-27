"""Tests for FallbackCorrectnessMetric without external model calls."""

import pytest

from deepeval.metrics.community import FallbackCorrectnessMetric
from deepeval.metrics.community.fallback_correctness.schema import (
    FallbackCorrectnessVerdict,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall


class FakeJudge(DeepEvalBaseLLM):
    def __init__(self, verdict: FallbackCorrectnessVerdict):
        self._verdict = verdict
        self.last_prompt = None
        super().__init__(model="fake-judge")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, *args, schema=None, **kwargs):
        self.last_prompt = prompt
        return self._verdict

    async def a_generate(self, prompt, *args, schema=None, **kwargs):
        self.last_prompt = prompt
        return self._verdict

    def get_model_name(self, *args, **kwargs):
        return "fake-judge"


def make_test_case(actual_output: str) -> LLMTestCase:
    return LLMTestCase(
        input="Where is order 123, and when will it arrive?",
        actual_output=actual_output,
        context=[
            "The order-status tool timed out and returned no order data.",
            "The agent must not claim an order status or delivery date without tool evidence.",
        ],
        tools_called=[
            ToolCall(
                name="get_order_status",
                input_parameters={"order_id": "123"},
                output={"error": "timeout"},
            )
        ],
    )


def verdict(
    acknowledges_limitation: bool = True,
    avoids_unsupported_claims: bool = True,
    recovery_action_appropriate: bool = True,
    reasoning: str = "The response handles the failed lookup safely.",
) -> FallbackCorrectnessVerdict:
    return FallbackCorrectnessVerdict(
        acknowledges_limitation=acknowledges_limitation,
        avoids_unsupported_claims=avoids_unsupported_claims,
        recovery_action_appropriate=recovery_action_appropriate,
        reasoning=reasoning,
    )


def test_passes_safe_fallback_and_includes_runtime_evidence_in_prompt():
    judge = FakeJudge(verdict())
    metric = FallbackCorrectnessMetric(model=judge, async_mode=False)
    test_case = make_test_case(
        "I couldn't retrieve order 123 because the lookup timed out. "
        "I don't have a verified delivery date; please try again shortly."
    )

    score = metric.measure(test_case, _show_indicator=False)

    assert score == 1.0
    assert metric.is_successful() is True
    assert metric.score_breakdown == {
        "acknowledges_limitation": 1.0,
        "avoids_unsupported_claims": 1.0,
        "recovery_action_appropriate": 1.0,
    }
    assert "order-status tool timed out" in judge.last_prompt
    assert '"name": "get_order_status"' in judge.last_prompt
    assert '"error": "timeout"' in judge.last_prompt


def test_fails_when_agent_fabricates_success_after_tool_failure():
    judge = FakeJudge(
        verdict(
            acknowledges_limitation=False,
            avoids_unsupported_claims=False,
            recovery_action_appropriate=False,
            reasoning="The response invents a delivery status after the lookup failed.",
        )
    )
    metric = FallbackCorrectnessMetric(model=judge, async_mode=False)

    score = metric.measure(
        make_test_case("Order 123 shipped today and will arrive tomorrow."),
        _show_indicator=False,
    )

    assert score == 0.0
    assert metric.is_successful() is False
    assert "invents a delivery status" in metric.reason


def test_partial_score_and_custom_threshold():
    judge = FakeJudge(
        verdict(
            recovery_action_appropriate=False,
            reasoning="The response is honest but provides no useful next step.",
        )
    )
    metric = FallbackCorrectnessMetric(
        model=judge,
        async_mode=False,
        threshold=0.6,
    )

    score = metric.measure(
        make_test_case("I couldn't retrieve the order status."),
        _show_indicator=False,
    )

    assert score == pytest.approx(2 / 3)
    assert metric.is_successful() is True


def test_strict_mode_zeroes_partial_score():
    judge = FakeJudge(verdict(recovery_action_appropriate=False))
    metric = FallbackCorrectnessMetric(
        model=judge,
        async_mode=False,
        threshold=0.5,
        strict_mode=True,
    )

    score = metric.measure(
        make_test_case("I couldn't retrieve the order status."),
        _show_indicator=False,
    )

    assert score == 0
    assert metric.threshold == 1
    assert metric.is_successful() is False


def test_async_measure_matches_sync_and_can_omit_reason():
    judge = FakeJudge(verdict())
    metric = FallbackCorrectnessMetric(
        model=judge,
        async_mode=True,
        include_reason=False,
    )

    score = metric.measure(
        make_test_case("The lookup timed out; please try again shortly."),
        _show_indicator=False,
    )

    assert score == 1.0
    assert metric.reason is None

"""Tests for ThoughtActionConsistencyMetric.

These tests use a fake DeepEvalBaseLLM judge so they run without any API key.
They prove the metric:
- scores the fraction of steps whose tool call executes the intent the
  reasoning declared (mismatch -> partial score, clean trace -> 1.0);
- skips tool calls that carry no reasoning and keeps original trace indices;
- raises MissingTestCaseParamsError when no call carries reasoning;
- renders each step's reasoning + tool + arguments into the judge prompt;
- honors strict_mode (threshold 1.0, shortfall -> 0) and include_reason=False;
- surfaces a judge that returns no verdicts instead of contradicting itself.
"""

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import ThoughtActionConsistencyMetric
from deepeval.metrics.community.thought_action_consistency.schema import (
    ThoughtActionConsistencyVerdict,
    StepConsistencyVerdict,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall


class FakeJudge(DeepEvalBaseLLM):
    """Returns a preset verdict, capturing the prompt it was given."""

    def __init__(self, verdict: ThoughtActionConsistencyVerdict):
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


def make_test_case(calls) -> LLMTestCase:
    return LLMTestCase(
        input="Diagnose why checkouts are failing.",
        actual_output="The payment service is returning 500s.",
        tools_called=calls,
    )


LOG_QUERY_CALL = ToolCall(
    name="query_database",
    reasoning=(
        "I need to inspect the payment service's recent error logs to find "
        "the failing operation."
    ),
    input_parameters={
        "service": "payment-service",
        "query": "error logs last 1h",
    },
)
METRICS_API_CALL = ToolCall(
    name="get_service_metrics",
    reasoning=(
        "The logs show timeouts on charge requests, so I will pull the "
        "payment service's p99 latency to confirm saturation."
    ),
    input_parameters={"service": "payment-service", "metric": "p99_latency"},
)


def test_scores_partial_consistency():
    # Step 1 consistent, step 2 consistent, step 3 inconsistent -> 2/3.
    calls = [
        LOG_QUERY_CALL,
        METRICS_API_CALL,
        ToolCall(
            name="get_service_metrics",
            # Argument mismatch: reasoning says payment service, argument is
            # the auth service.
            reasoning="Now I will check the payment service error rate.",
            input_parameters={"service": "auth-api", "metric": "error_rate"},
        ),
    ]
    judge = FakeJudge(
        ThoughtActionConsistencyVerdict(
            verdicts=[
                StepConsistencyVerdict(
                    step_index=1, verdict="consistent", reasoning="ok"
                ),
                StepConsistencyVerdict(
                    step_index=2, verdict="consistent", reasoning="ok"
                ),
                StepConsistencyVerdict(
                    step_index=3,
                    verdict="inconsistent",
                    reasoning=(
                        "Argument mismatch: reasoning says payment service "
                        "but the service argument is auth-api."
                    ),
                ),
            ]
        )
    )
    metric = ThoughtActionConsistencyMetric(
        model=judge, async_mode=False, threshold=0.9
    )

    metric.measure(make_test_case(calls))

    assert metric.score == pytest.approx(2 / 3)
    assert metric.is_successful() is False
    assert "Step 3" in metric.reason


def test_passes_clean_trace():
    judge = FakeJudge(
        ThoughtActionConsistencyVerdict(
            verdicts=[
                StepConsistencyVerdict(step_index=1, verdict="consistent"),
                StepConsistencyVerdict(step_index=2, verdict="consistent"),
            ]
        )
    )
    metric = ThoughtActionConsistencyMetric(model=judge, async_mode=False)

    metric.measure(make_test_case([LOG_QUERY_CALL, METRICS_API_CALL]))

    assert metric.score == 1.0
    assert metric.is_successful() is True


def test_async_measure_matches_sync():
    judge = FakeJudge(
        ThoughtActionConsistencyVerdict(
            verdicts=[
                StepConsistencyVerdict(step_index=1, verdict="consistent"),
                StepConsistencyVerdict(step_index=2, verdict="inconsistent"),
            ]
        )
    )
    metric = ThoughtActionConsistencyMetric(model=judge, async_mode=True)

    metric.measure(make_test_case([LOG_QUERY_CALL, METRICS_API_CALL]))

    assert metric.score == pytest.approx(0.5)


def test_skips_calls_without_reasoning_keeps_original_indices():
    # A routed call with no reasoning must not be judged, and the judged steps
    # keep their original indices (1 and 3), matching the user's trace.
    routed_call = ToolCall(
        name="get_service_metrics",
        reasoning=None,
        input_parameters={"service": "payment-service", "metric": "cpu"},
    )
    calls = [LOG_QUERY_CALL, routed_call, METRICS_API_CALL]
    judge = FakeJudge(
        ThoughtActionConsistencyVerdict(
            verdicts=[
                StepConsistencyVerdict(step_index=1, verdict="consistent"),
                StepConsistencyVerdict(step_index=3, verdict="inconsistent"),
            ]
        )
    )
    metric = ThoughtActionConsistencyMetric(model=judge, async_mode=False)

    metric.measure(make_test_case(calls))

    # The clean kept steps score 1/2, not 2/3 — the routed call is excluded
    # from the denominator, and its index (2) is skipped in the prompt.
    assert metric.score == pytest.approx(0.5)
    assert "Step 1 — Reasoning:" in judge.last_prompt
    assert "Step 3 — Reasoning:" in judge.last_prompt
    assert "Step 2" not in judge.last_prompt
    # Verbose logs report the original positions.
    assert "Judged 2 of 3 steps with reasoning." in metric.verbose_logs


def test_no_reasoning_raises():
    calls = [
        ToolCall(name="query_database", input_parameters={"service": "x"}),
    ]
    judge = FakeJudge(ThoughtActionConsistencyVerdict(verdicts=[]))
    metric = ThoughtActionConsistencyMetric(model=judge, async_mode=False)

    with pytest.raises(MissingTestCaseParamsError, match="reasoning"):
        metric.measure(make_test_case(calls))


def test_prompt_renders_steps():
    judge = FakeJudge(
        ThoughtActionConsistencyVerdict(
            verdicts=[
                StepConsistencyVerdict(step_index=1, verdict="consistent"),
                StepConsistencyVerdict(step_index=2, verdict="consistent"),
            ]
        )
    )
    metric = ThoughtActionConsistencyMetric(model=judge, async_mode=False)

    metric.measure(make_test_case([LOG_QUERY_CALL, METRICS_API_CALL]))

    assert "Step 1 — Reasoning:" in judge.last_prompt
    assert "query_database" in judge.last_prompt
    assert '"service": "payment-service"' in judge.last_prompt
    assert "Step 2 — Reasoning:" in judge.last_prompt
    # Scope rules must be present to keep the judge on-task.
    assert "Do NOT judge whether the step was the best" in judge.last_prompt


def test_strict_mode_sets_threshold_to_one():
    judge = FakeJudge(
        ThoughtActionConsistencyVerdict(
            verdicts=[
                StepConsistencyVerdict(step_index=1, verdict="consistent"),
                StepConsistencyVerdict(step_index=2, verdict="inconsistent"),
            ]
        )
    )
    metric = ThoughtActionConsistencyMetric(
        model=judge, async_mode=False, strict_mode=True
    )

    metric.measure(make_test_case([LOG_QUERY_CALL, METRICS_API_CALL]))

    # 0.5 < strict threshold (1.0) -> score collapses to 0.
    assert metric.score == 0.0
    assert metric.is_successful() is False


def test_include_reason_false_returns_none_reason():
    judge = FakeJudge(
        ThoughtActionConsistencyVerdict(
            verdicts=[
                StepConsistencyVerdict(
                    step_index=1,
                    verdict="inconsistent",
                    reasoning="tool mismatch",
                ),
            ]
        )
    )
    metric = ThoughtActionConsistencyMetric(
        model=judge, async_mode=False, include_reason=False
    )

    metric.measure(make_test_case([LOG_QUERY_CALL]))

    assert metric.score == 0.0
    assert metric.reason is None


def test_empty_verdicts_does_not_contradict():
    judge = FakeJudge(ThoughtActionConsistencyVerdict(verdicts=[]))
    metric = ThoughtActionConsistencyMetric(model=judge, async_mode=False)

    metric.measure(make_test_case([LOG_QUERY_CALL]))

    # Score 0 but the reason must not claim everything was consistent.
    assert metric.score == 0.0
    assert "no verdicts" in metric.reason

"""Tests for ToolFailureRecoveryMetric.

These tests use a fake DeepEvalBaseLLM judge so they run without any API key
(same pattern as test_citation_faithfulness_metric.py). Trace dicts are built
by hand in the _trace_dict format (same pattern as test_agent_loop_detection.py).

The retry-discipline sub-signal is fully deterministic and is additionally
covered without any judge involvement: the clean-trace test proves the judge
is never called when the trace has no failures.
"""

import asyncio

import pytest

from deepeval.metrics.community import ToolFailureRecoveryMetric
from deepeval.metrics.community.tool_failure_recovery.schema import (
    HallucinatedSuccessVerdict,
    HallucinatedSuccessVerdicts,
    RecoveryVerdict,
    RecoveryVerdicts,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class FakeJudge(DeepEvalBaseLLM):
    """Returns a preset response per schema class, capturing every prompt."""

    def __init__(self, responses=None):
        # responses: {schema_cls: schema_instance}
        self._responses = responses or {}
        self.prompts = []
        self.call_count = 0
        super().__init__(model="fake-judge")

    def load_model(self, *args, **kwargs):
        return None

    def _respond(self, prompt, schema):
        self.call_count += 1
        self.prompts.append(prompt)
        if schema not in self._responses:
            raise AssertionError(
                f"Judge was called with unexpected schema {schema} — "
                "this test expected no LLM call for it."
            )
        return self._responses[schema]

    def generate(self, prompt, *args, schema=None, **kwargs):
        return self._respond(prompt, schema)

    async def a_generate(self, prompt, *args, schema=None, **kwargs):
        return self._respond(prompt, schema)

    def get_model_name(self, *args, **kwargs):
        return "fake-judge"


def _honest_and_recovered_judge(n_failures: int = 1) -> FakeJudge:
    """A judge that reports honest output and good recovery for every failure."""
    return FakeJudge(
        {
            HallucinatedSuccessVerdicts: HallucinatedSuccessVerdicts(
                verdicts=[
                    HallucinatedSuccessVerdict(
                        failure_index=i + 1,
                        verdict="honest",
                        reasoning="No downstream claim of results.",
                    )
                    for i in range(n_failures)
                ]
            ),
            RecoveryVerdicts: RecoveryVerdicts(
                verdicts=[
                    RecoveryVerdict(
                        failure_index=i + 1,
                        verdict="recovered",
                        reasoning="The agent adapted after the failure.",
                    )
                    for i in range(n_failures)
                ]
            ),
        }
    )


# ---------------------------------------------------------------------------
# Trace / test-case builder helpers (same shape as test_agent_loop_detection)
# ---------------------------------------------------------------------------


def _make_tool_span(
    name: str, input_data: dict, output=None, error=None, children=None
) -> dict:
    span = {
        "type": "tool",
        "name": name,
        "input": input_data,
        "children": children or [],
    }
    # Mirror create_nested_spans_dict: keys with None values are stripped.
    if output is not None:
        span["output"] = output
    if error is not None:
        span["error"] = error
    return span


def _make_llm_span(output: str, children=None) -> dict:
    return {
        "type": "llm",
        "name": "llm_call",
        "input": "some prompt",
        "output": output,
        "children": children or [],
    }


def _make_agent_span(name: str, children: list) -> dict:
    return {
        "type": "agent",
        "name": name,
        "input": "user query",
        "output": "agent answer",
        "children": children,
    }


def _make_test_case(
    trace_dict: dict, actual_output="test output"
) -> LLMTestCase:
    tc = LLMTestCase(input="test input", actual_output=actual_output)
    tc._trace_dict = trace_dict
    return tc


# ---------------------------------------------------------------------------
# Test 1: Clean trace (no failed tool calls) → 1.0, judge never called
# ---------------------------------------------------------------------------


def test_clean_trace_scores_one_without_llm_call():
    judge = FakeJudge()  # raises if any schema is requested
    trace = _make_agent_span(
        "planner",
        [
            _make_tool_span(
                "search_web", {"query": "Paris weather"}, output="Sunny, 22C"
            ),
            _make_tool_span(
                "get_forecast", {"city": "Paris"}, output="Sunny all week"
            ),
        ],
    )
    metric = ToolFailureRecoveryMetric(model=judge, async_mode=False)
    tc = _make_test_case(trace)

    metric.measure(tc)

    assert metric.score == 1.0
    assert metric.is_successful() is True
    assert "No failed tool calls" in metric.reason
    assert judge.call_count == 0
    assert metric.score_breakdown == {
        "retry_discipline": 1.0,
        "hallucinated_success": 1.0,
        "recovery_quality": 1.0,
    }


# ---------------------------------------------------------------------------
# Test 2: Blind retry storm → retry_discipline 0.0, metric fails
# ---------------------------------------------------------------------------


def test_blind_retry_storm_fails():
    """A failed call blindly retried 3x with identical args (allowance 1)."""
    failing = lambda: _make_tool_span(
        "fetch_price",
        {"ticker": "EQNR"},
        error="TimeoutError: upstream API timed out",
    )
    trace = _make_agent_span(
        "agent", [failing(), failing(), failing(), failing()]
    )
    judge = FakeJudge(
        {
            HallucinatedSuccessVerdicts: HallucinatedSuccessVerdicts(
                verdicts=[
                    HallucinatedSuccessVerdict(
                        failure_index=1, verdict="honest"
                    )
                ]
            ),
            RecoveryVerdicts: RecoveryVerdicts(
                verdicts=[RecoveryVerdict(failure_index=1, verdict="partial")]
            ),
        }
    )
    metric = ToolFailureRecoveryMetric(model=judge, async_mode=False)
    tc = _make_test_case(trace, actual_output="I could not fetch the price.")

    metric.measure(tc)

    assert metric.score_breakdown["retry_discipline"] == 0.0
    assert metric.score == 0.0
    assert metric.is_successful() is False
    assert "retried" in metric.reason


# ---------------------------------------------------------------------------
# Test 3: Failed call + honest degradation → passes
# ---------------------------------------------------------------------------


def test_failed_call_with_honest_degradation_passes():
    trace = _make_agent_span(
        "agent",
        [
            _make_tool_span(
                "fetch_weather",
                {"city": "Paris"},
                error="ConnectionError: service unavailable",
            ),
            _make_llm_span(
                "The weather service is unavailable; I will answer with what "
                "I have and tell the user the forecast could not be fetched."
            ),
        ],
    )
    judge = _honest_and_recovered_judge(n_failures=1)
    metric = ToolFailureRecoveryMetric(model=judge, async_mode=False)
    tc = _make_test_case(
        trace,
        actual_output=(
            "I couldn't retrieve the live forecast (the weather service was "
            "unavailable). Here is what I can tell you without it: ..."
        ),
    )

    metric.measure(tc)

    assert metric.score == 1.0
    assert metric.is_successful() is True
    # Failed call must be surfaced to the judge, tagged for join-back.
    assert any("FAILURE #1" in p for p in judge.prompts)
    assert any("ConnectionError" in p for p in judge.prompts)
    assert "1 failed tool call" in metric.reason


# ---------------------------------------------------------------------------
# Test 4: Hallucinated success → score capped at 0 even with clean retries
# ---------------------------------------------------------------------------


def test_hallucinated_success_caps_score_at_zero():
    trace = _make_agent_span(
        "agent",
        [
            _make_tool_span(
                "fetch_price",
                {"ticker": "EQNR"},
                error="TimeoutError: upstream API timed out",
            ),
        ],
    )
    judge = FakeJudge(
        {
            HallucinatedSuccessVerdicts: HallucinatedSuccessVerdicts(
                verdicts=[
                    HallucinatedSuccessVerdict(
                        failure_index=1,
                        verdict="hallucinated",
                        reasoning="The final answer quotes a price although "
                        "the lookup timed out and nothing else fetched it.",
                    )
                ]
            ),
            # Even a 'recovered' verdict must not rescue a fabricated result.
            RecoveryVerdicts: RecoveryVerdicts(
                verdicts=[RecoveryVerdict(failure_index=1, verdict="recovered")]
            ),
        }
    )
    metric = ToolFailureRecoveryMetric(model=judge, async_mode=False)
    tc = _make_test_case(
        trace, actual_output="The current EQNR price is 312.40 NOK."
    )

    metric.measure(tc)

    assert metric.score == 0.0
    assert metric.is_successful() is False
    assert metric.score_breakdown["hallucinated_success"] == 0.0
    assert metric.score_breakdown["retry_discipline"] == 1.0
    assert "Hallucinated success" in metric.reason


# ---------------------------------------------------------------------------
# Test 5: Adjusted retry then success → passes (changed args never penalized)
# ---------------------------------------------------------------------------


def test_adjusted_retry_then_success_passes():
    trace = _make_agent_span(
        "agent",
        [
            _make_tool_span(
                "search_web",
                {"query": "pariss weathr"},
                error="SearchError: no results for query",
            ),
            _make_tool_span(
                "search_web",
                {"query": "paris weather"},
                output="Sunny, 22C",
            ),
        ],
    )
    judge = _honest_and_recovered_judge(n_failures=1)
    metric = ToolFailureRecoveryMetric(model=judge, async_mode=False)
    tc = _make_test_case(trace, actual_output="It is sunny and 22C in Paris.")

    metric.measure(tc)

    assert metric.score_breakdown["retry_discipline"] == 1.0
    assert metric.score == 1.0
    assert metric.is_successful() is True


# ---------------------------------------------------------------------------
# Test 6: Edge — failed call as the last span in the trace
# ---------------------------------------------------------------------------


def test_failed_call_as_last_span():
    trace = _make_agent_span(
        "agent",
        [
            _make_tool_span(
                "search_web", {"query": "paris hotels"}, output="10 results"
            ),
            _make_tool_span(
                "book_hotel",
                {"hotel_id": "h-42"},
                error="HTTP 500: booking service crashed",
            ),
        ],
    )
    judge = _honest_and_recovered_judge(n_failures=1)
    metric = ToolFailureRecoveryMetric(model=judge, async_mode=False)
    tc = _make_test_case(
        trace,
        actual_output=(
            "I found 10 hotels but the booking step failed, so nothing was "
            "booked. You can book hotel h-42 manually."
        ),
    )

    metric.measure(tc)

    assert metric.score == 1.0
    assert metric.is_successful() is True
    assert "1 failed tool call" in metric.reason


# ---------------------------------------------------------------------------
# Test 7: One blind retry is within the allowance (transient errors exist)
# ---------------------------------------------------------------------------


def test_single_blind_retry_is_allowed():
    trace = _make_agent_span(
        "agent",
        [
            _make_tool_span(
                "fetch_weather",
                {"city": "Paris"},
                error="TimeoutError: request timed out",
            ),
            # One identical retry that succeeds — a reasonable transient-retry.
            _make_tool_span(
                "fetch_weather", {"city": "Paris"}, output="Sunny, 22C"
            ),
        ],
    )
    judge = _honest_and_recovered_judge(n_failures=1)
    metric = ToolFailureRecoveryMetric(model=judge, async_mode=False)
    tc = _make_test_case(trace, actual_output="It is sunny and 22C in Paris.")

    metric.measure(tc)

    assert metric.score_breakdown["retry_discipline"] == 1.0
    assert metric.score == 1.0


# ---------------------------------------------------------------------------
# Test 8: Silent drop → recovery 'ignored' fails the metric
# ---------------------------------------------------------------------------


def test_silently_ignored_failure_fails():
    trace = _make_agent_span(
        "agent",
        [
            _make_tool_span(
                "fetch_weather",
                {"city": "Paris"},
                error="ConnectionError: service unavailable",
            ),
            _make_tool_span(
                "search_restaurants", {"city": "Paris"}, output="5 results"
            ),
        ],
    )
    judge = FakeJudge(
        {
            HallucinatedSuccessVerdicts: HallucinatedSuccessVerdicts(
                verdicts=[
                    HallucinatedSuccessVerdict(
                        failure_index=1, verdict="honest"
                    )
                ]
            ),
            RecoveryVerdicts: RecoveryVerdicts(
                verdicts=[
                    RecoveryVerdict(
                        failure_index=1,
                        verdict="ignored",
                        reasoning="The answer never mentions the weather "
                        "sub-task and just lists restaurants.",
                    )
                ]
            ),
        }
    )
    metric = ToolFailureRecoveryMetric(model=judge, async_mode=False)
    tc = _make_test_case(
        trace, actual_output="Here are 5 restaurants in Paris."
    )

    metric.measure(tc)

    assert metric.score_breakdown["recovery_quality"] == 0.0
    assert metric.score == 0.0
    assert metric.is_successful() is False


# ---------------------------------------------------------------------------
# Test 9: async paths — measure(async_mode=True) and a_measure directly
# ---------------------------------------------------------------------------


def test_async_measure_matches_sync():
    trace = _make_agent_span(
        "agent",
        [
            _make_tool_span(
                "fetch_weather",
                {"city": "Paris"},
                error="ConnectionError: service unavailable",
            ),
        ],
    )
    judge = _honest_and_recovered_judge(n_failures=1)
    metric = ToolFailureRecoveryMetric(model=judge, async_mode=True)
    tc = _make_test_case(
        trace, actual_output="I couldn't fetch the weather; here's the rest."
    )

    metric.measure(tc)

    assert metric.score == 1.0
    assert metric.is_successful() is True
    assert judge.call_count == 2  # hallucination + recovery judges


def test_a_measure_direct_call():
    trace = _make_agent_span(
        "agent",
        [
            _make_tool_span(
                "fetch_weather",
                {"city": "Paris"},
                error="ConnectionError: service unavailable",
            ),
        ],
    )
    judge = _honest_and_recovered_judge(n_failures=1)
    metric = ToolFailureRecoveryMetric(model=judge)
    tc = _make_test_case(
        trace, actual_output="I couldn't fetch the weather; here's the rest."
    )

    score = asyncio.run(metric.a_measure(tc, _show_indicator=False))

    assert score == 1.0
    assert metric.is_successful() is True


# ---------------------------------------------------------------------------
# Test 10: No trace data → 0.0 with descriptive reason (agent_loop parity)
# ---------------------------------------------------------------------------


def test_no_trace_returns_zero():
    judge = FakeJudge()
    metric = ToolFailureRecoveryMetric(model=judge, async_mode=False)
    tc = LLMTestCase(input="x", actual_output="y")
    tc._trace_dict = None

    metric.measure(tc)

    assert metric.score == 0.0
    assert "No trace data" in metric.reason
    assert judge.call_count == 0


# ---------------------------------------------------------------------------
# Test 11: Deterministic failure detection heuristics (no LLM involved)
# ---------------------------------------------------------------------------


def test_failure_detection_heuristics():
    judge = FakeJudge()
    metric = ToolFailureRecoveryMetric(model=judge, async_mode=False)

    # Explicit error field
    assert metric._is_failed_tool_span(_make_tool_span("t", {}, error="boom"))
    # Dict output with truthy error key
    assert metric._is_failed_tool_span(
        _make_tool_span("t", {}, output={"error": "rate limited"})
    )
    # Dict output with failed status
    assert metric._is_failed_tool_span(
        _make_tool_span("t", {}, output={"status": "timeout"})
    )
    # String output matching an anchored error pattern
    assert metric._is_failed_tool_span(
        _make_tool_span("t", {}, output="Error: connection refused")
    )
    assert metric._is_failed_tool_span(
        _make_tool_span("t", {}, output="TimeoutError: took too long")
    )
    # Healthy outputs are NOT failures — even ones mentioning errors mid-text
    assert not metric._is_failed_tool_span(
        _make_tool_span("t", {}, output="Scan complete. No errors found.")
    )
    assert not metric._is_failed_tool_span(
        _make_tool_span("t", {}, output={"status": "ok", "error": None})
    )


# ---------------------------------------------------------------------------
# Test 12: Canonical signature — shape normalization (no LLM involved)
# ---------------------------------------------------------------------------


def test_call_signature_shape_normalization():
    sig = ToolFailureRecoveryMetric._call_signature

    # Key order does not matter
    assert sig(_make_tool_span("t", {"a": 1, "b": 2})) == sig(
        _make_tool_span("t", {"b": 2, "a": 1})
    )
    # Whitespace inside string args is collapsed
    assert sig(_make_tool_span("t", {"q": "paris   weather"})) == sig(
        _make_tool_span("t", {"q": "paris weather"})
    )
    # A JSON-string input equals its dict equivalent
    assert sig(_make_tool_span("t", '{"a": 1}')) == sig(
        _make_tool_span("t", {"a": 1})
    )
    # Changed argument values are different signatures
    assert sig(_make_tool_span("t", {"q": "paris"})) != sig(
        _make_tool_span("t", {"q": "oslo"})
    )
    # Different tool names are different signatures
    assert sig(_make_tool_span("t1", {"q": "paris"})) != sig(
        _make_tool_span("t2", {"q": "paris"})
    )


# ---------------------------------------------------------------------------
# Test 13: strict_mode forces threshold 1.0 and binary scoring
# ---------------------------------------------------------------------------


def test_strict_mode_binary_score():
    trace = _make_agent_span(
        "agent",
        [
            _make_tool_span("fetch_weather", {"city": "Paris"}, error="boom"),
        ],
    )
    judge = FakeJudge(
        {
            HallucinatedSuccessVerdicts: HallucinatedSuccessVerdicts(
                verdicts=[
                    HallucinatedSuccessVerdict(
                        failure_index=1, verdict="honest"
                    )
                ]
            ),
            RecoveryVerdicts: RecoveryVerdicts(
                verdicts=[RecoveryVerdict(failure_index=1, verdict="partial")]
            ),
        }
    )
    metric = ToolFailureRecoveryMetric(
        model=judge, async_mode=False, strict_mode=True
    )
    tc = _make_test_case(trace, actual_output="Some answer.")

    metric.measure(tc)

    assert metric.threshold == 1
    # partial recovery → 0.5 < 1.0 → strict mode forces 0.0
    assert metric.score == 0.0
    assert metric.is_successful() is False

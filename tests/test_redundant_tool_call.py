"""Deterministic tests for RedundantToolCallMetric. No API key required."""

import pytest

from deepeval.metrics import RedundantToolCallMetric
from deepeval.test_case import LLMTestCase


def _tool(name, input=None, output=None, children=None):
    span = {"name": name, "type": "tool", "children": children or []}
    if input is not None:
        span["input"] = input
    if output is not None:
        span["output"] = output
    return span


def _llm(name, input=None, output=None, children=None):
    span = {
        "name": name,
        "type": "llm",
        "model": "gpt-4o",
        "children": children or [],
    }
    if input is not None:
        span["input"] = input
    if output is not None:
        span["output"] = output
    return span


def _agent(name="agent", children=None):
    return {"name": name, "type": "base", "children": children or []}


def _test_case(trace_dict):
    tc = LLMTestCase(input="q", actual_output="a")
    tc._trace_dict = trace_dict
    return tc


LONG_A = "Refunds are allowed within thirty days of purchase."
LONG_B = "The warranty covers manufacturing defects for two years."


def test_no_trace_scores_zero():
    metric = RedundantToolCallMetric()
    tc = LLMTestCase(input="q", actual_output="a")
    assert metric.measure(tc) == 0.0
    assert metric.is_successful() is False
    assert "No trace data found" in metric.reason


def test_distinct_consumed_calls_score_one():
    trace = _agent(
        children=[
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _tool("lookup", input={"q": "warranty"}, output=LONG_B),
            _llm("answer", input=f"{LONG_A} {LONG_B}", output="done"),
        ]
    )
    metric = RedundantToolCallMetric()
    assert metric.measure(_test_case(trace)) == 1.0
    assert metric.is_successful() is True
    assert metric.reason == "No redundant tool usage detected."


def test_duplicate_read_only_call_penalized():
    trace = _agent(
        children=[
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _llm("answer", input=LONG_A, output="done"),
        ]
    )
    metric = RedundantToolCallMetric()
    score = metric.measure(_test_case(trace))
    # 2 read-only calls, 1 of them a redundant re-fetch -> 1/2
    assert score == 0.5
    assert metric.score_breakdown["duplicate_calls"] == 0.5
    assert metric.is_successful() is False
    assert "redundant re-fetches" in metric.reason


def test_duplicate_write_tool_not_penalized():
    """Re-calling a tool outside `read_only_tools` is intentional, not waste."""
    trace = _agent(
        children=[
            _tool("send_email", input={"to": "a@b.c"}, output=LONG_A),
            _tool("send_email", input={"to": "a@b.c"}, output=LONG_A),
            _llm("answer", input=LONG_A, output="done"),
        ]
    )
    metric = RedundantToolCallMetric(read_only_tools=["search"])
    assert metric.measure(_test_case(trace)) == 1.0
    assert metric.score_breakdown["duplicate_calls"] == 1.0


def test_empty_read_only_list_disables_duplicate_detection():
    """An empty allowlist declares that no tool is cacheable, so no repeat
    can be redundant. Distinct from `None`, which treats every tool as
    read-only."""
    trace = _agent(
        children=[
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _llm("answer", input=LONG_A, output="done"),
        ]
    )
    metric = RedundantToolCallMetric(read_only_tools=[])
    assert metric.measure(_test_case(trace)) == 1.0
    assert metric.score_breakdown["duplicate_calls"] == 1.0


def test_three_identical_calls_scale_with_waste():
    """Redundancy is proportional: 2 of 3 calls wasted -> 1/3."""
    trace = _agent(
        children=[
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _llm("answer", input=LONG_A, output="done"),
        ]
    )
    metric = RedundantToolCallMetric()
    assert metric.measure(_test_case(trace)) == pytest.approx(1 / 3)


def test_duplicate_with_different_args_not_redundant():
    trace = _agent(
        children=[
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _tool("search", input={"q": "warranty"}, output=LONG_B),
            _llm("answer", input=f"{LONG_A} {LONG_B}", output="done"),
        ]
    )
    metric = RedundantToolCallMetric()
    assert metric.measure(_test_case(trace)) == 1.0


def test_arg_order_does_not_affect_identity():
    """Same args in a different key order is still the same call."""
    trace = _agent(
        children=[
            _tool("search", input={"a": "1", "b": "2"}, output=LONG_A),
            _tool("search", input={"b": "2", "a": "1"}, output=LONG_A),
            _llm("answer", input=LONG_A, output="done"),
        ]
    )
    metric = RedundantToolCallMetric()
    assert metric.measure(_test_case(trace)) == 0.5


def test_json_string_input_normalized_like_dict():
    trace = _agent(
        children=[
            _tool("search", input='{"q": "refund"}', output=LONG_A),
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _llm("answer", input=LONG_A, output="done"),
        ]
    )
    metric = RedundantToolCallMetric()
    assert metric.measure(_test_case(trace)) == 0.5


def test_unconsumed_output_penalized():
    trace = _agent(
        children=[
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _tool("lookup", input={"q": "warranty"}, output=LONG_B),
            # Only LONG_A reaches the model; LONG_B was fetched and dropped.
            _llm("answer", input=LONG_A, output="done"),
        ]
    )
    metric = RedundantToolCallMetric()
    score = metric.measure(_test_case(trace))
    assert score == 0.5
    assert metric.score_breakdown["unconsumed_outputs"] == 0.5
    assert metric.score_breakdown["unconsumed_tools"] == ["lookup"]
    assert "never passed to an LLM span" in metric.reason


def test_output_consumed_by_llm_with_list_message_input():
    """LLM span input is a list of message dicts in real traces."""
    trace = _agent(
        children=[
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _llm(
                "answer",
                input=[{"role": "user", "content": f"Context: {LONG_A}"}],
                output="done",
            ),
        ]
    )
    metric = RedundantToolCallMetric()
    assert metric.measure(_test_case(trace)) == 1.0


def test_ancestor_llm_input_does_not_count_as_consumption():
    """A parent LLM's input was fixed before its child tool ran, so it cannot
    have consumed that tool's output even though the text coincides."""
    trace = _agent(
        children=[
            _llm(
                "decide",
                input=LONG_A,
                output="call search",
                children=[_tool("search", input={"q": "r"}, output=LONG_A)],
            ),
        ]
    )
    metric = RedundantToolCallMetric()
    score = metric.measure(_test_case(trace))
    assert score == 0.0
    assert metric.score_breakdown["unconsumed_tools"] == ["search"]


def test_short_output_skipped_from_consumption_check():
    """Outputs too short to match distinctively are not judged."""
    trace = _agent(
        children=[
            _tool("ping", input={}, output="OK"),
            _llm("answer", input="unrelated prose", output="done"),
        ]
    )
    metric = RedundantToolCallMetric()
    assert metric.measure(_test_case(trace)) == 1.0
    assert metric.score_breakdown["unconsumed_tools"] == []


def test_non_string_tool_output_is_flattened_not_crashed():
    payload = {"policy": "Refunds are allowed within thirty days of purchase."}
    trace = _agent(
        children=[
            _tool("search", input={"q": "refund"}, output=payload),
            _llm(
                "answer",
                input=[
                    {
                        "role": "user",
                        "content": '{"policy": "Refunds are allowed within '
                        'thirty days of purchase."}',
                    }
                ],
                output="done",
            ),
        ]
    )
    metric = RedundantToolCallMetric()
    assert metric.measure(_test_case(trace)) == 1.0


def test_missing_output_key_does_not_crash():
    trace = _agent(
        children=[
            _tool("search", input={"q": "refund"}),
            _llm("answer", input="prose", output="done"),
        ]
    )
    metric = RedundantToolCallMetric()
    assert metric.measure(_test_case(trace)) == 1.0


def test_both_signals_firing_takes_minimum():
    trace = _agent(
        children=[
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _tool("lookup", input={"q": "warranty"}, output=LONG_B),
            _llm("answer", input=LONG_A, output="done"),
        ]
    )
    metric = RedundantToolCallMetric()
    score = metric.measure(_test_case(trace))
    # duplicates: 1 redundant of 3 -> 2/3; unconsumed: LONG_B of 3 -> 2/3
    assert metric.score_breakdown["duplicate_calls"] == pytest.approx(2 / 3)
    assert metric.score_breakdown["unconsumed_outputs"] == pytest.approx(2 / 3)
    assert score == pytest.approx(2 / 3)


def test_disable_duplicate_check():
    trace = _agent(
        children=[
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _llm("answer", input=LONG_A, output="done"),
        ]
    )
    metric = RedundantToolCallMetric(check_duplicate_calls=False)
    assert metric.measure(_test_case(trace)) == 1.0


def test_disable_unconsumed_check():
    trace = _agent(
        children=[
            _tool("lookup", input={"q": "warranty"}, output=LONG_B),
            _llm("answer", input="unrelated", output="done"),
        ]
    )
    metric = RedundantToolCallMetric(check_unconsumed_output=False)
    assert metric.measure(_test_case(trace)) == 1.0


def test_disabling_both_checks_raises():
    with pytest.raises(ValueError):
        RedundantToolCallMetric(
            check_duplicate_calls=False, check_unconsumed_output=False
        )


def test_strict_mode_is_binary():
    trace = _agent(
        children=[
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _llm("answer", input=LONG_A, output="done"),
        ]
    )
    metric = RedundantToolCallMetric(strict_mode=True)
    assert metric.measure(_test_case(trace)) == 0.0
    assert metric.threshold == 1.0


def test_no_tool_spans_scores_one():
    trace = _agent(children=[_llm("answer", input="q", output="done")])
    metric = RedundantToolCallMetric()
    assert metric.measure(_test_case(trace)) == 1.0


def test_nested_spans_are_walked_recursively():
    """Duplicates buried at different depths must still be found."""
    trace = _agent(
        children=[
            _agent(
                "sub",
                children=[_tool("search", input={"q": "r"}, output=LONG_A)],
            ),
            _agent(
                "sub2",
                children=[
                    _agent(
                        "deep",
                        children=[
                            _tool("search", input={"q": "r"}, output=LONG_A)
                        ],
                    )
                ],
            ),
            _llm("answer", input=LONG_A, output="done"),
        ]
    )
    metric = RedundantToolCallMetric()
    assert metric.measure(_test_case(trace)) == 0.5


def test_real_trace_dict_from_tracing_module():
    """Guard the fakes above against drift in the real trace dict shape."""
    import time

    from deepeval.tracing.tracing import trace_manager
    from deepeval.tracing.types import BaseSpan, LlmSpan, ToolSpan

    def mk(cls, name, children=None, **kw):
        return cls(
            uuid=name,
            trace_uuid="tr",
            parent_uuid=None,
            start_time=time.perf_counter(),
            end_time=time.perf_counter(),
            status="SUCCESS",
            children=children or [],
            name=name,
            **kw,
        )

    tool_1 = mk(ToolSpan, "search", input={"q": "refund"}, output=LONG_A)
    tool_2 = mk(ToolSpan, "search", input={"q": "refund"}, output=LONG_A)
    llm = mk(
        LlmSpan,
        "answer",
        model="gpt-4o",
        input=[{"role": "user", "content": LONG_A}],
        output="done",
    )
    root = mk(BaseSpan, "agent", children=[tool_1, tool_2, llm])
    trace_dict = trace_manager.create_nested_spans_dict(root)

    metric = RedundantToolCallMetric()
    # Real trace: identical read-only re-fetch -> 1 of 2 calls redundant.
    assert metric.measure(_test_case(trace_dict)) == 0.5


@pytest.mark.asyncio
async def test_a_measure_matches_measure():
    trace = _agent(
        children=[
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _tool("search", input={"q": "refund"}, output=LONG_A),
            _llm("answer", input=LONG_A, output="done"),
        ]
    )
    metric = RedundantToolCallMetric()
    assert await metric.a_measure(_test_case(trace)) == 0.5

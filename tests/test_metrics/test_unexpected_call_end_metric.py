import pytest

from deepeval.metrics import UnexpectedCallEndMetric
from tests.test_metrics.helpers import (
    AGENT_HUNG_UP,
    agent_turn,
    caller_turn,
    clean_turns,
    measure,
    without_place,
)


def test_the_agent_hanging_up_on_a_question_is_an_unexpected_end():
    turns = clean_turns() + [caller_turn("Can you check my order?")]
    metric = measure(UnexpectedCallEndMetric(), turns, AGENT_HUNG_UP)
    assert metric.score == 0.0
    assert metric.reason == "1 of 1 calls had an unexpected end."
    assert [without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "unexpected_end", "turn": 6, "end_reason": "agent_ended_call"}
    ]
    # The last instant of timed audio; the final caller line has no timestamp.
    [event] = metric.score_breakdown["events"]
    assert event["mark"] == "moment"
    assert event["start_time"] == pytest.approx(20.5, abs=0.05)
    assert event["duration_seconds"] == 0.0


@pytest.mark.parametrize(
    "last_turn",
    [
        caller_turn("Great, thanks, bye"),
        agent_turn("Thanks for calling, goodbye!"),
    ],
)
def test_the_agent_hanging_up_after_a_goodbye_is_a_normal_end(last_turn):
    turns = clean_turns() + [last_turn]
    metric = measure(UnexpectedCallEndMetric(), turns, AGENT_HUNG_UP)
    assert metric.score == 1.0
    assert metric.reason == "No call had an unexpected end."


@pytest.mark.parametrize("end_reason", ["ERROR", "idle_timeout"])
def test_an_error_or_timeout_always_ends_the_call_badly(end_reason):
    turns = clean_turns() + [agent_turn("Thanks for calling, goodbye!")]
    [event] = measure(
        UnexpectedCallEndMetric(), turns, {"end_reason": end_reason}
    ).score_breakdown["events"]
    assert event["end_reason"] == end_reason


def test_a_platform_agent_hangup_counts_like_the_simulators():
    turns = clean_turns() + [caller_turn("Can you check my order?")]
    metric = measure(
        UnexpectedCallEndMetric(), turns, {"endReason": "AGENT_HANGUP"}
    )
    assert metric.score == 0.0

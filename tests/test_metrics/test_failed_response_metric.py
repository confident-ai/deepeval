import pytest

from deepeval.metrics import FailedResponseMetric
from deepeval.test_case import Turn
from tests.test_metrics.helpers import (
    caller_turn,
    clean_turns,
    measure,
    without_place,
)


def test_a_question_left_at_the_end_of_the_call_fails():
    turns = clean_turns() + [caller_turn("Can you check my order?")]
    metric = measure(FailedResponseMetric(), turns)
    assert metric.score == pytest.approx(3 / 4)
    assert metric.reason == "1 of 4 caller lines had no reply from the agent."
    assert [without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "call_ended_without_reply", "turn": 6}
    ]


def test_a_closing_line_needs_no_reply():
    turns = clean_turns() + [caller_turn("Thanks, bye")]
    metric = measure(FailedResponseMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3


def test_an_empty_agent_turn_is_no_reply():
    turns = clean_turns()
    turns[3] = Turn(role="assistant", content="")
    [event] = measure(FailedResponseMetric(), turns).score_breakdown["events"]
    assert without_place(event) == {
        "type": "empty_reply",
        "turn": 2,
        "reply_turn": 3,
    }


def test_every_caller_line_after_the_last_reply_fails():
    turns = clean_turns() + [
        caller_turn("Hello?"),
        caller_turn("Are you there?"),
    ]
    metric = measure(FailedResponseMetric(), turns)
    assert [event["turn"] for event in metric.score_breakdown["events"]] == [
        6,
        7,
    ]

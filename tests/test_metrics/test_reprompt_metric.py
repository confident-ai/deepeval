import pytest

from deepeval.metrics import RepromptMetric
from tests.test_metrics.helpers import (
    caller_turn,
    clean_turns,
    measure,
    timed_turn,
    without_place,
)


def test_a_caller_repeating_a_line_before_the_reply_fails_that_line():
    turns = clean_turns()
    turns.insert(3, caller_turn("Hello? Can you hear me?"))
    metric = measure(RepromptMetric(), turns)
    assert metric.score == pytest.approx(3 / 4)
    assert metric.reason == (
        "1 of 4 caller lines had to be repeated before the agent replied."
    )
    assert [without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "user_reprompted", "turn": 2, "reprompt_turn": 3}
    ]


def test_closing_lines_in_a_row_need_no_reply():
    turns = clean_turns() + [caller_turn("Thanks"), caller_turn("Bye")]
    metric = measure(RepromptMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3


def test_lines_never_answered_are_left_to_failed_response():
    turns = clean_turns() + [
        caller_turn("Hello?"),
        caller_turn("Are you there?"),
    ]
    metric = measure(RepromptMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 5


def test_one_sentence_split_into_turns_is_one_line():
    turns = [
        timed_turn("assistant", 2.0, 0.0),
        timed_turn("user", 2.0, 3.0),
        timed_turn("user", 1.0, 5.5, seed=1),
        timed_turn("assistant", 2.0, 7.5, seed=2),
    ]
    metric = measure(RepromptMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 1


def test_speaking_again_after_a_silence_is_a_reprompt():
    turns = [
        timed_turn("assistant", 2.0, 0.0),
        timed_turn("user", 2.0, 3.0),
        timed_turn("user", 1.0, 8.0, seed=1),
        timed_turn("assistant", 2.0, 10.0, seed=2),
    ]
    metric = measure(RepromptMetric(), turns)
    assert metric.score == pytest.approx(1 / 2)
    assert [without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "user_reprompted", "turn": 1, "reprompt_turn": 2}
    ]
    # The silence the caller sat through before saying it again.
    [event] = metric.score_breakdown["events"]
    assert event["mark"] == "range"
    assert event["start_time"] == pytest.approx(5.0, abs=0.05)
    assert event["duration_seconds"] == pytest.approx(3.0, abs=0.05)
    assert event["description"] == "Caller repeated themselves after 3.0s"

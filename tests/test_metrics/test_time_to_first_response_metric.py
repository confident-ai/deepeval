import pytest

from deepeval.metrics import TimeToFirstResponseMetric
from tests.test_metrics.helpers import (
    CLEAN_GAPS,
    measure,
    measure_recording,
    speech_like,
    timed_call,
    undecodable,
)


def test_an_undecodable_greeting_is_still_timed():
    turns = timed_call(CLEAN_GAPS, greeting_at=5.0)
    undecodable(turns[0])
    assert (
        measure(
            TimeToFirstResponseMetric(use_speech_detection=False), turns
        ).score
        == 0.0
    )


def test_a_prompt_greeting_passes_time_to_first_response():
    metric = measure(
        TimeToFirstResponseMetric(use_speech_detection=False),
        timed_call(CLEAN_GAPS, greeting_at=1.0),
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 1


def test_a_slow_greeting_fails_time_to_first_response():
    metric = measure(
        TimeToFirstResponseMetric(use_speech_detection=False),
        timed_call(CLEAN_GAPS, greeting_at=5.0),
    )
    assert metric.score == 0.0
    assert metric.score_breakdown["events"] == [
        {
            "type": "slow_first_response",
            "turn": 0,
            "first_response_seconds": 5.0,
            "start_time": 0.0,
            "duration_seconds": 5.0,
            "mark": "range",
            "timing_source": "turn_timestamps",
            "description": "Agent took 5.0s to first respond",
        }
    ]


def test_a_call_the_caller_opens_has_no_first_response_score():
    metric = measure(
        TimeToFirstResponseMetric(use_speech_detection=False),
        timed_call(CLEAN_GAPS),
    )
    assert metric.score is None
    assert metric.reason == (
        "The caller spoke first, so the agent was not expected to greet."
    )


def test_a_slow_answer_to_the_callers_first_words(tmp_path):
    metric = measure_recording(
        TimeToFirstResponseMetric(use_speech_detection=False),
        tmp_path,
        caller=[(0.5, speech_like(1.0))],
        agent=[(5.0, speech_like(2.0))],
    )
    assert metric.score == 0.0
    assert "after the caller's first words" in metric.reason
    [event] = metric.score_breakdown["events"]
    assert event["first_response_seconds"] == pytest.approx(3.75, abs=0.05)


def test_a_prompt_greeting_on_a_recording(tmp_path):
    metric = measure_recording(
        TimeToFirstResponseMetric(use_speech_detection=False),
        tmp_path,
        agent=[(1.0, speech_like(2.0))],
        caller=[(4.0, speech_like(1.0))],
    )
    assert metric.score == 1.0
    assert metric.reason == (
        "No agent reply had a first response over 3 seconds after the call "
        "connected."
    )


def test_a_caller_left_in_silence_before_saying_hello_waited_for_the_agent(
    tmp_path,
):
    metric = measure_recording(
        TimeToFirstResponseMetric(use_speech_detection=False),
        tmp_path,
        caller=[(6.0, speech_like(0.5))],
        agent=[(6.5, speech_like(2.0))],
    )
    assert metric.score == 0.0
    assert "after the call connected" in metric.reason
    [event] = metric.score_breakdown["events"]
    assert event["first_response_seconds"] == pytest.approx(6.0, abs=0.05)

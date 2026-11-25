import pytest

from deepeval.metrics import UserInterruptionMetric
from tests.test_metrics.helpers import (
    CLEAN_GAPS,
    STAMPS_NOTE,
    measure,
    measure_recording,
    speech_like,
    timed_call,
    timed_turn,
)


def test_a_caller_who_waits_for_the_agent_interrupts_no_one():
    metric = measure(
        UserInterruptionMetric(), timed_call(CLEAN_GAPS, greeting_at=1.0)
    )
    assert metric.score == 1.0
    assert metric.reason == (
        f"No caller turn had the caller interrupting the agent.{STAMPS_NOTE}"
    )
    assert metric.score_breakdown["eligible_turns"] == 3


def test_the_caller_talking_over_the_agent_from_turn_timestamps():
    turns = [
        timed_turn("assistant", 3.0, 0.0),
        timed_turn("user", 2.0, 2.0),
        timed_turn("assistant", 2.0, 4.5, seed=1),
        timed_turn("user", 2.0, 7.0, seed=1),
    ]
    metric = measure(UserInterruptionMetric(), turns)
    assert metric.score == 0.5
    assert metric.reason == (
        "1 of 2 caller turns had the caller interrupting the agent."
        f"{STAMPS_NOTE}"
    )
    [event] = metric.score_breakdown["events"]
    assert event["type"] == "talk_over"
    assert event["turn"] == 1


def test_the_caller_talking_over_the_agent_on_a_recording(tmp_path):
    metric = measure_recording(
        UserInterruptionMetric(use_speech_detection=False),
        tmp_path,
        agent=[(0.0, speech_like(4.0))],
        caller=[(2.0, speech_like(3.0)), (6.0, speech_like(2.0))],
    )
    assert metric.score == 0.5
    assert metric.score_breakdown["timing_source"] == "call_recording"
    [event] = metric.score_breakdown["events"]
    assert event["type"] == "talk_over"
    assert event["caller_start"] == pytest.approx(2.0, abs=0.05)
    assert event["start_time"] == pytest.approx(2.0, abs=0.05)
    assert event["duration_seconds"] == pytest.approx(
        event["caller_seconds"], abs=0.01
    )
    assert event["overlap_seconds"] == pytest.approx(1.75, abs=0.1)


def test_a_caller_who_cuts_in_and_backs_off_made_a_false_start(tmp_path):
    metric = measure_recording(
        UserInterruptionMetric(use_speech_detection=False),
        tmp_path,
        agent=[(0.0, speech_like(4.0))],
        caller=[(1.5, speech_like(1.2))],
    )
    assert metric.score == 0.0
    [event] = metric.score_breakdown["events"]
    assert event["type"] == "false_start"


def test_an_mm_hm_while_the_agent_talks_is_not_an_interruption(tmp_path):
    metric = measure_recording(
        UserInterruptionMetric(use_speech_detection=False),
        tmp_path,
        agent=[(0.0, speech_like(4.0))],
        caller=[(1.5, speech_like(0.6)), (5.0, speech_like(2.0))],
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 1


def test_a_call_where_the_caller_never_spoke_is_skipped(tmp_path):
    metric = measure_recording(
        UserInterruptionMetric(use_speech_detection=False),
        tmp_path,
        agent=[(0.0, speech_like(4.0))],
        caller=[(1.5, speech_like(0.6))],
    )
    assert metric.score is None
    assert metric.reason == "The caller never spoke for 1 second or more."

import pytest

from deepeval.metrics import UserInterruptionMetric
from deepeval.test_case import Turn
from tests.test_metrics.system_one_fakes import FakeAudioDecisionModel
from tests.test_metrics.helpers import (
    CLEAN_GAPS,
    STAMPS_NOTE,
    measure,
    measure_recording,
    speech_like,
    timed_call,
    timed_turn,
    to_audio,
    zeros,
)


def test_a_caller_turn_with_a_pause_inside_its_audio_is_still_one_turn():
    line = speech_like(1.5)
    line.extend(zeros(1.5))
    line.extend(speech_like(1.5, seed=1))
    turns = [
        timed_turn("assistant", 2.0, 0.0),
        Turn(
            role="user",
            content="Yes. And another thing.",
            audio=to_audio(line, start_time=2.5),
        ),
        timed_turn("assistant", 2.0, 7.5, seed=1),
    ]
    metric = measure(UserInterruptionMetric(use_speech_detection=False), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 1


def test_a_caller_who_waits_for_the_agent_interrupts_no_one():
    metric = measure(
        UserInterruptionMetric(use_speech_detection=False),
        timed_call(CLEAN_GAPS, greeting_at=1.0),
    )
    assert metric.score == 1.0
    assert metric.reason == (
        f"No caller turn had the caller interrupting the agent.{STAMPS_NOTE}"
    )
    assert metric.score_breakdown["eligible_turns"] == 3


def test_the_caller_talking_over_the_agent_from_turn_timestamps():
    turns = [
        timed_turn("assistant", 3.5, 0.0),
        timed_turn("user", 2.0, 2.0),
        timed_turn("assistant", 2.0, 4.5, seed=1),
        timed_turn("user", 2.0, 7.0, seed=1),
    ]
    metric = measure(UserInterruptionMetric(use_speech_detection=False), turns)
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
    # Marks only where both talked: from the caller's start to the agent's end.
    assert event["mark"] == "range"
    assert event["start_time"] == pytest.approx(2.0, abs=0.05)
    assert event["duration_seconds"] == pytest.approx(
        event["overlap_seconds"], abs=0.05
    )
    assert event["overlap_seconds"] == pytest.approx(1.75, abs=0.1)
    assert event["description"].startswith("Caller talked over the agent")


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


def test_a_caller_interruption_the_decision_model_dismisses_is_dropped(
    tmp_path,
):
    model = FakeAudioDecisionModel(probability=0.2)
    metric = measure_recording(
        UserInterruptionMetric(
            use_speech_detection=False,
            eval_mode="hybrid",
            audio_decision_model=model,
        ),
        tmp_path,
        agent=[(0.0, speech_like(4.0))],
        caller=[(2.0, speech_like(3.0)), (6.0, speech_like(2.0))],
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["dismissed_events"] == 1
    assert len(model.clips) == 1

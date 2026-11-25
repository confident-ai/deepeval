import pytest

from deepeval.metrics import ResponseLatencyMetric
from tests.test_metrics.helpers import (
    TAIL,
    measure,
    measure_recording,
    speech_like,
    timed_call,
    to_audio,
    tone,
    undecodable,
    zeros,
)


def test_a_reply_after_undecodable_caller_audio_is_skipped():
    turns = timed_call([0.5, 0.5, 0.5, 0.5])
    undecodable(turns[2])
    metric = measure(ResponseLatencyMetric(use_speech_detection=False), turns)
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["skipped"] == [
        {
            "turn": 3,
            "error": "The caller's audio before this reply could not be "
            "decoded to time it.",
        }
    ]


def _speech_runs_across_turns_2_and_3():
    turns = timed_call([0.5, 0.0, 0.5])
    turns[2].audio = to_audio(tone(2.0, 220), start_time=5.0)
    turns[3].audio = to_audio(tone(2.0, 300), start_time=7.0)
    return turns


def test_the_recorded_latency_is_the_wait_when_a_reply_has_one():
    turns = timed_call([0.5, 0.5, 0.5])
    turns[3].latency_ms = 5000.0
    [event] = measure(
        ResponseLatencyMetric(use_speech_detection=False), turns
    ).score_breakdown["events"]
    reply_start = turns[3].audio.start_time
    assert event == {
        "type": "slow_response",
        "turn": 3,
        "response_delay_seconds": 5.0,
        # The wait the caller sat through, ending where the reply starts.
        "start_time": reply_start - 5.0,
        "duration_seconds": 5.0,
        "mark": "range",
        "timing_source": "turn_timestamps",
        "description": "Agent took 5.0s to reply",
    }


def test_speech_running_across_touching_clips_without_a_latency_is_skipped():
    metric = measure(
        ResponseLatencyMetric(use_speech_detection=False),
        _speech_runs_across_turns_2_and_3(),
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [3]


def _pause_inside_the_callers_clip():
    # Clips cut from one recording at each turn hold the caller's pause.
    turns = timed_call([0.5])
    caller = speech_like(2.0)
    caller.extend(zeros(4.5))
    turns[0].audio = to_audio(caller, start_time=0.0)
    turns[1].audio = to_audio(speech_like(2.0), start_time=6.5)
    return turns


def test_without_a_latency_the_silence_heard_inside_touching_clips_is_the_wait():
    [event] = measure(
        ResponseLatencyMetric(use_speech_detection=False),
        _pause_inside_the_callers_clip(),
    ).score_breakdown["events"]
    assert event["response_delay_seconds"] == pytest.approx(
        4.5 + TAIL, abs=0.05
    )
    assert event["start_time"] == pytest.approx(2.0 - TAIL, abs=0.05)


def test_the_recorded_latency_wins_over_the_silence_heard():
    turns = _pause_inside_the_callers_clip()
    turns[1].latency_ms = 500.0
    metric = measure(ResponseLatencyMetric(use_speech_detection=False), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["events"] == []


def test_a_recording_replaces_the_turn_timestamps(tmp_path):
    metric = measure_recording(
        ResponseLatencyMetric(use_speech_detection=False),
        tmp_path,
        agent=[(0.0, speech_like(1.5)), (8.0, speech_like(2.0))],
        caller=[(2.0, speech_like(2.0))],
    )
    assert metric.score == 0.0
    assert metric.reason == (
        "1 of 1 agent replies had a wait over 4 seconds before replying."
    )
    assert metric.score_breakdown["timing_source"] == "call_recording"
    [event] = metric.score_breakdown["events"]
    assert event["reply_start"] == pytest.approx(8.0, abs=0.05)
    assert event["response_delay_seconds"] == pytest.approx(4.25, abs=0.05)
    assert event["start_time"] + event["duration_seconds"] == pytest.approx(
        event["reply_start"], abs=0.01
    )

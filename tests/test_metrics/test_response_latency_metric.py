import pytest

from deepeval.metrics import ResponseLatencyMetric
from tests.test_metrics.helpers import (
    measure,
    measure_recording,
    speech_like,
    timed_call,
    undecodable,
)


def test_a_reply_after_undecodable_caller_audio_is_skipped():
    turns = timed_call([0.5, 0.5, 0.5, 0.5])
    undecodable(turns[2])
    metric = measure(ResponseLatencyMetric(), turns)
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["skipped"] == [
        {
            "turn": 3,
            "error": "The caller's audio before this reply could not be "
            "decoded to time it.",
        }
    ]


def test_touching_clips_fall_back_to_the_recorded_latency():
    turns = timed_call([0.5, 0.0, 0.5])
    turns[3].latency_ms = 5000.0
    [event] = measure(ResponseLatencyMetric(), turns).score_breakdown["events"]
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


def test_touching_clips_without_a_latency_are_skipped():
    metric = measure(ResponseLatencyMetric(), timed_call([0.5, 0.0, 0.5]))
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [3]


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

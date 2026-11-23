import pytest

from deepeval.metrics import SilentReplyMetric
from deepeval.test_case import Audio, Turn
from tests.test_metrics.helpers import (
    clean_turns,
    measure,
    to_audio,
    without_place,
    zeros,
)


def test_a_reply_with_text_but_no_audio_is_silent():
    turns = clean_turns()
    turns[3].audio = None
    metric = measure(SilentReplyMetric(), turns)
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == (
        "1 of 3 agent replies had a reply the caller could not hear."
    )
    assert [without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "no_audio", "turn": 3}
    ]


def test_a_reply_whose_audio_holds_no_speech_is_silent():
    turns = clean_turns()
    turns[3].audio = to_audio(zeros(2.0), start_time=turns[3].audio.start_time)
    [event] = measure(SilentReplyMetric(), turns).score_breakdown["events"]
    assert without_place(event) == {"type": "no_speech", "turn": 3}
    assert event["offset_seconds"] == 0.0


def test_undecodable_agent_audio_is_skipped_not_failed():
    turns = clean_turns()
    turns[3].audio = Audio.from_bytes(b"not-a-wav", "audio/wav")
    metric = measure(SilentReplyMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [3]


def test_an_agent_turn_with_no_text_is_not_a_silent_reply():
    turns = clean_turns()
    turns[3] = Turn(role="assistant", content="")
    metric = measure(SilentReplyMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2


def test_a_reply_with_audio_but_no_text_can_still_be_silent():
    turns = clean_turns()
    turns[3] = Turn(role="assistant", content="", audio=to_audio(zeros(2.0)))
    metric = measure(SilentReplyMetric(), turns)
    assert [without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "no_speech", "turn": 3}
    ]

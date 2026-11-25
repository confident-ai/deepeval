import pytest

from deepeval.metrics import VoiceReliabilityMetric
from tests.test_metrics.helpers import (
    AGENT_HUNG_UP,
    caller_turn,
    clean_turns,
    loop_clip,
    measure,
    undecodable,
    with_reply_audio,
)


def test_undecodable_agent_audio_fails_the_call():
    turns = clean_turns()
    undecodable(turns[3])
    metric = measure(VoiceReliabilityMetric(use_speech_detection=False), turns)
    assert metric.score == 0.0
    assert metric.score_breakdown["critical_failure"] is True


def test_audio_and_responsiveness_issues_count_per_exchange():
    turns = with_reply_audio(3, loop_clip()) + [
        caller_turn("Can you check my order?")
    ]
    metric = measure(
        VoiceReliabilityMetric(use_speech_detection=False), turns, AGENT_HUNG_UP
    )
    assert metric.score == pytest.approx(2 / 4)
    assert metric.reason == (
        "2 of 4 exchanges had an issue: Failed Response (1), "
        "Unexpected Call End (1), Audio Loop (1)."
    )

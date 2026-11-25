from deepeval.metrics import VoiceNaturalnessMetric
from tests.test_metrics.helpers import (
    clean_turns,
    clipped,
    measure,
    speech_like,
    with_reply_audio,
)


def test_two_issues_on_one_reply_count_once():
    turns = with_reply_audio(
        3, speech_like(1.0) + clipped(0.5) + speech_like(1.0)
    )
    turns[3].content = " ".join(["word"] * 30)
    metric = measure(VoiceNaturalnessMetric(), turns)
    assert metric.score_breakdown["turns_with_events"] == 1
    assert metric.reason == (
        "1 of 3 agent replies had an issue: Audio Clipping (1), "
        "Speaking Rate (1)."
    )
    assert {event["metric"] for event in metric.score_breakdown["events"]} == {
        "Audio Clipping",
        "Speaking Rate",
    }


def test_a_check_that_cannot_judge_short_replies_is_left_out():
    turns = clean_turns()
    for index in (1, 3, 5):
        with_reply_audio(index, speech_like(1.0, base_hz=230.0), turns)
    metric = measure(VoiceNaturalnessMetric(), turns)
    assert metric.score == 1.0
    parts = metric.score_breakdown["metrics"]
    assert parts["Background Noise"]["score"] is None
    assert parts["Audio Clipping"]["score"] == 1.0

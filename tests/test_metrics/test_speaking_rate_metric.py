import pytest

from deepeval.metrics import SpeakingRateMetric
from deepeval.metrics.voice.prosody.speaking_rate.utils import (
    speech_span_rate_wpm,
)
from deepeval.metrics.voice.utils.analysis import analyze_audio
from tests.test_metrics.helpers import (
    agent_speech,
    edit_reply,
    measure,
    set_content,
    to_audio,
)


def test_digits_are_counted_as_spoken_words():
    reply = analyze_audio(to_audio(agent_speech()))
    with_digits = speech_span_rate_wpm("Call 555-123-4567 now.", reply)
    assert with_digits == pytest.approx(12 / (reply.speech_span_seconds / 60))


def test_a_rushed_reply_is_too_fast():
    turns = edit_reply(1, set_content("Call 555-123-4567 right now please."))
    [event] = measure(SpeakingRateMetric(), turns).score_breakdown["events"]
    assert event["type"] == "too_fast"


def test_speaking_rate_limits_are_configurable():
    turns = edit_reply(1, set_content("Hello."))
    assert measure(SpeakingRateMetric(min_wpm=10), turns).score == 1.0
    with pytest.raises(ValueError):
        SpeakingRateMetric(min_wpm=200, max_wpm=110)


def test_a_reply_without_text_is_skipped_for_speaking_rate():
    turns = edit_reply(1, set_content(""))
    metric = measure(SpeakingRateMetric(), turns)
    assert metric.score == 1.0
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]

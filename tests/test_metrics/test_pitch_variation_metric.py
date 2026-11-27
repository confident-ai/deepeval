import pytest

from deepeval.metrics import PitchVariationMetric
from tests.test_metrics.helpers import (
    agent_speech,
    edit_reply,
    leaping_speech,
    measure,
    set_audio,
)


@pytest.mark.parametrize("semitones", [7, 12])
def test_a_wide_ranging_reply_is_not_flagged_by_default(semitones):
    turns = edit_reply(1, set_audio(leaping_speech(semitones)))
    metric = measure(PitchVariationMetric(), turns)
    assert metric.score == 1.0
    assert metric.reason == (
        "No agent reply had pitch varying under 1 semitones."
    )


def test_a_wide_ranging_reply_is_erratic_once_a_limit_is_set():
    turns = edit_reply(1, set_audio(leaping_speech(7)))
    metric = measure(PitchVariationMetric(max_spread_semitones=8), turns)
    assert metric.reason == (
        "1 of 3 agent replies had pitch varying outside 1 to 8 semitones."
    )
    [event] = metric.score_breakdown["events"]
    assert event["type"] == "erratic"
    assert event["pitch_spread_semitones"] > 8


def test_a_pitch_spread_under_the_set_limit_is_not_erratic():
    turns = edit_reply(1, set_audio(leaping_speech(5)))
    metric = measure(PitchVariationMetric(max_spread_semitones=8), turns)
    assert metric.score == 1.0
    [event] = measure(
        PitchVariationMetric(max_spread_semitones=6), turns
    ).score_breakdown["events"]
    assert 6 < event["pitch_spread_semitones"] < 8


@pytest.mark.parametrize(
    "seconds,score",
    [(3.9, 1.0), (4.0, pytest.approx(2 / 3)), (5.0, pytest.approx(2 / 3))],
)
def test_a_flat_reply_is_monotone_once_it_runs_four_seconds(seconds, score):
    turns = edit_reply(1, set_audio(agent_speech(seconds, glide_hz=0)))
    metric = measure(PitchVariationMetric(), turns)
    assert metric.score == score
    assert metric.score_breakdown["eligible_turns"] == 3
    assert [event["type"] for event in metric.score_breakdown["events"]] == (
        [] if score == 1.0 else ["monotone"]
    )


def test_pitch_variation_limits_are_configurable():
    turns = edit_reply(1, set_audio(leaping_speech(7)))
    assert (
        measure(PitchVariationMetric(max_spread_semitones=11), turns).score
        == 1.0
    )
    flat = edit_reply(1, set_audio(agent_speech(5.0, glide_hz=0)))
    assert (
        measure(PitchVariationMetric(min_spread_semitones=0), flat).score == 1.0
    )
    with pytest.raises(ValueError):
        PitchVariationMetric(min_spread_semitones=6, max_spread_semitones=6)
    with pytest.raises(ValueError):
        PitchVariationMetric(min_spread_semitones=-1)

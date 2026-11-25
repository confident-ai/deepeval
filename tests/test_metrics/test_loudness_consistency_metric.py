import pytest

from deepeval.metrics import LoudnessConsistencyMetric
from tests.test_metrics.helpers import (
    agent_speech,
    clean_turns,
    edit_reply,
    measure,
    set_audio,
    to_audio,
    with_gain,
    zeros,
)


def test_loudness_consistency_limit_is_configurable():
    turns = edit_reply(1, set_audio(with_gain(agent_speech(), 2.5)))
    assert (
        measure(LoudnessConsistencyMetric(max_shift_db=10), turns).score == 1.0
    )
    with pytest.raises(ValueError):
        LoudnessConsistencyMetric(max_shift_db=0)


def test_only_the_odd_reply_is_flagged_not_its_neighbours():
    turns = clean_turns(10)
    turns[5].audio = to_audio(
        with_gain(agent_speech(), 2.5), start_time=turns[5].audio.start_time
    )
    metric = measure(LoudnessConsistencyMetric(), turns)
    assert [event["turn"] for event in metric.score_breakdown["events"]] == [5]


def test_an_inaudible_reply_is_not_a_loudness_shift():
    turns = edit_reply(1, set_audio(zeros(3.0)), clean_turns(8))
    metric = measure(LoudnessConsistencyMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]

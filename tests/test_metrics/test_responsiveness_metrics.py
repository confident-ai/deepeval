import pytest

from deepeval.metrics import (
    FailedResponseMetric,
    RepromptMetric,
    SilentReplyMetric,
    UnexpectedCallEndMetric,
)
from deepeval.test_case import Turn
from tests.test_metrics.helpers import (
    AGENT_HUNG_UP,
    caller_turn,
    clean_turns,
    measure,
)


METRICS = [
    FailedResponseMetric,
    RepromptMetric,
    SilentReplyMetric,
    UnexpectedCallEndMetric,
]


def _metric(metric_class):
    if metric_class in (FailedResponseMetric, RepromptMetric):
        return metric_class(use_speech_detection=False)
    return metric_class()


@pytest.mark.parametrize("metric_class", METRICS)
def test_a_clean_call_scores_one(metric_class):
    assert measure(_metric(metric_class), clean_turns()).score == 1.0


@pytest.mark.parametrize("metric_class", METRICS)
def test_a_call_without_audio_is_skipped(metric_class):
    turns = [
        Turn(role="user", content="Hello?"),
        Turn(role="assistant", content="Hi, how can I help?"),
    ]
    metric = measure(_metric(metric_class), turns)
    assert metric.score is None
    assert metric.reason == "The call has no audio to evaluate."


def test_hanging_up_on_a_question_fails_both_metrics():
    turns = clean_turns() + [caller_turn("Can you check my order?")]
    assert (
        measure(
            FailedResponseMetric(use_speech_detection=False),
            turns,
            AGENT_HUNG_UP,
        ).score
        < 1.0
    )
    assert measure(UnexpectedCallEndMetric(), turns, AGENT_HUNG_UP).score == 0


@pytest.mark.parametrize("metric_class", METRICS)
def test_events_hold_plain_values(metric_class):
    turns = clean_turns() + [caller_turn("Can you check my order?")]
    turns[3].audio = None
    for event in measure(
        _metric(metric_class), turns, AGENT_HUNG_UP
    ).score_breakdown["events"]:
        for value in event.values():
            assert type(value) in (str, int, float)

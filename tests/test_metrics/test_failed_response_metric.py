import pytest

from deepeval.metrics import FailedResponseMetric
from deepeval.test_case import ConversationalTestCase, Turn
from tests.test_metrics.helpers import (
    caller_turn,
    clean_turns,
    measure,
    recording,
    stereo_recording,
    timed_turn,
    without_place,
)
from tests.test_metrics.system_one_fakes import FakeAudioDecisionModel


def test_a_question_left_at_the_end_of_the_call_fails():
    turns = clean_turns() + [caller_turn("Can you check my order?")]
    metric = measure(FailedResponseMetric(use_speech_detection=False), turns)
    assert metric.score == pytest.approx(3 / 4)
    assert metric.reason == "1 of 4 caller lines had no reply from the agent."
    assert [without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "call_ended_without_reply", "turn": 6}
    ]


def test_a_closing_line_needs_no_reply():
    turns = clean_turns() + [caller_turn("Thanks, bye")]
    metric = measure(FailedResponseMetric(use_speech_detection=False), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3


def test_an_empty_agent_turn_is_no_reply():
    turns = clean_turns()
    turns[3] = Turn(role="assistant", content="")
    [event] = measure(
        FailedResponseMetric(use_speech_detection=False), turns
    ).score_breakdown["events"]
    assert without_place(event) == {
        "type": "empty_reply",
        "turn": 2,
        "reply_turn": 3,
    }


def test_one_sentence_split_into_turns_is_one_line():
    turns = [
        timed_turn("assistant", 2.0, 0.0),
        timed_turn("user", 2.0, 3.0),
        timed_turn("user", 1.0, 5.5, seed=1),
    ]
    metric = measure(FailedResponseMetric(use_speech_detection=False), turns)
    assert metric.score == 0.0
    assert [without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "call_ended_without_reply", "turn": 1}
    ]
    # Nothing followed the line, so the mark runs from its last words to where
    # the call's audio ends.
    [event] = metric.score_breakdown["events"]
    assert event["mark"] == "range"
    assert event["start_time"] == pytest.approx(6.25, abs=0.05)
    assert event["duration_seconds"] == pytest.approx(0.25, abs=0.05)


def test_every_caller_line_after_the_last_reply_fails():
    turns = clean_turns() + [
        caller_turn("Hello?"),
        caller_turn("Are you there?"),
    ]
    metric = measure(FailedResponseMetric(use_speech_detection=False), turns)
    assert [event["turn"] for event in metric.score_breakdown["events"]] == [
        6,
        7,
    ]


@pytest.mark.parametrize("probability,score", [(0.3, 1.0), (0.9, 0.75)])
def test_the_decision_model_hears_the_agent_side_of_an_unanswered_line(
    tmp_path, probability, score
):
    turns = clean_turns() + [timed_turn("user", 2.0, 21.0)]
    path = stereo_recording(tmp_path / "call.wav", seconds=26.0)
    model = FakeAudioDecisionModel(probability=probability)
    metric = FailedResponseMetric(
        use_speech_detection=False,
        eval_mode="hybrid",
        audio_decision_model=model,
    )
    metric.measure(
        ConversationalTestCase(turns=turns, audio=recording(path)),
        _show_indicator=False,
    )
    assert metric.score == score
    assert model.questions == [FailedResponseMetric.event_question]

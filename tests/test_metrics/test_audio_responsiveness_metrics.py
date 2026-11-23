import pytest

from deepeval.metrics import (
    FailedResponseMetric,
    SilentReplyMetric,
    UnexpectedCallEndMetric,
)
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from tests.test_metrics.helpers import (
    clean_turns,
    speech_like,
    to_audio,
    zeros,
)

METRICS = [FailedResponseMetric, SilentReplyMetric, UnexpectedCallEndMetric]
AGENT_HUNG_UP = {"Stop reason": "The agent ended the call"}


def _measure(metric, turns, metadata=None):
    metric.measure(
        ConversationalTestCase(turns=turns, metadata=metadata),
        _show_indicator=False,
    )
    return metric


def _caller(content):
    return Turn(role="user", content=content, audio=to_audio(speech_like(1.0)))


def _agent(content):
    return Turn(
        role="assistant", content=content, audio=to_audio(speech_like(1.0))
    )


@pytest.mark.parametrize("metric_class", METRICS)
def test_a_clean_call_scores_one(metric_class):
    assert _measure(metric_class(), clean_turns()).score == 1.0


@pytest.mark.parametrize("metric_class", METRICS)
def test_a_call_without_audio_is_skipped(metric_class):
    turns = [
        Turn(role="user", content="Hello?"),
        Turn(role="assistant", content="Hi, how can I help?"),
    ]
    metric = _measure(metric_class(), turns)
    assert metric.score is None
    assert metric.reason == "The call has no audio to evaluate."


def test_a_question_left_at_the_end_of_the_call_fails():
    turns = clean_turns() + [_caller("Can you check my order?")]
    metric = _measure(FailedResponseMetric(), turns)
    assert metric.score == pytest.approx(3 / 4)
    assert metric.reason == "1 of 4 caller lines had no reply from the agent."
    assert [_without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "call_ended_without_reply", "turn": 6}
    ]


def test_a_closing_line_needs_no_reply():
    turns = clean_turns() + [_caller("Thanks, bye")]
    metric = _measure(FailedResponseMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3


def test_an_empty_agent_turn_is_no_reply():
    turns = clean_turns()
    turns[3] = Turn(role="assistant", content="")
    [event] = _measure(FailedResponseMetric(), turns).score_breakdown["events"]
    assert _without_place(event) == {
        "type": "empty_reply",
        "turn": 2,
        "reply_turn": 3,
    }


def test_every_caller_line_after_the_last_reply_fails():
    turns = clean_turns() + [_caller("Hello?"), _caller("Are you there?")]
    metric = _measure(FailedResponseMetric(), turns)
    assert [event["turn"] for event in metric.score_breakdown["events"]] == [
        6,
        7,
    ]


def test_a_reply_with_text_but_no_audio_is_silent():
    turns = clean_turns()
    turns[3].audio = None
    metric = _measure(SilentReplyMetric(), turns)
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == (
        "1 of 3 agent replies had a reply the caller could not hear."
    )
    assert [_without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "no_audio", "turn": 3}
    ]


def test_a_reply_whose_audio_holds_no_speech_is_silent():
    turns = clean_turns()
    turns[3].audio = to_audio(zeros(2.0), start_time=turns[3].audio.start_time)
    [event] = _measure(SilentReplyMetric(), turns).score_breakdown["events"]
    assert _without_place(event) == {"type": "no_speech", "turn": 3}
    assert event["offset_seconds"] == 0.0


def test_undecodable_agent_audio_is_skipped_not_failed():
    turns = clean_turns()
    turns[3].audio = Audio.from_bytes(b"not-a-wav", "audio/wav")
    metric = _measure(SilentReplyMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [3]


def test_an_agent_turn_with_no_text_is_not_a_silent_reply():
    turns = clean_turns()
    turns[3] = Turn(role="assistant", content="")
    metric = _measure(SilentReplyMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2


def test_the_agent_hanging_up_on_a_question_is_an_unexpected_end():
    turns = clean_turns() + [_caller("Can you check my order?")]
    metric = _measure(UnexpectedCallEndMetric(), turns, AGENT_HUNG_UP)
    assert metric.score == 0.0
    assert metric.reason == "1 of 1 calls had an unexpected end."
    assert [_without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "unexpected_end", "turn": 6, "end_reason": "agent_ended_call"}
    ]


def test_hanging_up_on_a_question_fails_both_metrics():
    turns = clean_turns() + [_caller("Can you check my order?")]
    assert _measure(FailedResponseMetric(), turns, AGENT_HUNG_UP).score < 1.0
    assert _measure(UnexpectedCallEndMetric(), turns, AGENT_HUNG_UP).score == 0


@pytest.mark.parametrize(
    "last_turn",
    [_caller("Great, thanks, bye"), _agent("Thanks for calling, goodbye!")],
)
def test_the_agent_hanging_up_after_a_goodbye_is_a_normal_end(last_turn):
    turns = clean_turns() + [last_turn]
    metric = _measure(UnexpectedCallEndMetric(), turns, AGENT_HUNG_UP)
    assert metric.score == 1.0
    assert metric.reason == "No call had an unexpected end."


def _without_place(event):
    """The event minus where it sits, which is the turn's own audio span."""
    return {
        key: value
        for key, value in event.items()
        if key not in ("offset_seconds", "duration_seconds", "start_time")
    }


@pytest.mark.parametrize("end_reason", ["ERROR", "idle_timeout"])
def test_an_error_or_timeout_always_ends_the_call_badly(end_reason):
    turns = clean_turns() + [_agent("Thanks for calling, goodbye!")]
    [event] = _measure(
        UnexpectedCallEndMetric(), turns, {"end_reason": end_reason}
    ).score_breakdown["events"]
    assert event["end_reason"] == end_reason


def test_a_platform_agent_hangup_counts_like_the_simulators():
    turns = clean_turns() + [_caller("Can you check my order?")]
    metric = _measure(
        UnexpectedCallEndMetric(), turns, {"endReason": "AGENT_HANGUP"}
    )
    assert metric.score == 0.0


@pytest.mark.parametrize("metric_class", METRICS)
def test_events_hold_plain_values(metric_class):
    turns = clean_turns() + [_caller("Can you check my order?")]
    turns[3].audio = None
    for event in _measure(metric_class(), turns, AGENT_HUNG_UP).score_breakdown[
        "events"
    ]:
        for value in event.values():
            assert type(value) in (str, int, float)


def test_a_reply_with_audio_but_no_text_can_still_be_silent():
    turns = clean_turns()
    turns[3] = Turn(role="assistant", content="", audio=to_audio(zeros(2.0)))
    metric = _measure(SilentReplyMetric(), turns)
    assert [_without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "no_speech", "turn": 3}
    ]

import pytest

from deepeval.metrics import (
    AgentInterruptionMetric,
    ResponseDelayConsistencyMetric,
    ResponseLatencyMetric,
    TimeToFirstResponseMetric,
)
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from tests.test_metrics.audio_fixtures import speech_like, timed_call, to_audio

CLEAN_GAPS = [0.5, 0.5, 0.5]

TIMING_ISSUES = [
    (
        ResponseLatencyMetric,
        "a wait over 4 seconds before replying",
        "slow_response",
        [0.5, 6.0, 0.5],
    ),
    (
        ResponseDelayConsistencyMetric,
        "a reply delay over 2 seconds off the agent's typical delay",
        "delay_shift",
        [0.5, 4.0, 0.5],
    ),
    (
        AgentInterruptionMetric,
        "the agent talking over the caller",
        "agent_overlap",
        [0.5, -1.0, 0.5],
    ),
]
METRICS = [metric_class for metric_class, _, _, _ in TIMING_ISSUES] + [
    TimeToFirstResponseMetric
]


def _measure(metric, turns):
    metric.measure(ConversationalTestCase(turns=turns), _show_indicator=False)
    return metric


@pytest.mark.parametrize("metric_class,label,_,__", TIMING_ISSUES)
def test_a_clean_call_scores_one(metric_class, label, _, __):
    metric = _measure(metric_class(), timed_call(CLEAN_GAPS))
    assert metric.score == 1.0
    assert metric.reason == f"No agent reply had {label}."
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["events"] == []


@pytest.mark.parametrize("metric_class,label,event_type,gaps", TIMING_ISSUES)
def test_score_is_the_share_of_agent_replies_without_the_issue(
    metric_class, label, event_type, gaps
):
    metric = _measure(metric_class(), timed_call(gaps))
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == f"1 of 3 agent replies had {label}."
    [event] = metric.score_breakdown["events"]
    assert event["type"] == event_type
    assert event["turn"] == 3


@pytest.mark.parametrize("metric_class", METRICS)
def test_a_call_missing_a_start_time_is_skipped(metric_class):
    turns = timed_call(CLEAN_GAPS, greeting_at=1.0)
    turns[1].audio.start_time = None
    metric = _measure(metric_class(), turns)
    assert metric.score is None
    assert "start_time" in metric.reason


@pytest.mark.parametrize("metric_class", METRICS)
def test_a_call_without_agent_audio_is_skipped(metric_class):
    turns = [
        Turn(
            role="user",
            content="Hello?",
            audio=to_audio(speech_like(1.0), start_time=0.0),
        ),
        Turn(role="assistant", content="Hi, how can I help?"),
    ]
    metric = _measure(metric_class(), turns)
    assert metric.score is None
    assert metric.skipped is True


def _undecodable(turn):
    turn.audio = Audio.from_bytes(
        b"not-a-wav", "audio/wav", start_time=turn.audio.start_time
    )


@pytest.mark.parametrize("metric_class,_,__,___", TIMING_ISSUES)
def test_undecodable_agent_audio_is_skipped_not_failed(
    metric_class, _, __, ___
):
    turns = timed_call([0.5, 0.5, 0.5, 0.5])
    _undecodable(turns[3])
    metric = _measure(metric_class(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["skipped"] == [
        {
            "turn": 3,
            "error": "Reply audio could not be decoded to measure its length.",
        }
    ]


def test_a_reply_after_undecodable_caller_audio_is_skipped():
    turns = timed_call([0.5, 0.5, 0.5, 0.5])
    _undecodable(turns[2])
    metric = _measure(ResponseLatencyMetric(), turns)
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["skipped"] == [
        {
            "turn": 3,
            "error": "The caller's audio before this reply could not be "
            "decoded to time it.",
        }
    ]


def test_an_undecodable_greeting_is_still_timed():
    turns = timed_call(CLEAN_GAPS, greeting_at=5.0)
    _undecodable(turns[0])
    assert _measure(TimeToFirstResponseMetric(), turns).score == 0.0


def test_touching_clips_fall_back_to_the_recorded_latency():
    turns = timed_call([0.5, 0.0, 0.5])
    turns[3].latency_ms = 5000.0
    [event] = _measure(ResponseLatencyMetric(), turns).score_breakdown["events"]
    assert event == {
        "type": "slow_response",
        "turn": 3,
        "response_delay_seconds": 5.0,
    }


def test_touching_clips_without_a_latency_are_skipped():
    metric = _measure(ResponseLatencyMetric(), timed_call([0.5, 0.0, 0.5]))
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [3]


def test_talking_over_the_caller_fails_only_agent_interruption():
    turns = timed_call([0.5, -1.0, 0.5, 0.5])
    assert _measure(AgentInterruptionMetric(), turns).score == 0.75
    for metric_class in (ResponseLatencyMetric, ResponseDelayConsistencyMetric):
        metric = _measure(metric_class(), turns)
        assert metric.score == 1.0
        assert metric.score_breakdown["skipped"] == [
            {"turn": 3, "error": "Agent started before the caller finished."}
        ]


def test_a_brief_overlap_is_not_an_interruption():
    turns = timed_call([0.5, -0.3, 0.5])
    assert _measure(AgentInterruptionMetric(), turns).score == 1.0
    metric = _measure(ResponseLatencyMetric(), turns)
    assert metric.score_breakdown["eligible_turns"] == 3


def test_an_instant_reply_among_slow_ones_is_inconsistent():
    turns = timed_call([3.0, 3.0, 0.0, 3.0])
    turns[5].latency_ms = 0.0
    [event] = _measure(ResponseDelayConsistencyMetric(), turns).score_breakdown[
        "events"
    ]
    assert event["turn"] == 5
    assert event["shift_seconds"] == -3.0


def test_delay_consistency_needs_three_replies():
    metric = _measure(ResponseDelayConsistencyMetric(), timed_call([0.5, 4.0]))
    assert metric.score is None
    assert metric.reason == (
        "At least 3 timed assistant replies are needed to compare their "
        "delays."
    )


def test_a_prompt_greeting_passes_time_to_first_response():
    metric = _measure(
        TimeToFirstResponseMetric(), timed_call(CLEAN_GAPS, greeting_at=1.0)
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 1


def test_a_slow_greeting_fails_time_to_first_response():
    metric = _measure(
        TimeToFirstResponseMetric(), timed_call(CLEAN_GAPS, greeting_at=5.0)
    )
    assert metric.score == 0.0
    assert metric.score_breakdown["events"] == [
        {
            "type": "slow_first_response",
            "turn": 0,
            "first_response_seconds": 5.0,
        }
    ]


def test_a_call_the_caller_opens_has_no_first_response_score():
    metric = _measure(TimeToFirstResponseMetric(), timed_call(CLEAN_GAPS))
    assert metric.score is None
    assert metric.reason == (
        "The caller spoke first, so the agent was not expected to greet."
    )


def test_timing_limits_are_configurable():
    slow = timed_call([0.5, 4.0, 0.5])
    assert (
        _measure(ResponseLatencyMetric(max_latency_seconds=3), slow).score
        == pytest.approx(2 / 3)
    )
    assert (
        _measure(
            ResponseDelayConsistencyMetric(max_shift_seconds=4), slow
        ).score
        == 1.0
    )
    late = timed_call(CLEAN_GAPS, greeting_at=5.0)
    assert (
        _measure(TimeToFirstResponseMetric(max_wait_seconds=6), late).score
        == 1.0
    )
    for metric_class, setting in (
        (ResponseLatencyMetric, "max_latency_seconds"),
        (ResponseDelayConsistencyMetric, "max_shift_seconds"),
        (TimeToFirstResponseMetric, "max_wait_seconds"),
    ):
        with pytest.raises(ValueError):
            metric_class(**{setting: 0})


@pytest.mark.parametrize("metric_class,_,__,gaps", TIMING_ISSUES)
def test_timing_events_hold_plain_numbers(metric_class, _, __, gaps):
    [event] = _measure(metric_class(), timed_call(gaps)).score_breakdown[
        "events"
    ]
    for value in event.values():
        assert type(value) in (str, int, float)

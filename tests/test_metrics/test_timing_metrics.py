import pytest

from deepeval.metrics import (
    AgentInterruptionMetric,
    ResponseDelayConsistencyMetric,
    ResponseLatencyMetric,
    TimeToFirstResponseMetric,
    UserInterruptionMetric,
)
from deepeval.test_case import Turn
from tests.test_metrics.helpers import (
    CLEAN_GAPS,
    STAMPS_NOTE,
    agent_backs_off_call,
    measure,
    measure_recording,
    speech_like,
    timed_call,
    to_audio,
    undecodable,
)


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
        "the agent interrupting the caller",
        "talk_over",
        [0.5, -1.5, 0.5],
    ),
]


METRICS = [metric_class for metric_class, _, _, _ in TIMING_ISSUES] + [
    TimeToFirstResponseMetric,
    UserInterruptionMetric,
]


@pytest.mark.parametrize("metric_class,label,_,__", TIMING_ISSUES)
def test_a_clean_call_scores_one(metric_class, label, _, __):
    metric = measure(
        metric_class(use_speech_detection=False),
        timed_call(CLEAN_GAPS, greeting_at=1.0),
    )
    assert metric.score == 1.0
    assert metric.reason == f"No agent reply had {label}.{STAMPS_NOTE}"
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["events"] == []
    assert metric.score_breakdown["timing_source"] == "turn_timestamps"


@pytest.mark.parametrize("metric_class,label,event_type,gaps", TIMING_ISSUES)
def test_score_is_the_share_of_agent_replies_without_the_issue(
    metric_class, label, event_type, gaps
):
    metric = measure(
        metric_class(use_speech_detection=False),
        timed_call(gaps, greeting_at=1.0),
    )
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == f"1 of 3 agent replies had {label}.{STAMPS_NOTE}"
    [event] = metric.score_breakdown["events"]
    assert event["type"] == event_type
    assert event["turn"] == 4


@pytest.mark.parametrize("metric_class", METRICS)
def test_a_call_missing_a_start_time_is_skipped(metric_class):
    turns = timed_call(CLEAN_GAPS, greeting_at=1.0)
    turns[1].audio.start_time = None
    metric = measure(metric_class(use_speech_detection=False), turns)
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
    metric = measure(metric_class(use_speech_detection=False), turns)
    assert metric.score is None
    assert metric.skipped is True


@pytest.mark.parametrize("metric_class,_,__,___", TIMING_ISSUES)
def test_undecodable_agent_audio_is_skipped_not_failed(
    metric_class, _, __, ___
):
    turns = timed_call([0.5, 0.5, 0.5, 0.5], greeting_at=1.0)
    undecodable(turns[4])
    metric = measure(metric_class(use_speech_detection=False), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["skipped"] == [
        {
            "turn": 4,
            "error": "Reply audio could not be decoded to measure its length.",
        }
    ]


def test_talking_over_the_caller_fails_only_agent_interruption():
    turns = timed_call([0.5, -1.5, 0.5, 0.5], greeting_at=1.0)
    assert (
        measure(
            AgentInterruptionMetric(use_speech_detection=False), turns
        ).score
        == 0.75
    )
    for metric_class in (ResponseLatencyMetric, ResponseDelayConsistencyMetric):
        metric = measure(metric_class(use_speech_detection=False), turns)
        assert metric.score == 1.0
        assert metric.score_breakdown["skipped"] == [
            {"turn": 4, "error": "Agent started before the caller finished."}
        ]


def test_a_brief_overlap_is_not_an_interruption():
    turns = timed_call([0.5, -0.3, 0.5])
    assert (
        measure(
            AgentInterruptionMetric(use_speech_detection=False), turns
        ).score
        == 1.0
    )
    metric = measure(ResponseLatencyMetric(use_speech_detection=False), turns)
    assert metric.score_breakdown["eligible_turns"] == 3


def test_timing_limits_are_configurable():
    slow = timed_call([0.5, 4.0, 0.5])
    assert measure(
        ResponseLatencyMetric(
            max_latency_seconds=3, use_speech_detection=False
        ),
        slow,
    ).score == pytest.approx(2 / 3)
    assert (
        measure(
            ResponseDelayConsistencyMetric(
                max_shift_seconds=4, use_speech_detection=False
            ),
            slow,
        ).score
        == 1.0
    )
    late = timed_call(CLEAN_GAPS, greeting_at=5.0)
    assert (
        measure(
            TimeToFirstResponseMetric(
                max_wait_seconds=6, use_speech_detection=False
            ),
            late,
        ).score
        == 1.0
    )
    for metric_class, setting in (
        (ResponseLatencyMetric, "max_latency_seconds"),
        (ResponseDelayConsistencyMetric, "max_shift_seconds"),
        (TimeToFirstResponseMetric, "max_wait_seconds"),
        (AgentInterruptionMetric, "max_caller_silence_seconds"),
    ):
        with pytest.raises(ValueError):
            metric_class(**{setting: 0})


@pytest.mark.parametrize("metric_class,_,__,gaps", TIMING_ISSUES)
def test_timing_events_hold_plain_numbers(metric_class, _, __, gaps):
    [event] = measure(
        metric_class(use_speech_detection=False), timed_call(gaps)
    ).score_breakdown["events"]
    for value in event.values():
        assert type(value) in (str, int, float)


def test_a_collision_is_only_an_agent_interruption():
    turns = agent_backs_off_call(1.4)
    assert (
        measure(
            AgentInterruptionMetric(use_speech_detection=False), turns
        ).score
        == 0.5
    )
    metric = measure(UserInterruptionMetric(use_speech_detection=False), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2


@pytest.mark.parametrize("caller_at", [2.7, 3.0, 3.3])
def test_starting_within_half_a_second_of_the_agent_is_a_collision(
    tmp_path, caller_at
):
    metric = measure_recording(
        UserInterruptionMetric(use_speech_detection=False),
        tmp_path,
        agent=[(0.0, speech_like(1.5)), (3.0, speech_like(1.0))],
        caller=[(caller_at, speech_like(2.0))],
    )
    assert metric.score == 1.0
    agent = measure_recording(
        AgentInterruptionMetric(use_speech_detection=False),
        tmp_path,
        agent=[(0.0, speech_like(1.5)), (3.0, speech_like(1.0))],
        caller=[(caller_at, speech_like(2.0))],
    )
    assert agent.score == 0.0

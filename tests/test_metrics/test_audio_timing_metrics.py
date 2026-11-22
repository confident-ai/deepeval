import pytest

from deepeval.metrics import (
    AgentInterruptionMetric,
    ResponseDelayConsistencyMetric,
    ResponseLatencyMetric,
    TimeToFirstResponseMetric,
)
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from tests.test_metrics.helpers import (
    speech_like,
    stereo_recording,
    timed_call,
    to_audio,
)

CLEAN_GAPS = [0.5, 0.5, 0.5]
STAMPS_NOTE = (
    " Judged from each turn's own audio and timestamps, as no stereo call "
    "recording was available."
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
    TimeToFirstResponseMetric
]


def _measure(metric, turns):
    metric.measure(ConversationalTestCase(turns=turns), _show_indicator=False)
    return metric


@pytest.mark.parametrize("metric_class,label,_,__", TIMING_ISSUES)
def test_a_clean_call_scores_one(metric_class, label, _, __):
    metric = _measure(metric_class(), timed_call(CLEAN_GAPS, greeting_at=1.0))
    assert metric.score == 1.0
    assert metric.reason == f"No agent reply had {label}.{STAMPS_NOTE}"
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["events"] == []
    assert metric.score_breakdown["timing_source"] == "turn_timestamps"


@pytest.mark.parametrize("metric_class,label,event_type,gaps", TIMING_ISSUES)
def test_score_is_the_share_of_agent_replies_without_the_issue(
    metric_class, label, event_type, gaps
):
    metric = _measure(metric_class(), timed_call(gaps, greeting_at=1.0))
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == f"1 of 3 agent replies had {label}.{STAMPS_NOTE}"
    [event] = metric.score_breakdown["events"]
    assert event["type"] == event_type
    assert event["turn"] == 4


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
    turns = timed_call([0.5, 0.5, 0.5, 0.5], greeting_at=1.0)
    _undecodable(turns[4])
    metric = _measure(metric_class(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["skipped"] == [
        {
            "turn": 4,
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
    turns = timed_call([0.5, -1.5, 0.5, 0.5], greeting_at=1.0)
    assert _measure(AgentInterruptionMetric(), turns).score == 0.75
    for metric_class in (ResponseLatencyMetric, ResponseDelayConsistencyMetric):
        metric = _measure(metric_class(), turns)
        assert metric.score == 1.0
        assert metric.score_breakdown["skipped"] == [
            {"turn": 4, "error": "Agent started before the caller finished."}
        ]


def test_the_agents_opening_line_is_not_an_interruption():
    metric = _measure(AgentInterruptionMetric(), timed_call([-1.5, 0.5]))
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 1


def test_an_agent_that_backs_off_from_the_caller_made_a_false_start():
    turns = timed_call([0.5, 0.5, 0.5], reply_seconds=1.0, greeting_at=1.0)
    assert _measure(AgentInterruptionMetric(), turns).score == 1.0
    metric = _measure(
        AgentInterruptionMetric(caller_start_window_ms=2000), turns
    )
    assert metric.score == pytest.approx(1 / 3)
    assert {event["type"] for event in metric.score_breakdown["events"]} == {
        "false_start"
    }


def test_a_false_start_after_a_long_caller_silence_is_excused():
    turns = timed_call([0.5, 5.0, 0.5], reply_seconds=1.0, greeting_at=1.0)
    window = {"caller_start_window_ms": 2000}
    metric = _measure(AgentInterruptionMetric(**window), turns)
    assert metric.score == pytest.approx(2 / 3)
    assert 4 not in [e["turn"] for e in metric.score_breakdown["events"]]
    metric = _measure(
        AgentInterruptionMetric(**window, max_caller_silence_seconds=7), turns
    )
    assert metric.score == pytest.approx(1 / 3)


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
    assert _measure(
        ResponseLatencyMetric(max_latency_seconds=3), slow
    ).score == pytest.approx(2 / 3)
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
        (AgentInterruptionMetric, "caller_start_window_ms"),
        (AgentInterruptionMetric, "max_caller_silence_seconds"),
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


def _recorded(tmp_path, *, caller, agent):
    path = stereo_recording(tmp_path / "call.wav", caller=caller, agent=agent)
    return timed_call(CLEAN_GAPS), path


def _measure_recording(metric, tmp_path, *, caller, agent):
    turns, path = _recorded(tmp_path, caller=caller, agent=agent)
    metric.measure(
        ConversationalTestCase(turns=turns, call_recording_path=path),
        _show_indicator=False,
    )
    return metric


def test_a_recording_replaces_the_turn_timestamps(tmp_path):
    metric = _measure_recording(
        ResponseLatencyMetric(),
        tmp_path,
        agent=[(0.0, speech_like(1.5)), (8.0, speech_like(2.0))],
        caller=[(2.0, speech_like(2.0))],
    )
    assert metric.score == 0.0
    assert metric.reason == (
        "1 of 1 agent replies had a wait over 4 seconds before replying."
    )
    assert metric.score_breakdown["timing_source"] == "call_recording"
    [event] = metric.score_breakdown["events"]
    assert event["at_seconds"] == pytest.approx(8.0, abs=0.05)
    assert event["response_delay_seconds"] == pytest.approx(4.25, abs=0.05)


def test_delay_consistency_reads_the_same_waits_from_a_recording(tmp_path):
    agent = [(0.0, speech_like(1.0))]
    caller = []
    cursor = 1.5
    for wait in (0.5, 0.5, 4.0, 0.5):
        caller.append((cursor, speech_like(1.0)))
        agent.append((cursor + 0.75 + wait, speech_like(1.0)))
        cursor += 0.75 + wait + 2.0
    metric = _measure_recording(
        ResponseDelayConsistencyMetric(), tmp_path, caller=caller, agent=agent
    )
    assert metric.score == 0.75
    [event] = metric.score_breakdown["events"]
    assert event["shift_seconds"] == pytest.approx(3.5, abs=0.05)


def test_a_false_start_on_a_recording(tmp_path):
    metric = _measure_recording(
        AgentInterruptionMetric(),
        tmp_path,
        agent=[
            (0.0, speech_like(1.5)),
            (3.0, speech_like(1.0)),
            (7.0, speech_like(2.0)),
        ],
        caller=[(2.0, speech_like(4.0))],
    )
    assert metric.score == 0.5
    [event] = metric.score_breakdown["events"]
    assert event["type"] == "false_start"
    assert event["at_seconds"] == pytest.approx(3.0, abs=0.05)


def test_talking_over_the_caller_on_a_recording(tmp_path):
    metric = _measure_recording(
        AgentInterruptionMetric(),
        tmp_path,
        agent=[(0.0, speech_like(1.5)), (3.0, speech_like(3.0))],
        caller=[(2.0, speech_like(4.0))],
    )
    [event] = metric.score_breakdown["events"]
    assert event["type"] == "talk_over"
    assert event["overlap_seconds"] == pytest.approx(2.75, abs=0.05)


def test_a_short_reply_after_the_caller_finished_is_not_a_false_start(
    tmp_path,
):
    metric = _measure_recording(
        AgentInterruptionMetric(),
        tmp_path,
        agent=[(0.0, speech_like(1.5)), (4.0, speech_like(0.5))],
        caller=[(2.0, speech_like(1.5)), (6.0, speech_like(1.0))],
    )
    assert metric.score == 1.0


@pytest.mark.parametrize("silence_limit,score", [(4, 1.0), (6, 0.0)])
def test_a_false_start_after_four_seconds_of_caller_silence_is_excused(
    tmp_path, silence_limit, score
):
    metric = _measure_recording(
        AgentInterruptionMetric(max_caller_silence_seconds=silence_limit),
        tmp_path,
        agent=[(0.0, speech_like(1.5)), (8.0, speech_like(0.5))],
        caller=[(2.0, speech_like(1.5)), (8.2, speech_like(1.5))],
    )
    assert metric.score == score


def test_the_callers_whole_pause_decides_whether_a_false_start_is_excused(
    tmp_path,
):
    # Silent 3.8 s before the agent starts, 4.2 s before speaking again.
    metric = _measure_recording(
        AgentInterruptionMetric(),
        tmp_path,
        agent=[(0.0, speech_like(1.5)), (7.05, speech_like(0.5))],
        caller=[(2.0, speech_like(1.5)), (7.45, speech_like(1.5))],
    )
    assert metric.score == 1.0


def test_a_slow_answer_to_the_callers_first_words(tmp_path):
    metric = _measure_recording(
        TimeToFirstResponseMetric(),
        tmp_path,
        caller=[(0.5, speech_like(1.0))],
        agent=[(5.0, speech_like(2.0))],
    )
    assert metric.score == 0.0
    assert "after the caller's first words" in metric.reason
    [event] = metric.score_breakdown["events"]
    assert event["first_response_seconds"] == pytest.approx(3.75, abs=0.05)


def test_a_prompt_greeting_on_a_recording(tmp_path):
    metric = _measure_recording(
        TimeToFirstResponseMetric(),
        tmp_path,
        agent=[(1.0, speech_like(2.0))],
        caller=[(4.0, speech_like(1.0))],
    )
    assert metric.score == 1.0
    assert metric.reason == (
        "No agent reply had a first response over 3 seconds after the call "
        "connected."
    )


def test_a_caller_left_in_silence_before_saying_hello_waited_for_the_agent(
    tmp_path,
):
    metric = _measure_recording(
        TimeToFirstResponseMetric(),
        tmp_path,
        caller=[(6.0, speech_like(0.5))],
        agent=[(6.5, speech_like(2.0))],
    )
    assert metric.score == 0.0
    assert "after the call connected" in metric.reason
    [event] = metric.score_breakdown["events"]
    assert event["first_response_seconds"] == pytest.approx(6.0, abs=0.05)

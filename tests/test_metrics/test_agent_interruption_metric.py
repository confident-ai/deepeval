import pytest

from deepeval.metrics import AgentInterruptionMetric
from deepeval.test_case import Turn
from tests.test_metrics.system_one_fakes import FakeAudioDecisionModel
from tests.test_metrics.helpers import (
    agent_backs_off_call,
    clip_shape,
    measure,
    measure_recording,
    speech_like,
    timed_call,
    timed_turn,
    to_audio,
    zeros,
)


def test_a_reply_with_a_pause_inside_its_audio_is_still_one_reply():
    reply = speech_like(1.0)
    reply.extend(zeros(1.5))
    reply.extend(speech_like(1.0, seed=1))
    turns = [
        timed_turn("user", 2.0, 0.0),
        Turn(
            role="assistant",
            content="Sure. One more thing.",
            audio=to_audio(reply, start_time=2.5),
        ),
        timed_turn("user", 2.0, 6.5, seed=1),
        timed_turn("assistant", 2.0, 9.0, seed=1),
    ]
    metric = measure(AgentInterruptionMetric(use_speech_detection=False), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2


def test_the_agents_opening_line_is_not_an_interruption():
    metric = measure(
        AgentInterruptionMetric(use_speech_detection=False),
        timed_call([0.5, 0.5], greeting_at=0.5),
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2


def test_a_first_reply_over_a_caller_who_opened_the_call_is_an_interruption():
    metric = measure(
        AgentInterruptionMetric(use_speech_detection=False),
        timed_call([-1.5, 0.5]),
    )
    assert metric.score == 0.5
    assert metric.score_breakdown["eligible_turns"] == 2


def test_an_agent_that_backs_off_from_the_caller_made_a_false_start():
    metric = measure(
        AgentInterruptionMetric(use_speech_detection=False),
        agent_backs_off_call(1.4),
    )
    assert metric.score == 0.5
    [event] = metric.score_breakdown["events"]
    assert event["type"] == "false_start"
    assert event["turn"] == 2


def test_a_false_start_after_a_long_caller_silence_is_excused():
    turns = agent_backs_off_call(5.4)
    assert (
        measure(
            AgentInterruptionMetric(use_speech_detection=False), turns
        ).score
        == 1.0
    )
    metric = measure(
        AgentInterruptionMetric(
            max_caller_silence_seconds=7, use_speech_detection=False
        ),
        turns,
    )
    assert metric.score == 0.5


def test_a_false_start_on_a_recording(tmp_path):
    metric = measure_recording(
        AgentInterruptionMetric(use_speech_detection=False),
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
    assert event["reply_start"] == pytest.approx(3.0, abs=0.05)
    assert event["start_time"] == pytest.approx(3.0, abs=0.05)
    assert event["duration_seconds"] == pytest.approx(
        event["agent_seconds"], abs=0.01
    )


def test_talking_over_the_caller_on_a_recording(tmp_path):
    metric = measure_recording(
        AgentInterruptionMetric(use_speech_detection=False),
        tmp_path,
        agent=[(0.0, speech_like(1.5)), (3.0, speech_like(3.0))],
        caller=[(2.0, speech_like(4.0))],
    )
    [event] = metric.score_breakdown["events"]
    assert event["type"] == "talk_over"
    assert event["overlap_seconds"] == pytest.approx(2.75, abs=0.05)
    assert event["start_time"] == pytest.approx(3.0, abs=0.05)
    assert event["duration_seconds"] == pytest.approx(
        event["overlap_seconds"], abs=0.05
    )
    assert event["timing_source"] == "call_recording"


def test_a_short_reply_after_the_caller_finished_is_not_a_false_start(
    tmp_path,
):
    metric = measure_recording(
        AgentInterruptionMetric(use_speech_detection=False),
        tmp_path,
        agent=[(0.0, speech_like(1.5)), (4.0, speech_like(0.5))],
        caller=[(2.0, speech_like(1.5)), (6.0, speech_like(1.0))],
    )
    assert metric.score == 1.0


@pytest.mark.parametrize("silence_limit,score", [(4, 1.0), (6, 0.0)])
def test_a_false_start_after_four_seconds_of_caller_silence_is_excused(
    tmp_path, silence_limit, score
):
    metric = measure_recording(
        AgentInterruptionMetric(
            max_caller_silence_seconds=silence_limit,
            use_speech_detection=False,
        ),
        tmp_path,
        agent=[(0.0, speech_like(1.5)), (8.0, speech_like(0.5))],
        caller=[(2.0, speech_like(1.5)), (8.2, speech_like(1.5))],
    )
    assert metric.score == score


def test_the_callers_whole_pause_decides_whether_a_false_start_is_excused(
    tmp_path,
):
    # Silent 3.8 s before the agent starts, 4.2 s before speaking again.
    metric = measure_recording(
        AgentInterruptionMetric(use_speech_detection=False),
        tmp_path,
        agent=[(0.0, speech_like(1.5)), (7.05, speech_like(0.5))],
        caller=[(2.0, speech_like(1.5)), (7.45, speech_like(1.5))],
    )
    assert metric.score == 1.0


def _talk_over(tmp_path, model):
    return measure_recording(
        AgentInterruptionMetric(
            use_speech_detection=False,
            eval_mode="hybrid",
            audio_decision_model=model,
        ),
        tmp_path,
        agent=[(0.0, speech_like(1.5)), (3.0, speech_like(3.0))],
        caller=[(2.0, speech_like(4.0))],
    )


def test_an_interruption_the_decision_model_hears_no_caller_in_is_dismissed(
    tmp_path,
):
    model = FakeAudioDecisionModel(probability=0.1)
    metric = _talk_over(tmp_path, model)
    assert metric.score == 1.0
    assert metric.score_breakdown["dismissed_events"] == 1
    assert metric.reason.endswith(
        "1 flagged event was dismissed by the audio decision model."
    )
    assert metric.evaluation_model == "fake-hume"
    assert metric.evaluation_cost == pytest.approx(0.001)
    [clip] = model.clips
    channels, seconds = clip_shape(clip)
    # Context around the 2.75 s overlap, cut short by the recording's end.
    assert channels == 1
    assert 2.75 < seconds <= 5.0


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(FakeAudioDecisionModel(probability=0.9), id="confirmed"),
        pytest.param(
            FakeAudioDecisionModel(error=RuntimeError("down")), id="failed"
        ),
    ],
)
def test_an_interruption_the_decision_model_confirms_or_cannot_judge_stays(
    tmp_path, model
):
    metric = _talk_over(tmp_path, model)
    assert metric.score == 0.0
    assert "dismissed_events" not in metric.score_breakdown


def test_a_failed_decision_check_is_logged(tmp_path, caplog):
    _talk_over(tmp_path, FakeAudioDecisionModel(error=RuntimeError("down")))
    assert (
        "Agent Interruption kept a flagged event its audio decision check "
        "failed on: down"
    ) in caplog.text

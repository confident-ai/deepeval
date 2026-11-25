import asyncio
import threading
from array import array

import pytest

from deepeval.metrics import EchoMetric
from deepeval.models import OpenAIAudioDecisionModel
from deepeval.models.system_one.openai_audio_model import AudioVerdict
from deepeval.test_case import ConversationalTestCase
from tests.test_metrics.system_one_fakes import FakeAudioDecisionModel
from tests.test_metrics.helpers import (
    recording,
    speech_like,
    stereo_recording,
    timed_call,
)


def _scaled(clip, gain):
    return array("h", (int(value * gain) for value in clip))


def _echo_on_recording(tmp_path, *, caller, metric=None):
    reply = speech_like(3.0)
    path = stereo_recording(
        tmp_path / "call.wav",
        agent=[(0.0, speech_like(0.5, seed=4)), (1.0, reply)],
        caller=caller(reply),
        seconds=7.0,
    )
    metric = metric or EchoMetric()
    metric.measure(
        ConversationalTestCase(turns=timed_call([0.5]), audio=recording(path)),
        _show_indicator=False,
    )
    return metric


def test_the_agent_heard_back_through_the_callers_speaker_is_an_echo(
    tmp_path,
):
    metric = _echo_on_recording(
        tmp_path, caller=lambda reply: [(1.45, _scaled(reply, 0.2))]
    )
    assert metric.score == 0.0
    assert metric.score_breakdown["timing_source"] == "call_recording"
    [event] = metric.score_breakdown["events"]
    assert event["type"] == "echo"
    assert event["reply_start"] == pytest.approx(1.0, abs=0.05)
    # The whole reply the echo was found on, not the echoed copy.
    assert event["mark"] == "whole_reply"
    assert event["start_time"] == pytest.approx(1.0, abs=0.05)
    assert event["delay_seconds"] == pytest.approx(0.45, abs=0.01)
    assert event["level_db"] == pytest.approx(-14.0, abs=1.5)


@pytest.mark.parametrize(
    "caller",
    [
        pytest.param(
            lambda reply: [(1.45, _scaled(reply, 0.005))], id="too_faint"
        ),
        pytest.param(lambda reply: [(2.0, _scaled(reply, 0.2))], id="too_late"),
        pytest.param(
            lambda reply: [(1.2, speech_like(3.0, base_hz=140.0, seed=3))],
            id="the_caller_talking",
        ),
    ],
)
def test_sound_on_the_callers_side_that_is_not_the_agent_heard_back(
    tmp_path, caller
):
    assert _echo_on_recording(tmp_path, caller=caller).score == 1.0


def test_echo_needs_a_call_recording():
    metric = EchoMetric()
    metric.measure(
        ConversationalTestCase(turns=timed_call([0.5])),
        _show_indicator=False,
    )
    assert metric.score is None
    assert "stereo call recording" in metric.reason


def test_an_echo_the_decision_model_cannot_hear_is_dismissed(tmp_path):
    model = FakeAudioDecisionModel(probability=0.3)
    metric = _echo_on_recording(
        tmp_path,
        caller=lambda reply: [(1.45, _scaled(reply, 0.2))],
        metric=EchoMetric(eval_mode="hybrid", audio_decision_model=model),
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["dismissed_events"] == 1
    assert len(model.clips) == 1


class _ThreadRecordingEchoMetric(EchoMetric):
    def defect_share(self, test_case):
        self.scored_on = threading.current_thread()
        return super().defect_share(test_case)


def test_an_async_measure_with_a_decision_model_runs_off_the_loop(tmp_path):
    reply = speech_like(3.0)
    path = stereo_recording(
        tmp_path / "call.wav",
        agent=[(0.0, speech_like(0.5, seed=4)), (1.0, reply)],
        caller=[(1.45, _scaled(reply, 0.2))],
        seconds=7.0,
    )
    metric = _ThreadRecordingEchoMetric(
        eval_mode="hybrid",
        audio_decision_model=FakeAudioDecisionModel(probability=0.9),
    )

    async def measure():
        await metric.a_measure(
            ConversationalTestCase(
                turns=timed_call([0.5]), audio=recording(path)
            ),
            _show_indicator=False,
        )
        return threading.current_thread()

    loop_thread = asyncio.run(measure())
    assert metric.score == 0.0
    assert metric.scored_on is not loop_thread


def test_an_openai_audio_model_that_says_no_dismisses_the_echo(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        OpenAIAudioDecisionModel,
        "_verdict",
        lambda self, messages: (
            AudioVerdict(answer="no", confidence=0.7),
            0.001,
        ),
    )
    metric = _echo_on_recording(
        tmp_path,
        caller=lambda reply: [(1.45, _scaled(reply, 0.2))],
        metric=EchoMetric(
            eval_mode="hybrid",
            audio_decision_model=OpenAIAudioDecisionModel(api_key="k"),
        ),
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["dismissed_events"] == 1
    assert metric.evaluation_model == "OpenAI gpt-audio"
    assert metric.evaluation_cost == 0.001

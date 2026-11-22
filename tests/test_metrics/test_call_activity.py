import pytest

from deepeval.metrics.voice._call_activity import (
    agent_starts,
    call_activity,
    recording_activity,
)
from deepeval.test_case import ConversationalTestCase
from deepeval.voice.connectors import audio_utils
from tests.test_metrics.helpers import (
    RATE,
    speech_like,
    stereo_recording,
    timed_call,
)

# speech_like's syllables leave the last quarter second of a clip silent.
TAIL = 0.25


def _starts(path):
    return agent_starts(recording_activity(path))


def test_spans_come_from_each_channels_loudness(tmp_path):
    path = stereo_recording(
        tmp_path / "call.wav",
        caller=[(0.5, speech_like(2.0))],
        agent=[(3.0, speech_like(2.0, base_hz=230.0))],
    )
    activity = recording_activity(path)
    assert activity.source == "call_recording"
    [caller] = activity.caller
    [agent] = activity.agent
    assert caller.start == pytest.approx(0.5, abs=0.03)
    assert caller.end == pytest.approx(2.5 - TAIL, abs=0.03)
    assert agent.start == pytest.approx(3.0, abs=0.03)
    assert agent.end == pytest.approx(5.0 - TAIL, abs=0.03)


def test_a_missing_or_mono_recording_falls_back_to_turn_timestamps(
    tmp_path,
):
    mono = tmp_path / "mono.wav"
    mono.write_bytes(
        audio_utils.pcm16_to_wav_bytes(speech_like(1.0).tobytes(), RATE, 1)
    )
    for path in (str(tmp_path / "missing.wav"), str(mono)):
        assert recording_activity(path) is None
        test_case = ConversationalTestCase(
            turns=timed_call([0.5]), call_recording_path=path
        )
        activity = call_activity(test_case)
        assert activity.source == "turn_timestamps"
        assert [span.turn for span in activity.agent] == [1]


def test_an_agent_start_records_what_the_caller_was_doing(tmp_path):
    path = stereo_recording(
        tmp_path / "call.wav",
        agent=[(0.0, speech_like(1.0)), (4.0, speech_like(2.0))],
        caller=[(1.5, speech_like(2.0))],
    )
    _, reply = _starts(path)
    assert reply.caller_spoke_last is True
    assert reply.caller_talking is False
    assert reply.caller_silent_for == pytest.approx(4.0 - 3.25, abs=0.05)
    assert reply.caller_runs_past == 0.0
    assert reply.caller_starts_after is None


def test_an_agent_start_over_the_caller_records_the_overlap(tmp_path):
    path = stereo_recording(
        tmp_path / "call.wav",
        agent=[(0.0, speech_like(1.0)), (3.0, speech_like(2.0))],
        caller=[(1.5, speech_like(3.0))],
    )
    _, reply = _starts(path)
    assert reply.caller_talking is True
    assert reply.caller_starts_after == 0.0
    assert reply.caller_silent_for == 0.0
    assert reply.caller_pause_seconds == 0.0
    assert reply.overlap_seconds == pytest.approx(4.25 - 3.0, abs=0.05)


def test_the_callers_pause_runs_from_stopping_to_starting_again(tmp_path):
    path = stereo_recording(
        tmp_path / "call.wav",
        agent=[(0.0, speech_like(1.0)), (6.0, speech_like(0.5))],
        caller=[(1.5, speech_like(1.0)), (6.5, speech_like(1.0))],
    )
    _, reply = _starts(path)
    assert reply.caller_silent_for == pytest.approx(6.0 - 2.25, abs=0.05)
    assert reply.caller_pause_seconds == pytest.approx(6.5 - 2.25, abs=0.05)


@pytest.mark.parametrize("pause,starts", [(0.75, 1), (1.5, 2)])
def test_a_pause_under_a_second_does_not_end_the_agents_stretch(
    tmp_path, pause, starts
):
    path = stereo_recording(
        tmp_path / "call.wav",
        agent=[(0.0, speech_like(1.0)), (1.0 - TAIL + pause, speech_like(1.0))],
    )
    assert len(_starts(path)) == starts

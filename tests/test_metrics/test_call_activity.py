import pytest

from deepeval.metrics.voice.utils.call_activity import (
    Span,
    agent_starts,
    call_activity,
    call_end,
    caller_lines,
    interruption_place,
    recording_activity,
    silence_after_line,
    turn_activity,
)
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from deepeval.voice.connectors import audio_utils
from tests.test_metrics.helpers import (
    recording,
    RATE,
    TAIL,
    speech_like,
    stereo_recording,
    timed_call,
    timed_turn,
    to_audio,
    zeros,
)


def _starts(path):
    return agent_starts(
        recording_activity(recording(path), use_speech_detection=False)
    )


def test_spans_come_from_each_channels_loudness(tmp_path):
    path = stereo_recording(
        tmp_path / "call.wav",
        caller=[(0.5, speech_like(2.0))],
        agent=[(3.0, speech_like(2.0, base_hz=230.0))],
    )
    activity = recording_activity(recording(path), use_speech_detection=False)
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
    assert recording_activity(None, use_speech_detection=False) is None
    for audio in (None, recording(mono)):
        assert recording_activity(audio, use_speech_detection=False) is None
        test_case = ConversationalTestCase(turns=timed_call([0.5]), audio=audio)
        activity = call_activity(test_case, use_speech_detection=False)
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


def test_an_interruption_is_placed_where_both_speakers_talked():
    other = [Span(0.0, 2.5), Span(2.8, 3.2), Span(6.0, 7.0)]
    assert interruption_place(other, 2.0, 4.0) == {
        "start_time": 2.0,
        "duration_seconds": 1.2,
    }


def test_an_interruption_with_no_overlap_is_the_interrupters_stretch():
    assert interruption_place([Span(5.0, 6.0)], 2.0, 3.0) == {
        "start_time": 2.0,
        "duration_seconds": 1.0,
    }


def test_the_silence_after_a_line_runs_from_its_speech_to_the_next():
    turns = [
        timed_turn("user", 2.0, 0.0),
        timed_turn("user", 1.0, 5.0, seed=1),
    ]
    silence = silence_after_line(turns, 0, use_speech_detection=False)
    assert silence["start_time"] == pytest.approx(2.0 - TAIL, abs=0.05)
    assert silence["duration_seconds"] == pytest.approx(3.0 + TAIL, abs=0.05)
    assert call_end(turns) == pytest.approx(6.0, abs=0.05)


def test_an_untimed_line_has_no_silence_to_place():
    turns = timed_call([0.5])
    for turn in turns:
        turn.audio.start_time = None
    assert silence_after_line(turns, 0, use_speech_detection=False) == {}
    assert call_end(turns) is None


def _turn_activity(turns):
    return turn_activity(
        ConversationalTestCase(turns=turns), use_speech_detection=False
    )


def test_turn_audio_counts_only_where_the_speaker_is_heard():
    activity = _turn_activity(timed_call([0.5]))
    assert activity.source == "turn_timestamps"
    [caller] = activity.caller
    [agent] = activity.agent
    assert (caller.turn, agent.turn) == (0, 1)
    assert caller.start == pytest.approx(0.0, abs=0.03)
    assert caller.end == pytest.approx(2.0 - TAIL, abs=0.03)
    assert agent.start == pytest.approx(2.5, abs=0.03)
    assert agent.end == pytest.approx(4.5 - TAIL, abs=0.03)


@pytest.mark.parametrize(
    "audio",
    [
        to_audio(zeros(2.0), start_time=2.5),
        Audio.from_bytes(
            b"not-a-wav", "audio/wav", duration=2.0, start_time=2.5
        ),
    ],
    ids=["silent", "undecodable"],
)
def test_turn_audio_with_no_speech_to_find_counts_whole(audio):
    turns = timed_call([0.5])
    turns[1].audio = audio
    [agent] = _turn_activity(turns).agent
    assert (agent.start, agent.end, agent.turn) == (2.5, 4.5, 1)


def test_a_pause_inside_turn_audio_splits_it_into_stretches_of_that_turn():
    reply = speech_like(1.0)
    reply.extend(zeros(1.5))
    reply.extend(speech_like(1.0, seed=1))
    turns = timed_call([0.5])
    turns[1].audio = to_audio(reply, start_time=2.5)
    first, second = _turn_activity(turns).agent
    assert (first.turn, second.turn) == (1, 1)
    assert first.end == pytest.approx(3.5 - TAIL, abs=0.03)
    assert second.start == pytest.approx(5.0, abs=0.03)


@pytest.mark.parametrize(
    "pause,lines", [(0.3, [(0, 1)]), (2.0, [(0, 0), (1, 1)])]
)
def test_back_to_back_caller_turns_are_one_line_only_after_a_short_pause(
    pause, lines
):
    # The caller's pause sits inside their first turn's audio, as when one
    # recording is cut at each turn.
    first = speech_like(1.0)
    first.extend(zeros(pause))
    turns = [
        Turn(role="user", content="Hi", audio=to_audio(first, start_time=0.0)),
        timed_turn("user", 1.0, 1.0 + pause, seed=1),
    ]
    assert caller_lines(turns, use_speech_detection=False) == lines


def test_call_recording_path_still_works_and_warns(tmp_path):
    path = stereo_recording(
        tmp_path / "call.wav", agent=[(0.5, speech_like(1.0))]
    )
    with pytest.warns(DeprecationWarning, match="call_recording_path"):
        test_case = ConversationalTestCase(
            turns=timed_call([0.5]), call_recording_path=path
        )
    assert test_case.audio is not None
    activity = call_activity(test_case, use_speech_detection=False)
    assert activity.source == "call_recording"
    with pytest.warns(DeprecationWarning):
        missing = ConversationalTestCase(
            turns=timed_call([0.5]),
            call_recording_path=str(tmp_path / "missing.wav"),
        )
    assert missing.audio is None


def test_agent_channel_says_which_side_is_the_agent(tmp_path):
    # Left channel speaks at 0.5 s, right at 3.0 s. Here the agent is on the
    # left, the reverse of the usual layout.
    path = stereo_recording(
        tmp_path / "left-agent.wav",
        caller=[(0.5, speech_like(2.0))],
        agent=[(3.0, speech_like(2.0, base_hz=230.0))],
    )
    default = recording_activity(recording(path), False)
    left = recording_activity(recording(path), False, agent_channel="left")
    assert default.agent[0].start == pytest.approx(3.0, abs=0.03)
    assert left.agent[0].start == pytest.approx(0.5, abs=0.03)
    assert left.caller[0].start == pytest.approx(3.0, abs=0.03)

    test_case = ConversationalTestCase(
        turns=timed_call([0.5]), audio=recording(path), agent_channel="left"
    )
    activity = call_activity(test_case, use_speech_detection=False)
    assert activity.agent[0].start == pytest.approx(0.5, abs=0.03)

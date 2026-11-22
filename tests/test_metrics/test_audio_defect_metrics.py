from array import array

import pytest

from deepeval.metrics import (
    AbruptCutoffMetric,
    AudioClippingMetric,
    AudioDropoutMetric,
    AudioLoopMetric,
    DeadStreamMetric,
    EchoMetric,
    HeldToneMetric,
)
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from tests.test_metrics.helpers import (
    RATE,
    clean_turns,
    clipped,
    cut_short,
    speech_like,
    spoken_words,
    stereo_recording,
    timed_call,
    to_audio,
    tone,
    with_noise,
    zeros,
)


def dropout_clip():
    clip = spoken_words()
    clip[int(1.1 * RATE) : int(1.2 * RATE)] = zeros(0.1)
    return clip


def loop_clip():
    syllable = speech_like(0.25, seed=1)
    return syllable + syllable + syllable + syllable


DEFECTS = [
    (AudioDropoutMetric, "audio dropouts", dropout_clip),
    (
        AudioClippingMetric,
        "clipped audio",
        lambda: speech_like(1.0) + clipped(0.5) + speech_like(1.0),
    ),
    (AudioLoopMetric, "looping audio", loop_clip),
    (
        AbruptCutoffMetric,
        "an abrupt cut-off",
        lambda: cut_short(speech_like(3.0), 1.125),
    ),
    (
        HeldToneMetric,
        "a held tone",
        lambda: speech_like(2.0) + tone(1.5, 388.0) + speech_like(1.0),
    ),
    (
        DeadStreamMetric,
        "a dead stream",
        lambda: speech_like(1.0) + zeros(0.7) + speech_like(1.0, seed=5),
    ),
]
METRICS = [metric_class for metric_class, _, _ in DEFECTS]
STAMPS_NOTE = (
    " Judged from each turn's own audio and timestamps, as no stereo call "
    "recording was available."
)


def _reason(metric_class, reason):
    return reason + (STAMPS_NOTE if metric_class is AbruptCutoffMetric else "")


def _measure(metric_class, turns):
    metric = metric_class()
    metric.measure(ConversationalTestCase(turns=turns), _show_indicator=False)
    return metric


def _with_defect(index, build_clip):
    turns = clean_turns()
    turns[index].audio = to_audio(build_clip(), start_time=index * 3.5)
    return turns


@pytest.mark.parametrize("metric_class,label,_", DEFECTS)
def test_a_clean_call_scores_one(metric_class, label, _):
    metric = _measure(metric_class, clean_turns())
    assert metric.score == 1.0
    assert metric.success is True
    assert metric.reason == _reason(
        metric_class, f"No agent reply had {label}."
    )
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["events"] == []


@pytest.mark.parametrize("metric_class,label,build_clip", DEFECTS)
def test_score_is_the_share_of_agent_replies_without_the_defect(
    metric_class, label, build_clip
):
    metric = _measure(metric_class, _with_defect(1, build_clip))
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == _reason(
        metric_class, f"1 of 3 agent replies had {label}."
    )
    assert metric.score_breakdown["turns_with_events"] == 1
    assert {event["turn"] for event in metric.score_breakdown["events"]} == {1}


@pytest.mark.parametrize("metric_class,_,build_clip", DEFECTS)
def test_the_callers_audio_is_ignored(metric_class, _, build_clip):
    metric = _measure(metric_class, _with_defect(0, build_clip))
    assert metric.score == 1.0


@pytest.mark.parametrize("metric_class", METRICS)
def test_a_call_without_agent_audio_is_skipped(metric_class):
    turns = [
        Turn(role="user", content="Hello?", audio=to_audio(speech_like(1.0))),
        Turn(role="assistant", content="Hi, how can I help?"),
    ]
    metric = _measure(metric_class, turns)
    assert metric.score is None
    assert metric.skipped is True


@pytest.mark.parametrize("metric_class", METRICS)
def test_undecodable_agent_audio_is_skipped_not_failed(metric_class):
    turns = clean_turns()
    turns[1].audio = Audio.from_bytes(b"not-a-wav", "audio/wav", start_time=3.5)
    metric = _measure(metric_class, turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]


def _agent_stream_cut_call():
    turns = clean_turns()
    # The agent's reply stops mid-word and the agent speaks again 0.5 s later.
    turns[1].audio = to_audio(
        cut_short(speech_like(3.0, base_hz=230.0), 1.125), start_time=3.5
    )
    turns[2] = Turn(
        role="assistant",
        content="...so your appointment is confirmed.",
        audio=to_audio(
            speech_like(1.5, base_hz=230.0, seed=6), start_time=5.125
        ),
    )
    return turns


def test_a_stream_cut_counts_as_an_abrupt_cut_off_not_a_dead_stream():
    turns = _agent_stream_cut_call()
    cutoff = _measure(AbruptCutoffMetric, turns)
    (event,) = cutoff.score_breakdown["events"]
    assert event["type"] == "stream_cut"
    assert event["turn"] == 1
    assert event["resumed_turn"] == 2
    assert cutoff.score == pytest.approx(3 / 4)
    assert _measure(DeadStreamMetric, turns).score == 1.0


def test_a_reply_ending_mid_sound_is_an_abrupt_cut_off_without_a_timeline():
    turns = _with_defect(1, lambda: cut_short(speech_like(3.0), 1.125))
    for turn in turns:
        turn.audio.start_time = None
    metric = _measure(AbruptCutoffMetric, turns)
    (event,) = metric.score_breakdown["events"]
    assert event == {"type": "abrupt_cutoff", "turn": 1}


def test_a_reply_the_caller_barged_into_is_not_an_abrupt_cut_off():
    turns = _with_defect(1, lambda: cut_short(speech_like(3.0), 1.125))
    turns[1].interrupted = True
    assert _measure(AbruptCutoffMetric, turns).score == 1.0


def test_a_reply_ending_in_steady_background_noise_is_not_a_cut_off():
    turns = _with_defect(1, lambda: with_noise(speech_like(3.0), -30))
    assert _measure(AbruptCutoffMetric, turns).score == 1.0


def test_a_reply_cut_mid_word_over_background_noise_is_a_cut_off():
    turns = _with_defect(
        1, lambda: cut_short(with_noise(speech_like(4.0), -30), 2.125)
    )
    assert _measure(AbruptCutoffMetric, turns).score == pytest.approx(2 / 3)


def test_a_dead_stretch_mid_reply_is_a_dead_stream_not_a_cut_off():
    turns = _with_defect(
        1, lambda: speech_like(1.0) + zeros(0.7) + speech_like(1.0, seed=5)
    )
    assert _measure(DeadStreamMetric, turns).score == pytest.approx(2 / 3)
    assert _measure(AbruptCutoffMetric, turns).score == 1.0


def _scaled(clip, gain):
    return array("h", (int(value * gain) for value in clip))


def _echo_on_recording(tmp_path, *, caller):
    reply = speech_like(3.0)
    path = stereo_recording(
        tmp_path / "call.wav",
        agent=[(0.0, speech_like(0.5, seed=4)), (1.0, reply)],
        caller=caller(reply),
        seconds=7.0,
    )
    metric = EchoMetric()
    metric.measure(
        ConversationalTestCase(
            turns=timed_call([0.5]), call_recording_path=path
        ),
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
    assert event["at_seconds"] == pytest.approx(1.0, abs=0.05)
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


def _recorded_cutoff(tmp_path, *, agent, caller=(), seconds=None):
    metric = AbruptCutoffMetric(use_speech_detection=False)
    path = stereo_recording(
        tmp_path / "call.wav", caller=caller, agent=agent, seconds=seconds
    )
    metric.measure(
        ConversationalTestCase(
            turns=timed_call([0.5]), call_recording_path=path
        ),
        _show_indicator=False,
    )
    return metric


def test_a_reply_cut_mid_sound_on_a_recording_is_a_cutoff(tmp_path):
    metric = _recorded_cutoff(
        tmp_path,
        agent=[
            (0.0, speech_like(2.0)),
            (3.0, cut_short(speech_like(3.0), 1.125)),
        ],
        caller=[(6.0, speech_like(1.0))],
    )
    assert metric.score == 0.5
    assert metric.score_breakdown["timing_source"] == "call_recording"
    [event] = metric.score_breakdown["events"]
    assert event["at_seconds"] == pytest.approx(3.0, abs=0.05)
    assert event["stopped_at_seconds"] == pytest.approx(4.125, abs=0.05)


def test_an_agent_cut_off_by_the_caller_speaking_is_excused(tmp_path):
    metric = _recorded_cutoff(
        tmp_path,
        agent=[
            (0.0, speech_like(2.0)),
            (3.0, cut_short(speech_like(3.0), 1.125)),
        ],
        caller=[(3.6, speech_like(2.0))],
    )
    assert metric.score == 1.0


def test_a_reply_that_fades_out_on_a_recording_is_not_a_cutoff(tmp_path):
    metric = _recorded_cutoff(
        tmp_path,
        agent=[(0.0, speech_like(2.0)), (3.0, speech_like(2.0))],
        caller=[(6.0, speech_like(1.0))],
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2


def test_sound_running_to_the_end_of_the_recording_is_not_judged(tmp_path):
    metric = _recorded_cutoff(
        tmp_path,
        agent=[
            (0.0, speech_like(2.0)),
            (3.0, cut_short(speech_like(3.0), 1.125)),
        ],
        seconds=4.125,
    )
    assert metric.score == 1.0

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
from tests.test_metrics.audio_fixtures import (
    RATE,
    clean_turns,
    clipped,
    cut_short,
    speech_like,
    spoken_words,
    to_audio,
    tone,
    zeros,
)


def dropout_clip():
    clip = spoken_words()
    clip[int(1.1 * RATE) : int(1.2 * RATE)] = zeros(0.1)
    return clip


def echo_clip(delay=0.15, gain=0.4):
    clip = speech_like(3.0)
    shift = int(delay * RATE)
    for index in range(len(clip) - 1, shift - 1, -1):
        clip[index] = max(
            -32768, min(32767, clip[index] + int(gain * clip[index - shift]))
        )
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
    (EchoMetric, "an echo", echo_clip),
]
METRICS = [metric_class for metric_class, _, _ in DEFECTS]


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
    assert metric.reason == f"No agent reply had {label}."
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["events"] == []


@pytest.mark.parametrize("metric_class,label,build_clip", DEFECTS)
def test_score_is_the_share_of_agent_replies_without_the_defect(
    metric_class, label, build_clip
):
    metric = _measure(metric_class, _with_defect(1, build_clip))
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == f"1 of 3 agent replies had {label}."
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


def test_a_dead_stretch_mid_reply_is_a_dead_stream_not_a_cut_off():
    turns = _with_defect(
        1, lambda: speech_like(1.0) + zeros(0.7) + speech_like(1.0, seed=5)
    )
    assert _measure(DeadStreamMetric, turns).score == pytest.approx(2 / 3)
    assert _measure(AbruptCutoffMetric, turns).score == 1.0


def test_an_echo_is_reported_with_its_delay():
    (event,) = _measure(EchoMetric, _with_defect(1, echo_clip)).score_breakdown[
        "events"
    ]
    assert event["type"] == "echo"
    assert event["delay_seconds"] == pytest.approx(0.15, abs=0.005)


def test_a_faint_echo_is_not_flagged():
    turns = _with_defect(1, lambda: echo_clip(gain=0.05))
    assert _measure(EchoMetric, turns).score == 1.0


def test_a_looping_reply_counts_as_a_loop_not_an_echo():
    turns = _with_defect(1, loop_clip)
    assert _measure(AudioLoopMetric, turns).score == pytest.approx(2 / 3)
    assert _measure(EchoMetric, turns).score == 1.0

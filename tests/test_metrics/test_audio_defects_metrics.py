import pytest

from deepeval.metrics import (
    AbruptCutoffMetric,
    AudioClippingMetric,
    AudioDropoutMetric,
    AudioLoopMetric,
    DeadStreamMetric,
    HeldToneMetric,
)
from deepeval.models.base_model import DeepEvalBaseSTT
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from tests.test_metrics.helpers import (
    RATE,
    STAMPS_NOTE,
    clean_turns,
    clipped,
    cut_short,
    loop_clip,
    measure,
    speech_like,
    spoken_words,
    stereo_recording,
    to_audio,
    tone,
    with_defect,
    zeros,
    recording,
)


def dropout_clip():
    clip = spoken_words(word=0.5, rhythm=0.5)
    clip[int(1.35 * RATE) : int(1.55 * RATE)] = zeros(0.2)
    return clip


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


def _reason(metric_class, reason):
    return reason + (STAMPS_NOTE if metric_class is AbruptCutoffMetric else "")


@pytest.mark.parametrize("metric_class,label,_", DEFECTS)
def test_a_clean_call_scores_one(metric_class, label, _):
    metric = measure(metric_class(), clean_turns())
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
    metric = measure(metric_class(), with_defect(1, build_clip))
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == _reason(
        metric_class, f"1 of 3 agent replies had {label}."
    )
    assert metric.score_breakdown["turns_with_events"] == 1
    assert {event["turn"] for event in metric.score_breakdown["events"]} == {1}


@pytest.mark.parametrize("metric_class,_,build_clip", DEFECTS)
def test_the_callers_audio_is_ignored(metric_class, _, build_clip):
    metric = measure(metric_class(), with_defect(0, build_clip))
    assert metric.score == 1.0


@pytest.mark.parametrize("metric_class", METRICS)
def test_a_call_without_agent_audio_is_skipped(metric_class):
    turns = [
        Turn(role="user", content="Hello?", audio=to_audio(speech_like(1.0))),
        Turn(role="assistant", content="Hi, how can I help?"),
    ]
    metric = measure(metric_class(), turns)
    assert metric.score is None
    assert metric.skipped is True


@pytest.mark.parametrize("metric_class", METRICS)
def test_undecodable_agent_audio_is_skipped_not_failed(metric_class):
    turns = clean_turns()
    turns[1].audio = Audio.from_bytes(b"not-a-wav", "audio/wav", start_time=3.5)
    metric = measure(metric_class(), turns)
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
    cutoff = measure(AbruptCutoffMetric(), turns)
    (event,) = cutoff.score_breakdown["events"]
    assert event["type"] == "stream_cut"
    assert event["turn"] == 1
    assert event["resumed_turn"] == 2
    assert cutoff.score == pytest.approx(3 / 4)
    assert measure(DeadStreamMetric(), turns).score == 1.0


def test_a_dead_stretch_mid_reply_is_a_dead_stream_not_a_cut_off():
    turns = with_defect(
        1, lambda: speech_like(1.0) + zeros(0.7) + speech_like(1.0, seed=5)
    )
    assert measure(DeadStreamMetric(), turns).score == pytest.approx(2 / 3)
    assert measure(AbruptCutoffMetric(), turns).score == 1.0


def test_a_dropout_is_described_in_milliseconds():
    metric = measure(AudioDropoutMetric(), with_defect(1, dropout_clip))
    (event,) = metric.score_breakdown["events"]
    assert event["description"] == "Audio dropped out for 180 ms"


class StubSTT(DeepEvalBaseSTT):
    """Returns one scripted transcription per clip."""

    def __init__(self, heard):
        self.scripted = list(heard)
        super().__init__(model="stub-stt")

    def load_model(self):
        return self

    def transcribe(self, audio, **kwargs):
        return self.scripted.pop(0), None

    async def a_transcribe(self, audio, **kwargs):
        return self.transcribe(audio, **kwargs)

    def get_model_name(self):
        return "stub-stt"


def _dead_stretch_call(tmp_path):
    reply = speech_like(1.0) + zeros(0.7) + speech_like(1.0, seed=5)
    turns = with_defect(1, lambda: reply)
    path = stereo_recording(
        tmp_path / "call.wav", agent=[(turns[1].audio.start_time, reply)]
    )
    return ConversationalTestCase(turns=turns, audio=recording(path))


def _dead_streams_heard(tmp_path, before, across):
    metric = DeadStreamMetric(stt_model=StubSTT([before, across]))
    metric.measure(_dead_stretch_call(tmp_path), _show_indicator=False)
    return metric.score_breakdown["events"]


def test_a_pause_after_a_finished_sentence_is_not_a_dead_stream(tmp_path):
    assert (
        _dead_streams_heard(
            tmp_path,
            "Let me check on that for you.",
            "Let me check on that for you. Your order shipped today.",
        )
        == []
    )


@pytest.mark.parametrize(
    "before,across",
    [
        (
            "Tell me about a time you",
            "Tell me about a time you encountered a problem.",
        ),
        (
            "Or would you prefer...",
            "Or would you prefer... a later time?",
        ),
    ],
)
def test_audio_dying_mid_sentence_is_a_dead_stream(tmp_path, before, across):
    [event] = _dead_streams_heard(tmp_path, before, across)
    assert event["type"] == "dead_stream"


def test_a_hole_with_no_words_before_it_stays_a_dead_stream(tmp_path):
    assert len(_dead_streams_heard(tmp_path, "", "")) == 1

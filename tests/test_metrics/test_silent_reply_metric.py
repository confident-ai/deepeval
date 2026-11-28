import pytest

from deepeval.metrics import SilentReplyMetric
from deepeval.models.base_model import DeepEvalBaseSTT
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from tests.test_metrics.helpers import (
    clean_turns,
    measure,
    stereo_recording,
    to_audio,
    without_place,
    zeros,
    recording,
)
from tests.test_metrics.system_one_fakes import FakeAudioDecisionModel


def test_a_reply_with_text_but_no_audio_is_silent():
    turns = clean_turns()
    turns[3].audio = None
    metric = measure(SilentReplyMetric(), turns)
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == (
        "1 of 3 agent replies had a reply the caller could not hear."
    )
    assert [without_place(e) for e in metric.score_breakdown["events"]] == [
        {"type": "no_audio", "turn": 3}
    ]


def test_a_reply_whose_audio_holds_no_speech_is_silent():
    turns = clean_turns()
    turns[3].audio = to_audio(zeros(2.0), start_time=turns[3].audio.start_time)
    [event] = measure(SilentReplyMetric(), turns).score_breakdown["events"]
    assert without_place(event) == {"type": "no_speech", "turn": 3}
    assert event["offset_seconds"] == 0.0


def test_undecodable_agent_audio_is_skipped_not_failed():
    turns = clean_turns()
    turns[3].audio = Audio.from_bytes(b"not-a-wav", "audio/wav")
    metric = measure(SilentReplyMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [3]


def test_an_agent_turn_with_no_text_is_not_a_silent_reply():
    turns = clean_turns()
    turns[3] = Turn(role="assistant", content="")
    metric = measure(SilentReplyMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2


def test_a_silent_agent_turn_with_no_text_is_not_a_silent_reply():
    turns = clean_turns()
    turns[3] = Turn(role="assistant", content="  ", audio=to_audio(zeros(2.0)))
    metric = measure(SilentReplyMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["events"] == []


class StubSTT(DeepEvalBaseSTT):
    """Hears the same scripted text in every clip."""

    def __init__(self, heard):
        self.heard = heard
        super().__init__(model="stub-stt")

    def load_model(self):
        return self

    def transcribe(self, audio, **kwargs):
        return self.heard, None

    async def a_transcribe(self, audio, **kwargs):
        return self.transcribe(audio, **kwargs)

    def get_model_name(self):
        return "stub-stt"


def _silent_reply_events(tmp_path, content, heard):
    turns = clean_turns()
    turns[3] = Turn(
        role="assistant",
        content=content,
        audio=to_audio(zeros(2.0), start_time=turns[3].audio.start_time),
    )
    path = stereo_recording(tmp_path / "call.wav", seconds=20.0)
    metric = SilentReplyMetric(stt_model=StubSTT(heard))
    metric.measure(
        ConversationalTestCase(turns=turns, audio=recording(path)),
        _show_indicator=False,
    )
    return metric.score_breakdown["events"]


REPLY = "your order ships to the store on friday"


def test_a_reply_heard_elsewhere_on_the_call_is_not_silent(tmp_path):
    heard = "okay so " + REPLY + " anything else"
    assert _silent_reply_events(tmp_path, REPLY, heard) == []


def test_common_words_heard_nearby_do_not_count_as_the_reply(tmp_path):
    heard = "on the call you said to the store your friday was"
    [event] = _silent_reply_events(tmp_path, REPLY, heard)
    assert event["turn"] == 3


def test_a_silent_reply_the_decision_model_hears_speech_in_is_dismissed():
    turns = clean_turns()
    turns[3].audio = to_audio(zeros(2.0), start_time=turns[3].audio.start_time)
    model = FakeAudioDecisionModel(probability=0.3)
    metric = measure(
        SilentReplyMetric(eval_mode="hybrid", audio_decision_model=model),
        turns,
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["dismissed_events"] == 1
    assert model.questions == [SilentReplyMetric.event_question]

import pytest

from deepeval.metrics import AgentResponsivenessMetric
from deepeval.models.system_one.schema import NoulAnswer, SystemOneAnswers
from deepeval.models.base_model import DeepEvalBaseSTT
from deepeval.test_case import ConversationalTestCase, Turn
from tests.test_metrics.system_one_fakes import FakeSystemOneModel
from tests.test_metrics.helpers import (
    reprompted_turns,
    AGENT_HUNG_UP,
    caller_turn,
    clean_turns,
    measure,
    timed_turn,
    recording,
    stereo_recording,
    to_audio,
    zeros,
)


def test_hanging_up_on_a_question_counts_once_for_that_exchange():
    turns = clean_turns() + [caller_turn("Can you check my order?")]
    metric = measure(
        AgentResponsivenessMetric(use_speech_detection=False),
        turns,
        AGENT_HUNG_UP,
    )
    assert metric.score == pytest.approx(3 / 4)
    assert metric.reason == (
        "1 of 4 exchanges had an issue: Failed Response (1), "
        "Unexpected Call End (1)."
    )


def test_a_reply_without_audio_fails_the_line_it_answers():
    turns = clean_turns()
    turns[3].audio = None
    metric = measure(
        AgentResponsivenessMetric(use_speech_detection=False), turns
    )
    assert metric.score == pytest.approx(2 / 3)
    (event,) = metric.score_breakdown["events"]
    assert event["metric"] == "Silent Reply"
    assert event["turn"] == 3


def test_a_caller_line_split_into_turns_is_one_exchange():
    turns = [
        timed_turn("assistant", 2.0, 0.0),
        timed_turn("user", 2.0, 3.0),
        timed_turn("user", 1.0, 5.5, seed=1),
    ]
    metric = measure(
        AgentResponsivenessMetric(use_speech_detection=False), turns
    )
    assert metric.score == pytest.approx(1 / 2)
    assert metric.reason == (
        "1 of 2 exchanges had an issue: Failed Response (1)."
    )


def test_a_call_without_audio_is_skipped():
    metric = measure(
        AgentResponsivenessMetric(use_speech_detection=False),
        [
            Turn(role="user", content="Hello?"),
            Turn(role="assistant", content="Hi, how can I help?"),
        ],
    )
    assert metric.score is None
    assert metric.reason == "The call has no audio to evaluate."


def test_reprompt_decides_with_the_roll_up_s_decision_model():
    model = FakeSystemOneModel(
        answer_fn=lambda questions: SystemOneAnswers(
            nouls={key: NoulAnswer(probability=0.9) for key in questions}
        ),
        cost=0.003,
    )
    turns = reprompted_turns()
    metric = measure(
        AgentResponsivenessMetric(
            use_speech_detection=False, system_one_model=model
        ),
        turns,
    )
    assert metric.score == pytest.approx(3 / 4)
    (event,) = metric.score_breakdown["events"]
    assert event["metric"] == "Reprompt"
    assert metric.evaluation_model == "fake-jev"
    assert metric.evaluation_cost == pytest.approx(0.003)


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


def _silent_reply_events(tmp_path, stt_model):
    turns = clean_turns()
    turns[3] = Turn(
        role="assistant",
        content="Your order ships to the store on Friday.",
        audio=to_audio(zeros(2.0), start_time=turns[3].audio.start_time),
    )
    path = stereo_recording(tmp_path / "call.wav", seconds=20.0)
    metric = AgentResponsivenessMetric(
        use_speech_detection=False, stt_model=stt_model
    )
    metric.measure(
        ConversationalTestCase(turns=turns, audio=recording(path)),
        _show_indicator=False,
    )
    return [
        event
        for event in metric.score_breakdown["events"]
        if event["metric"] == "Silent Reply"
    ]


def test_silent_reply_inside_the_roll_up_uses_its_stt_model(tmp_path):
    assert _silent_reply_events(tmp_path, None)
    heard = StubSTT("okay so your order ships to the store on friday")
    assert _silent_reply_events(tmp_path, heard) == []

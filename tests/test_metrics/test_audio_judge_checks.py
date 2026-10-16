from types import SimpleNamespace

import pytest

from deepeval.metrics import ConversationalGEval
from deepeval.metrics.utils import check_conversational_test_case_params
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms import AnthropicModel, OpenAIModel
from deepeval.test_case import (
    Audio,
    ConversationalTestCase,
    MultiTurnParams,
    Turn,
)

PARAMS = [MultiTurnParams.ROLE, MultiTurnParams.CONTENT, MultiTurnParams.AUDIO]


def audio_test_case():
    clip = Audio.from_bytes(b"RIFF-wav-bytes", "audio/wav")
    return ConversationalTestCase(
        turns=[Turn(role="user", content="Hi", audio=clip)]
    )


def stub_metric():
    return SimpleNamespace(__name__="Stub Metric", error=None)


def gpt_audio():
    return OpenAIModel(model="gpt-audio", api_key="k")


def test_rejects_a_judge_without_audio_input_when_sending_audio():
    metric = stub_metric()

    with pytest.raises(ValueError, match="does not accept audio input"):
        check_conversational_test_case_params(
            audio_test_case(),
            PARAMS,
            metric,
            model=AnthropicModel(model="claude-sonnet-4-5", api_key="k"),
            audio=True,
        )
    assert "does not accept audio input" in metric.error


def test_rejects_an_audio_only_judge_when_sending_no_audio():
    with pytest.raises(ValueError, match="only accepts requests that"):
        check_conversational_test_case_params(
            audio_test_case(),
            [MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
            stub_metric(),
            model=gpt_audio(),
        )


def test_accepts_an_audio_only_judge_when_sending_audio():
    check_conversational_test_case_params(
        audio_test_case(), PARAMS, stub_metric(), model=gpt_audio(), audio=True
    )


def test_conversational_g_eval_needs_steps_for_an_audio_only_judge():
    with pytest.raises(
        ValueError, match="Pass `evaluation_steps` instead of `criteria`"
    ):
        ConversationalGEval(
            name="Audio",
            criteria="Is the assistant polite?",
            evaluation_params=list(PARAMS),
            model=gpt_audio(),
        )

    ConversationalGEval(
        name="Audio",
        evaluation_steps=["Check the assistant is polite."],
        evaluation_params=list(PARAMS),
        model=gpt_audio(),
    )


class RecordingModel(DeepEvalBaseLLM):
    """A custom judge that records the audio of each call and gives one reply."""

    def __init__(self, name: str, reply: str):
        self.reply = reply
        self.audio_per_call = []
        super().__init__(name)

    def load_model(self):
        return self

    def generate(self, prompt, schema=None, audio=None):
        self.audio_per_call.append(audio)
        return self.reply

    async def a_generate(self, prompt, schema=None, audio=None):
        return self.generate(prompt, schema, audio)

    def get_model_name(self):
        return self.name


def test_unused_audio_model_is_accepted_and_never_called():
    # Without AUDIO in the params nothing carries audio, so `model` scores.
    text_model = RecordingModel("text", '{"score": 8, "reason": "Polite."}')
    audio_model = RecordingModel("audio", '{"score": 0, "reason": "Unused."}')
    metric = ConversationalGEval(
        name="Audio",
        evaluation_steps=["Check the assistant is polite."],
        evaluation_params=[MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
        model=text_model,
        _audio_model=audio_model,
        async_mode=False,
    )

    metric.measure(audio_test_case())

    assert metric.score == pytest.approx(0.8)
    assert text_model.audio_per_call == [None]
    assert audio_model.audio_per_call == []


def test_audio_model_lets_criteria_work_with_an_audio_only_judge():
    metric = ConversationalGEval(
        name="Audio",
        criteria="Is the assistant polite?",
        evaluation_params=list(PARAMS),
        model=OpenAIModel(model="gpt-4.1", api_key="k"),
        _audio_model=gpt_audio(),
    )

    assert metric.evaluation_model == "gpt-4.1 (audio: gpt-audio)"


@pytest.mark.parametrize("async_mode", [False, True])
def test_steps_use_model_and_scoring_uses_audio_model(async_mode):
    # Each fake only knows its own job, so a call routed to the wrong model
    # would fail to parse.
    text_model = RecordingModel("text", '{"steps": ["Check politeness."]}')
    audio_model = RecordingModel("audio", '{"score": 8, "reason": "Polite."}')
    test_case = audio_test_case()
    metric = ConversationalGEval(
        name="Audio",
        criteria="Is the assistant polite?",
        evaluation_params=list(PARAMS),
        model=text_model,
        _audio_model=audio_model,
        async_mode=async_mode,
    )

    metric.measure(test_case)

    assert metric.score == pytest.approx(0.8)
    assert text_model.audio_per_call == [None]
    assert audio_model.audio_per_call == [[test_case.turns[0].audio]]

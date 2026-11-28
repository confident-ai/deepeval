import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from deepeval.errors import DeepEvalError
from deepeval.models import OpenAIAudioDecisionModel
from deepeval.models.llms.openai_model import OpenAIModel
from deepeval.models.system_one.openai_audio_model import (
    AUDIO_DECISION_SYSTEM_PROMPT,
    AudioVerdict,
)
from deepeval.models.system_one.schema import ChoiceQuestion, NoulQuestion


def _completion(reply):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
        usage=SimpleNamespace(
            prompt_tokens=150,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(audio_tokens=100),
        ),
    )


def _replying(monkeypatch, replies):
    requests = []
    replies = iter(replies)

    def create(**request):
        requests.append(request)
        return _completion(next(replies))

    async def a_create(**request):
        return create(**request)

    def load_model(self, async_mode=False):
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=a_create if async_mode else create
                )
            )
        )

    monkeypatch.setattr(OpenAIModel, "load_model", load_model)
    return requests


def _verdict(answer, confidence):
    return json.dumps({"answer": answer, "confidence": confidence})


def test_sends_the_clip_before_the_question_under_a_system_message(
    monkeypatch,
):
    requests = _replying(monkeypatch, [_verdict("yes", 0.9)])
    model = OpenAIAudioDecisionModel(model="gpt-audio-mini", api_key="k")

    answers, cost = model.decide(
        b"RIFF-clip",
        {
            "event": NoulQuestion(
                instructions="Is a person speaking?",
                true="Someone talks",
                false="Only noise",
            )
        },
        "A phone call.",
    )

    (request,) = requests
    assert request["model"] == "gpt-audio-mini"
    assert request["modalities"] == ["text"]
    system, user = request["messages"]
    assert system == {"role": "system", "content": AUDIO_DECISION_SYSTEM_PROMPT}
    audio, text = user["content"]
    assert audio["type"] == "input_audio"
    assert base64.b64decode(audio["input_audio"]["data"]) == b"RIFF-clip"
    assert audio["input_audio"]["format"] == "wav"
    assert "A phone call." in text["text"]
    assert "Is a person speaking?" in text["text"]
    assert 'Answer "no" when: Only noise' in text["text"]
    assert answers.nouls["event"].probability == 0.9
    # 50 text and 100 audio input tokens, 20 output, at gpt-audio-mini prices.
    assert cost == pytest.approx(50 * 0.6e-6 + 100 * 10e-6 + 20 * 2.4e-6)
    assert model.get_model_name() == "OpenAI gpt-audio-mini"


def test_a_confident_no_is_a_low_probability(monkeypatch):
    requests = _replying(monkeypatch, [_verdict("no", 0.8)])
    answers, _ = asyncio.run(
        OpenAIAudioDecisionModel(api_key="k").a_decide(
            b"clip", {"event": NoulQuestion(instructions="Is it real?")}
        )
    )
    assert answers.nouls["event"].probability == pytest.approx(0.2)
    assert requests[0]["messages"][0]["role"] == "system"


def test_answers_several_questions_and_adds_their_costs(monkeypatch):
    _replying(monkeypatch, [_verdict("yes", 0.6), _verdict("no", 1.0)])
    model = OpenAIAudioDecisionModel(model="gpt-audio-mini", api_key="k")
    answers, cost = model.decide(
        b"clip",
        {
            "a": NoulQuestion(instructions="First?"),
            "b": NoulQuestion(instructions="Second?"),
        },
    )
    assert answers.nouls["a"].probability == 0.6
    assert answers.nouls["b"].probability == 0.0
    one_request = 50 * 0.6e-6 + 100 * 10e-6 + 20 * 2.4e-6
    assert cost == pytest.approx(2 * one_request)


def test_a_refusal_instead_of_json_fails_the_check(monkeypatch):
    _replying(monkeypatch, ["I'm sorry, but I can't listen to audio."])
    with pytest.raises(DeepEvalError):
        OpenAIAudioDecisionModel(api_key="k").decide(
            b"clip", {"event": NoulQuestion(instructions="Is it real?")}
        )


def test_rejects_pick_one_questions(monkeypatch):
    requests = _replying(monkeypatch, [])
    with pytest.raises(DeepEvalError, match="yes/no questions only"):
        OpenAIAudioDecisionModel(api_key="k").decide(
            b"clip",
            {
                "pick": ChoiceQuestion(
                    instructions="Which?", options={"a": "", "b": ""}
                )
            },
        )
    assert requests == []


def test_a_confidence_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError):
        AudioVerdict(answer="yes", confidence=85)

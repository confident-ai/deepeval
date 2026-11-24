import json

import pytest

import deepeval.models.system_one.hume_model as hume_model
from deepeval.errors import DeepEvalError
from deepeval.models import HumeDecisionModel
from deepeval.models.system_one.schema import ChoiceQuestion, NoulQuestion


class FakeResponse:
    def __init__(self, data, status_code=200, headers=None):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data)
        self.headers = headers or {}

    def json(self):
        return self._data


def _fake_post(monkeypatch, *responses):
    calls = []
    queue = list(responses)

    def post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return queue.pop(0)

    monkeypatch.setattr(hume_model.requests, "post", post)
    monkeypatch.setattr(hume_model.time, "sleep", lambda _seconds: None)
    return calls


def _answered(answers, tokens=1_000_000):
    return FakeResponse(
        {"answers": answers, "received": {"input_tokens": tokens}}
    )


def test_sends_the_clip_and_questions_as_multipart(monkeypatch):
    calls = _fake_post(
        monkeypatch, _answered({"event": {"type": "noul", "noul": 0.8}})
    )
    answers, cost = HumeDecisionModel(api_key="hume-key").decide(
        b"wav",
        {
            "event": NoulQuestion(
                instructions="Is someone speaking?", true="Yes"
            )
        },
        context="A phone call.",
    )

    (call,) = calls
    assert call["url"] == "https://api.cloud.hume.ai/v0/decisions/file"
    assert call["headers"] == {"X-Hume-Api-Key": "hume-key"}
    assert call["files"]["file"] == ("audio.wav", b"wav", "audio/wav")
    request = json.loads(call["files"]["request"][1])
    assert request == {
        "questions": {
            "event": {
                "type": "noul",
                "instructions": "Is someone speaking?",
                "criteria": {"true": "Yes"},
            }
        },
        "context": "A phone call.",
    }
    assert answers.nouls["event"].probability == 0.8
    assert cost == pytest.approx(0.47)


def test_maps_shortened_choice_names_back(monkeypatch):
    long = "x" * 60
    calls = _fake_post(monkeypatch, _answered({}))
    HumeDecisionModel(api_key="k").decide(
        b"wav",
        {
            "pick": ChoiceQuestion(
                instructions="Pick.", options={long: "d", "b": None}
            )
        },
    )
    sent = json.loads(calls[0]["files"]["request"][1])["questions"]["pick"]
    name = sent["criteria"][0]["name"]
    assert len(name) == 50

    _fake_post(
        monkeypatch,
        _answered(
            {
                "pick": {
                    "type": "choice",
                    "choice": name,
                    "confidence": 0.7,
                    "probabilities": {name: 0.8, "b": 0.2},
                }
            }
        ),
    )
    answers, _ = HumeDecisionModel(api_key="k").decide(
        b"wav",
        {
            "pick": ChoiceQuestion(
                instructions="Pick.", options={long: "d", "b": None}
            )
        },
    )
    assert answers.choices["pick"].choice == long
    assert answers.choices["pick"].probabilities == {long: 0.8, "b": 0.2}


def test_retries_once_when_busy(monkeypatch):
    calls = _fake_post(
        monkeypatch,
        FakeResponse({}, 503, {"Retry-After": "2"}),
        _answered({"event": {"type": "noul", "noul": 0.4}}),
    )
    answers, _ = HumeDecisionModel(api_key="k").decide(
        b"wav", {"event": NoulQuestion(instructions="Q?")}
    )
    assert len(calls) == 2
    assert answers.nouls["event"].probability == 0.4


def test_a_failed_request_raises(monkeypatch):
    _fake_post(monkeypatch, FakeResponse({"code": "payment_required"}, 402))
    with pytest.raises(DeepEvalError, match="HTTP 402"):
        HumeDecisionModel(api_key="k").decide(
            b"wav", {"event": NoulQuestion(instructions="Q?")}
        )


def test_needs_a_key(monkeypatch):
    monkeypatch.delenv("HUME_API_KEY", raising=False)
    model = HumeDecisionModel()
    model.api_key = None
    with pytest.raises(DeepEvalError):
        model.decide(b"wav", {"event": NoulQuestion(instructions="Q?")})


def test_caps_questions_per_request():
    questions = {str(i): NoulQuestion(instructions="Q?") for i in range(21)}
    with pytest.raises(DeepEvalError, match="at most 20"):
        HumeDecisionModel(api_key="k").decide(b"wav", questions)

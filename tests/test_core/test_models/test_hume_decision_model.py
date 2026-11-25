import pytest

from deepeval.errors import DeepEvalError
from deepeval.models import HumeDecisionModel
from deepeval.models.speech import SpeechHTTPError
from deepeval.models.system_one.schema import ChoiceQuestion, NoulQuestion
from tests.test_core.test_models.speech_stubs import FakeTransport


def _model(*responses) -> HumeDecisionModel:
    model = HumeDecisionModel(api_key="hume-key")
    model.model = FakeTransport(json_sequence=list(responses))
    return model


def _answered(answers, tokens=1_000_000):
    return {"answers": answers, "received": {"input_tokens": tokens}}


def test_sends_the_clip_and_questions_as_multipart():
    model = _model(_answered({"event": {"type": "noul", "noul": 0.8}}))
    answers, cost = model.decide(
        b"wav",
        {
            "event": NoulQuestion(
                instructions="Is someone speaking?", true="Yes"
            )
        },
        context="A phone call.",
    )

    call = model.model.last
    assert (call["method"], call["path"]) == ("POST", "/decisions/file")
    multipart = call["multipart"]
    assert (multipart.file_field, multipart.content) == ("file", b"wav")
    assert multipart.content_type == "audio/wav"
    assert multipart.fields["request"] == {
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


@pytest.mark.asyncio
async def test_a_decide_posts_through_the_async_transport():
    model = _model(_answered({"event": {"type": "noul", "noul": 0.3}}))
    answers, _ = await model.a_decide(
        b"wav", {"event": NoulQuestion(instructions="Q?")}
    )
    assert answers.nouls["event"].probability == 0.3


def test_maps_shortened_choice_names_back():
    long = "x" * 60
    question = ChoiceQuestion(
        instructions="Pick.", options={long: "d", "b": None}
    )
    model = _model(_answered({}))
    model.decide(b"wav", {"pick": question})
    sent = model.model.last["multipart"].fields["request"]["questions"]["pick"]
    name = sent["criteria"][0]["name"]
    assert len(name) == 50

    model = _model(
        _answered(
            {
                "pick": {
                    "type": "choice",
                    "choice": name,
                    "confidence": 0.7,
                    "probabilities": {name: 0.8, "b": 0.2},
                }
            }
        )
    )
    answers, _ = model.decide(b"wav", {"pick": question})
    assert answers.choices["pick"].choice == long
    assert answers.choices["pick"].probabilities == {long: 0.8, "b": 0.2}
    assert answers.choices["pick"].confidence == 0.7


def test_skips_answers_outside_the_probability_range():
    model = _model(_answered({"event": {"type": "noul", "noul": 1.4}}))
    answers, _ = model.decide(
        b"wav", {"event": NoulQuestion(instructions="Q?")}
    )
    assert answers.nouls == {}


def test_a_failed_request_raises():
    model = HumeDecisionModel(api_key="k")

    class Failing(FakeTransport):
        def request_json(self, method, path, **kwargs):
            raise SpeechHTTPError("Hume request failed with HTTP 402")

    model.model = Failing()
    with pytest.raises(DeepEvalError, match="HTTP 402"):
        model.decide(b"wav", {"event": NoulQuestion(instructions="Q?")})


def test_needs_a_key(monkeypatch):
    monkeypatch.delenv("HUME_API_KEY", raising=False)
    monkeypatch.setattr(
        "deepeval.models.system_one.hume_model.get_settings",
        lambda: type("S", (), {"HUME_API_KEY": None})(),
    )
    with pytest.raises(DeepEvalError):
        HumeDecisionModel()


def test_caps_questions_per_request():
    questions = {str(i): NoulQuestion(instructions="Q?") for i in range(21)}
    with pytest.raises(DeepEvalError, match="at most 20"):
        _model().decide(b"wav", questions)


def test_caps_choice_options():
    options = {str(i): None for i in range(21)}
    with pytest.raises(DeepEvalError, match="2 to 20"):
        _model().decide(
            b"wav", {"pick": ChoiceQuestion(instructions="Q?", options=options)}
        )

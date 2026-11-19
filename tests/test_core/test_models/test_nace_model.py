import json

import httpx
import pytest

from deepeval._version import __version__
from deepeval.config.settings import reset_settings
from deepeval.errors import DeepEvalError
from deepeval.metrics.utils import initialize_system_one_model
from deepeval.models import NaceModel, TypeSafeModel
from deepeval.models.system_one import NoulQuestion

QUESTIONS = {"q": NoulQuestion(instructions="Is the output relevant?")}


def _drex(requests, answers):
    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "answers": answers,
                "usage": {"input_tokens": 1000, "output_tokens": 10},
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_decides_through_nace_with_attribution():
    requests = []
    answers = [{"type": "predicate", "name": "q", "probability": 0.9}]
    model = NaceModel(api_key="nace-key", http_client=_drex(requests, answers))

    result, cost = model.decide("x", QUESTIONS)

    request = requests[0]
    assert str(request.url) == "https://console.nace.ai/v1/decisions"
    assert request.headers["authorization"] == "Bearer nace-key"
    assert request.headers["x-nace-client"] == f"deepeval/{__version__}"
    assert request.headers["user-agent"] == f"deepeval/{__version__}"
    assert json.loads(request.content)["model"] == "drex-max-v1.1"
    assert result.nouls["q"].probability == 0.9
    assert cost == pytest.approx(1000 * 0.03 / 1e6)
    assert model.get_model_name() == "drex-max-v1.1 (Nace)"


def test_errors_name_the_drex_model():
    refused = [{"type": "refusal", "name": "q"}]
    model = NaceModel(api_key="nace-key", http_client=_drex([], refused))

    with pytest.raises(DeepEvalError, match=r"drex-max-v1.1 \(Nace\) refused"):
        model.decide("x", QUESTIONS)


def test_never_sends_the_openai_key(monkeypatch):
    monkeypatch.delenv("NACE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    reset_settings(reload_dotenv=False)
    with pytest.raises(DeepEvalError, match="NACE_API_KEY"):
        NaceModel()

    monkeypatch.setenv("NACE_API_KEY", "nace-key")
    reset_settings(reload_dotenv=False)
    assert NaceModel().load_model().api_key == "nace-key"


def test_decision_provider_selects_nace(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_DECISION_PROVIDER", "nace")
    monkeypatch.setenv("NACE_API_KEY", "nace-key")
    monkeypatch.setenv("NACE_MODEL_NAME", "drex-max-finance")
    reset_settings(reload_dotenv=False)

    model = initialize_system_one_model(None, "system_one")

    assert isinstance(model, NaceModel)
    assert model.name == "drex-max-finance"


def test_a_drex_model_name_picks_nace_whatever_the_provider(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "nace-key")
    reset_settings(reload_dotenv=False)

    model = initialize_system_one_model("drex-v2.0", "hybrid")

    assert isinstance(model, NaceModel)
    assert model.name == "drex-v2.0"


def test_decision_provider_defaults_to_typesafe(monkeypatch):
    pytest.importorskip("typesafe_sdk")
    monkeypatch.delenv("DEEPEVAL_DECISION_PROVIDER", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_test")
    reset_settings(reload_dotenv=False)

    assert isinstance(
        initialize_system_one_model(None, "system_one"), TypeSafeModel
    )


def test_unusable_nace_says_how_to_configure_it(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_DECISION_PROVIDER", "nace")
    monkeypatch.delenv("NACE_API_KEY", raising=False)
    reset_settings(reload_dotenv=False)

    with pytest.raises(DeepEvalError, match="deepeval set-nace"):
        initialize_system_one_model(None, "hybrid")

import pytest

from deepeval.errors import DeepEvalError
from deepeval.models.llms.thegrid_model import TheGridModel


def _seed_settings(settings):
    with settings.edit(persist=False):
        settings.THEGRID_MODEL_NAME = "text-standard"
        settings.THEGRID_API_KEY = "thegrid-secret"


#####################################
# __init__ / configuration behavior #
#####################################


def test_thegrid_model_prefers_explicit_params_over_settings(settings):
    _seed_settings(settings)

    model = TheGridModel(
        model="agent-max",
        api_key="explicit-secret",
        base_url="https://explicit.example.com/",
    )

    assert model.name == "agent-max"
    assert model.base_url == "https://explicit.example.com"  # slash stripped
    assert model.api_key.get_secret_value() == "explicit-secret"


def test_thegrid_model_uses_settings_when_params_missing(settings):
    _seed_settings(settings)

    model = TheGridModel()

    assert model.name == "text-standard"
    assert model.api_key.get_secret_value() == "thegrid-secret"


def test_thegrid_model_defaults_base_url(settings):
    _seed_settings(settings)

    model = TheGridModel()

    assert model.base_url == "https://api.thegrid.ai/v1"


def test_thegrid_model_raises_if_model_missing(settings):
    _seed_settings(settings)
    with settings.edit(persist=False):
        settings.THEGRID_MODEL_NAME = None

    with pytest.raises(DeepEvalError) as exc:
        TheGridModel(model=None)

    msg = str(exc.value)
    assert "The Grid is missing a required parameter" in msg
    assert "THEGRID_MODEL_NAME" in msg
    assert "model" in msg


def test_thegrid_model_rejects_negative_temperature(settings):
    _seed_settings(settings)

    with pytest.raises(DeepEvalError):
        TheGridModel(model="text-standard", temperature=-1)


def test_thegrid_model_cost_is_unknown_without_user_pricing(settings):
    """The Grid publishes no per-token price, so cost stays unresolved."""
    _seed_settings(settings)

    model = TheGridModel()

    assert model.cost_per_input_token is None
    assert model.cost_per_output_token is None

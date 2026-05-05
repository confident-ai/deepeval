from pydantic import SecretStr

from deepeval.confident import api as confident_api
from deepeval.confident.api import is_confident, get_confident_api_key


def test_confident_boundary_off_in_core():
    assert get_confident_api_key() is None
    assert is_confident() is False


def test_confident_api_key_takes_precedence(monkeypatch):
    class DummySettings:
        CONFIDENT_API_KEY = SecretStr("legacy-unprefixed-confident-key")

    monkeypatch.setattr(confident_api, "get_settings", lambda: DummySettings())

    assert get_confident_api_key() == "legacy-unprefixed-confident-key"
    assert is_confident() is True


def test_confident_api_key_field_is_required():
    from deepeval.config.settings import Settings

    assert "API_KEY" not in Settings.model_fields

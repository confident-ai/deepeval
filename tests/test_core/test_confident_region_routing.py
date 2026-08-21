import pytest
from pydantic import ValidationError

from deepeval.confident import api as confident_api
from deepeval.config.settings import get_settings, reset_settings


def _set_env_region(monkeypatch, region, api_key=None):
    """Load CONFIDENT_REGION (and optionally a key) through real Settings."""
    monkeypatch.setenv("CONFIDENT_REGION", region)
    if api_key is not None:
        monkeypatch.setenv("CONFIDENT_API_KEY", api_key)
    reset_settings(reload_dotenv=False)


def _stub_keystore(monkeypatch, region):
    monkeypatch.setattr(
        confident_api.KEY_FILE_HANDLER,
        "fetch_data",
        lambda *args, **kwargs: region,
    )


def test_explicit_region_wins_over_conflicting_key_prefix(monkeypatch):
    """CONFIDENT_REGION=EU must beat a US-prefixed API key."""
    _set_env_region(monkeypatch, "EU", api_key="confident_us_org_xxx")
    _stub_keystore(monkeypatch, None)

    assert confident_api.resolve_backend() == confident_api.BackendResolution(
        confident_api.API_BASE_URL_EU, "EU", "explicit_region"
    )


def test_explicit_us_region_wins_over_eu_key_prefix(monkeypatch):
    """Precedence holds in both directions, not just towards EU."""
    _set_env_region(monkeypatch, "US", api_key="confident_eu_org_xxx")
    _stub_keystore(monkeypatch, None)

    assert confident_api.resolve_backend() == confident_api.BackendResolution(
        confident_api.API_BASE_URL, "US", "explicit_region"
    )


def test_explicit_region_is_case_insensitive(monkeypatch):
    _set_env_region(monkeypatch, "eu")
    _stub_keystore(monkeypatch, None)

    assert get_settings().CONFIDENT_REGION == "EU"
    assert confident_api.resolve_backend().base_url == (
        confident_api.API_BASE_URL_EU
    )


def test_unsupported_region_raises_at_settings_load(monkeypatch):
    """A typo must fail loudly rather than silently routing to US."""
    monkeypatch.setenv("CONFIDENT_REGION", "EUROPE")

    with pytest.raises(
        ValidationError, match="CONFIDENT_REGION must be one of"
    ):
        reset_settings(reload_dotenv=False)


def test_explicit_region_wins_over_keystore_region(monkeypatch):
    """Settings (env/.env) outrank the legacy JSON keystore."""
    _set_env_region(monkeypatch, "EU")
    _stub_keystore(monkeypatch, "US")

    assert confident_api.resolve_backend() == confident_api.BackendResolution(
        confident_api.API_BASE_URL_EU, "EU", "explicit_region"
    )


def test_keystore_region_used_when_settings_region_unset(monkeypatch):
    """`deepeval set-confident-region` keystore setups must not regress."""
    monkeypatch.delenv("CONFIDENT_REGION", raising=False)
    reset_settings(reload_dotenv=False)
    _stub_keystore(monkeypatch, "EU")

    assert confident_api.resolve_backend() == confident_api.BackendResolution(
        confident_api.API_BASE_URL_EU, "EU", "keystore_region"
    )


def test_unroutable_keystore_region_falls_through_to_key_prefix(monkeypatch):
    """A stale keystore region we no longer serve must not resolve to US."""
    monkeypatch.delenv("CONFIDENT_REGION", raising=False)
    monkeypatch.setenv("CONFIDENT_API_KEY", "confident_eu_org_xxx")
    reset_settings(reload_dotenv=False)
    _stub_keystore(monkeypatch, "AU")

    assert confident_api.resolve_backend() == confident_api.BackendResolution(
        confident_api.API_BASE_URL_EU, "EU", "api_key_prefix"
    )


def test_custom_base_url_outranks_explicit_region(monkeypatch):
    _set_env_region(monkeypatch, "EU")
    monkeypatch.setenv("CONFIDENT_BASE_URL", "https://self-hosted.example.com/")
    reset_settings(reload_dotenv=False)
    _stub_keystore(monkeypatch, None)

    assert confident_api.resolve_backend() == confident_api.BackendResolution(
        "https://self-hosted.example.com", None, "custom_base_url"
    )

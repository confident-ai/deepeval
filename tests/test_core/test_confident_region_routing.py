from pydantic import SecretStr

from deepeval.confident import api as confident_api


def test_resolve_backend_respects_explicit_env_region_over_key_prefix(
    monkeypatch,
):
    """
    CONFIDENT_REGION=EU set via settings (env/.env) must win over a
    conflicting US-prefixed API key.
    """

    class DummySettings:
        CONFIDENT_BASE_URL = None
        CONFIDENT_REGION = "EU"
        CONFIDENT_API_KEY = SecretStr("confident_us_xxx")

    monkeypatch.setattr(confident_api, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        confident_api.KEY_FILE_HANDLER,
        "fetch_data",
        lambda *args, **kwargs: None,
    )

    assert confident_api.resolve_backend() == confident_api.BackendResolution(
        confident_api.API_BASE_URL_EU, "EU", "explicit_region"
    )


def test_resolve_backend_respects_explicit_env_region_without_key_prefix(
    monkeypatch,
):
    """
    CONFIDENT_REGION=AU set via settings must be honored even when the API
    key carries no recognizable region prefix.
    """

    class DummySettings:
        CONFIDENT_BASE_URL = None
        CONFIDENT_REGION = "AU"
        CONFIDENT_API_KEY = SecretStr("abc123")

    monkeypatch.setattr(confident_api, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        confident_api.KEY_FILE_HANDLER,
        "fetch_data",
        lambda *args, **kwargs: None,
    )

    assert confident_api.resolve_backend() == confident_api.BackendResolution(
        confident_api.API_BASE_URL_AU, "AU", "explicit_region"
    )


def test_resolve_backend_falls_back_to_keystore_region(monkeypatch):
    """
    Legacy keystore region is still honored when settings carries no
    CONFIDENT_REGION attribute (existing behavior must not regress).
    """

    class DummySettings:
        CONFIDENT_BASE_URL = None
        CONFIDENT_API_KEY = SecretStr("confident_eu_6M_dummy")

    monkeypatch.setattr(confident_api, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        confident_api.KEY_FILE_HANDLER,
        "fetch_data",
        lambda *args, **kwargs: "EU",
    )

    assert confident_api.resolve_backend() == confident_api.BackendResolution(
        confident_api.API_BASE_URL_EU, "EU", "explicit_region"
    )


def test_resolve_backend_keystore_fallback_when_settings_region_none(
    monkeypatch,
):
    """
    When CONFIDENT_REGION is explicitly None in settings, the keystore
    region is used as a fallback.
    """

    class DummySettings:
        CONFIDENT_BASE_URL = None
        CONFIDENT_REGION = None
        CONFIDENT_API_KEY = SecretStr("confident_eu_6M_dummy")

    monkeypatch.setattr(confident_api, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        confident_api.KEY_FILE_HANDLER,
        "fetch_data",
        lambda *args, **kwargs: "EU",
    )

    assert confident_api.resolve_backend() == confident_api.BackendResolution(
        confident_api.API_BASE_URL_EU, "EU", "explicit_region"
    )

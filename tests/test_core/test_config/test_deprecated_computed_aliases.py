import pytest

from deepeval.config.settings import (
    get_settings,
    reset_settings,
    _DEPRECATED_TO_OVERRIDE,
)


# helper to clear just the keys we touch
def _clear_deprecated_and_overrides(monkeypatch):
    for old_key, override_key in _DEPRECATED_TO_OVERRIDE.items():
        monkeypatch.delenv(old_key, raising=False)
        monkeypatch.delenv(override_key, raising=False)


@pytest.mark.parametrize(
    "old_key,override_key,raw",
    [
        (
            "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS",
            "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE",
            "42",
        ),
        (
            "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS",
            "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
            "5",
        ),
        (
            "DEEPEVAL_TASK_GATHER_BUFFER_SECONDS",
            "DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE",
            "12",
        ),
    ],
)
def test_deprecated_env_applies_to_override_when_override_missing(
    monkeypatch, caplog, old_key, override_key, raw
):
    _clear_deprecated_and_overrides(monkeypatch)
    # only deprecated key set
    monkeypatch.setenv(old_key, raw)

    # rebuild settings from env
    reset_settings(reload_dotenv=False)
    setting = get_settings()

    # Override should be set and coerced to float
    val = getattr(setting, override_key)
    assert isinstance(val, float)
    assert val == float(raw)

    # assert that we logged a warning
    msgs = [
        rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"
    ]
    assert any(
        old_key in m and override_key in m and "deprecated" in m.lower()
        for m in msgs
    )


def test_deprecated_env_ignored_when_override_already_set(monkeypatch, caplog):
    _clear_deprecated_and_overrides(monkeypatch)

    # both present, so override must win
    monkeypatch.setenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS", "999")
    monkeypatch.setenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "7")

    reset_settings(reload_dotenv=False)
    s = get_settings()

    assert s.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE == 7.0  # override wins

    msgs = [
        rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"
    ]
    assert any(
        "deprecated" in m.lower()
        and "ignored because" in m.lower()
        and "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS" in m
        and "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE" in m
        for m in msgs
    )


@pytest.mark.parametrize(
    "old_key,override_key",
    list(_DEPRECATED_TO_OVERRIDE.items()),
)
def test_deprecated_empty_string_is_ignored(monkeypatch, old_key, override_key):
    _clear_deprecated_and_overrides(monkeypatch)

    # empty string should be treated as unset
    monkeypatch.setenv(old_key, "")

    reset_settings(reload_dotenv=False)
    setting = get_settings()

    assert getattr(setting, override_key) is None


def test_deprecated_invalid_value_warns_and_skips(monkeypatch, caplog):
    _clear_deprecated_and_overrides(monkeypatch)

    # feed invalid text
    monkeypatch.setenv("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS", "bogus")

    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # not applied due to coercion failure
    assert settings.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE is None

    msgs = [
        rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"
    ]
    assert any(
        "could not be applied" in m.lower()
        and "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS" in m
        and "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE" in m
        for m in msgs
    )

import os
import pytest

from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _reset_key_file_handler():
    """
    Reset KEY_FILE_HANDLER singleton state before and after each CLI test.
    This prevents in-memory cache pollution between tests.
    """
    from deepeval.key_handler import KEY_FILE_HANDLER

    # Clear in-memory cache before test
    KEY_FILE_HANDLER.data = {}
    yield
    # Clear again after test
    KEY_FILE_HANDLER.data = {}


@pytest.fixture(autouse=True)
def _protect_critical_env_vars():
    """
    Backup and restore critical environment variables that CLI tests might overwrite.
    This prevents CLI tests from affecting other test modules.
    """
    critical_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "GOOGLE_API_KEY",
        "CONFIDENT_API_KEY",
        "CONFIDENTAI_API_KEY",
        "GROK_API_KEY",
        "MOONSHOT_API_KEY",
        "DEEPSEEK_API_KEY",
        "LITELLM_API_KEY",
        "LOCAL_MODEL_API_KEY",
        "LOCAL_EMBEDDING_API_KEY",
    ]

    # Backup original values
    backup = {}
    for var in critical_vars:
        backup[var] = os.environ.get(var)

    yield

    # Restore original values after test
    for var, value in backup.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


# @pytest.fixture(autouse=True)
# def _fresh_settings_env(monkeypatch):
#     # Settings is a singleton, so we need to do some cleanup between tests
#     # Reset the singleton so each test gets a fresh Settings instance
#     reset_settings_env(monkeypatch)
#     yield
#     teardown_settings_singleton()

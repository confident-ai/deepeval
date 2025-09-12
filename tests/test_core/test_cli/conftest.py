import pytest

from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# @pytest.fixture(autouse=True)
# def _fresh_settings_env(monkeypatch):
#     # Settings is a singleton, so we need to do some cleanup between tests
#     # Reset the singleton so each test gets a fresh Settings instance
#     reset_settings_env(monkeypatch)
#     yield
#     teardown_settings_singleton()

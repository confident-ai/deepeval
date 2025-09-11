import pytest
from tests.test_core.helpers import (
    reset_settings_env,
    teardown_settings_singleton,
)


@pytest.fixture(autouse=True)
def _fresh_settings_env(monkeypatch):
    reset_settings_env(monkeypatch)
    yield
    teardown_settings_singleton()

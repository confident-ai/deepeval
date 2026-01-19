import pytest


@pytest.fixture(autouse=True)
def _offline_deterministic_env(monkeypatch: pytest.MonkeyPatch):
    # Prevent dotenv loading (could pull real API keys/configs) and avoid browser open.
    monkeypatch.setenv("DEEPEVAL_DISABLE_DOTENV", "1")
    monkeypatch.setenv("CONFIDENT_OPEN_BROWSER", "0")
    # Keep stable even if unset.
    monkeypatch.setenv("DEEPEVAL_RESULTS_FOLDER", "")

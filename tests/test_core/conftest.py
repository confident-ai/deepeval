try:
    import sys, pysqlite3 as sqlite3  # type: ignore

    sys.modules["sqlite3"] = sqlite3
    sys.modules["sqlite3.dbapi2"] = sqlite3.dbapi2
except Exception:
    pass

import pytest
import test

from pathlib import Path
from typer.testing import CliRunner


# Silence telemetry for all tests so we don't have to deal with the noise
@pytest.fixture(autouse=True)
def _telemetry_opt_out(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    yield


# Run every test in its own temp CWD so .deepeval/.deepeval is sandboxed
@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield


# Default dotenv path most tests can reuse; override in tests as needed
@pytest.fixture
def env_path(tmp_path: Path) -> Path:
    return tmp_path / ".env.local"


@pytest.fixture
def hidden_store_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".deepeval"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(autouse=True)
def _fresh_settings_env(monkeypatch):
    # Settings is a singleton, so we need to do some cleanup between tests
    # Reset the singleton so each test gets a fresh Settings instance
    import deepeval.config.settings as settings_mod

    settings_mod._settings_singleton = None

    # drop any env vars that map to Settings fields to avoid cross test contamination
    from deepeval.config.settings import Settings

    for k in Settings.model_fields.keys():
        monkeypatch.delenv(k, raising=False)

    # also ensure no implicit default save path is carried over
    monkeypatch.delenv("DEEPEVAL_DEFAULT_SAVE", raising=False)

    yield

    # clean after the test too
    settings_mod._settings_singleton = None

try:
    import sys
    import pysqlite3 as sqlite3  # type: ignore

    sys.modules["sqlite3"] = sqlite3
    sys.modules["sqlite3.dbapi2"] = sqlite3.dbapi2
except Exception:
    pass

import pytest
import tenacity

from pathlib import Path


@pytest.fixture(autouse=True)
def _ensure_hidden_store_dir(tmp_path: Path):
    d = tmp_path / ".deepeval"
    d.mkdir(exist_ok=True)
    # some code expects the file to be there after a run,
    # but at minimum the directory must exist to avoid FileNotFoundError
    yield


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
def no_sleep(monkeypatch):
    monkeypatch.setattr(tenacity.nap, "sleep", lambda _: None, raising=True)

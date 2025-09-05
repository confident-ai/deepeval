from pathlib import Path
import pytest
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


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# Default dotenv path most tests can reuse; override in tests as needed
@pytest.fixture
def env_path(tmp_path: Path) -> Path:
    return tmp_path / ".env.local"


@pytest.fixture
def hidden_store_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".deepeval"
    d.mkdir(parents=True, exist_ok=True)
    return d

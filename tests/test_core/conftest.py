try:
    import sys
    import pysqlite3 as sqlite3  # type: ignore

    sys.modules["sqlite3"] = sqlite3
    sys.modules["sqlite3.dbapi2"] = sqlite3.dbapi2
except Exception:
    pass

import os
import pytest
import tenacity

from typing import TYPE_CHECKING
from pathlib import Path

from deepeval.tracing.tracing import trace_manager
from deepeval.config.settings import get_settings, reset_settings, Settings


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


PRESERVE_FROM_BASELINE = {
    name
    for name in Settings.model_fields.keys()
    if name.endswith("_API_KEY") and not name.startswith("CONFIDENT")
}


@pytest.fixture(autouse=True)
def _ensure_hidden_store_dir(tmp_path: Path):
    d = tmp_path / ".deepeval"
    d.mkdir(exist_ok=True)
    # some code expects the file to be there after a run,
    # but at minimum the directory must exist to avoid FileNotFoundError
    yield


@pytest.fixture
def hidden_store_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".deepeval"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
def env_path(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("ENV_DIR_PATH", str(tmp_path))
    return tmp_path / ".env.local"


@pytest.fixture
def env_dir(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("ENV_DIR_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(tenacity.nap, "sleep", lambda _: None, raising=True)


@pytest.fixture()
def settings():
    settings = get_settings()
    yield settings


@pytest.fixture(scope="session")
def _session_env_baseline():
    # capture the environment as it existed when pytest started
    return os.environ.copy()


def _restore_env_to(baseline: dict[str, str]) -> None:
    # remove any keys not in the baseline
    for k in list(os.environ.keys()):
        if k not in baseline:
            os.environ.pop(k, None)
    # update differing values to match the baseline
    for k, v in baseline.items():
        if os.environ.get(k) != v:
            os.environ[k] = v


@pytest.fixture(autouse=True)
def _env_sandbox(_session_env_baseline, request, monkeypatch):
    # Start from the session baseline (CI secrets included)
    _restore_env_to(_session_env_baseline)

    # Save whitelisted secrets from the session baseline
    preserved = {
        k: v
        for k, v in _session_env_baseline.items()
        if k in PRESERVE_FROM_BASELINE and isinstance(v, str) and v.strip()
    }

    # Clear ALL Settings keys to avoid leaking config (file system mode, default save, etc.)
    for setting_key in list(Settings.model_fields.keys()):
        monkeypatch.delenv(setting_key, raising=False)

    # Re-inject only the secrets we explicitly want to preserve
    for k, v in preserved.items():
        monkeypatch.setenv(k, v)

    # Disable dotenv by default unless the test opts in via @pytest.mark.enable_dotenv
    if not request.node.get_closest_marker("enable_dotenv"):
        monkeypatch.setenv("DEEPEVAL_DISABLE_DOTENV", "1")
    else:
        monkeypatch.delenv("DEEPEVAL_DISABLE_DOTENV", raising=False)

    # Fresh Settings for this test
    reset_settings(reload_dotenv=False)

    yield

    # Restore to the session baseline after the test
    _restore_env_to(_session_env_baseline)
    reset_settings(reload_dotenv=False)


@pytest.fixture(autouse=True)
def _core_mode_no_confident(monkeypatch, request: "FixtureRequest"):

    # Ensure no Confident keys come from the process env in this test
    for key in ("CONFIDENT_API_KEY", "CONFIDENTAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    # Prevent dotenv from re-injecting keys from files during the test
    # core tests shouldn’t depend on local .env anyway
    if not request.node.get_closest_marker("enable_dotenv"):
        monkeypatch.setenv("DEEPEVAL_DISABLE_DOTENV", "1")

    # Rebuild the Settings singleton from the now-clean process env
    reset_settings(reload_dotenv=False)

    # Clear the in-memory Settings fields (no persistence)
    s = get_settings()
    with s.edit(persist=False) as ctx:
        ctx.s.API_KEY = None
        ctx.s.CONFIDENT_API_KEY = None

    # Yield control to the test
    yield


@pytest.fixture()
def enable_dotenv(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_DISABLE_DOTENV", "0")
    # rebuild Settings after changing the env
    reset_settings(reload_dotenv=False)


@pytest.fixture(autouse=False)
def unpatch_openai_after():
    from deepeval.openai.patch import unpatch_openai_classes

    yield
    unpatch_openai_classes()


@pytest.fixture(autouse=True)
def _reset_tracing_state():
    trace_manager.clear_traces()
    trace_manager.traces_to_evaluate_order.clear()
    trace_manager.traces_to_evaluate.clear()
    trace_manager.integration_traces_to_evaluate.clear()
    trace_manager.trace_uuid_to_golden.clear()
    try:
        trace_manager.task_bindings.clear()
    except Exception:
        pass
    trace_manager.evaluating = False
    trace_manager.evaluation_loop = False
    yield

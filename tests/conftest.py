try:
    import sys
    import pysqlite3 as sqlite3  # type: ignore

    sys.modules["sqlite3"] = sqlite3
    sys.modules["sqlite3.dbapi2"] = sqlite3.dbapi2
except Exception:
    pass

import pytest

from typing import TYPE_CHECKING
from pathlib import Path

from deepeval.tracing.tracing import trace_manager
from deepeval.config.settings import get_settings, reset_settings


if TYPE_CHECKING:
    pass


# Silence telemetry for all tests so we don't have to deal with the noise
@pytest.fixture(autouse=True)
def _telemetry_opt_out(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    yield


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


@pytest.fixture()
def settings():
    settings = get_settings()
    yield settings


@pytest.fixture()
def enable_dotenv(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_DISABLE_DOTENV", "0")
    # rebuild Settings after changing the env
    reset_settings(reload_dotenv=False)


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

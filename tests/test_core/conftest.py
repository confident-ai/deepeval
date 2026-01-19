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

from deepeval.config.settings import get_settings, reset_settings, Settings


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


PRESERVE_FROM_BASELINE = {
    name
    for name in Settings.model_fields.keys()
    if name.endswith("_API_KEY") and not name.startswith("CONFIDENT")
}


# Default dotenv path most tests can reuse; override in tests as needed
@pytest.fixture
def env_path(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("ENV_DIR_PATH", str(tmp_path))
    return tmp_path / ".env.local"


@pytest.fixture
def env_dir(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("ENV_DIR_PATH", str(tmp_path))
    return tmp_path


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

    # Never open the Confident AI browser UI during tests
    monkeypatch.setenv("CONFIDENT_OPEN_BROWSER", "0")

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
def _core_mode_no_confident(
    _env_sandbox, monkeypatch, request: "FixtureRequest"
):

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


@pytest.fixture(autouse=False)
def unpatch_openai_after():
    from deepeval.openai.patch import unpatch_openai_classes

    yield
    unpatch_openai_classes()


# Run every test in its own temp CWD so .deepeval/.deepeval is sandboxed
@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(tenacity.nap, "sleep", lambda _: None, raising=True)

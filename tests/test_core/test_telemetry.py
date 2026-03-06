import os
import pytest
import shutil
from unittest.mock import patch, MagicMock

import deepeval.telemetry as telemetry_mod

from pathlib import Path


def _no_hidden_store_dir(base: Path):
    deepeval_path = base / ".deepeval"
    shutil.rmtree(deepeval_path, ignore_errors=True)


def _reset_telemetry_state():
    telemetry_mod._posthog_client = None
    telemetry_mod._sentry_initialized = False
    telemetry_mod._anonymous_public_ip = None
    telemetry_mod._ip_resolved = False
    telemetry_mod._error_hook_installed = False


def test_telemetry_writes_create_dir_when_missing(tmp_path, monkeypatch):
    _no_hidden_store_dir(tmp_path)

    os.path
    # Ensure opt-out is not set
    monkeypatch.delenv("DEEPEVAL_TELEMETRY_OPT_OUT", raising=False)

    # Run from a clean CWD with no .deepeval
    monkeypatch.chdir(tmp_path)
    assert not os.path.exists(".deepeval")

    uid = telemetry_mod.get_unique_id()
    assert isinstance(uid, str) and len(uid) > 0
    assert os.path.exists(".deepeval/.deepeval_telemetry.txt")


def test_no_external_calls_at_import_time(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    _reset_telemetry_state()

    assert telemetry_mod._posthog_client is None
    assert telemetry_mod._sentry_initialized is False
    assert telemetry_mod._ip_resolved is False


def test_opt_out_skips_initialization(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    _reset_telemetry_state()

    telemetry_mod._ensure_telemetry_initialized()

    assert telemetry_mod._posthog_client is None
    assert telemetry_mod._sentry_initialized is False
    assert telemetry_mod._ip_resolved is False


def test_opt_out_context_managers_skip_posthog(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    _reset_telemetry_state()

    with telemetry_mod.capture_evaluation_run("evaluate()"):
        pass

    assert telemetry_mod._posthog_client is None


@patch("deepeval.telemetry.get_anonymous_public_ip", return_value="1.2.3.4")
@patch("posthog.Posthog", autospec=True)
def test_lazy_init_when_telemetry_enabled(
    mock_posthog_cls, mock_ip, monkeypatch
):
    monkeypatch.delenv("DEEPEVAL_TELEMETRY_OPT_OUT", raising=False)
    _reset_telemetry_state()

    mock_client = MagicMock()
    mock_posthog_cls.return_value = mock_client

    with patch("deepeval.telemetry.telemetry_opt_out", return_value=False):
        with patch("sentry_sdk.init"):
            telemetry_mod._ensure_telemetry_initialized()

    assert telemetry_mod._posthog_client is mock_client
    assert telemetry_mod._sentry_initialized is True
    assert telemetry_mod._ip_resolved is True
    assert telemetry_mod._anonymous_public_ip == "1.2.3.4"
    mock_posthog_cls.assert_called_once()


def test_lazy_init_only_runs_once(monkeypatch):
    _reset_telemetry_state()

    sentinel = MagicMock()
    telemetry_mod._posthog_client = sentinel

    telemetry_mod._ensure_telemetry_initialized()

    assert telemetry_mod._posthog_client is sentinel

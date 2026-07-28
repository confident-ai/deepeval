import os
from pathlib import Path
import subprocess
import sys
import shutil

import pytest

import deepeval.telemetry as telemetry_mod


def _no_hidden_store_dir(base: Path):
    deepeval_path = base / ".deepeval"
    shutil.rmtree(deepeval_path, ignore_errors=True)


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


def test_telemetry_opt_out_skips_error_reporting_firewall_probe():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["DEEPEVAL_DISABLE_DOTENV"] = "1"
    env["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
    env["DEEPEVAL_UPDATE_WARNING_OPT_IN"] = "0"
    env["ERROR_REPORTING"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repo_root), env.get("PYTHONPATH")])
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import socket\n"
                "def fail_if_socket_opened(*args, **kwargs):\n"
                "    raise AssertionError('telemetry opt-out should not open sockets')\n"
                "socket.create_connection = fail_if_socket_opened\n"
                "import deepeval.telemetry\n"
            ),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr

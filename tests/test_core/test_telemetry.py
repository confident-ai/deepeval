import os
import pytest
import shutil

import deepeval.telemetry as telemetry_mod

from pathlib import Path


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

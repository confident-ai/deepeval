import os
import pytest
import deepeval.telemetry as telemetry_mod


def test_telemetry_writes_create_dir_when_missing(tmp_path, monkeypatch):
    # Ensure opt-out is not set
    monkeypatch.delenv("DEEPEVAL_TELEMETRY_OPT_OUT", raising=False)

    # Run from a clean CWD with no .deepeval
    monkeypatch.chdir(tmp_path)
    assert not os.path.exists(".deepeval")

    # After the fix, this should NOT raise; it should create the dir and write the file
    uid = telemetry_mod.get_unique_id()
    assert isinstance(uid, str) and len(uid) > 0
    assert os.path.exists(".deepeval/.deepeval_telemetry.txt")

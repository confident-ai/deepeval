# tests/conftest.py
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def deepeval_isolated_no_disk(tmp_path, monkeypatch):
    hidden = tmp_path / ".deepeval"
    hidden.mkdir(parents=True, exist_ok=True)

    # import the modules we need to patch
    import deepeval.constants as consts
    import deepeval.key_handler as keyh
    import deepeval.test_run.test_run as tr
    import deepeval.dataset.dataset as ds

    # point both constants modules at our isolated dir
    monkeypatch.setattr(consts, "HIDDEN_DIR", str(hidden), raising=False)
    monkeypatch.setattr(keyh, "HIDDEN_DIR", str(hidden), raising=False)

    tmp_temp = hidden / ".temp_test_run_data.json"
    tmp_latest = hidden / ".latest_test_run.json"

    # patch both modules that reference these file paths:
    for mod in (tr, ds):
        monkeypatch.setattr(mod, "TEMP_FILE_PATH", str(tmp_temp), raising=False)
        monkeypatch.setattr(
            mod, "LATEST_TEST_RUN_FILE_PATH", str(tmp_latest), raising=False
        )

    # make sure the manager uses our temp file path,
    # and disable writes and uploads
    tr.global_test_run_manager.temp_file_path = str(tmp_temp)
    tr.global_test_run_manager.save_to_disk = False
    tr.global_test_run_manager.disable_request = True

    # at the class level ensure no disk writing methods so a plugin
    # or code path can’t write anyway.
    monkeypatch.setattr(
        tr.TestRunManager,
        "save_test_run",
        lambda self, *a, **k: None,
        raising=False,
    )
    monkeypatch.setattr(
        tr.TestRunManager,
        "save_final_test_run_link",
        lambda self, *a, **k: None,
        raising=False,
    )
    monkeypatch.setattr(
        tr.TestRunManager,
        "save_test_run_locally",
        lambda self: None,
        raising=False,
    )

    # ensure the dir exists before portalocker could be touched by anything else
    hidden.mkdir(parents=True, exist_ok=True)

    yield

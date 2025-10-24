from pathlib import Path
import pytest

@pytest.fixture(autouse=True)
def deepeval_isolated_no_disk(tmp_path, monkeypatch):
    # isolated hidden dir per test
    hidden = tmp_path / ".deepeval"
    hidden.mkdir(parents=True, exist_ok=True)

    # point deepeval at the isolated dir and files
    import deepeval.constants as consts
    import deepeval.key_handler as keyh
    import deepeval.test_run.test_run as tr

    monkeypatch.setattr(consts, "HIDDEN_DIR", str(hidden), raising=False)
    monkeypatch.setattr(keyh,   "HIDDEN_DIR", str(hidden), raising=False)
    monkeypatch.setattr(tr, "TEMP_FILE_PATH",            str(hidden / ".temp_test_run_data.json"), raising=False)
    monkeypatch.setattr(tr, "LATEST_TEST_RUN_FILE_PATH", str(hidden / ".latest_test_run.json"),     raising=False)

    # update the global manager with path, disable writing to disk and network uploads
    tr.global_test_run_manager.temp_file_path = tr.TEMP_FILE_PATH
    tr.global_test_run_manager.save_to_disk   = False
    tr.global_test_run_manager.disable_request = True
    yield

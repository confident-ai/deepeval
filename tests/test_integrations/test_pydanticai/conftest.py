from pathlib import Path
import pytest

@pytest.fixture(autouse=True)
def isolated_deepeval_dir(tmp_path, monkeypatch):
    """
    Give each test its own .deepeval/ inside pytest's tmp_path and point all places
    that read and write these files to the isolated dir.
    """
    hidden = tmp_path / ".deepeval"
    hidden.mkdir(parents=True, exist_ok=True)

    # patch modules that read and write using these constants paths
    import deepeval.constants as consts
    import deepeval.key_handler as keyh
    import deepeval.test_run.test_run as tr

    # redirect constants used by some modules
    monkeypatch.setattr(consts, "HIDDEN_DIR", str(hidden), raising=False)
    monkeypatch.setattr(keyh, "HIDDEN_DIR", str(hidden), raising=False)

    # redirect module level file paths
    monkeypatch.setattr(tr, "TEMP_FILE_PATH", str(hidden / ".temp_test_run_data.json"), raising=False)
    monkeypatch.setattr(tr, "LATEST_TEST_RUN_FILE_PATH", str(hidden / ".latest_test_run.json"), raising=False)

    # the manager captures the path at import
    # so repoint its instance attribute to the patched path
    tr.global_test_run_manager.temp_file_path = tr.TEMP_FILE_PATH

    # and make sure the parent dir exists before any portalocker calls
    hidden.mkdir(parents=True, exist_ok=True)

    yield

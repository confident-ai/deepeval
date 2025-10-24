from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def clean_deepeval_hidden_dir(tmp_path, monkeypatch):
    """
    Ensure each test runs with an empty, isolated .deepeval/ directory,
    and that DeepEval's temp files point inside it.
    """
    hidden_dir = tmp_path / ".deepeval"
    hidden_dir.mkdir(parents=True, exist_ok=True)

    # patch the places that read and write to .deepeval
    import deepeval.constants as consts
    import deepeval.key_handler as keyh
    import deepeval.test_run.test_run as tr

    # point all modules at the hidden dir
    monkeypatch.setattr(consts, "HIDDEN_DIR", str(hidden_dir), raising=False)
    monkeypatch.setattr(keyh, "HIDDEN_DIR", str(hidden_dir), raising=False)

    # redirect temp files into the dir
    monkeypatch.setattr(
        tr, "TEMP_FILE_PATH", str(hidden_dir / ".temp_test_run_data.json"), raising=False
    )
    monkeypatch.setattr(
        tr, "LATEST_TEST_RUN_FILE_PATH", str(hidden_dir / ".latest_test_run.json"), raising=False
    )

    # the tmp_path is empty at the start of each test
    # and pytest will clean it up afterwards.
    yield

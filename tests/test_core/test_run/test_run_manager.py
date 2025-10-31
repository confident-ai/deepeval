import os
import portalocker
from deepeval.test_case import LLMTestCase
from deepeval.test_run.test_run import TestRunManager, LLMApiTestCase


def test_get_test_run_preserves_valid_instance_on_read_lock(tmp_path):
    p = tmp_path / "temp_test_run_data.json"
    p.write_text("{}")

    trm = TestRunManager()
    trm.save_to_disk = True
    trm.temp_file_path = str(p)

    trm.create_test_run(identifier="repro-2243")

    # Now simulate a read lock so get_test_run() hits LockException
    lock = portalocker.Lock(
        str(p), mode="w", flags=portalocker.LOCK_EX | portalocker.LOCK_NB
    )
    lock.acquire()
    try:
        out = trm.get_test_run(identifier="repro-2243")
        assert out is not None
    finally:
        lock.release()


def test_get_test_run_preserves_instance_when_file_missing(
    tmp_path, monkeypatch
):
    p = tmp_path / "missing.json"

    trm = TestRunManager()
    trm.save_to_disk = True
    trm.temp_file_path = str(p)

    trm.create_test_run(identifier="first-run")
    # simulate file vanished between create and read
    if os.path.exists(p):
        os.remove(p)

    out = trm.get_test_run(identifier="first-run")
    assert out is not None  # preserves in-memory object


def test_get_test_run_preserves_instance_on_malformed_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json]")

    trm = TestRunManager()
    trm.save_to_disk = True
    trm.temp_file_path = str(p)

    trm.create_test_run(identifier="bad-json")

    out = trm.get_test_run(identifier="bad-json")
    assert out is not None


def test_update_test_run_falls_back_in_memory_on_read_failure(tmp_path):
    p = tmp_path / "run.json"

    trm = TestRunManager()
    trm.save_to_disk = True
    trm.temp_file_path = str(p)

    # create a valid run and write it to disk once
    trm.create_test_run(identifier="fallback")

    # corrupt the file so the subsequent read in update_test_run() JSON-decodes and fails
    p.write_text("{not valid json]")

    api_tc = LLMApiTestCase(
        name="t1",
        input="in",
        actual_output="out",
        order=0,
        metrics_data=[],
        trace=None,
    )
    llm_tc = LLMTestCase(input="in", actual_output="out")

    # this should hit the except branch and fall back to in-memory update
    trm.update_test_run(api_tc, llm_tc)

    out = trm.get_test_run()
    assert out is not None
    assert any(tc.name == "t1" for tc in out.test_cases)

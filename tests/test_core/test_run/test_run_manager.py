import os
import portalocker

import deepeval.test_run.test_run as tr_mod
import deepeval.test_run.cache as cache_mod
import deepeval.utils as utils_mod

from types import SimpleNamespace

from deepeval.test_case import LLMTestCase
from deepeval.test_run.test_run import TestRunManager, LLMApiTestCase
from deepeval.test_run.cache import CachedTestRun
from deepeval.utils import suppress_evaluation_output
from tests.test_core.helpers import _make_fake_portalocker
from tests.test_core.stubs import RecordingPortalockerLock


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


def test_save_test_run_with_save_under_key_flushes_and_syncs(
    monkeypatch, tmp_path
):
    """
    When save_under_key is used, TestRunManager.save_test_run calls json.dump
    directly. We want to ensure that path flushes and fsyncs the file before releasing
    the portalocker lock.
    """
    # Patch portalocker inside the module under test
    monkeypatch.setattr(
        tr_mod, "portalocker", _make_fake_portalocker(), raising=False
    )

    # Track fsync calls
    fsync_calls: list[int] = []

    def fake_fsync(fd: int) -> None:
        fsync_calls.append(fd)

    monkeypatch.setattr(tr_mod.os, "fsync", fake_fsync)

    # Minimal "test_run" stub: only needs model_dump/dict for this path
    dummy_test_run = SimpleNamespace(
        model_dump=lambda **kwargs: {"foo": "bar"},
        dict=lambda **kwargs: {"foo": "bar"},
        save=lambda f: None,
    )

    # Minimal "self" stub: save_to_disk + test_run
    dummy_manager = SimpleNamespace(
        save_to_disk=True,
        test_run=dummy_test_run,
    )

    path = tmp_path / "run.json"

    # Call the real implementation as an unbound method
    TestRunManager.save_test_run(
        dummy_manager,
        str(path),
        save_under_key="wrapped_key",
    )

    f = RecordingPortalockerLock.last_file
    assert f is not None, "RecordingPortalockerLock did not capture a file"

    assert f.flushed, (
        "save_test_run(..., save_under_key=...) should call file.flush() "
        "after json.dump(...)"
    )
    assert (
        fsync_calls
    ), "save_test_run(..., save_under_key=...) should call os.fsync(file.fileno())"
    assert fsync_calls[-1] == f.fileno()


def test_post_test_run_quiet_mode_uploads_without_output(monkeypatch, capsys):
    class FakeApi:
        def send_request(self, **kwargs):
            return {"id": "run-123"}, "https://example.test/runs/run-123"

    manager = TestRunManager()
    test_run = tr_mod.TestRun(
        testCases=[
            LLMApiTestCase(
                name="quiet-upload",
                input="input",
                actualOutput="output",
                success=True,
                metricsData=[],
            )
        ]
    )
    saved_links = []
    opened_links = []
    monkeypatch.setattr(tr_mod, "Api", FakeApi)
    monkeypatch.setattr(manager, "save_final_test_run_link", saved_links.append)
    monkeypatch.setattr(tr_mod, "open_browser", opened_links.append)

    result = manager.post_test_run(test_run, print_results=False)

    assert result == ("https://example.test/runs/run-123", "run-123")
    assert saved_links == ["https://example.test/runs/run-123"]
    assert opened_links == ["https://example.test/runs/run-123"]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_quiet_wrap_up_supports_legacy_manager_override_signatures(
    monkeypatch, capsys
):
    class LegacyManager(TestRunManager):
        def __init__(self):
            super().__init__()
            self.calls = []

        def save_test_run_locally(self):
            self.calls.append("save")

        def post_test_run(self, test_run):
            self.calls.append("post")
            return "https://example.test/runs/legacy", "legacy-run"

    manager = LegacyManager()
    manager.set_test_run(
        tr_mod.TestRun(
            testCases=[
                LLMApiTestCase(
                    name="legacy-override",
                    input="input",
                    actualOutput="output",
                    success=True,
                    metricsData=[],
                )
            ]
        )
    )
    monkeypatch.setattr(tr_mod, "is_confident", lambda: True)
    monkeypatch.setattr(tr_mod, "delete_file_if_exists", lambda path: None)
    monkeypatch.setattr(
        tr_mod.global_test_run_cache_manager,
        "wrap_up_cached_test_run",
        lambda: None,
    )

    result = manager.wrap_up_test_run(
        0.1, display_table=False, print_results=False
    )

    assert result == ("https://example.test/runs/legacy", "legacy-run")
    assert manager.calls == ["save", "post"]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_quiet_wrap_up_suppresses_temp_file_delete_error(
    tmp_path, monkeypatch, capsys
):
    temp_path = tmp_path / "temp-run.json"
    temp_path.write_text("{}", encoding="utf-8")
    manager = TestRunManager()
    manager.temp_file_path = str(temp_path)
    manager.set_test_run(tr_mod.TestRun())

    def fail_to_remove(path):
        raise OSError("simulated delete failure")

    monkeypatch.setattr(utils_mod.os, "remove", fail_to_remove)

    manager.wrap_up_test_run(0.1, display_table=False, print_results=False)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    utils_mod.delete_file_if_exists(temp_path)
    assert "simulated delete failure" in capsys.readouterr().out


def test_evaluation_output_context_suppresses_cache_io_error(
    monkeypatch, capsys
):
    class FailingLock:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise OSError("simulated cache failure")

        def __exit__(self, exc_type, exc, traceback):
            return False

    manager = cache_mod.TestRunCacheManager()
    manager.disable_write_cache = False
    manager.cached_test_run = CachedTestRun()
    monkeypatch.setattr(
        cache_mod,
        "portalocker",
        SimpleNamespace(Lock=FailingLock),
    )

    with suppress_evaluation_output():
        manager.save_cached_test_run()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    manager.save_cached_test_run()
    assert "simulated cache failure" in capsys.readouterr().err

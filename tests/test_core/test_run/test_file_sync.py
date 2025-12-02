import types


import deepeval.test_run.test_run as tr
import deepeval.test_run.cache as cache_mod
from tests.test_core.stubs import RecordingFile


def _make_dummy_self():
    """
    Minimal stand-in for TestRun / CachedTestRun.

    The save() methods only care that .model_dump() or .dict()
    produce something JSON-serializable, so we provide just that.
    """

    class Dummy:
        def model_dump(self, **kwargs):
            return {"dummy": True}

    return Dummy()


def test_test_run_save_flushes_and_syncs(monkeypatch):
    """
    TestRun.save(self, f) must flush Python buffers and fsync OS buffers.

    This fails on current main, because TestRun.save() only calls json.dump
    and never flushes or fsyncs. It passes after you add:

        f.flush()
        os.fsync(f.fileno())
    """
    fsynced = {"called": False}

    def fake_fsync(fd: int) -> None:
        fsynced["called"] = True

    # Patch os.fsync as seen from the test_run module
    monkeypatch.setattr(
        tr, "os", types.SimpleNamespace(**vars(tr.os)), raising=False
    )
    monkeypatch.setattr(tr.os, "fsync", fake_fsync, raising=False)

    f = RecordingFile()
    dummy_self = _make_dummy_self()

    # Call the real implementation on a dummy "self"
    tr.TestRun.save(dummy_self, f)

    assert (
        f.flushed
    ), "TestRun.save() should call f.flush() after json.dump(...)"
    assert fsynced["called"], "TestRun.save() should call os.fsync(f.fileno())"


def test_cached_test_run_save_flushes_and_syncs(monkeypatch):
    """
    CachedTestRun.save(self, f) must also flush and fsync.

    This mirrors the same durability requirement for the cached
    on-disk representation.
    """
    fsynced = {"called": False}

    def fake_fsync(fd: int) -> None:
        fsynced["called"] = True

    # Patch os.fsync as seen from the cache module
    monkeypatch.setattr(
        cache_mod,
        "os",
        types.SimpleNamespace(**vars(cache_mod.os)),
        raising=False,
    )
    monkeypatch.setattr(cache_mod.os, "fsync", fake_fsync, raising=False)

    f = RecordingFile()
    dummy_self = _make_dummy_self()

    cache_mod.CachedTestRun.save(dummy_self, f)

    assert (
        f.flushed
    ), "CachedTestRun.save() should call f.flush() after json.dump(...)"
    assert fsynced[
        "called"
    ], "CachedTestRun.save() should call os.fsync(f.fileno())"

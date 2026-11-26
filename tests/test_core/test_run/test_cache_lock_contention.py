import functools

import portalocker

from deepeval.test_case import LLMTestCase
from deepeval.test_run import cache as cache_module


def _manager(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = cache_module.TestRunCacheManager()
    manager.disable_write_cache = False
    manager.create_cached_test_run(temp=True)
    manager.temp_cached_test_run = None
    return manager


def _fail_fast_on_lock(monkeypatch):
    # portalocker retries a contended lock for 5s by default. Patch it to give
    # up immediately so these tests don't each sit through that wait. Call this
    # after the test's own holder lock is taken, since it patches Lock globally.
    real_lock = portalocker.Lock
    monkeypatch.setattr(
        portalocker,
        "Lock",
        functools.partial(real_lock, timeout=0),
    )


def test_cache_test_case_skips_when_cache_file_is_locked(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    test_case = LLMTestCase(input="a", actual_output="b")

    # Another process holding the file makes get_cached_test_run return None.
    with portalocker.Lock(manager.temp_cache_file_name, mode="w"):
        _fail_fast_on_lock(monkeypatch)
        manager.cache_test_case(
            test_case, cache_module.CachedTestCase(), None, to_temp=True
        )


def test_get_cached_test_case_is_a_miss_when_cache_file_is_locked(
    tmp_path, monkeypatch
):
    manager = _manager(tmp_path, monkeypatch)
    manager.create_cached_test_run()
    manager.cached_test_run = None
    test_case = LLMTestCase(input="a", actual_output="b")

    with portalocker.Lock(manager.cache_file_name, mode="w"):
        _fail_fast_on_lock(monkeypatch)
        assert manager.get_cached_test_case(test_case, None) is None

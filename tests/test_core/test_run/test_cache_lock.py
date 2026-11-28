import portalocker

from deepeval.test_case import LLMTestCase
from deepeval.test_run.cache import (
    CachedTestCase,
    CachedTestRun,
    TestRunCacheManager,
)


def test_cache_test_case_skips_when_temp_cache_locked(tmp_path):
    manager = TestRunCacheManager()
    manager.disable_write_cache = False
    manager.cache_file_name = str(tmp_path / "cache.json")
    manager.temp_cache_file_name = str(tmp_path / "temp_cache.json")
    manager.create_cached_test_run(temp=True)
    manager.temp_cached_test_run = None

    case = LLMTestCase(input="a", actual_output="b")
    with portalocker.Lock(
        manager.temp_cache_file_name,
        mode="w",
        flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
    ):
        manager.cache_test_case(case, CachedTestCase(), None, to_temp=True)

    assert manager.temp_cached_test_run is None


def test_cache_manager_handles_none_cached_test_run(tmp_path, monkeypatch):
    manager = TestRunCacheManager()
    manager.disable_write_cache = False
    cache_path = tmp_path / "cache.json"
    temp_path = tmp_path / "temp_cache.json"
    cache_path.write_text("existing-cache", encoding="utf-8")
    temp_path.write_text("existing-temp", encoding="utf-8")
    manager.cache_file_name = str(cache_path)
    manager.temp_cache_file_name = str(temp_path)

    monkeypatch.setattr(manager, "get_cached_test_run", lambda **kwargs: None)
    case = LLMTestCase(input="a", actual_output="b")

    assert manager.get_cached_test_case(case, None) is None
    manager.cache_test_case(case, CachedTestCase(), None, to_temp=True)
    manager.save_cached_test_run(to_temp=False)
    manager.save_cached_test_run(to_temp=True)
    assert cache_path.read_text(encoding="utf-8") == "existing-cache"

    manager.wrap_up_cached_test_run()
    assert cache_path.read_text(encoding="utf-8") == "existing-cache"
    assert not temp_path.exists()

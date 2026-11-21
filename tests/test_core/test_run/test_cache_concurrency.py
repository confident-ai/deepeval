import portalocker

from deepeval.test_case import LLMTestCase
from deepeval.test_run.cache import (
    CachedTestCase,
    CachedTestRun,
    TestRunCacheManager,
)


def test_cache_test_case_when_get_cached_test_run_returns_none(monkeypatch):
    """Test that cache_test_case does not crash with AttributeError when get_cached_test_run returns None."""
    manager = TestRunCacheManager()
    monkeypatch.setattr(manager, "get_cached_test_run", lambda **kwargs: None)
    monkeypatch.setattr(manager, "save_cached_test_run", lambda **kwargs: None)

    case = LLMTestCase(input="Hello", actual_output="World")
    cached_case = CachedTestCase()

    # Must not raise AttributeError: 'NoneType' object has no attribute 'test_cases_lookup_map'
    manager.cache_test_case(case, cached_case, None)


def test_cache_test_case_when_lookup_map_is_none(monkeypatch):
    """Test that cache_test_case safely initializes test_cases_lookup_map when it is None."""
    manager = TestRunCacheManager()
    cached_run = CachedTestRun(test_cases_lookup_map=None)
    monkeypatch.setattr(
        manager, "get_cached_test_run", lambda **kwargs: cached_run
    )
    monkeypatch.setattr(manager, "save_cached_test_run", lambda **kwargs: None)

    case = LLMTestCase(input="Hello", actual_output="World")
    cached_case = CachedTestCase()

    manager.cache_test_case(case, cached_case, None)
    assert cached_run.test_cases_lookup_map is not None
    assert len(cached_run.test_cases_lookup_map) == 1


def test_get_cached_test_case_when_cached_test_run_is_none(monkeypatch):
    """Test that get_cached_test_case returns None safely when get_cached_test_run returns None."""
    manager = TestRunCacheManager()
    monkeypatch.setattr(manager, "get_cached_test_run", lambda **kwargs: None)

    case = LLMTestCase(input="Hello", actual_output="World")
    result = manager.get_cached_test_case(case, None)
    assert result is None


def test_get_cached_api_test_case_when_lookup_map_is_none():
    """Test that CachedTestRun.get_cached_api_test_case handles None lookup map without AttributeError."""
    cached_run = CachedTestRun(test_cases_lookup_map=None)
    result = cached_run.get_cached_api_test_case("some_key")
    assert result is None


def test_get_cached_test_run_on_lock_exception(monkeypatch):
    """Test that get_cached_test_run gracefully falls back to an initialized CachedTestRun on LockException."""
    manager = TestRunCacheManager()

    def mock_lock(*args, **kwargs):
        raise portalocker.exceptions.LockException(
            "File locked by concurrent process"
        )

    monkeypatch.setattr(portalocker, "Lock", mock_lock)
    monkeypatch.setattr("os.path.exists", lambda path: True)

    cached_run = manager.get_cached_test_run()
    assert cached_run is not None
    assert cached_run.test_cases_lookup_map is not None


def test_test_case_cache_module_export():
    """Verify that deepeval.test_case.cache exports TestRunCacheManager and cache entities."""
    from deepeval.test_case.cache import TestRunCacheManager as ExportedManager
    from deepeval.test_case.cache import CachedTestRun as ExportedRun

    assert ExportedManager is not None
    assert ExportedRun is not None

import json
from pathlib import Path

import pytest

from deepeval.test_case import LLMTestCase
from deepeval.test_run.cache import (
    CachedTestCase,
    CachedTestRun,
    TestRunCacheManager as _TestRunCacheManager,
)


@pytest.mark.parametrize("from_temp", [False, True])
@pytest.mark.parametrize(
    "content",
    ["", '{"test_cases_lookup_map": null}'],
    ids=["empty-file", "null-lookup-map"],
)
def test_cache_write_recovers_corrupt_file(
    tmp_path: Path, from_temp: bool, content: str
):
    manager = _TestRunCacheManager()
    manager.cache_file_name = str(tmp_path / "cache.json")
    manager.temp_cache_file_name = str(tmp_path / "temp-cache.json")
    cache_path = Path(
        manager.temp_cache_file_name if from_temp else manager.cache_file_name
    )
    cache_path.write_text(content)

    test_case = LLMTestCase(input="question", actual_output="answer")
    cached_test_case = CachedTestCase()
    manager.cache_test_case(
        test_case, cached_test_case, hyperparameters=None, to_temp=from_temp
    )

    saved_run = CachedTestRun.load(json.loads(cache_path.read_text()))
    assert list(saved_run.test_cases_lookup_map.values()) == [cached_test_case]


@pytest.mark.parametrize("from_temp", [False, True])
def test_cache_write_preserves_existing_entries(
    tmp_path: Path, from_temp: bool
):
    manager = _TestRunCacheManager()
    manager.cache_file_name = str(tmp_path / "cache.json")
    manager.temp_cache_file_name = str(tmp_path / "temp-cache.json")
    cache_path = Path(
        manager.temp_cache_file_name if from_temp else manager.cache_file_name
    )
    cache_path.write_text('{"test_cases_lookup_map": {"existing": {}}}')

    manager.cache_test_case(
        LLMTestCase(input="question", actual_output="answer"),
        CachedTestCase(),
        hyperparameters=None,
        to_temp=from_temp,
    )

    saved_run = CachedTestRun.load(json.loads(cache_path.read_text()))
    assert "existing" in saved_run.test_cases_lookup_map
    assert len(saved_run.test_cases_lookup_map) == 2

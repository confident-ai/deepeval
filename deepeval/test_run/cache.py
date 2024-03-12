import portalocker
import sys
import json
from typing import List, Optional, Union
from pydantic import Field

from deepeval.test_run.api import MetricsMetadata
from deepeval.test_run.test_run import TestRun

CACHE_FILE_NAME = ".deepeval-cache"

class CachedTestRun:
    # metrics lookup map will lookup whether the existing metrics already exist according to
    # the dictionary key str. The key str should be a recursively serialized llm test case params
    metrics_lookup_map: Optional[dict[str: List[MetricsMetadata]]] = Field(None)

    # never use the cache when there is a hyperparameter change
    hyperparameters: Optional[dict[str, Union[str, float, int]]] = Field(None)

    #saves to file (this happens at the very end of a test run)
    def save(self, f):
        json.dump(self.model_dump(by_alias=True, exclude_none=True), f)
        return self

    #load from file (this happens initially during a test run)
    @classmethod
    def load(cls, f):
        return cls(**json.load(f))


class TestRunCacheManager:
    def __init__(self):
        self.cached_test_run = None
        self.cache_file_name = CACHE_FILE_NAME

    def create_cached_test_run(self, cache: dict[str: List[MetricsMetadata]]):
        self.cached_test_run = CachedTestRun(
            cache_file=self.cache_file_name,
            cache=cache
        )

        self.save_cached_test_run()

    def get_cached_test_run(self) -> Union[CachedTestRun, None]:
        # Subsequent calls to get cached test run (if any) doesn't go through disk
        if self.cached_test_run:
            return self.cached_test_run

        # first create cached test run object
        self.cached_test_run = CachedTestRun(cache_file=self.cache_file_name)
        try:
            with portalocker.Lock(
                self.cache_file_name, mode="r", timeout=5
            ) as file:
                # populate cached hyperparameters and metrics if any from .deepeval-cache
                self.cached_test_run = self.cached_test_run.load(file)
        except (
            FileNotFoundError,
            portalocker.exceptions.LockException,
        ) as e:
            print(f"Error loading test run from disk: {e}", file=sys.stderr)
            self.cached_test_run = None

        # return cached, can be None
        return self.cached_test_run

    def save_cached_test_run(self, test_run: TestRun):
        # do some processing to turn test_run into cached_test_run
        self.cached_test_run.metrics_lookup_map = ...
        
        # can just update directly
        self.cached_test_run.hyperparameters = test_run.hyperparameters
        try:
            with portalocker.Lock(
                self.cache_file_name, mode="w", timeout=5
            ) as file:
                self.cached_test_run = self.cached_test_run.save(file)
                print("Cached saved!")
        except portalocker.exceptions.LockException:
            print("Error saving test run to disk", file=sys.stderr)


test_run_cache_manager = TestRunCacheManager()
cached_test_run = test_run_cache_manager.get_cached_test_run()

if cached_test_run is None:
    ...

else:
    cache logic...


### execute test run...
    

test_run_cache_manager.save_cached_test_run(test_run)
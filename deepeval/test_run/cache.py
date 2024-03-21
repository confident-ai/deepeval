import portalocker
import sys
import json
import os
from typing import List, Optional, Union
from pydantic import BaseModel, Field

from deepeval.test_run.api import MetricsMetadata
from deepeval.test_run.test_run import TestRun
from deepeval.test_run.api import APITestCase
from deepeval.utils import delete_file_if_exists

CACHE_FILE_NAME = ".deepeval-cache.json"
TEMP_CACHE_FILE_NAME = ".temp-deepeval-cache.json"

class CachedTestRun(BaseModel):
    # metrics lookup map will lookup whether the existing metrics already exist according to
    # the dictionary key str. The key str should be a recursively serialized llm test case params
    test_cases_lookup_map: Optional[dict[str, APITestCase]] = Field(None, alias="testCasesLookupMap")

    # never use the cache when there is a hyperparameter change
    hyperparameters: Optional[dict[str, Union[str, float, int]]] = Field(None, alias="hyperparameters")

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
        self.cached_test_run: Optional[CachedTestRun] = None
        self.temp_cached_test_run: Optional[CachedTestRun] = None
        self.cache_file_name: str = CACHE_FILE_NAME 
        self.temp_cache_file_name: str = TEMP_CACHE_FILE_NAME
    
    def reset(self):
        self.temp_cached_test_run = None

    def set_temp_cached_test_run(self, temp_cached_test_run: CachedTestRun):
        self.temp_cached_test_run = temp_cached_test_run

    def set_cached_test_run(self, cached_test_run: CachedTestRun):
        self.cached_test_run = cached_test_run

    def create_cached_test_run(self):
        cached_test_run = CachedTestRun(
            testCasesLookupMap={},
            hyperparameters={}
        )
        self.set_cached_test_run(cached_test_run)
        self.save_cached_test_run()
    
    def create_temp_cached_test_run(self):
        temp_cached_test_run = CachedTestRun(
            testCasesLookupMap={},
            hyperparameters={}
        )
        self.set_temp_cached_test_run(temp_cached_test_run)
        self.save_temp_cached_test_run()

    def get_cached_test_run(self) -> Union[CachedTestRun, None]:
        # Subsequent calls to get cached test run (if any) doesn't go through disk
        # This occurs when you are in the same test_run but encountering new test cases
        if self.cached_test_run:
            return self.cached_test_run
        
        # Check if the cache file exists
        if not os.path.exists(self.cache_file_name):
            # If the file does not exist, create a new CachedTestRun instance and save it to disk
            self.create_cached_test_run()

        # Load cached_test_run from disk
        try:
            with portalocker.Lock(
                self.cache_file_name, mode="r", timeout=5
            ) as file:
                # populate cached hyperparameters and metrics if any from .deepeval-cache
                # cache file could exist but self.cached_test_run = none when running a new test_run
                # but test_cases have already been saved. That's why 'CachedTestRun.load(file)' is necessary
                self.cached_test_run = CachedTestRun.load(file)
        except portalocker.exceptions.LockException as e:
            print(f"Lock acquisition failed: {e}", file=sys.stderr)
            # Optionally, handle the retry mechanism here if the lock cannot be acquired
        except FileNotFoundError as e:
            # This should not happen since we check if the file exists,
            # but it's here as a safety net
            print(f"File not found: {e}", file=sys.stderr)
            self.create_test_run()


        # return cached, can be None
        return self.cached_test_run
    
    def get_temp_cached_test_run(self) -> Union[CachedTestRun, None]:
        if self.temp_cached_test_run:
            return self.temp_cached_test_run

        if not os.path.exists(self.temp_cache_file_name):
            self.create_temp_cached_test_run()

        try:
            with portalocker.Lock(
                self.temp_cache_file_name, mode="r", timeout=5
            ) as file:
                self.temp_cached_test_run = CachedTestRun.load(file)
        except portalocker.exceptions.LockException as e:
            print(f"Lock acquisition failed: {e}", file=sys.stderr)
        except FileNotFoundError as e:
            print(f"File not found: {e}", file=sys.stderr)
            self.create_temp_cached_test_run()
            
        return self.temp_cached_test_run

    def save_cached_test_run(self):
        try:
            with portalocker.Lock(
                self.cache_file_name, mode="w", timeout=5
            ) as file:
                self.cached_test_run = self.cached_test_run.save(file)
        except portalocker.exceptions.LockException:
            print("Error saving test run to disk", file=sys.stderr)
    
    def save_temp_cached_test_run(self):
            try:
                with portalocker.Lock(
                    self.temp_cache_file_name, mode="w", timeout=5
                ) as file:
                    self.temp_cached_test_run = self.temp_cached_test_run.save(file)
            except portalocker.exceptions.LockException:
                print("Error saving test run to disk", file=sys.stderr)
    
    def wrap_up_test_run(self):
        try:
            with portalocker.Lock(
                self.cache_file_name, mode="w", timeout=5
            ) as file:
                self.temp_cached_test_run = self.temp_cached_test_run.save(file)
        except portalocker.exceptions.LockException:
            print("Error saving test run to disk", file=sys.stderr)
        
        delete_file_if_exists(self.temp_cache_file_name)
        
test_run_cache_manager = TestRunCacheManager()
import portalocker
import sys
import json
import os
from typing import List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.embeddings import Embeddings

from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.types import Languages
from deepeval.test_run.api import MetricsMetadata
from deepeval.utils import delete_file_if_exists
from deepeval.metrics import BaseMetric

CACHE_FILE_NAME = ".deepeval-cache.json"
TEMP_CACHE_FILE_NAME = ".temp-deepeval-cache.json"

class MetricConfiguration(BaseModel):
    ##### Required fields #####
    criteria: Optional[str] = None
    include_reason: Optional[bool] = None
    strict_mode: Optional[bool] = None
    n: Optional[int] = None

    ##### Optional fields #####
    evaluation_steps: Optional[List[str]] = None
    assessment_questions: Optional[List[str]] = None
    language: Optional[Languages] = None
    embeddings: Optional[str] = None
    evaluation_params: Optional[List[LLMTestCaseParams]] = None

    class Config:
        arbitrary_types_allowed = True

class CachedMetricsMetadata(BaseModel):
     metric_metadata: MetricsMetadata
     metric_configuration: MetricConfiguration

class CachedAPITestCase(BaseModel):
    cached_metrics_metadata: List[CachedMetricsMetadata] = Field(
        ..., alias="metricsMetadata"
    )

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, BaseModel):
            return obj.model_dump(by_alias=True, exclude_none=True)
        return json.JSONEncoder.default(self, obj)

class CachedTestRun(BaseModel):
    # metrics lookup map will lookup whether the existing metrics already exist according to
    # the dictionary key str. The key str should be a recursively serialized llm test case params
    test_cases_lookup_map: Optional[dict[str, CachedAPITestCase]] = Field(None, alias="testCasesLookupMap")
    # never use the cache when there is a hyperparameter change
    hyperparameters: Optional[dict[str, Union[str, float, int]]] = Field(None, alias="hyperparameters")
    
    #saves to file (this happens at the very end of a test run)
    def save(self, file):
        json.dump(self, file, cls=CustomEncoder, ensure_ascii=False, indent=4)
        return self
    
    #load from file (this happens initially during a test run)
    @classmethod
    def load(cls, file):
        return cls(**json.load(file))
    
    def get_cached_api_test_case(self, key: str) -> CachedAPITestCase:
        return self.test_cases_lookup_map.get(key, None)

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

############ Helper Functions #############

class Cache:

    @staticmethod
    def generate_cache_key(test_case: LLMTestCase):
            context = sorted(test_case.context) if test_case.context else []
            retrieval_context = sorted(test_case.retrieval_context) if test_case.retrieval_context else []
            cache_data = {
                "input": test_case.input,
                "expected_output": test_case.expected_output,
                "context": context,
                "retrieval_context": retrieval_context,
            }
            sorted_cache_data = dict(sorted(cache_data.items()))
            cache_key = json.dumps(sorted_cache_data)
            return cache_key

    @staticmethod
    def get_metrics_metadata_from_cache(
            metric: BaseMetric, 
            cached_api_test_case: Optional[CachedAPITestCase]
            ) -> Optional[MetricsMetadata]:
        metadata_fields = [
            'threshold',
            'evaluation_model']
        config_fields = [
            'n',
            'criteria', 
            'language',
            'embeddings',
            'strict_mode', 
            'include_reason', 
            'evaluation_steps', 
            'evaluation_params',
            'assessment_questions'
            ]
        if not cached_api_test_case:
            return None
        for c in cached_api_test_case.cached_metrics_metadata:
            if c.metric_metadata.metric == metric.__name__ and \
            Cache.same_fields(metric, c.metric_metadata, metadata_fields) and \
            Cache.same_fields(metric, c.metric_configuration, config_fields):
                return c.metric_metadata
        return None

    @staticmethod
    def same_fields(
            metric: BaseMetric,
            cached_object: Union[MetricsMetadata, 
                                MetricConfiguration], 
            fields) -> bool:
        for field in fields:
            if hasattr(metric, field):
                metric_value = getattr(metric, field, None)
                cached_value = getattr(cached_object, field, None)
                if field == 'embeddings' and metric_value is not None:
                    metric_value = metric_value.__class__.__name__
                if metric_value != cached_value:
                    return False
        return True

    @staticmethod
    def create_metric_configuration(metric: BaseMetric) -> MetricConfiguration:
        config_kwargs = {}
        config_fields = [
            'n', # checked
            'criteria', # checked
            'language', # can't check
            'embeddings', # 
            'include_reason', # checked
            'strict_mode',  #checked
            'evaluation_steps',  #checked
            'evaluation_params', #checked
            'assessment_questions' #checked
            ]
        for field in config_fields:
            if hasattr(metric, field):
                value  = getattr(metric, field)
                if field == 'embeddings' and value is not None:
                    value = value.__class__.__name__
                config_kwargs[field] = value
       
        return MetricConfiguration(**config_kwargs)
    
    @staticmethod 
    def create_cached_api_test_case():
        return CachedAPITestCase(metricsMetadata=[])



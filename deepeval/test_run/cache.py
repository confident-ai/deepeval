import portalocker
import sys
import json
import os
from typing import List, Optional, Union, Dict
from enum import Enum
from pydantic import BaseModel, Field

from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.types import Languages
from deepeval.test_run.api import MetricMetadata
from deepeval.utils import delete_file_if_exists, serialize
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


class CachedMetricData(BaseModel):
    metric_metadata: MetricMetadata
    metric_configuration: MetricConfiguration


class CachedTestCase(BaseModel):
    cached_metrics_data: List[CachedMetricData] = Field(
        default_factory=lambda: []
    )
    hyperparameters: Optional[str] = Field(None)
    model: Optional[str] = Field(None)
    user_prompt_template: Optional[str] = Field(None)


class CachedTestRun(BaseModel):
    test_cases_lookup_map: Optional[dict[str, CachedTestCase]] = Field(
        default_factory=lambda: {}
    )

    # saves to file (this happens at the very end of a test run)
    def save(self, f):
        try:
            body = self.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = self.dict(by_alias=True, exclude_none=True)
        json.dump(body, f)
        return self

    # load from file (this happens initially during a test run)
    @classmethod
    def load(cls, data):
        return cls(**data)

    def get_cached_api_test_case(self, key: str) -> CachedTestCase:
        return self.test_cases_lookup_map.get(key, None)


class TestRunCacheManager:
    def __init__(self):
        self.cached_test_run: Optional[CachedTestRun] = None
        self.cache_file_name: str = CACHE_FILE_NAME
        self.temp_cached_test_run: Optional[CachedTestRun] = None
        self.temp_cache_file_name: str = TEMP_CACHE_FILE_NAME

    def get_cached_test_case(
        self,
        test_case: LLMTestCase,
        hyperparameters: Union[Dict, None],
        model: str,
        user_prompt_template: str,
    ) -> Union[CachedTestCase, None]:
        cached_test_run = self.get_cached_test_run()
        cache_dict = {
            LLMTestCaseParams.INPUT.value: test_case.input,
            LLMTestCaseParams.ACTUAL_OUTPUT.value: test_case.actual_output,
            LLMTestCaseParams.EXPECTED_OUTPUT.value: test_case.expected_output,
            LLMTestCaseParams.CONTEXT.value: test_case.context,
            LLMTestCaseParams.RETRIEVAL_CONTEXT.value: test_case.retrieval_context,
            "hyperparameters": hyperparameters,
            "model": model,
            "user_prompt_template": user_prompt_template,
        }
        test_case_cache_key = serialize(cache_dict)
        cached_test_case = cached_test_run.get_cached_api_test_case(
            test_case_cache_key
        )
        return cached_test_case

    def cache_test_case(
        self,
        test_case: LLMTestCase,
        new_cache_test_case: CachedTestCase,
        hyperparameters: Union[Dict, None],
        model: str,
        user_prompt_template: str,
        to_temp: bool = False,
    ):
        cache_dict = {
            LLMTestCaseParams.INPUT.value: test_case.input,
            LLMTestCaseParams.ACTUAL_OUTPUT.value: test_case.actual_output,
            LLMTestCaseParams.EXPECTED_OUTPUT.value: test_case.expected_output,
            LLMTestCaseParams.CONTEXT.value: test_case.context,
            LLMTestCaseParams.RETRIEVAL_CONTEXT.value: test_case.retrieval_context,
            "hyperparameters": hyperparameters,
            "model": model,
            "user_prompt_template": user_prompt_template,
        }
        test_case_cache_key = serialize(cache_dict)
        cached_test_run = self.get_cached_test_run(from_temp=to_temp)
        cached_test_run.test_cases_lookup_map[test_case_cache_key] = (
            new_cache_test_case
        )
        self.save_cached_test_run(to_temp=to_temp)

    def set_cached_test_run(
        self, cached_test_run: CachedTestRun, temp: bool = False
    ):
        if temp:
            self.temp_cached_test_run = cached_test_run
        else:
            self.cached_test_run = cached_test_run

    def save_cached_test_run(self, to_temp: bool = False):
        if to_temp:
            try:
                with portalocker.Lock(
                    self.temp_cache_file_name, mode="w", timeout=5
                ) as file:
                    self.temp_cached_test_run = self.temp_cached_test_run.save(
                        file
                    )
            except Exception as e:
                print(
                    f"In save_cached_test_run, temp={to_temp}, Error saving test run to disk {e}",
                    file=sys.stderr,
                )
        else:
            try:
                with portalocker.Lock(
                    self.cache_file_name, mode="w", timeout=5
                ) as file:
                    self.cached_test_run = self.cached_test_run.save(file)
            except Exception as e:
                print(
                    f"In save_cached_test_run, temp={to_temp}, Error saving test run to disk {e}",
                    file=sys.stderr,
                )

    def create_cached_test_run(self, temp: bool = False):
        cached_test_run = CachedTestRun()
        self.set_cached_test_run(cached_test_run, temp)
        self.save_cached_test_run(to_temp=temp)

    def get_cached_test_run(
        self, from_temp: bool = False
    ) -> Union[CachedTestRun, None]:
        should_create_cached_test_run = False
        if from_temp:
            if self.temp_cached_test_run:
                return self.temp_cached_test_run

            if not os.path.exists(self.temp_cache_file_name):
                self.create_cached_test_run(temp=from_temp)

            try:
                with portalocker.Lock(
                    self.temp_cache_file_name, mode="r", timeout=5
                ) as file:
                    content = file.read().strip()
                    try:
                        data = json.loads(content)
                        self.temp_cached_test_run = CachedTestRun.load(data)
                    except Exception as e:
                        should_create_cached_test_run = True
            except portalocker.exceptions.LockException as e:
                print(
                    f"In get_cached_test_run, temp={from_temp}, Lock acquisition failed: {e}",
                    file=sys.stderr,
                )

            return self.temp_cached_test_run
        else:
            if self.cached_test_run:
                return self.cached_test_run

            if not os.path.exists(self.cache_file_name):
                self.create_cached_test_run()

            try:
                with portalocker.Lock(
                    self.cache_file_name, mode="r", timeout=5
                ) as file:
                    content = file.read().strip()
                    try:
                        data = json.loads(content)
                        self.cached_test_run = CachedTestRun.load(data)
                    except Exception:
                        should_create_cached_test_run = True

            except portalocker.exceptions.LockException as e:
                print(
                    f"In get_cached_test_run, temp={from_temp}, Lock acquisition failed: {e}",
                    file=sys.stderr,
                )

            if should_create_cached_test_run:
                self.create_cached_test_run(temp=from_temp)

            return self.cached_test_run

    def wrap_up_cached_test_run(self):
        self.get_cached_test_run(from_temp=True)
        try:
            with portalocker.Lock(
                self.cache_file_name, mode="w", timeout=5
            ) as file:
                self.temp_cached_test_run = self.temp_cached_test_run.save(file)
        except Exception as e:
            print(
                f"In wrap_up_cached_test_run,Error saving test run to disk, {e}",
                file=sys.stderr,
            )
        finally:
            delete_file_if_exists(self.temp_cache_file_name)


test_run_cache_manager = TestRunCacheManager()

############ Helper Functions #############


class Cache:
    @staticmethod
    def get_metric_data(
        metric: BaseMetric, cached_api_test_case: Optional[CachedTestCase]
    ) -> Optional[CachedMetricData]:
        metadata_fields = ["threshold", "evaluation_model", "strict_mode"]
        config_fields = [
            "n",
            "criteria",
            "language",
            "embeddings",
            "strict_mode",
            "include_reason",
            "evaluation_steps",
            "evaluation_params",
            "assessment_questions",
        ]
        if not cached_api_test_case:
            return None
        for c in cached_api_test_case.cached_metrics_data:
            if (
                c.metric_metadata.metric == metric.__name__
                and Cache.same_fields(
                    metric, c.metric_metadata, metadata_fields
                )
                and Cache.same_fields(
                    metric, c.metric_configuration, config_fields
                )
            ):
                return c
        return None

    @staticmethod
    def same_fields(
        metric: BaseMetric,
        cached_object: Union[MetricMetadata, MetricConfiguration],
        fields,
    ) -> bool:
        for field in fields:
            if hasattr(metric, field):
                metric_value = getattr(metric, field, None)
                cached_value = getattr(cached_object, field, None)
                if field == "embeddings" and metric_value is not None:
                    metric_value = metric_value.__class__.__name__
                if metric_value != cached_value:
                    return False
        return True

    @staticmethod
    def create_metric_configuration(metric: BaseMetric) -> MetricConfiguration:
        config_kwargs = {}
        config_fields = [
            "n",  # checked
            "criteria",  # checked
            "language",  # can't check
            "embeddings",  #
            "include_reason",  # checked
            "strict_mode",  # checked
            "evaluation_steps",  # checked
            "evaluation_params",  # checked
            "assessment_questions",  # checked
        ]
        for field in config_fields:
            if hasattr(metric, field):
                value = getattr(metric, field)
                if field == "embeddings" and value is not None:
                    value = value.__class__.__name__
                config_kwargs[field] = value

        return MetricConfiguration(**config_kwargs)

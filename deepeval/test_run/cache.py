import logging
import sys
import json
import os
from typing import List, Optional, Dict, Union
from enum import Enum
from pydantic import BaseModel, Field

from deepeval.utils import make_model_config

from deepeval.test_case import SingleTurnParams, LLMTestCase, ToolCallParams
from deepeval.test_run.api import MetricData
from deepeval.test_run.cache_identity import CacheIdentity
from deepeval.utils import (
    delete_file_if_exists,
    is_read_only_env,
    serialize,
)
from deepeval.constants import HIDDEN_DIR

logger = logging.getLogger(__name__)


portalocker = None
if not is_read_only_env():
    try:
        import portalocker
    except Exception as e:
        logger.warning("failed to import portalocker: %s", e)
else:
    logger.warning("READ_ONLY filesystem: skipping disk cache for test runs.")


CACHE_FILE_NAME = f"{HIDDEN_DIR}/.deepeval-cache.json"
TEMP_CACHE_FILE_NAME = f"{HIDDEN_DIR}/.temp-deepeval-cache.json"


class MetricConfiguration(BaseModel):
    model_config = make_model_config(arbitrary_types_allowed=True)

    ##### Required fields #####
    threshold: Union[float, str]
    evaluation_model: Optional[str] = None
    strict_mode: Union[bool, str] = False
    criteria: Optional[str] = None
    include_reason: Optional[Union[bool, str]] = None
    n: Optional[Union[int, str]] = None

    ##### Optional fields #####
    language: Optional[str] = None
    evaluation_steps: Optional[Union[List[str], str]] = None
    assessment_questions: Optional[Union[List[str], str]] = None
    embeddings: Optional[str] = None
    evaluation_params: Optional[
        Union[List[SingleTurnParams], List[ToolCallParams], str]
    ] = None
    # Stores constructor-parameter fingerprints/sentinels for cache compatibility.
    # This field intentionally never contains raw constructor values.
    custom_parameters: Optional[Dict[str, str]] = None


class CachedMetricData(BaseModel):
    metric_data: MetricData
    metric_configuration: MetricConfiguration


class CachedTestCase(BaseModel):
    cached_metrics_data: List[CachedMetricData] = Field(
        default_factory=lambda: []
    )
    hyperparameters: Optional[str] = Field(None)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, BaseModel):
            return obj.model_dump(by_alias=True, exclude_none=True)
        return json.JSONEncoder.default(self, obj)


class CachedTestRun(BaseModel):
    test_cases_lookup_map: Optional[Dict[str, CachedTestCase]] = Field(
        default_factory=lambda: {}
    )

    # saves to file (this happens at the very end of a test run)
    def save(self, f):
        try:
            body = self.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = self.dict(by_alias=True, exclude_none=True)
        json.dump(body, f, cls=CustomEncoder)
        f.flush()
        os.fsync(f.fileno())
        return self

    # load from file (this happens initially during a test run)
    @classmethod
    def load(cls, data):
        return cls(**data)

    def get_cached_api_test_case(self, key: str) -> CachedTestCase:
        return self.test_cases_lookup_map.get(key, None)


class TestRunCacheManager:
    def __init__(self):
        self.disable_write_cache: Optional[bool] = None
        self.cached_test_run: Optional[CachedTestRun] = None
        self.cache_file_name: str = CACHE_FILE_NAME
        self.temp_cached_test_run: Optional[CachedTestRun] = None
        self.temp_cache_file_name: str = TEMP_CACHE_FILE_NAME

    def get_cached_test_case(
        self, test_case: LLMTestCase, hyperparameters: Union[Dict, None]
    ) -> Union[CachedTestCase, None]:
        if self.disable_write_cache or portalocker is None:
            return None

        cached_test_run = self.get_cached_test_run()
        cache_dict = {
            SingleTurnParams.INPUT.value: test_case.input,
            SingleTurnParams.ACTUAL_OUTPUT.value: test_case.actual_output,
            SingleTurnParams.EXPECTED_OUTPUT.value: test_case.expected_output,
            SingleTurnParams.CONTEXT.value: test_case.context,
            SingleTurnParams.RETRIEVAL_CONTEXT.value: (
                [
                    rc.context if hasattr(rc, "context") else rc
                    for rc in test_case.retrieval_context
                ]
                if test_case.retrieval_context
                else None
            ),
            "hyperparameters": hyperparameters,
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
        to_temp: bool = False,
    ):
        if self.disable_write_cache or portalocker is None:
            return
        cache_dict = {
            SingleTurnParams.INPUT.value: test_case.input,
            SingleTurnParams.ACTUAL_OUTPUT.value: test_case.actual_output,
            SingleTurnParams.EXPECTED_OUTPUT.value: test_case.expected_output,
            SingleTurnParams.CONTEXT.value: test_case.context,
            SingleTurnParams.RETRIEVAL_CONTEXT.value: (
                [
                    rc.context if hasattr(rc, "context") else rc
                    for rc in test_case.retrieval_context
                ]
                if test_case.retrieval_context
                else None
            ),
            "hyperparameters": hyperparameters,
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
        if self.disable_write_cache or portalocker is None:
            return

        if temp:
            self.temp_cached_test_run = cached_test_run
        else:
            self.cached_test_run = cached_test_run

    def save_cached_test_run(self, to_temp: bool = False):
        if self.disable_write_cache or portalocker is None:
            return

        if to_temp:
            try:
                with portalocker.Lock(
                    self.temp_cache_file_name, mode="w"
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
                with portalocker.Lock(self.cache_file_name, mode="w") as file:
                    self.cached_test_run = self.cached_test_run.save(file)
            except Exception as e:
                print(
                    f"In save_cached_test_run, temp={to_temp}, Error saving test run to disk {e}",
                    file=sys.stderr,
                )

    def create_cached_test_run(self, temp: bool = False):
        if self.disable_write_cache or portalocker is None:
            return

        cached_test_run = CachedTestRun()
        self.set_cached_test_run(cached_test_run, temp)
        self.save_cached_test_run(to_temp=temp)

    def get_cached_test_run(
        self, from_temp: bool = False
    ) -> Union[CachedTestRun, None]:
        if self.disable_write_cache or portalocker is None:
            return

        should_create_cached_test_run = False
        if from_temp:
            if self.temp_cached_test_run:
                return self.temp_cached_test_run

            if not os.path.exists(self.temp_cache_file_name):
                self.create_cached_test_run(temp=from_temp)

            try:
                with portalocker.Lock(
                    self.temp_cache_file_name,
                    mode="r",
                    flags=portalocker.LOCK_SH | portalocker.LOCK_NB,
                ) as file:
                    content = file.read().strip()
                    try:
                        data = json.loads(content)
                        self.temp_cached_test_run = CachedTestRun.load(data)
                    except Exception:
                        should_create_cached_test_run = True
            except portalocker.exceptions.LockException as e:
                print(
                    f"In get_cached_test_run, temp={from_temp}, Lock acquisition failed: {e}",
                    file=sys.stderr,
                )

            if should_create_cached_test_run:
                self.create_cached_test_run(temp=from_temp)

            return self.temp_cached_test_run
        else:
            if self.cached_test_run:
                return self.cached_test_run

            if not os.path.exists(self.cache_file_name):
                self.create_cached_test_run()

            try:
                with portalocker.Lock(
                    self.cache_file_name,
                    mode="r",
                    flags=portalocker.LOCK_SH | portalocker.LOCK_NB,
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
        if portalocker is None:
            return

        if self.disable_write_cache:
            # Clear cache if write cache is disabled
            delete_file_if_exists(self.cache_file_name)
            delete_file_if_exists(self.temp_cache_file_name)
            return

        self.get_cached_test_run(from_temp=True)
        try:
            with portalocker.Lock(self.cache_file_name, mode="w") as file:
                self.temp_cached_test_run = self.temp_cached_test_run.save(file)
        except Exception as e:
            print(
                f"In wrap_up_cached_test_run, Error saving test run to disk, {e}",
                file=sys.stderr,
            )
        finally:
            delete_file_if_exists(self.temp_cache_file_name)


global_test_run_cache_manager = TestRunCacheManager()

############ Helper Functions #############


CacheIdentity._METRIC_CONFIGURATION_CLASS = MetricConfiguration


class Cache(CacheIdentity):
    pass

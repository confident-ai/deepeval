import logging
import sys
import json
import os
import hashlib
import inspect
from typing import List, Optional, Dict, Union
from enum import Enum
from pydantic import BaseModel, Field

from deepeval.utils import make_model_config

from deepeval.test_case import SingleTurnParams, LLMTestCase, ToolCallParams
from deepeval.test_run.api import MetricData
from deepeval.utils import (
    delete_file_if_exists,
    is_read_only_env,
    serialize,
)
from deepeval.metrics import BaseMetric
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
    threshold: Optional[float] = None
    evaluation_model: Optional[str] = None
    strict_mode: bool = False
    criteria: Optional[str] = None
    include_reason: Optional[bool] = None
    n: Optional[int] = None

    ##### Optional fields #####
    evaluation_steps: Optional[List[str]] = None
    assessment_questions: Optional[List[str]] = None
    embeddings: Optional[str] = None
    evaluation_params: Optional[
        Union[List[SingleTurnParams], List[ToolCallParams]]
    ] = None

    ##### Cache identity fields #####
    # Fully qualified metric class name (module + qualname). Used to make
    # sure we never compare hashes across metric families.
    metric_class: Optional[str] = None
    # Deterministic SHA-256 digest over the configuration payload (for
    # built-ins) or over configuration + the metric's custom evaluation
    # method source code (for third-party metrics). When either the params
    # or the implementation changes the hash changes, so the cache entry
    # correctly invalidates.
    metric_hash: Optional[str] = None


_DEEPEVAL_METRICS_MODULE = "deepeval.metrics"
_HASH_VERSION = b"v1"


def _is_builtin_metric(metric_class: type) -> bool:
    """A metric is considered "built-in" when its implementation lives
    under the public `deepeval.metrics` package. For those we can rely on
    field-level config equality: the eval logic only changes when the
    package itself is re-installed (i.e. outside cache scope). For user
    subclasses or community metrics we need to hash the implementation
    source so edits to `measure()`/`a_measure()` immediately invalidate
    stale entries.
    """
    try:
        module = metric_class.__module__ or ""
    except AttributeError:
        return False
    return module.startswith(_DEEPEVAL_METRICS_MODULE)


def _hashable_repr(value) -> bytes:
    """
    Serialize arbitrary metric-config values into a stable byte string.

    We don't need round-tripping — only determinism across objects that
    compare equal. dicts/lists/tuples are walked recursively; `None` is
    encoded explicitly so "field absent" and "field = None" don't collide
    with "empty list". The payload is JSON-ish but we keep nesting marks
    explicit because JSON alone would collapse ``[None, [None]]`` into
    something that re-serializes the same as ``[[]]`` once fields move —
    we want ordering and nesting to affect the digest.
    """
    if value is None:
        return b"n"
    if isinstance(value, bool):
        # bools are ints; check BEFORE int so True != "1" vs "0".
        return b"t" if value else b"f"
    if isinstance(value, (int, float)):
        return ("n:" + repr(value)).encode("utf-8")
    if isinstance(value, str):
        return ("s:" + value).encode("utf-8")
    if isinstance(value, Enum):
        return ("e:" + type(value).__qualname__ + "." + value.name).encode(
            "utf-8"
        )
    if isinstance(value, bytes):
        return b"b:" + value
    if isinstance(value, (list, tuple)):
        parts = [b"L" if isinstance(value, list) else b"T"]
        for item in value:
            r = _hashable_repr(item)
            parts.append(len(r).to_bytes(4, "little"))
            parts.append(r)
        return b"".join(parts)
    if isinstance(value, dict):
        parts = [b"D"]
        # key order must be deterministic for cross-process stability.
        # Sort by str(key) to be safe (keys aren't guaranteed comparable).
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        for k, v in items:
            kr = _hashable_repr(k)
            vr = _hashable_repr(v)
            parts.append(len(kr).to_bytes(4, "little"))
            parts.append(kr)
            parts.append(len(vr).to_bytes(4, "little"))
            parts.append(vr)
        return b"".join(parts)
    if isinstance(value, BaseModel):
        # model_dump(mode="json") gives a plain dict with no pydantic
        # types leaking through, and the field order is declared order.
        try:
            return _hashable_repr(value.model_dump(mode="json"))
        except Exception:
            return _hashable_repr(dict(value))
    if hasattr(value, "__class__") and hasattr(value, "__dict__"):
        # Fall back to class name + sorted attribute dict for plain
        # objects. Values like embeddings class names (strings above)
        # won't hit this branch, but for arbitrary user-provided
        # config objects we at least don't silently ignore structure.
        return _hashable_repr(
            {
                "__class__": (
                    getattr(type(value), "__module__", ""),
                    type(value).__qualname__,
                ),
                **{
                    k: v
                    for k, v in vars(value).items()
                    if not k.startswith("_")
                },
            }
        )
    # str() is the last resort; it's better than crashing.
    return ("x:" + str(value)).encode("utf-8")


def _compute_metric_hash(
    metric: BaseMetric, configuration_payload: Dict
) -> Optional[str]:
    """Compute the deterministic SHA-256 digest for a metric instance.

    The hash covers:
      * a per-class identifier (module + qualname, not `__name__` so
        nested community classes still differ),
      * the full configuration payload (threshold, model, strict_mode,
        steps, params, criteria, ...),
      * for non-built-in metrics, the source code of
        ``measure`` and ``a_measure`` as reported by ``inspect.getsource``.

    Returns ``None`` on systems where ``inspect.getsource`` can't locate
    the source (e.g. frozen REPL / embedded eval). Callers then fall back
    to the legacy field-by-field comparison so caching still works.
    """
    metric_class = type(metric)
    class_parts = (
        getattr(metric_class, "__module__", ""),
        metric_class.__qualname__,
    )
    class_bytes = _hashable_repr(class_parts)

    payload_bytes = _hashable_repr(configuration_payload)

    builtin = _is_builtin_metric(metric_class)
    source_bytes = b""
    if not builtin:
        sources: List[bytes] = []
        for method_name in ("measure", "a_measure"):
            method = getattr(metric_class, method_name, None)
            if method is None:
                continue
            try:
                src = inspect.getsource(method)
            except (OSError, TypeError):
                # REPL / packaged .so / py-less distributions can't give us
                # source. Signal "unhashable source" by returning None from
                # the outer call so the legacy compare path kicks in.
                return None
            sources.append(method_name.encode("utf-8"))
            sources.append(src.encode("utf-8"))
        source_bytes = b"".join(sources)

    hasher = hashlib.sha256()
    hasher.update(_HASH_VERSION)
    hasher.update(b"cls")
    hasher.update(len(class_bytes).to_bytes(4, "little"))
    hasher.update(class_bytes)
    hasher.update(b"cfg")
    hasher.update(len(payload_bytes).to_bytes(4, "little"))
    hasher.update(payload_bytes)
    if not builtin:
        hasher.update(b"src")
        hasher.update(len(source_bytes).to_bytes(4, "little"))
        hasher.update(source_bytes)
    return hasher.hexdigest()


def _qualified_metric_class_name(metric_class: type) -> str:
    return (
        f"{getattr(metric_class, '__module__', '')}:{metric_class.__qualname__}"
    )


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


class Cache:
    @staticmethod
    def get_metric_data(
        metric: BaseMetric, cached_test_case: Optional[CachedTestCase]
    ) -> Optional[CachedMetricData]:
        if not cached_test_case:
            return None
        for cached_metric_data in cached_test_case.cached_metrics_data:
            if (
                cached_metric_data.metric_data.name == metric.__name__
                and Cache.same_metric_configs(
                    metric,
                    cached_metric_data.metric_configuration,
                )
            ):
                return cached_metric_data
        return None

    @staticmethod
    def same_metric_configs(
        metric: BaseMetric,
        metric_configuration: MetricConfiguration,
    ) -> bool:
        metric_class = type(metric)
        cached_hash = metric_configuration.metric_hash
        cached_class = metric_configuration.metric_class
        # --- Fast path: both sides carry the new identity fields. A
        # single hash comparison is O(1) and implicitly covers every
        # configurable field, so we never again drift between what
        # create_metric_configuration stores and what we compare.
        if cached_hash is not None and cached_class is not None:
            if cached_class != _qualified_metric_class_name(metric_class):
                return False
            # Build the exact same payload used when writing the cache so
            # hashes are bit-identical. This deliberately duplicates the
            # field set from create_metric_configuration — if the pair
            # ever drifts, the hash won't match and tests will catch it.
            config_fields = [
                "threshold",
                "evaluation_model",
                "strict_mode",
                "include_reason",
                "n",
                "criteria",
                "language",
                "embeddings",
                "evaluation_steps",
                "evaluation_params",
                "assessment_questions",
            ]
            payload = {}
            for field in config_fields:
                value = getattr(metric, field, None)
                if field == "embeddings" and value is not None:
                    value = value.__class__.__name__
                payload[field] = value
            current_hash = _compute_metric_hash(metric, payload)
            if current_hash is None:
                # Source couldn't be retrieved (REPL / packaged module).
                # Fall through to the field-level check — safe because
                # the legacy comparison at least covers parameters.
                pass
            else:
                return current_hash == cached_hash

        # --- Backward-compatible fallback for caches written before the
        # identity fields were added. Preserves the old behaviour so
        # users don't lose their whole cache on upgrade.
        config_fields = [
            "threshold",
            "evaluation_model",
            "strict_mode",
            "include_reason",
            "n",
            "language",
            "embeddings",
            "evaluation_params",
            "assessment_questions",
            "evaluation_steps",
        ]

        for field in config_fields:
            metric_value = getattr(metric, field, None)
            cached_value = getattr(metric_configuration, field, None)

            if field == "evaluation_steps":
                if metric_value is not None:
                    if metric_value == cached_value:
                        continue
                else:
                    try:
                        # For GEval only
                        if metric.criteria is not None:
                            criteria_value = getattr(metric, "criteria", None)
                            cached_criteria_value = getattr(
                                metric_configuration, "criteria", None
                            )
                            if criteria_value != cached_criteria_value:
                                return False
                            continue
                    except Exception:
                        # For non-GEval
                        continue

            if field == "embeddings" and metric_value is not None:
                metric_value = metric_value.__class__.__name__

            if metric_value != cached_value:
                return False

        return True

    @staticmethod
    def create_metric_configuration(metric: BaseMetric) -> MetricConfiguration:
        config_kwargs = {}
        config_fields = [
            "threshold",
            "evaluation_model",
            "strict_mode",
            "include_reason",  # checked
            "n",  # checked
            "criteria",  # checked
            "language",  # can't check
            "embeddings",  #
            "strict_mode",  # checked
            "evaluation_steps",  # checked
            "evaluation_params",  # checked
            "assessment_questions",  # checked
        ]
        for field in config_fields:
            value = getattr(metric, field, None)
            if field == "embeddings" and value is not None:
                value = value.__class__.__name__
            config_kwargs[field] = value

        metric_class = type(metric)
        metric_hash = _compute_metric_hash(metric, config_kwargs.copy())
        # Identity fields must NOT be part of the hashed payload (they
        # exist only to short-circuit the comparison). Inject them AFTER
        # computing the digest so `same_metric_configs` — which rebuilds
        # the payload from scratch — lands on the same byte stream.
        config_kwargs["metric_class"] = _qualified_metric_class_name(
            metric_class
        )
        config_kwargs["metric_hash"] = metric_hash

        return MetricConfiguration(**config_kwargs)

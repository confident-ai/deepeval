import hashlib
import inspect
import json
import math
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional, get_args, get_origin
from urllib.parse import parse_qsl, unquote, unquote_plus, urlsplit

from pydantic import BaseModel

from deepeval.metrics import BaseConversationalMetric, BaseMetric

try:
    from pydantic.v1 import BaseModel as LegacyBaseModel
except ImportError:
    LegacyBaseModel = None


class CacheIdentity:
    _MISSING_ATTRIBUTE = object()
    _UNSAFE_ATTRIBUTE = object()
    _METRIC_CONFIGURATION_CLASS: Any = None
    _CONFIG_FIELDS = [
        "threshold",
        "evaluation_model",
        "strict_mode",
        "criteria",
        "include_reason",
        "n",
        "language",
        "embeddings",
        "evaluation_params",
        "assessment_questions",
        "evaluation_steps",
    ]
    _CREATE_CONFIG_FIELDS = _CONFIG_FIELDS
    _FREE_FORM_CONFIG_FIELDS = {
        "criteria",
        "assessment_questions",
        "evaluation_steps",
    }
    _FINGERPRINTED_CONFIG_FIELDS = _FREE_FORM_CONFIG_FIELDS | {
        "embeddings",
        "evaluation_params",
    }
    _CONSTRUCTOR_PARAMETER_EXCLUDE_FIELDS = set(_CREATE_CONFIG_FIELDS) | {
        # Identity/name is already compared through MetricData.name.
        "name",
        # Execution/display controls do not change metric scoring semantics.
        "async_mode",
        "verbose_mode",
        # Model objects are represented by evaluation_model when relevant.
        "model",
        "using_native_model",
        # Runtime result fields must not affect cache compatibility.
        "score",
        "score_breakdown",
        "reason",
        "success",
        "error",
        "evaluation_cost",
        "input_tokens",
        "output_tokens",
        "verbose_logs",
        "skipped",
        # Internal telemetry/display toggles do not change metric scoring.
        "_track",
        "_include_g_eval_suffix",
        "_include_dag_suffix",
    }
    _SENSITIVE_PARAMETER_NAMES = {
        "api_key",
        "apikey",
        "authorization",
        "auth_header",
        "auth_headers",
        "bearer",
        "headers",
        "password",
        "passwd",
        "pwd",
        "token",
        "cookie",
        "private_key",
        "client_cert",
        "session_id",
        "connection_string",
        "conn_str",
        "dsn",
        "database_url",
        "broker_url",
        "jdbc_url",
        "redis_url",
    }
    _SENSITIVE_PARAMETER_PARTS = {
        "auth",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "credentials",
        "signature",
        "sig",
        "password",
        "secret",
        "token",
    }
    _SENSITIVE_PARAMETER_SUFFIXES = (
        "_api_key",
        "_access_key",
        "_access_key_id",
        "_secret_key",
        "_access_token",
        "_refresh_token",
        "_id_token",
        "_private_key",
        "_client_cert",
        "_session_id",
    )
    _SENSITIVE_PARAMETER_COMPACT_NAMES = {
        "apikey",
        "accesskey",
        "accesskeyid",
        "secretkey",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "authtoken",
        "sessiontoken",
        "clientsecret",
        "privatekey",
        "clientcert",
        "sessionid",
        "connectionstring",
        "connstr",
        "dsn",
        "databaseurl",
        "brokerurl",
        "jdbcurl",
        "redisurl",
        "token",
        "secret",
        "credential",
        "credentials",
        "signature",
        "sig",
        "password",
        "jwt",
        "pat",
        "key",
    }
    _SENSITIVE_PARAMETER_COMPACT_SUFFIXES = tuple(
        sorted(
            _SENSITIVE_PARAMETER_COMPACT_NAMES - {"key"},
            key=len,
            reverse=True,
        )
    )
    _PYDANTIC_FREE_FORM_SCHEMA_METADATA_FIELDS = (
        "title",
        "description",
        "examples",
        "json_schema_extra",
        "discriminator",
    )
    _PYDANTIC_LEGACY_CONFIG_FIELDS = (
        "allow_mutation",
        "allow_population_by_field_name",
        "anystr_lower",
        "anystr_strip_whitespace",
        "anystr_upper",
        "arbitrary_types_allowed",
        "extra",
        "frozen",
        "max_anystr_length",
        "min_anystr_length",
        "orm_mode",
        "smart_union",
        "use_enum_values",
        "validate_all",
        "validate_assignment",
    )
    _NATIVE_MODEL_CACHE_IDENTITY_FIELDS = (
        "name",
        "model_name",
        "base_url",
        "deployment_name",
        "api_version",
        "temperature",
        "max_tokens",
        "_max_tokens",
        "generation_kwargs",
        "kwargs",
        "async_http_client",
        "model_data",
        "project",
        "location",
        "region",
        "use_vertexai",
        "model_safety_settings",
    )
    _KNOWN_DAG_NODE_TYPES = {
        "deepeval.metrics.dag.nodes.VerdictNode",
        "deepeval.metrics.dag.nodes.TaskNode",
        "deepeval.metrics.dag.nodes.BinaryJudgementNode",
        "deepeval.metrics.dag.nodes.NonBinaryJudgementNode",
        "deepeval.metrics.conversational_dag.nodes.ConversationalVerdictNode",
        "deepeval.metrics.conversational_dag.nodes.ConversationalTaskNode",
        "deepeval.metrics.conversational_dag.nodes.ConversationalBinaryJudgementNode",
        "deepeval.metrics.conversational_dag.nodes.ConversationalNonBinaryJudgementNode",
    }
    _MAX_CACHE_PARAMETER_DEPTH = 24
    _MAX_CACHE_PARAMETER_ITEMS = 2048
    _MAX_CACHE_PARAMETER_COLLECTION_ITEMS = 256
    _MAX_CACHE_PARAMETER_STRING_LENGTH = 8192
    _MAX_URL_DECODE_PASSES = 4
    _MAX_DAG_CACHE_NODES = 512
    _UNCACHEABLE_CACHE_VALUE = "<uncacheable>"

    @staticmethod
    def get_metric_data(metric: BaseMetric, cached_test_case: Any) -> Any:
        if not cached_test_case:
            return None
        for cached_metric_data in cached_test_case.cached_metrics_data:
            if (
                cached_metric_data.metric_data.name == metric.__name__
                and CacheIdentity.same_metric_configs(
                    metric,
                    cached_metric_data.metric_configuration,
                )
            ):
                return cached_metric_data
        return None

    @staticmethod
    def same_metric_configs(
        metric: BaseMetric,
        metric_configuration: Any,
    ) -> bool:
        for field in CacheIdentity._CONFIG_FIELDS:
            metric_value = CacheIdentity._cache_config_field_value(
                metric, field
            )
            cached_value = getattr(metric_configuration, field, None)

            if (
                metric_value == CacheIdentity._UNCACHEABLE_CACHE_VALUE
                or cached_value == CacheIdentity._UNCACHEABLE_CACHE_VALUE
            ):
                return False

            # Preserve GEval's legacy criteria fallback when
            # evaluation_steps is absent.
            if field == "evaluation_steps":
                if metric_value is not None:
                    if metric_value == cached_value:
                        continue
                else:
                    try:
                        # For GEval only
                        criteria_value = (
                            CacheIdentity._cache_config_field_value(
                                metric, "criteria"
                            )
                        )
                        if criteria_value is not None:
                            cached_criteria_value = getattr(
                                metric_configuration, "criteria", None
                            )
                            if (
                                criteria_value
                                == CacheIdentity._UNCACHEABLE_CACHE_VALUE
                                or cached_criteria_value
                                == CacheIdentity._UNCACHEABLE_CACHE_VALUE
                            ):
                                return False
                            if criteria_value != cached_criteria_value:
                                return False
                            continue
                    except Exception:
                        # For non-GEval
                        continue

            if metric_value != cached_value:
                return False

        constructor_parameters = CacheIdentity._constructor_parameters(metric)
        cached_parameters = metric_configuration.custom_parameters
        if CacheIdentity._has_uncacheable_parameter(
            constructor_parameters
        ) or CacheIdentity._has_uncacheable_parameter(cached_parameters):
            return False

        if constructor_parameters != cached_parameters:
            # Older cache files do not have constructor fingerprints. Treat
            # those entries as misses for constructor-sensitive metrics rather
            # than reusing a potentially stale custom/built-in metric result.
            return False

        return True

    @staticmethod
    def create_metric_configuration(metric: BaseMetric) -> Any:
        config_kwargs = {}
        for field in CacheIdentity._CREATE_CONFIG_FIELDS:
            config_kwargs[field] = CacheIdentity._cache_config_field_value(
                metric, field
            )

        config_kwargs["custom_parameters"] = (
            CacheIdentity._constructor_parameters(metric)
        )

        try:
            return CacheIdentity._METRIC_CONFIGURATION_CLASS(**config_kwargs)
        except Exception:
            # Invalid or hostile config values should not abort cache writes or
            # lookups after a metric run. Store an uncacheable configuration so
            # comparisons miss closed instead.
            return CacheIdentity._METRIC_CONFIGURATION_CLASS(
                threshold=CacheIdentity._UNCACHEABLE_CACHE_VALUE,
                custom_parameters={
                    "__configuration__": CacheIdentity._UNCACHEABLE_CACHE_VALUE
                },
            )

    @staticmethod
    def _cache_config_field_value(metric: BaseMetric, field: str) -> Any:
        value = CacheIdentity._passive_metric_config_field_value(metric, field)
        if value is CacheIdentity._UNSAFE_ATTRIBUTE:
            return CacheIdentity._UNCACHEABLE_CACHE_VALUE
        if value is CacheIdentity._MISSING_ATTRIBUTE:
            if field == "threshold":
                return CacheIdentity._UNCACHEABLE_CACHE_VALUE
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return CacheIdentity._UNCACHEABLE_CACHE_VALUE
        if field == "threshold":
            if type(value) in (int, float):
                return value
            return CacheIdentity._UNCACHEABLE_CACHE_VALUE
        if field in ("strict_mode", "include_reason"):
            if value is None or type(value) is bool:
                return value
            return CacheIdentity._UNCACHEABLE_CACHE_VALUE
        if field == "n":
            if value is None or type(value) is int:
                return value
            return CacheIdentity._UNCACHEABLE_CACHE_VALUE
        if field == "language":
            if value is None:
                return None
            if isinstance(value, str) and not (
                CacheIdentity._is_sensitive_parameter_value(field, value)
            ):
                return value
            return CacheIdentity._UNCACHEABLE_CACHE_VALUE
        if field in CacheIdentity._FINGERPRINTED_CONFIG_FIELDS:
            if value is None:
                return None
            return CacheIdentity._cache_parameter_fingerprint(field, value)
        if field == "evaluation_model":
            model_name = CacheIdentity._metric_model_name(metric)
            if value is not None and not isinstance(value, str):
                return CacheIdentity._UNCACHEABLE_CACHE_VALUE
            if model_name is not None and value == model_name:
                # Derived string model identifiers are represented by
                # custom_parameters["model"].
                return None
            if (
                value is not None
                and CacheIdentity._metric_has_non_string_model(metric)
                and not CacheIdentity._metric_uses_native_model(metric)
            ):
                return CacheIdentity._UNCACHEABLE_CACHE_VALUE
            if (
                value is not None
                and CacheIdentity._is_sensitive_parameter_value(field, value)
            ):
                return CacheIdentity._UNCACHEABLE_CACHE_VALUE
        return value

    @staticmethod
    def _passive_metric_config_field_value(
        metric: BaseMetric, field: str
    ) -> Any:
        instance_fields = CacheIdentity._passive_instance_fields(metric)
        if instance_fields is None:
            return CacheIdentity._UNSAFE_ATTRIBUTE
        if field in instance_fields:
            return instance_fields[field]

        metric_type = type(metric)
        for cls in metric_type.__mro__:
            class_fields = CacheIdentity._passive_class_attribute(
                cls, "__dict__"
            )
            if not isinstance(class_fields, MappingProxyType):
                continue
            if field not in class_fields:
                continue
            value = class_fields[field]
            if hasattr(type(value), "__get__"):
                return CacheIdentity._UNSAFE_ATTRIBUTE
            return value

        return CacheIdentity._MISSING_ATTRIBUTE

    @staticmethod
    def _constructor_parameters(
        metric: BaseMetric,
    ) -> Optional[Dict[str, str]]:
        if type(metric).__init__ is object.__init__:
            return None

        try:
            signature = inspect.signature(type(metric).__init__)
        except (TypeError, ValueError):
            return {"__signature__": CacheIdentity._UNCACHEABLE_CACHE_VALUE}

        instance_fields = CacheIdentity._passive_instance_fields(metric)
        if instance_fields is None:
            return CacheIdentity._uncacheable_constructor_parameters(signature)

        parameters: Dict[str, str] = {}
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                parameters[name] = CacheIdentity._UNCACHEABLE_CACHE_VALUE
                continue
            if name == "model":
                # Model objects can share a deployment/name while differing in
                # behavior. Only string model identifiers are safe to fingerprint;
                # opaque model objects miss closed.
                if name in instance_fields:
                    value = instance_fields[name]
                elif f"_{name}" in instance_fields:
                    value = instance_fields[f"_{name}"]
                else:
                    parameters[name] = CacheIdentity._UNCACHEABLE_CACHE_VALUE
                    continue
                if not isinstance(value, str):
                    model_identity = CacheIdentity._native_model_cache_identity(
                        value
                    )
                    if (
                        CacheIdentity._metric_uses_native_model(metric)
                        and model_identity is not None
                    ):
                        # Real DeepEval metrics commonly initialize a native
                        # model object from a string/default constructor value.
                        # Fingerprint a small passive allowlist of behavior-
                        # relevant model fields so identical model names with
                        # different temperature/base URL/generation kwargs do
                        # not share stale cache entries.
                        parameters[name] = (
                            CacheIdentity._cache_parameter_fingerprint(
                                name, model_identity
                            )
                        )
                        continue
                    parameters[name] = CacheIdentity._UNCACHEABLE_CACHE_VALUE
                    continue
                parameters[name] = CacheIdentity._cache_parameter_fingerprint(
                    name, value
                )
                continue
            if name in CacheIdentity._CONSTRUCTOR_PARAMETER_EXCLUDE_FIELDS:
                continue
            if name in instance_fields:
                value = instance_fields[name]
            elif f"_{name}" in instance_fields:
                value = instance_fields[f"_{name}"]
            else:
                parameters[name] = CacheIdentity._UNCACHEABLE_CACHE_VALUE
                continue

            parameters[name] = CacheIdentity._cache_parameter_fingerprint(
                name, value
            )

        return parameters or None

    @staticmethod
    def _uncacheable_constructor_parameters(
        signature: inspect.Signature,
    ) -> Optional[Dict[str, str]]:
        parameters: Dict[str, str] = {}
        for name in signature.parameters:
            if name == "self":
                continue
            if name == "model":
                parameters[name] = CacheIdentity._UNCACHEABLE_CACHE_VALUE
                continue
            if name in CacheIdentity._CONSTRUCTOR_PARAMETER_EXCLUDE_FIELDS:
                continue
            parameters[name] = CacheIdentity._UNCACHEABLE_CACHE_VALUE
        return parameters or None

    @staticmethod
    def _cache_parameter_fingerprint(name: str, value: Any) -> str:
        normalized = CacheIdentity._normalize_cache_parameter(
            name, value, set()
        )
        if CacheIdentity._contains_uncacheable_parameter(normalized):
            return CacheIdentity._UNCACHEABLE_CACHE_VALUE
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _normalize_cache_parameter(
        name: str,
        value: Any,
        seen: set[int],
        depth: int = 0,
        budget: Optional[Dict[str, int]] = None,
    ) -> Any:
        if budget is None:
            budget = {"items": 0}
        budget["items"] += 1
        if (
            depth > CacheIdentity._MAX_CACHE_PARAMETER_DEPTH
            or budget["items"] > CacheIdentity._MAX_CACHE_PARAMETER_ITEMS
        ):
            return CacheIdentity._uncacheable_parameter(
                "budget-exceeded", value
            )

        if CacheIdentity._is_sensitive_parameter_name(
            name
        ) and not CacheIdentity._should_descend_sensitive_parameter(
            name, value
        ):
            return CacheIdentity._uncacheable_parameter("sensitive")

        if value is None:
            return value

        value_type = type(value)

        if issubclass(value_type, Enum):
            return {
                "__type__": CacheIdentity._type_name(value_type),
                "__value__": CacheIdentity._normalize_cache_parameter(
                    name, value.value, seen, depth + 1, budget
                ),
            }

        if issubclass(value_type, (int, bool)):
            return value

        if issubclass(value_type, str):
            if len(value) > CacheIdentity._MAX_CACHE_PARAMETER_STRING_LENGTH:
                return CacheIdentity._uncacheable_parameter(
                    "budget-exceeded", value
                )
            if CacheIdentity._is_sensitive_parameter_value(
                name, value
            ) and not CacheIdentity._should_descend_sensitive_parameter(
                name, value
            ):
                return CacheIdentity._uncacheable_parameter("sensitive")
            return value

        if issubclass(value_type, float):
            if not math.isfinite(value):
                return CacheIdentity._uncacheable_parameter(
                    "non-finite-float", value
                )
            return value

        if issubclass(value_type, Path):
            path_value = str(value)
            if (
                len(path_value)
                > CacheIdentity._MAX_CACHE_PARAMETER_STRING_LENGTH
            ):
                return CacheIdentity._uncacheable_parameter(
                    "budget-exceeded", value
                )
            if CacheIdentity._is_sensitive_parameter_value(
                name, path_value
            ) and not CacheIdentity._should_descend_sensitive_parameter(
                name, value
            ):
                return CacheIdentity._uncacheable_parameter("sensitive")
            return {
                "__type__": CacheIdentity._type_name(value_type),
                "__value__": path_value,
            }

        value_id = id(value)
        if value_id in seen:
            return CacheIdentity._uncacheable_parameter("circular", value)

        if value_type is dict:
            if len(value) > CacheIdentity._MAX_CACHE_PARAMETER_COLLECTION_ITEMS:
                return CacheIdentity._uncacheable_parameter(
                    "budget-exceeded", value
                )
            if any(not isinstance(key, str) for key in value):
                return CacheIdentity._uncacheable_parameter(
                    "non-string-mapping-key", value
                )
            seen.add(value_id)
            try:
                return {
                    key: CacheIdentity._normalize_cache_parameter(
                        key, item, seen, depth + 1, budget
                    )
                    for key, item in sorted(value.items())
                }
            finally:
                seen.remove(value_id)

        if value_type in (list, tuple):
            if len(value) > CacheIdentity._MAX_CACHE_PARAMETER_COLLECTION_ITEMS:
                return CacheIdentity._uncacheable_parameter(
                    "budget-exceeded", value
                )
            seen.add(value_id)
            try:
                return {
                    "__type__": CacheIdentity._type_name(value_type),
                    "__items__": [
                        CacheIdentity._normalize_cache_parameter(
                            name, item, seen, depth + 1, budget
                        )
                        for item in value
                    ],
                }
            finally:
                seen.remove(value_id)

        if value_type in (set, frozenset):
            if len(value) > CacheIdentity._MAX_CACHE_PARAMETER_COLLECTION_ITEMS:
                return CacheIdentity._uncacheable_parameter(
                    "budget-exceeded", value
                )
            seen.add(value_id)
            try:
                normalized_items = [
                    CacheIdentity._normalize_cache_parameter(
                        name, item, seen, depth + 1, budget
                    )
                    for item in value
                ]
                return {
                    "__type__": CacheIdentity._type_name(value_type),
                    "__items__": sorted(
                        normalized_items,
                        key=lambda item: json.dumps(
                            item, sort_keys=True, ensure_ascii=True
                        ),
                    ),
                }
            finally:
                seen.remove(value_id)

        if CacheIdentity._is_pydantic_model_class(value_type):
            seen.add(value_id)
            try:
                if (
                    CacheIdentity._pydantic_model_has_custom_getattribute(
                        value_type
                    )
                    or CacheIdentity._pydantic_model_has_private_attrs(
                        value_type
                    )
                    or CacheIdentity._pydantic_instance_has_private_values(
                        value
                    )
                ):
                    return CacheIdentity._uncacheable_parameter(
                        "pydantic-private", value
                    )
                return {
                    "__type__": CacheIdentity._type_name(value_type),
                    "__schema__": CacheIdentity._safe_pydantic_model_schema(
                        value_type
                    ),
                    "__fields__": CacheIdentity._normalize_cache_parameter(
                        name,
                        CacheIdentity._safe_pydantic_field_values(value),
                        seen,
                        depth + 1,
                        budget,
                    ),
                }
            except Exception:
                return CacheIdentity._uncacheable_parameter("opaque", value)
            finally:
                seen.remove(value_id)

        if is_dataclass(value_type) and not issubclass(value_type, type):
            seen.add(value_id)
            try:
                return {
                    "__type__": CacheIdentity._type_name(value_type),
                    "__fields__": {
                        field.name: CacheIdentity._normalize_cache_parameter(
                            field.name,
                            CacheIdentity._safe_dataclass_field_value(
                                value, field.name
                            ),
                            seen,
                            depth + 1,
                            budget,
                        )
                        for field in fields(value)
                    },
                }
            except Exception:
                return CacheIdentity._uncacheable_parameter("opaque", value)
            finally:
                seen.remove(value_id)

        if issubclass(value_type, type):
            if CacheIdentity._is_pydantic_model_class(value):
                if CacheIdentity._pydantic_model_has_custom_getattribute(
                    value
                ) or CacheIdentity._pydantic_model_has_private_attrs(value):
                    return CacheIdentity._uncacheable_parameter(
                        "pydantic-private", value
                    )
                return {
                    "__type__": CacheIdentity._type_name(value),
                    "__fields__": CacheIdentity._safe_pydantic_model_schema(
                        value
                    ),
                }
            return {"__type__": CacheIdentity._type_name(value)}

        if CacheIdentity._is_deep_acyclic_graph(value):
            seen.add(value_id)
            try:
                return {
                    "__type__": CacheIdentity._type_name(type(value)),
                    "__graph__": CacheIdentity._canonicalize_dag(value),
                }
            except Exception:
                return CacheIdentity._uncacheable_parameter("opaque", value)
            finally:
                seen.remove(value_id)

        # Avoid serializing arbitrary object graphs, reprs, or __dict__ values
        # into the cache. Those can contain secrets, clients, sessions, or
        # large runtime state. Unsupported objects are marked uncacheable so
        # cache lookup misses closed instead of colliding by type alone.
        return CacheIdentity._uncacheable_parameter("opaque", value)

    @staticmethod
    def _safe_pydantic_field_values(value: BaseModel) -> Any:
        raw_values = CacheIdentity._passive_instance_fields(value)
        if raw_values is None:
            return CacheIdentity._uncacheable_parameter(
                "pydantic-fields", value
            )
        fields_map = CacheIdentity._pydantic_fields_map(type(value))
        if fields_map is None:
            return CacheIdentity._uncacheable_parameter(
                "pydantic-fields", value
            )
        field_names = sorted(fields_map)
        values: Dict[str, Any] = {}
        for field_name, field in sorted(
            fields_map.items(), key=lambda item: item[0]
        ):
            if field_name not in raw_values:
                continue
            cache_name = CacheIdentity._pydantic_default_parameter_name(
                field_name, field
            )
            if cache_name is None:
                values[field_name] = CacheIdentity._uncacheable_parameter(
                    "pydantic-alias", raw_values[field_name]
                )
                continue
            if cache_name in values:
                values[cache_name] = CacheIdentity._uncacheable_parameter(
                    "pydantic-alias", raw_values[field_name]
                )
                continue
            values[cache_name] = raw_values[field_name]

        extra_values = CacheIdentity._passive_object_attribute(
            value, "__pydantic_extra__"
        )
        if extra_values is CacheIdentity._UNSAFE_ATTRIBUTE:
            return CacheIdentity._uncacheable_parameter("pydantic-extra", value)
        if type(extra_values) is dict:
            if any(type(key) is not str for key in extra_values):
                return CacheIdentity._uncacheable_parameter(
                    "pydantic-extra", value
                )
            values.update(extra_values)
        elif (
            extra_values is not None
            and extra_values is not CacheIdentity._MISSING_ATTRIBUTE
        ):
            return CacheIdentity._uncacheable_parameter("pydantic-extra", value)

        if not values and not field_names:
            values.update(
                {
                    key: item
                    for key, item in raw_values.items()
                    if not key.startswith("_")
                }
            )

        return values

    @staticmethod
    def _pydantic_field_names(model_cls: type) -> List[str]:
        fields_map = CacheIdentity._pydantic_fields_map(model_cls)
        if fields_map is None:
            return []
        return sorted(fields_map)

    @staticmethod
    def _pydantic_fields_map(model_cls: type) -> Optional[Dict[Any, Any]]:
        model_fields = CacheIdentity._passive_class_attribute(
            model_cls, "model_fields"
        )
        if type(model_fields) is dict:
            if any(type(key) is not str for key in model_fields):
                return None
            return model_fields
        if model_fields is CacheIdentity._UNSAFE_ATTRIBUTE:
            return None
        if (
            model_fields is not None
            and model_fields is not CacheIdentity._MISSING_ATTRIBUTE
        ):
            return None

        legacy_fields = CacheIdentity._passive_class_attribute(
            model_cls, "__fields__"
        )
        if type(legacy_fields) is dict:
            if any(type(key) is not str for key in legacy_fields):
                return None
            return legacy_fields
        if legacy_fields is CacheIdentity._UNSAFE_ATTRIBUTE:
            return None
        if (
            legacy_fields is not None
            and legacy_fields is not CacheIdentity._MISSING_ATTRIBUTE
        ):
            return None

        return {}

    @staticmethod
    def _safe_pydantic_model_schema(model_cls: type) -> Dict[str, Any]:
        fields_map = CacheIdentity._pydantic_fields_map(model_cls)
        if fields_map is None:
            return CacheIdentity._uncacheable_parameter(
                "pydantic-fields", model_cls
            )
        schema: Dict[str, Any] = {
            "fields": {
                str(name): CacheIdentity._safe_pydantic_field_schema(
                    name, field
                )
                for name, field in sorted(
                    fields_map.items(), key=lambda item: item[0]
                )
            },
            "config": CacheIdentity._safe_pydantic_model_config(model_cls),
        }
        if CacheIdentity._pydantic_model_has_validators(model_cls):
            schema["validators"] = CacheIdentity._uncacheable_parameter(
                "pydantic-validator", model_cls
            )
        return schema

    @staticmethod
    def _safe_pydantic_model_config(model_cls: type) -> Any:
        config: Dict[str, Any] = {}
        model_config = CacheIdentity._passive_class_attribute(
            model_cls, "model_config"
        )
        if type(model_config) is dict:
            config.update(model_config)
        elif model_config is CacheIdentity._UNSAFE_ATTRIBUTE:
            return CacheIdentity._uncacheable_parameter(
                "pydantic-model-config", model_cls
            )
        elif (
            model_config is not None
            and model_config is not CacheIdentity._MISSING_ATTRIBUTE
        ):
            return CacheIdentity._uncacheable_parameter(
                "pydantic-model-config", model_config
            )

        legacy_config = CacheIdentity._passive_class_attribute(
            model_cls, "__config__"
        )
        if legacy_config is CacheIdentity._UNSAFE_ATTRIBUTE:
            return CacheIdentity._uncacheable_parameter(
                "pydantic-model-config", model_cls
            )
        if (
            legacy_config is not None
            and legacy_config is not CacheIdentity._MISSING_ATTRIBUTE
        ):
            legacy_values = {}
            legacy_config_fields = CacheIdentity._passive_class_attribute(
                legacy_config, "__dict__"
            )
            if not isinstance(legacy_config_fields, MappingProxyType):
                return CacheIdentity._uncacheable_parameter(
                    "pydantic-model-config", legacy_config
                )
            for field in CacheIdentity._PYDANTIC_LEGACY_CONFIG_FIELDS:
                if field in legacy_config_fields:
                    legacy_values[field] = legacy_config_fields[field]
            if legacy_values:
                config["__legacy_config__"] = legacy_values

        return CacheIdentity._normalize_cache_parameter(
            "model_config", config, set()
        )

    @staticmethod
    def _pydantic_model_has_validators(model_cls: type) -> bool:
        decorators = CacheIdentity._passive_class_attribute(
            model_cls, "__pydantic_decorators__"
        )
        if decorators is CacheIdentity._UNSAFE_ATTRIBUTE:
            return True
        if (
            decorators is not None
            and decorators is not CacheIdentity._MISSING_ATTRIBUTE
        ):
            for attr in (
                "validators",
                "field_validators",
                "root_validators",
                "model_validators",
            ):
                value = CacheIdentity._passive_object_attribute(
                    decorators, attr
                )
                if CacheIdentity._safe_truthy_builtin(value):
                    return True

        for attr in (
            "__validators__",
            "__pre_root_validators__",
            "__post_root_validators__",
        ):
            legacy_validators = CacheIdentity._passive_class_attribute(
                model_cls, attr
            )
            if CacheIdentity._safe_truthy_builtin(legacy_validators):
                return True
        return False

    @staticmethod
    def _pydantic_model_has_private_attrs(model_cls: type) -> bool:
        private_attrs = CacheIdentity._passive_class_attribute(
            model_cls, "__private_attributes__"
        )
        if private_attrs is CacheIdentity._UNSAFE_ATTRIBUTE:
            return True
        if type(private_attrs) is dict:
            return bool(private_attrs)
        return (
            private_attrs is not None
            and private_attrs is not CacheIdentity._MISSING_ATTRIBUTE
        )

    @staticmethod
    def _pydantic_instance_has_private_values(value: BaseModel) -> bool:
        private_values = CacheIdentity._passive_object_attribute(
            value, "__pydantic_private__"
        )
        if private_values is CacheIdentity._UNSAFE_ATTRIBUTE:
            return True
        if type(private_values) is dict:
            return bool(private_values)
        return (
            private_values is not None
            and private_values is not CacheIdentity._MISSING_ATTRIBUTE
        )

    @staticmethod
    def _pydantic_model_has_custom_getattribute(model_cls: type) -> bool:
        getattribute = CacheIdentity._passive_class_attribute(
            model_cls, "__getattribute__"
        )
        legacy_getattribute = (
            getattr(LegacyBaseModel, "__getattribute__", None)
            if LegacyBaseModel is not None
            else None
        )
        return (
            getattribute is not object.__getattribute__
            and getattribute is not BaseModel.__getattribute__
            and getattribute is not legacy_getattribute
            and getattribute is not CacheIdentity._MISSING_ATTRIBUTE
            and getattribute is not CacheIdentity._UNSAFE_ATTRIBUTE
        )

    @staticmethod
    def _is_safe_pydantic_field_object(field: Any) -> bool:
        return CacheIdentity._type_name(type(field)) in {
            "pydantic.fields.FieldInfo",
            "pydantic.v1.fields.ModelField",
        }

    @staticmethod
    def _pydantic_field_attribute(
        field: Any,
        name: str,
        default: Any = None,
    ) -> Any:
        value = CacheIdentity._passive_object_attribute(field, name)
        if value is CacheIdentity._MISSING_ATTRIBUTE:
            return default
        return value

    @staticmethod
    def _pydantic_field_annotation(field: Any) -> Any:
        for name in ("annotation", "outer_type_", "type_"):
            value = CacheIdentity._pydantic_field_attribute(
                field, name, CacheIdentity._MISSING_ATTRIBUTE
            )
            if value is CacheIdentity._UNSAFE_ATTRIBUTE:
                return CacheIdentity._UNSAFE_ATTRIBUTE
            if value is not CacheIdentity._MISSING_ATTRIBUTE:
                return value
        return None

    @staticmethod
    def _safe_pydantic_field_schema(field_name: str, field: Any) -> Any:
        if not CacheIdentity._is_safe_pydantic_field_object(field):
            return CacheIdentity._uncacheable_parameter("pydantic-field", field)
        default = CacheIdentity._pydantic_field_attribute(
            field, "default", None
        )
        default_factory = CacheIdentity._pydantic_field_attribute(
            field, "default_factory", None
        )
        metadata = CacheIdentity._pydantic_field_attribute(
            field, "metadata", None
        )
        if metadata is None:
            field_info = CacheIdentity._pydantic_field_attribute(
                field, "field_info", None
            )
            metadata = CacheIdentity._pydantic_field_attribute(
                field_info, "metadata", None
            )
        if (
            default is CacheIdentity._UNSAFE_ATTRIBUTE
            or default_factory is CacheIdentity._UNSAFE_ATTRIBUTE
            or metadata is CacheIdentity._UNSAFE_ATTRIBUTE
        ):
            return CacheIdentity._uncacheable_parameter("pydantic-field", field)
        annotation = CacheIdentity._pydantic_field_annotation(field)
        if annotation is CacheIdentity._UNSAFE_ATTRIBUTE:
            return CacheIdentity._uncacheable_parameter("pydantic-field", field)

        return {
            "annotation": CacheIdentity._annotation_fingerprint(annotation),
            "alias": CacheIdentity._pydantic_alias_identity(
                CacheIdentity._pydantic_field_attribute(field, "alias", None)
            ),
            "validation_alias": CacheIdentity._pydantic_alias_identity(
                CacheIdentity._pydantic_field_attribute(
                    field, "validation_alias", None
                )
            ),
            "serialization_alias": CacheIdentity._pydantic_alias_identity(
                CacheIdentity._pydantic_field_attribute(
                    field, "serialization_alias", None
                )
            ),
            "required": CacheIdentity._pydantic_field_is_required(field),
            "default": CacheIdentity._safe_schema_default(
                CacheIdentity._pydantic_default_parameter_name(
                    field_name, field
                ),
                default,
            ),
            "default_factory": CacheIdentity._callable_fingerprint(
                default_factory
            ),
            "metadata": CacheIdentity._safe_pydantic_field_metadata(metadata),
            "json_schema": CacheIdentity._safe_pydantic_json_schema_metadata(
                field
            ),
        }

    @staticmethod
    def _safe_pydantic_field_metadata(metadata: Any) -> Any:
        if metadata is None:
            return []
        if type(metadata) not in (list, tuple):
            return CacheIdentity._uncacheable_parameter(
                "pydantic-field-metadata", metadata
            )
        metadata_items = list(metadata)
        if not metadata_items:
            return []
        return CacheIdentity._uncacheable_parameter(
            "pydantic-field-metadata", metadata_items
        )

    @staticmethod
    def _safe_pydantic_json_schema_metadata(field: Any) -> Dict[str, Any]:
        metadata = {
            "deprecated": CacheIdentity._normalize_cache_parameter(
                "deprecated",
                CacheIdentity._pydantic_field_attribute(
                    field, "deprecated", None
                ),
                set(),
            )
        }
        for name in CacheIdentity._PYDANTIC_FREE_FORM_SCHEMA_METADATA_FIELDS:
            value = CacheIdentity._pydantic_field_attribute(field, name, None)
            if value is None:
                metadata[name] = None
            else:
                # Free-form schema metadata can contain prompts, examples, or
                # secrets under generic names. Do not persist deterministic
                # fingerprints for it; miss closed instead.
                metadata[name] = CacheIdentity._uncacheable_parameter(
                    "pydantic-json-schema-metadata", value
                )
        return metadata

    @staticmethod
    def _pydantic_field_is_required(field: Any) -> Any:
        required = CacheIdentity._pydantic_field_attribute(
            field, "required", CacheIdentity._MISSING_ATTRIBUTE
        )
        if isinstance(required, bool):
            return required
        if (
            required is not None
            and required is not CacheIdentity._MISSING_ATTRIBUTE
            and required is not CacheIdentity._UNSAFE_ATTRIBUTE
        ):
            return CacheIdentity._uncacheable_parameter(
                "pydantic-required", field
            )
        default = CacheIdentity._pydantic_field_attribute(
            field, "default", CacheIdentity._MISSING_ATTRIBUTE
        )
        default_factory = CacheIdentity._pydantic_field_attribute(
            field, "default_factory", None
        )
        if (
            default is CacheIdentity._UNSAFE_ATTRIBUTE
            or default_factory is CacheIdentity._UNSAFE_ATTRIBUTE
        ):
            return CacheIdentity._uncacheable_parameter(
                "pydantic-required", field
            )
        return (
            CacheIdentity._is_pydantic_undefined(default)
            and default_factory is None
        )

    @staticmethod
    def _pydantic_default_parameter_name(
        field_name: str, field: Any
    ) -> Optional[str]:
        unknown_alias = False
        candidates: List[str] = [field_name]
        for alias in (
            CacheIdentity._pydantic_field_attribute(field, "alias", None),
            CacheIdentity._pydantic_field_attribute(
                field, "validation_alias", None
            ),
            CacheIdentity._pydantic_field_attribute(
                field, "serialization_alias", None
            ),
        ):
            if alias is CacheIdentity._UNSAFE_ATTRIBUTE:
                unknown_alias = True
                continue
            alias_candidates = CacheIdentity._pydantic_alias_candidates(alias)
            if alias is not None and not alias_candidates:
                unknown_alias = True
            candidates.extend(alias_candidates)

        for candidate in candidates:
            if CacheIdentity._is_sensitive_parameter_name(candidate):
                return candidate
        if unknown_alias:
            return None
        return field_name

    @staticmethod
    def _pydantic_alias_candidates(alias: Any) -> List[str]:
        if alias is None:
            return []
        if isinstance(alias, str):
            return [alias]
        if isinstance(alias, (list, tuple)):
            candidates: List[str] = []
            for item in alias:
                candidates.extend(
                    CacheIdentity._pydantic_alias_candidates(item)
                )
            return candidates

        for attribute in ("choices", "path"):
            value = CacheIdentity._pydantic_alias_attribute(alias, attribute)
            if (
                value is CacheIdentity._UNSAFE_ATTRIBUTE
                or value is CacheIdentity._MISSING_ATTRIBUTE
            ):
                continue
            if value is not None:
                return CacheIdentity._pydantic_alias_candidates(value)

        return []

    @staticmethod
    def _pydantic_alias_identity(alias: Any) -> Any:
        if alias is None:
            return None
        if isinstance(alias, str):
            return alias
        if isinstance(alias, (list, tuple)):
            return [
                CacheIdentity._pydantic_alias_identity(item) for item in alias
            ]

        for attribute in ("choices", "path"):
            value = CacheIdentity._pydantic_alias_attribute(alias, attribute)
            if (
                value is CacheIdentity._UNSAFE_ATTRIBUTE
                or value is CacheIdentity._MISSING_ATTRIBUTE
            ):
                continue
            if value is not None:
                return {
                    attribute: CacheIdentity._pydantic_alias_identity(value)
                }

        return CacheIdentity._uncacheable_parameter("pydantic-alias", alias)

    @staticmethod
    def _pydantic_alias_attribute(alias: Any, attribute: str) -> Any:
        alias_fields = CacheIdentity._passive_instance_fields(alias)
        if type(alias_fields) is dict:
            return alias_fields.get(attribute)

        type_name = CacheIdentity._type_name(type(alias))
        if (
            attribute == "choices"
            and type_name == "pydantic.aliases.AliasChoices"
        ) or (
            attribute == "path" and type_name == "pydantic.aliases.AliasPath"
        ):
            return CacheIdentity._passive_object_attribute(alias, attribute)

        return CacheIdentity._UNSAFE_ATTRIBUTE

    @staticmethod
    def _safe_schema_default(name: Optional[str], default: Any) -> Any:
        if CacheIdentity._is_pydantic_undefined(default):
            return {"__missing__": True}
        if name is None:
            return CacheIdentity._uncacheable_parameter(
                "pydantic-alias", default
            )
        return CacheIdentity._normalize_cache_parameter(name, default, set())

    @staticmethod
    def _is_pydantic_undefined(value: Any) -> bool:
        if value is CacheIdentity._MISSING_ATTRIBUTE:
            return True
        if value is CacheIdentity._UNSAFE_ATTRIBUTE:
            return False
        type_name = CacheIdentity._type_name(type(value))
        return type_name.endswith(
            ".PydanticUndefinedType"
        ) or type_name.endswith(".UndefinedType")

    @staticmethod
    def _annotation_fingerprint(annotation: Any) -> Any:
        if annotation is None:
            return None

        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is not None:
            return {
                "origin": CacheIdentity._annotation_origin_fingerprint(origin),
                "args": [
                    CacheIdentity._annotation_fingerprint(arg) for arg in args
                ],
            }

        if isinstance(annotation, type):
            return CacheIdentity._type_name(annotation)

        return CacheIdentity._annotation_literal_fingerprint(annotation)

    @staticmethod
    def _annotation_origin_fingerprint(origin: Any) -> Any:
        if isinstance(origin, type):
            return CacheIdentity._type_name(origin)

        module = getattr(origin, "__module__", None)
        qualname = getattr(origin, "__qualname__", None)
        name = getattr(origin, "_name", None)
        if isinstance(module, str) and isinstance(qualname, str):
            return f"{module}.{qualname}"
        if isinstance(module, str) and isinstance(name, str):
            return f"{module}.{name}"
        return CacheIdentity._uncacheable_parameter("annotation-origin", origin)

    @staticmethod
    def _annotation_literal_fingerprint(annotation: Any) -> Any:
        annotation_type = type(annotation)
        if issubclass(annotation_type, (str, int, bool, float, Enum, Path)):
            normalized = CacheIdentity._normalize_cache_parameter(
                "annotation", annotation, set()
            )
            if CacheIdentity._contains_uncacheable_parameter(normalized):
                return normalized
            return {
                "__type__": CacheIdentity._type_name(type(annotation)),
                "__value__": normalized,
            }
        return CacheIdentity._uncacheable_parameter("annotation", annotation)

    @staticmethod
    def _callable_fingerprint(value: Any) -> Any:
        if value is None:
            return None
        return CacheIdentity._uncacheable_parameter("callable", value)

    @staticmethod
    def _safe_dataclass_field_value(value: Any, name: str) -> Any:
        try:
            instance_fields = vars(value)
        except TypeError as exc:
            raise AttributeError(name) from exc
        if name in instance_fields:
            return instance_fields[name]
        raise AttributeError(name)

    @staticmethod
    def _uncacheable_parameter(
        reason: str, value: Optional[Any] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"__deepeval_uncacheable__": reason}
        if value is not None:
            payload["__type__"] = CacheIdentity._type_name(type(value))
        return payload

    @staticmethod
    def _contains_uncacheable_parameter(value: Any) -> bool:
        if isinstance(value, Mapping):
            if "__deepeval_uncacheable__" in value:
                return True
            return any(
                CacheIdentity._contains_uncacheable_parameter(item)
                for item in value.values()
            )

        if isinstance(value, (list, tuple, set, frozenset)):
            return any(
                CacheIdentity._contains_uncacheable_parameter(item)
                for item in value
            )

        return False

    @staticmethod
    def _has_uncacheable_parameter(
        parameters: Optional[Dict[str, str]],
    ) -> bool:
        if not parameters:
            return False
        return any(
            value == CacheIdentity._UNCACHEABLE_CACHE_VALUE
            for value in parameters.values()
        )

    @staticmethod
    def _metric_configuration_has_uncacheable_value(
        configuration: Any,
    ) -> bool:
        if CacheIdentity._has_uncacheable_parameter(
            configuration.custom_parameters
        ):
            return True
        return any(
            getattr(configuration, field, None)
            == CacheIdentity._UNCACHEABLE_CACHE_VALUE
            for field in CacheIdentity._CREATE_CONFIG_FIELDS
        )

    @staticmethod
    def _passive_object_attribute(value: Any, name: str) -> Any:
        try:
            return object.__getattribute__(value, name)
        except AttributeError:
            return CacheIdentity._MISSING_ATTRIBUTE
        except Exception:
            return CacheIdentity._UNSAFE_ATTRIBUTE

    @staticmethod
    def _passive_class_attribute(value: type, name: str) -> Any:
        try:
            return type.__getattribute__(value, name)
        except AttributeError:
            return CacheIdentity._MISSING_ATTRIBUTE
        except Exception:
            return CacheIdentity._UNSAFE_ATTRIBUTE

    @staticmethod
    def _safe_truthy_builtin(value: Any) -> bool:
        if value is CacheIdentity._UNSAFE_ATTRIBUTE:
            return True
        if value is None or value is CacheIdentity._MISSING_ATTRIBUTE:
            return value is CacheIdentity._UNSAFE_ATTRIBUTE
        value_type = type(value)
        if value_type in (dict, list, tuple, set, frozenset, str, bytes):
            return len(value) > 0
        if value_type in (bool, int, float):
            return bool(value)
        return True

    @staticmethod
    def _type_name(value_type: type) -> str:
        try:
            module = type.__getattribute__(value_type, "__module__")
            qualname = type.__getattribute__(value_type, "__qualname__")
        except Exception:
            return "<unknown>"
        return f"{module}.{qualname}"

    @staticmethod
    def _safe_text_identity(value: Any) -> str:
        return CacheIdentity._type_name(type(value))

    @staticmethod
    def _is_pydantic_model_class(value: type) -> bool:
        try:
            model_classes = (
                (BaseModel,)
                if LegacyBaseModel is None
                else (BaseModel, LegacyBaseModel)
            )
            return issubclass(value, model_classes)
        except TypeError:
            return False

    @staticmethod
    def _is_deep_acyclic_graph(value: Any) -> bool:
        return (
            CacheIdentity._type_name(type(value))
            == "deepeval.metrics.dag.graph.DeepAcyclicGraph"
        )

    @staticmethod
    def _canonicalize_dag(dag: Any) -> Any:
        dag_fields = CacheIdentity._passive_instance_fields(dag)
        if dag_fields is None or "root_nodes" not in dag_fields:
            return CacheIdentity._uncacheable_parameter("dag", dag)

        root_nodes = CacheIdentity._safe_dag_sequence(
            dag_fields.get("root_nodes"), default_empty=True
        )
        if root_nodes is None:
            return CacheIdentity._uncacheable_parameter("dag", dag)
        canonical_ids: Dict[int, str] = {}
        node_by_identity: Dict[int, Any] = {}

        def assign(node: Any, stack: set[int]) -> bool:
            if CacheIdentity._safe_dag_node_fields(
                node
            ) is None or not CacheIdentity._dag_node_children_are_supported(
                node
            ):
                return False
            node_identity = id(node)
            if node_identity in stack:
                return False
            if node_identity in canonical_ids:
                return True
            if len(stack) > CacheIdentity._MAX_CACHE_PARAMETER_DEPTH:
                return False
            if len(canonical_ids) >= CacheIdentity._MAX_DAG_CACHE_NODES:
                return False

            canonical_ids[node_identity] = f"n{len(canonical_ids)}"
            node_by_identity[node_identity] = node
            next_stack = set(stack)
            next_stack.add(node_identity)
            for child_node in CacheIdentity._live_dag_child_nodes(node):
                if not assign(child_node, next_stack):
                    return False
            return True

        for root_node in root_nodes:
            if not assign(root_node, set()):
                return CacheIdentity._uncacheable_parameter("dag", dag)

        return {
            "roots": [canonical_ids[id(root)] for root in root_nodes],
            "nodes": {
                canonical_ids[node_identity]: CacheIdentity._live_dag_node_spec(
                    node, canonical_ids
                )
                for node_identity, node in node_by_identity.items()
            },
        }

    @staticmethod
    def _safe_dag_node_fields(node: Any) -> Optional[Mapping[str, Any]]:
        if not CacheIdentity._is_dag_node(node):
            return None
        node_fields = CacheIdentity._passive_instance_fields(node)
        if type(node_fields) is not dict:
            return None
        if any(type(key) is not str for key in node_fields):
            return None
        return node_fields

    @staticmethod
    def _safe_dag_sequence(
        value: Any,
        default_empty: bool = False,
    ) -> Optional[Any]:
        if value is None:
            return () if default_empty else None
        if type(value) in (list, tuple):
            return value
        return None

    @staticmethod
    def _dag_node_children_are_supported(node: Any) -> bool:
        node_fields = CacheIdentity._safe_dag_node_fields(node)
        if node_fields is None:
            return False

        children = CacheIdentity._safe_dag_sequence(
            node_fields.get("children"), default_empty=True
        )
        if children is None:
            return False
        for child in children:
            if not CacheIdentity._is_dag_node(child):
                return False

        child = node_fields.get("child")
        return (
            child is None
            or CacheIdentity._is_dag_node(child)
            or CacheIdentity._is_metric_instance(child)
        )

    @staticmethod
    def _live_dag_child_nodes(node: Any) -> List[Any]:
        node_fields = CacheIdentity._safe_dag_node_fields(node)
        if node_fields is None:
            return []

        children: List[Any] = []
        child_nodes = CacheIdentity._safe_dag_sequence(
            node_fields.get("children"), default_empty=True
        )
        if child_nodes is None:
            return []
        for child in child_nodes:
            if CacheIdentity._is_dag_node(child):
                children.append(child)

        child = node_fields.get("child")
        if CacheIdentity._is_dag_node(child):
            children.append(child)

        return children

    @staticmethod
    def _live_dag_node_spec(node: Any, canonical_ids: Dict[int, str]) -> Any:
        node_fields = CacheIdentity._safe_dag_node_fields(node)
        if node_fields is None:
            return CacheIdentity._uncacheable_parameter("dag-node", node)

        spec: Dict[str, Any] = {"type": CacheIdentity._type_name(type(node))}
        for name, value in sorted(node_fields.items()):
            if name in {"children", "child"} or name.startswith("_"):
                continue
            spec[name] = CacheIdentity._normalize_cache_parameter(
                name, value, set()
            )

        if "children" in node_fields:
            children = CacheIdentity._safe_dag_sequence(
                node_fields.get("children"), default_empty=True
            )
            if children is None:
                return CacheIdentity._uncacheable_parameter("dag-node", node)
            spec["children"] = [
                canonical_ids[id(child)]
                for child in children
                if CacheIdentity._is_dag_node(child)
            ]

        if "child" in node_fields:
            child = node_fields.get("child")
            if child is not None:
                spec["child"] = CacheIdentity._live_dag_child_spec(
                    child, canonical_ids
                )

        return spec

    @staticmethod
    def _live_dag_child_spec(child: Any, canonical_ids: Dict[int, str]) -> Any:
        if CacheIdentity._is_dag_node(child):
            return {"type": "node", "ref": canonical_ids[id(child)]}
        if CacheIdentity._is_metric_instance(child):
            return CacheIdentity._metric_child_configuration(child)
        return CacheIdentity._normalize_cache_parameter("child", child, set())

    @staticmethod
    def _metric_child_configuration(metric: Any) -> Dict[str, Any]:
        try:
            configuration = CacheIdentity.create_metric_configuration(metric)
        except Exception:
            return CacheIdentity._uncacheable_parameter("metric-child", metric)

        if CacheIdentity._metric_configuration_has_uncacheable_value(
            configuration
        ):
            return CacheIdentity._uncacheable_parameter("metric-child", metric)

        return {
            "type": "metric",
            "metric_class": CacheIdentity._type_name(type(metric)),
            "metric_configuration": configuration.model_dump(mode="json"),
        }

    @staticmethod
    def _is_dag_node(value: Any) -> bool:
        if value is None:
            return False
        return (
            CacheIdentity._type_name(type(value))
            in CacheIdentity._KNOWN_DAG_NODE_TYPES
        )

    @staticmethod
    def _is_metric_instance(value: Any) -> bool:
        return isinstance(value, (BaseMetric, BaseConversationalMetric))

    @staticmethod
    def _should_descend_sensitive_parameter(name: str, value: Any) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return normalized in {"headers", "auth_headers"} and issubclass(
            type(value), Mapping
        )

    @staticmethod
    def _is_sensitive_parameter_name(name: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        compact = normalized.replace("_", "")
        parts = set(filter(None, normalized.split("_")))

        if normalized in CacheIdentity._SENSITIVE_PARAMETER_NAMES:
            return True
        if compact in CacheIdentity._SENSITIVE_PARAMETER_NAMES:
            return True
        if compact in CacheIdentity._SENSITIVE_PARAMETER_COMPACT_NAMES:
            return True
        if any(
            part in parts for part in CacheIdentity._SENSITIVE_PARAMETER_PARTS
        ):
            return True
        if any(
            normalized.endswith(suffix)
            for suffix in CacheIdentity._SENSITIVE_PARAMETER_SUFFIXES
        ):
            return True
        return any(
            compact.endswith(suffix)
            for suffix in CacheIdentity._SENSITIVE_PARAMETER_COMPACT_SUFFIXES
        )

    @staticmethod
    def _is_sensitive_parameter_value(name: str, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        stripped = value.strip()
        if CacheIdentity._is_sensitive_url_component_value(stripped):
            return True
        try:
            parsed = urlsplit(stripped)
            _ = parsed.port
        except ValueError:
            return True
        if any(
            CacheIdentity._url_variant_has_userinfo(decoded_value)
            for decoded_value in CacheIdentity._decoded_text_variants(
                stripped, plus=True
            )
        ):
            return True
        if "://" in stripped and any(char.isspace() for char in stripped):
            return True
        for (
            component_name,
            component_value,
        ) in CacheIdentity._url_key_value_pairs(
            parsed.query
        ) + CacheIdentity._url_key_value_pairs(
            parsed.fragment
        ):
            if (
                CacheIdentity._decode_budget_exhausted_with_percent_escape(
                    component_name, plus=True
                )
                or any(
                    CacheIdentity._is_sensitive_parameter_name(decoded_name)
                    for decoded_name in CacheIdentity._decoded_text_variants(
                        component_name, plus=True
                    )
                )
                or CacheIdentity._is_sensitive_url_component_value(
                    component_value
                )
            ):
                return True
        if CacheIdentity._is_sensitive_url_component_value(parsed.query or ""):
            return True
        if CacheIdentity._is_sensitive_url_component_value(
            parsed.fragment or ""
        ):
            return True
        if CacheIdentity._is_sensitive_url_component_value(parsed.path or ""):
            return True
        if not parsed.scheme or not parsed.netloc:
            return False
        if parsed.username or parsed.password:
            return True
        return False

    @staticmethod
    def _url_key_value_pairs(value: str) -> List[tuple[str, str]]:
        try:
            return parse_qsl(value, keep_blank_values=True)
        except ValueError:
            return []

    @staticmethod
    def _url_variant_has_userinfo(value: str) -> bool:
        try:
            parsed = urlsplit(value)
            _ = parsed.port
        except ValueError:
            return True
        return bool(parsed.netloc and (parsed.username or parsed.password))

    @staticmethod
    def _is_sensitive_url_component_value(value: str) -> bool:
        if CacheIdentity._decode_budget_exhausted_with_percent_escape(
            value, plus=True
        ):
            return True
        return any(
            CacheIdentity._has_secret_like_string_value(decoded_value)
            or CacheIdentity._has_credential_like_dsn_value(decoded_value)
            for decoded_value in CacheIdentity._decoded_text_variants(
                value, plus=True
            )
        )

    @staticmethod
    def _decoded_text_variants(value: str, plus: bool = False) -> List[str]:
        variants = [value]
        current = value
        for _ in range(CacheIdentity._MAX_URL_DECODE_PASSES):
            decoded = unquote_plus(current) if plus else unquote(current)
            if decoded == current:
                break
            variants.append(decoded)
            current = decoded
        return variants

    @staticmethod
    def _decode_budget_exhausted_with_percent_escape(
        value: str, plus: bool = False
    ) -> bool:
        current = value
        for _ in range(CacheIdentity._MAX_URL_DECODE_PASSES):
            decoded = unquote_plus(current) if plus else unquote(current)
            if decoded == current:
                return False
            current = decoded
        return bool(re.search(r"%[0-9a-fA-F]{2}", current))

    @staticmethod
    def _has_secret_like_string_value(value: str) -> bool:
        lowered = value.lower().replace("\\", "/")
        return bool(
            re.search(
                r"(^|[^a-z0-9])("
                r"bearer\s+\S+|"
                r"sk-[a-z0-9][a-z0-9_-]*|"
                r"xox[baprs]-[a-z0-9-]+|"
                r"gh[pousr]_[a-z0-9_]+|"
                r"github_pat_[a-z0-9_]+|"
                r"eyj[a-z0-9_-]+\.[a-z0-9_-]+\.[a-z0-9_-]+|"
                r"(api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|"
                r"auth[_-]?token|session[_-]?token|token|password|passwd|secret|sig|"
                r"signature|authorization)\s*[:=/]\s*\S+"
                r")",
                lowered,
            )
        )

    @staticmethod
    def _has_credential_like_dsn_value(value: str) -> bool:
        return bool(re.match(r"^[a-z][a-z0-9+.-]*:[^\s/][^\s]*@", value, re.I))

    @staticmethod
    def _model_object_name(model: Any) -> Optional[str]:
        if isinstance(model, str):
            return model
        return None

    @staticmethod
    def _metric_model_name(metric: BaseMetric) -> Optional[str]:
        instance_fields = CacheIdentity._passive_instance_fields(metric)
        if instance_fields is None:
            return None
        for field in ("model", "_model"):
            model = instance_fields.get(field)
            if model is None:
                continue
            model_name = CacheIdentity._model_object_name(model)
            if model_name is not None:
                return model_name
        return None

    @staticmethod
    def _metric_uses_native_model(metric: BaseMetric) -> bool:
        instance_fields = CacheIdentity._passive_instance_fields(metric)
        if instance_fields is None:
            return False
        if instance_fields.get("using_native_model") is not True:
            return False
        return any(
            field in instance_fields
            and CacheIdentity._is_native_model_object(instance_fields[field])
            for field in ("model", "_model")
        )

    @staticmethod
    def _is_native_model_object(model: Any) -> bool:
        return CacheIdentity._type_name(type(model)).startswith(
            "deepeval.models.llms."
        )

    @staticmethod
    def _native_model_cache_identity(model: Any) -> Optional[Dict[str, Any]]:
        if not CacheIdentity._is_native_model_object(model):
            return None
        model_fields = CacheIdentity._passive_instance_fields(model)
        if model_fields is None:
            return None

        identity: Dict[str, Any] = {
            "__type__": CacheIdentity._type_name(type(model))
        }
        for field in CacheIdentity._NATIVE_MODEL_CACHE_IDENTITY_FIELDS:
            if field in model_fields:
                identity[field] = model_fields[field]
        return identity

    @staticmethod
    def _metric_has_non_string_model(metric: BaseMetric) -> bool:
        instance_fields = CacheIdentity._passive_instance_fields(metric)
        if instance_fields is None:
            return False
        return any(
            field in instance_fields
            and instance_fields[field] is not None
            and not isinstance(instance_fields[field], str)
            for field in ("model", "_model")
        )

    @staticmethod
    def _passive_instance_fields(value: Any) -> Optional[Mapping[str, Any]]:
        try:
            return vars(value)
        except Exception:
            return None

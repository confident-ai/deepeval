import asyncio
import math
import os
import re
from typing import Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
from collections import deque
from deepeval.constants import CONFIDENT_TRACING_ENABLED
from deepeval.tracing.integrations import Provider

if TYPE_CHECKING:
    from deepeval.tracing.api import TraceApi


class Environment(Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    STAGING = "staging"
    TESTING = "testing"


def infer_provider_from_model(model: str) -> Optional[str]:
    if not model or not isinstance(model, str):
        return None
    clean_name = model.lower().strip().replace(":", "/")
    model_id = clean_name.split("/")[-1]

    mapping: Dict[str, str] = {
        "gpt": Provider.OPEN_AI.value,
        "o1": Provider.OPEN_AI.value,
        "o3": Provider.OPEN_AI.value,
        "gemini": Provider.GEMINI.value,
        "palm": Provider.GEMINI.value,
        "gecko": Provider.GEMINI.value,
        "claude": Provider.ANTHROPIC.value,
        "sonnet": Provider.ANTHROPIC.value,
        "opus": Provider.ANTHROPIC.value,
        "haiku": Provider.ANTHROPIC.value,
        "mistral": Provider.MISTRAL.value,
        "mixtral": Provider.MISTRAL.value,
        "pixtral": Provider.MISTRAL.value,
        "codestral": Provider.MISTRAL.value,
        "grok": Provider.X_AI.value,
        "deepseek": Provider.DEEP_SEEK.value,
    }
    for prefix, provider in mapping.items():
        if model_id.startswith(prefix):
            return provider

    for provider in set(mapping.values()):
        if provider.lower() in clean_name:
            return provider

    return None


def _normalize_provider_string(value: str) -> str:
    """Lowercase and remove non-alphanumerics for loose equality checks."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalize_span_provider_for_platform(raw: Optional[Any]) -> Optional[str]:
    """Map raw provider strings (e.g. LangChain ``\"openai\"``) to ``Provider`` values."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    normalized_raw = _normalize_provider_string(s)
    head = re.split(r"[\s./\\]+", s, maxsplit=1)[0]
    normalized_head = _normalize_provider_string(head)

    for provider in Provider:
        canonical = provider.value
        normalized_canonical = _normalize_provider_string(canonical)
        enum_key_name = _normalize_provider_string(provider.name)
        if normalized_raw in (normalized_canonical, enum_key_name):
            return canonical
        if normalized_head in (normalized_canonical, enum_key_name):
            return canonical

    return s


def normalize_trace_api_span_providers(trace_api: "TraceApi") -> None:
    """Normalize ``provider`` on all API spans before POST to Confident."""
    for spans in (
        trace_api.llm_spans,
        trace_api.base_spans,
        trace_api.agent_spans,
        trace_api.retriever_spans,
        trace_api.tool_spans,
    ):
        if not spans:
            continue
        for sp in spans:
            if sp.provider:
                sp.provider = normalize_span_provider_for_platform(sp.provider)


def _strip_nul(s: str) -> str:
    # Replace embedded NUL, which Postgres cannot store in text/jsonb
    # Do NOT try to escape as \u0000 because PG will still reject it.
    return s.replace("\x00", "")


def tracing_enabled():
    return os.getenv(CONFIDENT_TRACING_ENABLED, "YES").upper() == "YES"


def validate_environment(environment: str):
    if environment not in [env.value for env in Environment]:
        valid_values = ", ".join(f'"{env.value}"' for env in Environment)
        raise ValueError(
            f"Invalid environment: {environment}. Please use one of the following instead: {valid_values}"
        )


def validate_sampling_rate(sampling_rate: float):
    if sampling_rate < 0 or sampling_rate > 1:
        raise ValueError(
            f"Invalid sampling rate: {sampling_rate}. Please use a value between 0 and 1"
        )


def make_json_serializable(obj):
    """
    Recursively converts an object to a JSON‐serializable form,
    replacing circular references with "<circular>".
    """
    seen = set()  # Store `id` of objects we've visited

    def _serialize(o):
        oid = id(o)

        # strip Nulls
        if isinstance(o, str):
            return _strip_nul(o)

        # Replace non-finite floats (NaN, Infinity, -Infinity) with None
        if isinstance(o, float):
            return None if not math.isfinite(o) else o

        # Primitive types are already serializable
        if isinstance(o, (int, bool)) or o is None:
            return o

        # Detect circular reference
        if oid in seen:
            return "<circular>"

        # Mark current object as seen
        seen.add(oid)

        # Handle containers
        if isinstance(o, (list, tuple, set, deque)):  # TODO: check if more
            serialized = []
            for item in o:
                serialized.append(_serialize(item))

            return serialized

        if isinstance(o, dict):
            result = {}
            for key, value in o.items():
                # Convert key to string (JSON only allows string keys)
                result[str(key)] = _serialize(value)
            return result

        # Handle objects with __dict__
        if hasattr(o, "__dict__"):
            result = {}
            for key, value in vars(o).items():
                if not key.startswith("_"):
                    result[key] = _serialize(value)
            return result

        # Fallback: convert to string
        return _strip_nul(str(o))

    return _serialize(obj)


def make_json_serializable_for_metadata(obj):
    """
    Recursively converts an object to a JSON‐serializable form,
    replacing circular references with "<circular>".

    Primitive types (``bool``, ``int``, ``float``, ``None``) are preserved
    as their native JSON types so downstream consumers can filter / type-
    check metadata correctly. Earlier versions of this helper coerced
    primitives to ``str`` (e.g. ``True`` → ``"True"``, ``3.14`` → ``"3.14"``),
    which broke type fidelity for any user metadata containing booleans
    or numbers. Non-finite floats (NaN / ±Infinity) are still replaced
    with ``None`` because they are not valid JSON.
    """
    seen = set()  # Store `id` of objects we've visited

    def _serialize(o):
        oid = id(o)

        # strip Nulls
        if isinstance(o, str):
            return _strip_nul(o)

        # Replace non-finite floats (NaN, Infinity, -Infinity) with None
        if isinstance(o, float):
            return None if not math.isfinite(o) else o

        # Primitive types are already serializable
        if isinstance(o, (int, bool)) or o is None:
            return o

        # Detect circular reference
        if oid in seen:
            return "<circular>"

        # Mark current object as seen
        seen.add(oid)

        # Handle containers
        if isinstance(o, (list, tuple, set, deque)):  # TODO: check if more
            serialized = []
            for item in o:
                serialized.append(_serialize(item))

            return serialized

        if isinstance(o, dict):
            result = {}
            for key, value in o.items():
                # Convert key to string (JSON only allows string keys)
                result[str(key)] = _serialize(value)
            return result

        # Handle objects with __dict__
        if hasattr(o, "__dict__"):
            result = {}
            for key, value in vars(o).items():
                if not key.startswith("_"):
                    result[key] = _serialize(value)
            return result

        # Fallback: convert to string
        return _strip_nul(str(o))

    return _serialize(obj)


def to_zod_compatible_iso(
    dt: datetime, microsecond_precision: bool = False
) -> str:
    return (
        dt.astimezone(timezone.utc)
        .isoformat(
            timespec="microseconds" if microsecond_precision else "milliseconds"
        )
        .replace("+00:00", "Z")
    )


def perf_counter_to_datetime(perf_counter_value: float) -> datetime:
    """
    Convert a perf_counter value to a datetime object.

    Args:
        perf_counter_value: A float value from perf_counter()

    Returns:
        A datetime object representing the current time
    """
    # Get the current time
    current_time = datetime.now(timezone.utc)
    # Calculate the time difference in seconds
    time_diff = current_time.timestamp() - perf_counter()
    # Convert perf_counter value to a real timestamp
    timestamp = time_diff + perf_counter_value
    # Return as a datetime object
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def replace_self_with_class_name(obj):
    try:
        return f"<{obj.__class__.__name__}>"
    except:
        return f"<self>"


def prepare_tool_call_input_parameters(output: Any) -> Dict[str, Any]:
    res = make_json_serializable(output)
    if res and not isinstance(res, dict):
        res = {"output": res}
    return res


def is_async_context() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False

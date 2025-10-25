import asyncio
import os
from typing import Dict, Any
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
from collections import deque
from deepeval.constants import CONFIDENT_TRACING_ENABLED


class Environment(Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    STAGING = "staging"
    TESTING = "testing"


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

        # Primitive types are already serializable
        if isinstance(o, (str, int, float, bool)) or o is None:
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
    """
    seen = set()  # Store `id` of objects we've visited

    def _serialize(o):
        oid = id(o)

        # strip Nulls
        if isinstance(o, str):
            return _strip_nul(o)

        # Primitive types are already serializable
        if isinstance(o, (str, int, float, bool)) or o is None:
            return str(o)

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

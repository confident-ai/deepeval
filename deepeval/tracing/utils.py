import asyncio
import os
from contextlib import contextmanager
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
from typing import Any, Dict, Optional

from deepeval.constants import CONFIDENT_TRACING_ENABLED
from deepeval.tracing.context import current_trace_context, current_span_context


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
    except Exception:
        return "<self>"


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


@contextmanager
def bind_trace_and_span(
    *,
    trace_uuid: Optional[str],
    span_uuid: Optional[str] = None,
    parent_uuid: Optional[str] = None,
    reset_on_exit: bool = True,
):
    from deepeval.tracing.tracing import trace_manager

    trace_token = None
    span_token = None
    try:
        if trace_uuid:
            trace = trace_manager.get_trace_by_uuid(trace_uuid)
            if trace is not None:
                trace_token = current_trace_context.set(trace)

        # Prefer binding the span if it exists.
        if span_uuid is not None:
            span = trace_manager.get_span_by_uuid(span_uuid)
            if span is not None:
                span_token = current_span_context.set(span)
            else:
                # Span doesn't exist yet (e.g., *_start callbacks). Bind parent if possible.
                if parent_uuid is not None:
                    parent = trace_manager.get_span_by_uuid(parent_uuid)
                    if parent is not None:
                        span_token = current_span_context.set(parent)
                    elif reset_on_exit:
                        span_token = current_span_context.set(None)
                elif reset_on_exit:
                    span_token = current_span_context.set(None)

        elif parent_uuid is not None:
            parent = trace_manager.get_span_by_uuid(parent_uuid)
            if parent is not None:
                span_token = current_span_context.set(parent)
            elif reset_on_exit:
                span_token = current_span_context.set(None)

        else:
            if reset_on_exit:
                span_token = current_span_context.set(None)

        yield

    finally:
        if not reset_on_exit:
            return
        if span_token is not None:
            current_span_context.reset(span_token)
        if trace_token is not None:
            current_trace_context.reset(trace_token)

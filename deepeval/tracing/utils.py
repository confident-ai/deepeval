import os
import time
import inspect
import json
import sys
import difflib
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
import time
from collections import deque
from typing import Any, Dict, Optional, Sequence, Callable

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


def get_deepeval_trace_mode() -> Optional[str]:
    deepeval_trace_mode = None
    try:
        args = sys.argv
        for idx, arg in enumerate(args):
            if isinstance(arg, str) and arg.startswith(
                "--deepeval-trace-mode="
            ):
                deepeval_trace_mode = (
                    arg.split("=", 1)[1].strip().strip('"').strip("'").lower()
                )
                break
            if arg == "--deepeval-trace-mode" and idx + 1 < len(args):
                deepeval_trace_mode = (
                    str(args[idx + 1]).strip().strip('"').strip("'").lower()
                )
                break
    except Exception:
        deepeval_trace_mode = None

    return deepeval_trace_mode


def dump_body_to_json_file(
    body: Dict[str, Any], file_path: Optional[str] = None
) -> str:
    entry_file = None
    try:
        cmd0 = sys.argv[0] if sys.argv else None
        if cmd0 and cmd0.endswith(".py"):
            entry_file = cmd0
        else:
            for frame_info in reversed(inspect.stack()):
                fp = frame_info.filename
                if (
                    fp
                    and fp.endswith(".py")
                    and "deepeval/tracing" not in fp
                    and "site-packages" not in fp
                ):
                    entry_file = fp
                    break
    except Exception:
        entry_file = None

    if not entry_file:
        entry_file = "unknown.py"

    abs_entry = os.path.abspath(entry_file)
    dir_path = os.path.dirname(abs_entry)

    file_arg = None
    try:
        for idx, arg in enumerate(sys.argv):
            if isinstance(arg, str) and arg.startswith(
                "--deepeval-trace-file-name="
            ):
                file_arg = arg.split("=", 1)[1].strip().strip('"').strip("'")
                break
            if arg == "--deepeval-trace-file-name" and idx + 1 < len(sys.argv):
                file_arg = str(sys.argv[idx + 1]).strip().strip('"').strip("'")
                break
    except Exception:
        file_arg = None

    if file_path:
        dst_path = os.path.abspath(file_path)
    elif file_arg:
        dst_path = os.path.abspath(file_arg)
    else:
        base_name = os.path.splitext(os.path.basename(abs_entry))[0]
        dst_path = os.path.join(dir_path, f"{base_name}.json")

    actual_body = make_json_serializable(body)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(actual_body, f, ensure_ascii=False, indent=2, sort_keys=True)
    return dst_path

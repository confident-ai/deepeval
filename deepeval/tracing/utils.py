import os
import inspect
import json
import sys
import difflib
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
from collections import deque
from typing import Any, Dict

from deepeval.constants import CONFIDENT_TRACING_ENABLED

class Environment(Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    STAGING = "staging"
    TESTING = "testing"


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
        return str(o)

    return _serialize(obj)


def to_zod_compatible_iso(dt: datetime) -> str:
    return (
        dt.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
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

_IGNORE_DYNAMIC_KEYS = {
    "uuid",
    "parentUuid",
    "startTime",
    "endTime",
    # snake_case variants for safety
    "parent_uuid",
    "start_time",
    "end_time",
    "trace_uuid",
    "traceUuid",
    ""
}

def _sanitize_dynamic_fields(obj):
    if isinstance(obj, dict):
        return {
            k: _sanitize_dynamic_fields(v)
            for k, v in obj.items()
            if k not in _IGNORE_DYNAMIC_KEYS
        }
    if isinstance(obj, list):
        return [_sanitize_dynamic_fields(v) for v in obj]
    return obj

def test_trace_body(body: Dict[str, Any]):
    mode = os.getenv("TEST_TRACE")
    if not mode:
        return

    mode = mode.lower()

    # Resolve the entrypoint file from the python command
    entry_file = None
    try:
        cmd0 = sys.argv[0] if sys.argv else None
        if cmd0 and cmd0.endswith(".py"):
            entry_file = cmd0
        else:
            # Fallback: try to find a plausible caller .py from the stack
            for frame_info in reversed(inspect.stack()):
                fp = frame_info.filename
                if fp and fp.endswith(".py") and "deepeval/tracing" not in fp and "site-packages" not in fp:
                    entry_file = fp
                    break
    except Exception:
        entry_file = None

    if not entry_file:
        entry_file = "unknown.py"

    abs_entry = os.path.abspath(entry_file)
    dir_path = os.path.dirname(abs_entry)

    # Optional: --file-name=<name>.json or --file-name <name>.json
    file_arg = None
    try:
        for idx, arg in enumerate(sys.argv):
            if isinstance(arg, str) and arg.startswith("--file-name="):
                file_arg = arg.split("=", 1)[1].strip().strip('"').strip("'")
                break
            if arg == "--file-name" and idx + 1 < len(sys.argv):
                file_arg = str(sys.argv[idx + 1]).strip().strip('"').strip("'")
                break
    except Exception:
        file_arg = None

    if file_arg:
        file_name = os.path.basename(file_arg)
    else:
        base_name = os.path.splitext(os.path.basename(abs_entry))[0]
        file_name = f"{base_name}.json"

    file_path = os.path.join(dir_path, file_name)

    serializable_body = make_json_serializable(body)
    normalized_body = _sanitize_dynamic_fields(serializable_body)

    if mode == "gen":
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(normalized_body, f, ensure_ascii=False, indent=2, sort_keys=True)
        return

    if mode == "test":
        if not os.path.exists(file_path):
            raise AssertionError(f"Assertion file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            expected = json.load(f)

        expected_normalized = _sanitize_dynamic_fields(expected)

        if normalized_body != expected_normalized:
            try:
                expected_str = json.dumps(expected_normalized, ensure_ascii=False, indent=2, sort_keys=True)
                actual_str = json.dumps(normalized_body, ensure_ascii=False, indent=2, sort_keys=True)
                diff = "\n".join(
                    difflib.unified_diff(
                        expected_str.splitlines(),
                        actual_str.splitlines(),
                        fromfile="expected",
                        tofile="actual",
                        lineterm="",
                    )
                )
            except Exception:
                diff = "<diff unavailable>"
            raise AssertionError(f"Trace body does not match expected file: {file_path}\n{diff}")
        return
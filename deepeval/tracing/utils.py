import os
import inspect
import json
import sys
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
from collections import deque
from typing import Any, Dict, Optional, Sequence, Callable
from to_json_schema.to_json_schema import SchemaBuilder
import jsonschema
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
    except Exception:
        return "<self>"


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
    body: Dict[str, Any], file_path: str
):
    """
    Dumps a dictionary to a JSON file at the specified path.
    """
    
    with open(file_path, 'w') as f:
        json.dump(body, f, indent=2)
    


def check_the_structure_of_dict_with_json_file(expected_file_path: str, actual_file_path: str) -> bool:
    """
    Validate that `actual_json_obj` matches the structure and data types of the JSON at `expected_file_path`.

    Rules:
    - Dicts: keys must match exactly on both sides; values are validated recursively.
    - Lists: both must be lists; elements are compared pairwise (same length required).
    - Primitives: types must match exactly. Int/float are treated as interchangeable numeric types.
    - Returns False if the file cannot be read or the JSON is invalid.
    """
    try:
        with open(expected_file_path, 'r') as f:
            try:
                expected_json_obj = json.load(f)
            except json.JSONDecodeError:
                return False

        with open(actual_file_path, 'r') as f:
            try:
                actual_json_obj = json.load(f)
            except json.JSONDecodeError:
                return False
    except OSError:
        return False

    def _compare(a: Any, b: Any, path: str = "root") -> bool:
        # Dict vs Dict
        if isinstance(b, dict):
            if not isinstance(a, dict):
                print(f"❌ Type mismatch at '{path}':")
                print(f"   Expected: dict")
                print(f"   Got: {type(a).__name__}")
                print(f"   Value: {a}")
                return False
            
            # Check for missing or extra keys
            missing_keys = set(b.keys()) - set(a.keys())
            extra_keys = set(a.keys()) - set(b.keys())
            
            if missing_keys:
                print(f"❌ Missing keys at '{path}': {missing_keys}")
                return False
            if extra_keys:
                print(f"❌ Extra keys at '{path}': {extra_keys}")
                return False
            
            for k in b.keys():
                if not _compare(a[k], b[k], f"{path}.{k}"):
                    return False
            return True

        # List vs List (pairwise compare)
        if isinstance(b, list):
            if not isinstance(a, list):
                print(f"❌ Type mismatch at '{path}':")
                print(f"   Expected: list")
                print(f"   Got: {type(a).__name__}")
                print(f"   Value: {a}")
                return False
            if len(a) != len(b):
                print(f"❌ Length mismatch at '{path}': expected {len(b)}, got {len(a)}")
                return False
            for idx, (ae, be) in enumerate(zip(a, b)):
                if not _compare(ae, be, f"{path}[{idx}]"):
                    return False
            return True

        # Primitives: exact type match, except int/float interchangeable (bool is not numeric here)
        number_types = (int, float)
        if type(b) in number_types and type(a) in number_types and not isinstance(a, bool) and not isinstance(b, bool):
            return True
        
        if type(a) is not type(b):
            print(f"❌ Type mismatch at '{path}':")
            print(f"   Expected: {type(b).__name__}")
            print(f"   Got: {type(a).__name__}")
            print(f"   Expected value: {b}")
            print(f"   Actual value: {a}")
            return False
        
        return True

    return _compare(actual_json_obj, expected_json_obj)
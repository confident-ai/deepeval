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

_PLACEHOLDER = "<is_present>"

def _apply_placeholders(expected, actual, path=""):
    if expected == _PLACEHOLDER:
        return actual
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"Type mismatch at {path or '<root>'}: expected object, got {type(actual).__name__}")
        out = {}
        for k, v in expected.items():
            sub_path = f"{path}.{k}" if path else k
            if v == _PLACEHOLDER:
                if k not in actual:
                    raise AssertionError(f"Missing required key at {sub_path}")
                out[k] = actual[k]
            else:
                out[k] = _apply_placeholders(v, actual.get(k), sub_path)
        return out
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise AssertionError(f"Type mismatch at {path or '<root>'}: expected list, got {type(actual).__name__}")
        if len(expected) != len(actual):
            raise AssertionError(f"Length mismatch at {path or '<root>'}: expected {len(expected)}, got {len(actual)}")
        return [
            _apply_placeholders(ev, av, f"{path}[{i}]")
            for i, (ev, av) in enumerate(zip(expected, actual))
        ]
    return expected

def _mark_differences(expected, actual):
    if expected == _PLACEHOLDER:
        return _PLACEHOLDER
    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = set(expected.keys()) | set(actual.keys())
        out = {}
        for k in keys:
            ev = expected.get(k)
            av = actual.get(k)
            if ev == _PLACEHOLDER:
                out[k] = _PLACEHOLDER
            elif k not in expected:
                out[k] = _PLACEHOLDER
            elif k not in actual:
                out[k] = ev
            else:
                if isinstance(ev, dict) and isinstance(av, dict):
                    out[k] = _mark_differences(ev, av)
                elif isinstance(ev, list) and isinstance(av, list):
                    out[k] = _mark_differences(ev, av)
                else:
                    out[k] = ev if ev == av else _PLACEHOLDER
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return _PLACEHOLDER
        marked = []
        for ev, av in zip(expected, actual):
            if isinstance(ev, (dict, list)) or isinstance(av, (dict, list)):
                marked.append(_mark_differences(ev, av))
            else:
                marked.append(ev if ev == av else _PLACEHOLDER)
        return marked
    return expected if expected == actual else _PLACEHOLDER

def get_trace_mode(args: Optional[Sequence[str]] = None) -> Optional[str]:
    mode = None
    try:
        if args is None:
            args = sys.argv
        for idx, arg in enumerate(args):
            if isinstance(arg, str) and arg.startswith("--mode="):
                mode = arg.split("=", 1)[1].strip().strip('"').strip("'").lower()
                break
            if arg == "--mode" and idx + 1 < len(args):
                mode = str(args[idx + 1]).strip().strip('"').strip("'").lower()
                break
    except Exception:
        mode = None
    
    return mode

def test_trace_body(body: Dict[str, Any], mode: str):
    entry_file = None
    try:
        cmd0 = sys.argv[0] if sys.argv else None
        if cmd0 and cmd0.endswith(".py"):
            entry_file = cmd0
        else:
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
        # Respect the provided path (absolute or relative) instead of forcing the entry file's directory
        file_path = os.path.abspath(file_arg)
    else:
        base_name = os.path.splitext(os.path.basename(abs_entry))[0]
        file_path = os.path.join(dir_path, f"{base_name}.json")

    actual_body = make_json_serializable(body)

    if mode == "gen":
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(actual_body, f, ensure_ascii=False, indent=2, sort_keys=True)
        return

    if mode == "mark_dynamic":
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                expected = json.load(f)
            marked = _mark_differences(expected, actual_body)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(marked, f, ensure_ascii=False, indent=2, sort_keys=True)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(actual_body, f, ensure_ascii=False, indent=2, sort_keys=True)
        return

    if mode == "test":
        if not os.path.exists(file_path):
            raise AssertionError(f"Assertion file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            expected = json.load(f)

        try:
            expected_aligned = _apply_placeholders(expected, actual_body)
        except AssertionError as e:
            raise AssertionError(str(e))

        if actual_body != expected_aligned:
            try:
                expected_str = json.dumps(expected_aligned, ensure_ascii=False, indent=2, sort_keys=True)
                actual_str = json.dumps(actual_body, ensure_ascii=False, indent=2, sort_keys=True)
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
        
        print(f"Test trace body passed: {file_path}")
        return


def run_in_mode(mode: str, func: Callable, *args, file_path: Optional[str] = None, **kwargs):
    """
    Execute a callable while forcing trace mode for this process.

    This temporarily injects or overrides the `--mode` argument in sys.argv
    so that downstream tracing code sees the desired mode.

    Args:
        mode: One of "gen", "test", or "mark_dynamic".
        func: The function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The return value of `func(*args, **kwargs)`.
    """
    original_argv = list(sys.argv)
    try:
        # Build a modified argv that enforces the desired mode
        new_argv = []
        replaced = False
        file_specified = False
        i = 0
        while i < len(original_argv):
            arg = original_argv[i]
            if isinstance(arg, str) and arg.startswith("--mode="):
                new_argv.append(f"--mode={mode}")
                replaced = True
            elif arg == "--mode":
                new_argv.append("--mode")
                # Skip/replace the next token if present
                if i + 1 < len(original_argv):
                    new_argv.append(mode)
                replaced = True
                i += 1  # consume the value token as well
            elif isinstance(arg, str) and arg.startswith("--file-name="):
                file_specified = True
                new_argv.append(arg)
            elif arg == "--file-name":
                new_argv.append(arg)
                if i + 1 < len(original_argv):
                    new_argv.append(original_argv[i + 1])
                file_specified = True
                i += 1  # consume the value token as well
            else:
                new_argv.append(arg)
            i += 1

        if not replaced:
            new_argv.append(f"--mode={mode}")
        # Inject file path for test/gen modes if not already specified and provided
        if mode in ("test", "gen") and not file_specified and file_path:
            new_argv.append(f"--file-name={file_path}")

        sys.argv = new_argv
        result = func(*args, **kwargs)
        time.sleep(15)
        return result
    finally:
        sys.argv = original_argv


def run_in_test_mode(func: Callable, *args, **kwargs):
    """Convenience wrapper for run_in_mode("test", ...)."""
    return run_in_mode("test", func, *args, **kwargs)


def compare_trace_files(expected_file_path: str, actual_file_path: str):
    """
    Compare two JSON trace files applying placeholder semantics.
    Raises AssertionError with a unified diff on mismatch.
    """
    if not os.path.exists(expected_file_path):
        raise AssertionError(f"Assertion file not found: {expected_file_path}")
    if not os.path.exists(actual_file_path):
        raise AssertionError(f"Actual file not found: {actual_file_path}")

    with open(expected_file_path, "r", encoding="utf-8") as f:
        expected = json.load(f)
    with open(actual_file_path, "r", encoding="utf-8") as f:
        actual = json.load(f)

    try:
        expected_aligned = _apply_placeholders(expected, actual)
    except AssertionError as e:
        raise AssertionError(str(e))

    if actual != expected_aligned:
        try:
            expected_str = json.dumps(expected_aligned, ensure_ascii=False, indent=2, sort_keys=True)
            actual_str = json.dumps(actual, ensure_ascii=False, indent=2, sort_keys=True)
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
        raise AssertionError(f"Trace body does not match expected file: {expected_file_path}\n{diff}")

    print(f"Test trace body passed: {expected_file_path}")
    return

def dump_body_to_json_file(body: Dict[str, Any], file_path: Optional[str] = None) -> str:
    """
    Serialize `body` to a JSON file using the same resolution rules as test_trace_body's gen mode.
    Priority:
    1) file_path param (if provided)
    2) --file-name flag (if present in sys.argv)
    3) <entry_script_basename>.json in the entry script's directory

    Returns the absolute path of the written JSON file.
    """
    entry_file = None
    try:
        cmd0 = sys.argv[0] if sys.argv else None
        if cmd0 and cmd0.endswith(".py"):
            entry_file = cmd0
        else:
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
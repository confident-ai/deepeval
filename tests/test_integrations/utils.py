import asyncio
import json
import re
import os

from typing import Dict, Any
from functools import wraps
import inspect


def _compute_tools_used(obj: Dict[str, Any]) -> bool:
    """
    Compute whether tools were used in a trace object.

    Returns True if any of these conditions hold:
    - non-empty root.toolSpans
    - non-empty root.toolsCalled
    - any AI message with non-empty tool_calls
    - any baseSpan[*].toolsCalled non-empty
    """
    # Check root.toolSpans
    if obj.get("toolSpans") and len(obj["toolSpans"]) > 0:
        return True

    # Check root.toolsCalled
    if obj.get("toolsCalled") and len(obj["toolsCalled"]) > 0:
        return True

    # Check AI messages with tool_calls in various locations
    def check_messages(messages):
        if not messages:
            return False
        for msg in messages:
            if isinstance(msg, dict) and msg.get("type") == "ai":
                # LangChain drift: tool_calls may appear either at top-level or under additional_kwargs
                tool_calls = msg.get("tool_calls", [])
                if (
                    tool_calls
                    and isinstance(tool_calls, list)
                    and len(tool_calls) > 0
                ):
                    return True
                additional = msg.get("additional_kwargs", {})
                if isinstance(additional, dict):
                    tc2 = additional.get("tool_calls", [])
                    if tc2 and isinstance(tc2, list) and len(tc2) > 0:
                        return True
        return False

    # Check root input/output messages
    if obj.get("input") and isinstance(obj["input"], dict):
        if check_messages(obj["input"].get("messages")):
            return True
    if obj.get("output") and isinstance(obj["output"], dict):
        if check_messages(obj["output"].get("messages")):
            return True

    # Check baseSpans
    for span in obj.get("baseSpans", []):
        if isinstance(span, dict):
            if span.get("toolsCalled") and len(span["toolsCalled"]) > 0:
                return True
            # Also check messages inside baseSpans
            if span.get("input") and isinstance(span["input"], dict):
                if check_messages(span["input"].get("messages")):
                    return True
            if span.get("output") and isinstance(span["output"], dict):
                if check_messages(span["output"].get("messages")):
                    return True

    return False


def assert_json_object_structure(
    expected_json_obj: Dict[str, Any], actual_json_obj: Dict[str, Any]
) -> bool:
    """
    Validate that actual_json_obj matches the structure and data types of expected_json_obj.

    Rules:
    - Dicts: keys must match (with allowed drift for LangChain v1.x fields).
    - Lists: compared pairwise (same length required).
    - Primitives: types must match exactly. Int/float are interchangeable.
    - Preserves no-tools semantics: if expected implies no tools, actual must have no tools.
    """
    # Validate tools-used invariant at the top level before detailed comparison.
    # This ensures we never mask a regression where tools appear unexpectedly.
    expected_tools_used = _compute_tools_used(expected_json_obj)
    actual_tools_used = _compute_tools_used(actual_json_obj)

    if expected_tools_used != actual_tools_used:
        print("❌ Tools-used invariant violation:")
        print(f"   Expected tools_used: {expected_tools_used}")
        print(f"   Actual tools_used: {actual_tools_used}")
        if not expected_tools_used and actual_tools_used:
            print("   Regression: tools were called when none were expected")
        else:
            print(
                "   Regression: no tools were called when tools were expected"
            )
        return False

    def _require_dict_keys(d: Any, required_keys: set, path: str) -> bool:
        if not isinstance(d, dict):
            print(
                f"❌ Type mismatch at '{path}': expected dict, got {type(d).__name__}"
            )
            print(f"   Value: {d}")
            return False
        missing = required_keys - set(d.keys())
        if missing:
            print(f"❌ Missing required keys at '{path}': {missing}")
            return False
        return True

    def _require_str_field(d: Dict[str, Any], key: str, path: str) -> bool:
        v = d.get(key)
        if not isinstance(v, str):
            print(
                f"❌ Type mismatch at '{path}.{key}': expected str, got {type(v).__name__}"
            )
            print(f"   Value: {v}")
            return False
        return True

    def _compare(actual: Any, expected: Any, path: str = "root") -> bool:
        # Dict vs Dict
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                print(f"❌ Type mismatch at '{path}':")
                print("   Expected: dict")
                print(f"   Got: {type(actual).__name__}")
                print(f"   Value: {actual}")
                return False

            # Filter out keys to ignore globally
            keys_to_ignore = {"tokenIntervals"}
            expected_keys = set(expected.keys()) - keys_to_ignore
            actual_keys = set(actual.keys()) - keys_to_ignore

            # Schema drift handling for LangChain v1.x (narrow allowlist)
            schema_drift_config = {
                # response_metadata gained new fields in v1.x
                ".response_metadata": {
                    "allowed_extra": {"model_provider", "service_tier"},
                    "allowed_missing": set(),
                },
            }

            allowed_extras = set()
            allowed_missing = set()
            for suffix, config in schema_drift_config.items():
                if path.endswith(suffix):
                    allowed_extras = config.get("allowed_extra", set())
                    allowed_missing = config.get("allowed_missing", set())
                    break

            # Keys that are allowed to be extra on message objects
            # usage_metadata was added in later LangChain versions
            if re.search(r"\.messages\[\d+\]$", path):
                allowed_extras = allowed_extras | {"usage_metadata"}

            # In LangChain v1.x, tool_calls moved from additional_kwargs to top-level
            # on AI messages. Allow tool_calls to be missing from additional_kwargs.
            if re.search(r"\.messages\[\d+\]\.additional_kwargs$", path):
                allowed_missing = allowed_missing | {"tool_calls"}

            # At root level, toolsCalled key presence can vary due to tracer behavior.
            # The tools-used invariant check above ensures semantic correctness.
            # Evidence: test_multiple_tools, test_async_parallel_tools showed key
            # presence flipping while tools_used semantics remained consistent.
            if path == "root":
                allowed_extras = allowed_extras | {"toolsCalled"}
                allowed_missing = allowed_missing | {"toolsCalled"}

            # Check for missing or extra keys (accounting for schema drift)
            missing_keys = expected_keys - actual_keys - allowed_missing
            extra_keys = actual_keys - expected_keys - allowed_extras

            if missing_keys:
                print(f"❌ Missing keys at '{path}': {missing_keys}")
                return False
            if extra_keys:
                print(f"❌ Extra keys at '{path}': {extra_keys}")
                return False

            # Compare keys that exist in both (skip allowed_missing keys not in actual)
            for key in expected_keys:
                if key not in actual_keys and key in allowed_missing:
                    continue
                # Skip toolsCalled comparison at root since semantics are checked above
                if path == "root" and key == "toolsCalled":
                    # Still validate structure if both have it
                    if key in actual_keys and key in expected_keys:
                        if not _compare(
                            actual[key], expected[key], f"{path}.{key}"
                        ):
                            return False
                    continue
                if not _compare(actual[key], expected[key], f"{path}.{key}"):
                    return False
            return True

        # List vs List
        if isinstance(expected, list):
            if not isinstance(actual, list):
                print(f"❌ Type mismatch at '{path}':")
                print("   Expected: list")
                print(f"   Got: {type(actual).__name__}")
                print(f"   Value: {actual}")
                return False

            # For non-variable-length arrays, require exact length match
            if len(actual) != len(expected):
                print(
                    f"❌ Length mismatch at '{path}': expected {len(expected)}, got {len(actual)}"
                )
                return False

            for idx, (actual_elem, expected_elem) in enumerate(
                zip(actual, expected)
            ):
                if not _compare(actual_elem, expected_elem, f"{path}[{idx}]"):
                    return False
            return True

        # Primitives: exact type match, except int/float interchangeable
        number_types = (int, float)
        if (
            type(expected) in number_types
            and type(actual) in number_types
            and not isinstance(actual, bool)
            and not isinstance(expected, bool)
        ):
            return True

        if type(actual) is not type(expected):
            print(f"❌ Type mismatch at '{path}':")
            print(f"   Expected: {type(expected).__name__}")
            print(f"   Got: {type(actual).__name__}")
            print(f"   Expected value: {expected}")
            print(f"   Actual value: {actual}")
            return False

        return True

    return _compare(actual_json_obj, expected_json_obj)


def load_trace_data(file_path: str):
    with open(file_path, "r") as file:
        return json.load(file)


# Global storage for trace dicts - shared across all imports
_TRACE_STORAGE: Dict[str, Dict[str, Any]] = {}


def _store_trace_for_upload(trace_dict: Dict[str, Any]):
    """Store trace dict for upload by conftest.py hook."""
    # Get current test nodeid from pytest environment
    nodeid = os.environ.get("PYTEST_CURRENT_TEST", "")
    if nodeid:
        # PYTEST_CURRENT_TEST format: "path/to/test.py::TestClass::test_method (call)"
        # Strip the phase suffix
        nodeid = nodeid.rsplit(" ", 1)[0]

    if not nodeid:
        return

    # Store in module-level dict
    _TRACE_STORAGE[nodeid] = trace_dict


def get_stored_trace(nodeid: str) -> Dict[str, Any]:
    """Retrieve and remove a stored trace dict."""
    return _TRACE_STORAGE.pop(nodeid, None)


def generate_trace_json(json_path: str):
    """
    Decorator that generates and saves trace data to a JSON file.

    Usage:
        @generate_trace_json("path/to/output.json")
        async def my_function():
            await some_llm_app("input")

    Args:
        json_path: Path where the trace JSON will be saved
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            from deepeval.tracing.trace_test_manager import (
                trace_testing_manager,
            )

            try:
                trace_testing_manager.test_name = json_path
                result = await func(*args, **kwargs)
                actual_dict = await trace_testing_manager.wait_for_test_dict()

                with open(json_path, "w") as f:
                    json.dump(actual_dict, f, indent=2)

                return result
            finally:
                trace_testing_manager.test_name = None
                trace_testing_manager.test_dict = None

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            from deepeval.tracing.trace_test_manager import (
                trace_testing_manager,
            )

            try:
                trace_testing_manager.test_name = json_path
                result = func(*args, **kwargs)

                # For sync functions, we need to handle the async wait differently
                loop = asyncio.get_event_loop()
                actual_dict = loop.run_until_complete(
                    trace_testing_manager.wait_for_test_dict()
                )

                with open(json_path, "w") as f:
                    json.dump(actual_dict, f, indent=2)

                return result
            finally:
                trace_testing_manager.test_name = None
                trace_testing_manager.test_dict = None

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def assert_trace_json(json_path: str):
    """
    Decorator that tests trace data against an expected JSON file.

    Usage:
        @pytest.mark.asyncio
        @test_trace_json("path/to/expected.json")
        async def test_my_function():
            await some_llm_app("input")

    Args:
        json_path: Path to the expected trace JSON file

    Raises:
        AssertionError: If the actual trace doesn't match the expected structure
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            from deepeval.tracing.trace_test_manager import (
                trace_testing_manager,
            )

            try:
                trace_testing_manager.test_name = json_path
                result = await func(*args, **kwargs)
                actual_dict = await trace_testing_manager.wait_for_test_dict()
                expected_dict = load_trace_data(json_path)

                # Store trace for upload (does not mutate)
                _store_trace_for_upload(actual_dict)

                assert assert_json_object_structure(expected_dict, actual_dict)

                return result
            finally:
                trace_testing_manager.test_name = None
                trace_testing_manager.test_dict = None

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            from deepeval.tracing.trace_test_manager import (
                trace_testing_manager,
            )

            try:
                trace_testing_manager.test_name = json_path
                result = func(*args, **kwargs)

                # For sync functions, we need to handle the async wait differently
                loop = asyncio.get_event_loop()
                actual_dict = loop.run_until_complete(
                    trace_testing_manager.wait_for_test_dict()
                )
                expected_dict = load_trace_data(json_path)

                # Store trace for upload (does not mutate)
                _store_trace_for_upload(actual_dict)

                assert assert_json_object_structure(expected_dict, actual_dict)

                return result
            finally:
                trace_testing_manager.test_name = None
                trace_testing_manager.test_dict = None

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

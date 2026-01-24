import asyncio
import json
import re
import os

from typing import Dict, Any
from functools import wraps
import inspect


def assert_json_object_structure(
    expected_json_obj: Dict[str, Any], actual_json_obj: Dict[str, Any]
) -> bool:
    """
    Validate that `actual_json_obj` matches the structure and data types of the JSON at `expected_file_path`.

    Rules:
    - Dicts: keys must match exactly on both sides; values are validated recursively.
    - Lists: both must be lists; elements are compared pairwise (same length required).
    - Primitives: types must match exactly. Int/float are treated as interchangeable numeric types.
    - Returns False if the file cannot be read or the JSON is invalid.
    """

    def _compare(a: Any, b: Any, path: str = "root") -> bool:
        # Dict vs Dict
        if isinstance(b, dict):
            if not isinstance(a, dict):
                print(f"❌ Type mismatch at '{path}':")
                print("   Expected: dict")
                print(f"   Got: {type(a).__name__}")
                print(f"   Value: {a}")
                return False

            # Filter out keys to ignore globally
            keys_to_ignore = {"tokenIntervals"}
            b_keys = set(b.keys()) - keys_to_ignore
            a_keys = set(a.keys()) - keys_to_ignore

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

            # Keys that are allowed to be extra on message objects (any .messages[N] path)
            # usage_metadata was added in later LangChain versions
            if re.search(r"\.messages\[\d+\]$", path):
                allowed_extras = allowed_extras | {"usage_metadata"}

            # In LangChain v1.x, tool_calls moved from additional_kwargs to top-level
            # on AI messages. Allow tool_calls to be missing from additional_kwargs
            # when we're inside a message's additional_kwargs dict.
            if re.search(r"\.messages\[\d+\]\.additional_kwargs$", path):
                allowed_missing = allowed_missing | {"tool_calls"}

            # Check for missing or extra keys (accounting for schema drift)
            missing_keys = b_keys - a_keys - allowed_missing
            extra_keys = a_keys - b_keys - allowed_extras

            if missing_keys:
                print(f"❌ Missing keys at '{path}': {missing_keys}")
                return False
            if extra_keys:
                print(f"❌ Extra keys at '{path}': {extra_keys}")
                return False

            # Compare only keys that exist in both (skip allowed_missing keys not in actual)
            for k in b_keys:
                if k not in a_keys and k in allowed_missing:
                    # This key is allowed to be missing, skip comparison
                    continue
                if not _compare(a[k], b[k], f"{path}.{k}"):
                    return False
            return True

        # List vs List (pairwise compare)
        if isinstance(b, list):
            if not isinstance(a, list):
                print(f"❌ Type mismatch at '{path}':")
                print("   Expected: list")
                print(f"   Got: {type(a).__name__}")
                print(f"   Value: {a}")
                return False

            # LLM nondeterminism handling: certain arrays can vary in length
            # because the LLM may make different tool-calling decisions across
            # runs despite temperature=0 and seed settings. We validate structure
            # of the first element but allow count to vary.
            #
            # Observed in LangGraph v1.x: tool call counts vary for parallel
            # tool tests, multi-tool tests, and heavy workload tests.
            variable_length_arrays = {
                # Top-level span arrays (LLM may produce different span counts)
                "root.llmSpans",
                "root.baseSpans",
                "root.toolSpans",
                "root.agentSpans",
                "root.retrieverSpans",
                # Top-level toolsCalled (LLM tool decisions vary)
                "root.toolsCalled",
                # Message arrays (conversation length varies with tool calls)
                "root.output.messages",
                "root.input.messages",
            }

            # Also match toolsCalled within baseSpans at any index
            # Pattern: root.baseSpans[N].toolsCalled
            is_nested_tools_called = re.match(
                r"^root\.baseSpans\[\d+\]\.toolsCalled$", path
            )
            # Pattern: root.baseSpans[N].input.messages or output.messages
            is_nested_messages = re.match(
                r"^root\.baseSpans\[\d+\]\.(input|output)\.messages$", path
            )
            # Pattern: root.llmSpans[N].input (LLM input is conversation history)
            is_llm_input = re.match(r"^root\.llmSpans\[\d+\]\.input$", path)

            if (
                path in variable_length_arrays
                or is_nested_tools_called
                or is_nested_messages
                or is_llm_input
            ):
                # Validate: if expected has items, actual must have at least one
                if len(b) > 0 and len(a) == 0:
                    print(
                        f"❌ Empty array at '{path}': expected at least 1 item"
                    )
                    return False

                # Validate structure of first element only
                if len(a) > 0 and len(b) > 0:
                    if not _compare(a[0], b[0], f"{path}[0]"):
                        return False
                return True

            # For all other arrays, require exact length match
            if len(a) != len(b):
                print(
                    f"❌ Length mismatch at '{path}': expected {len(b)}, got {len(a)}"
                )
                return False

            for idx, (ae, be) in enumerate(zip(a, b)):
                if not _compare(ae, be, f"{path}[{idx}]"):
                    return False
            return True

        # Primitives: exact type match, except int/float interchangeable (bool is not numeric here)
        number_types = (int, float)
        if (
            type(b) in number_types
            and type(a) in number_types
            and not isinstance(a, bool)
            and not isinstance(b, bool)
        ):
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

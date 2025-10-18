import json
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

            # Filter out keys to ignore
            keys_to_ignore = {"tokenIntervals"}
            b_keys = set(b.keys()) - keys_to_ignore
            a_keys = set(a.keys()) - keys_to_ignore

            # Check for missing or extra keys
            missing_keys = b_keys - a_keys
            extra_keys = a_keys - b_keys

            if missing_keys:
                print(f"❌ Missing keys at '{path}': {missing_keys}")
                return False
            if extra_keys:
                print(f"❌ Extra keys at '{path}': {extra_keys}")
                return False

            # Compare only non-ignored keys
            for k in b_keys:
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
            import asyncio

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
            import asyncio

            try:
                trace_testing_manager.test_name = json_path
                result = func(*args, **kwargs)

                # For sync functions, we need to handle the async wait differently
                loop = asyncio.get_event_loop()
                actual_dict = loop.run_until_complete(
                    trace_testing_manager.wait_for_test_dict()
                )
                expected_dict = load_trace_data(json_path)

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

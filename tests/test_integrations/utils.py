import sys
import os
import json
import difflib
from typing import Optional, Sequence
import time

PLACEHOLDER = "<is_present>"

def _apply_placeholders(expected, actual, path=""):
    if expected == PLACEHOLDER:
        return actual
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"Type mismatch at {path or '<root>'}: expected object, got {type(actual).__name__}")
        out = {}
        for k, v in expected.items():
            sub_path = f"{path}.{k}" if path else k
            if v == PLACEHOLDER:
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
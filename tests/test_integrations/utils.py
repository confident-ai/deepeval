import os
import sys
import time
import json
import difflib
import re
from datetime import datetime
from typing import Callable

PLACEHOLDER = "<is_present>"
TIME_KEYS = {"startTime", "endTime"}


def _is_iso_timestamp(value: str) -> bool:
    if not isinstance(value, str):
        return False
    # Matches YYYY-MM-DDTHH:MM:SS(.ffffff)?(Z|+HH:MM|-HH:MM)
    pattern = re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
    )
    if not pattern.match(value):
        return False
    try:
        # Python's fromisoformat doesn't accept 'Z'; replace with +00:00
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def generate_test_json(func: Callable, name: str, *args, **kwargs):
    target_path = os.path.abspath(name)
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    tmp_path = target_path + ".tmp"

    def _run_in_gen_mode(output_path: str):
        original_argv = list(sys.argv)
        try:
            new_argv = []
            replaced_mode = False
            i = 0
            while i < len(original_argv):
                arg = original_argv[i]
                if isinstance(arg, str) and arg.startswith(
                    "--deepeval-trace-mode="
                ):
                    new_argv.append("--deepeval-trace-mode=gen")
                    replaced_mode = True
                elif arg == "--deepeval-trace-mode":
                    new_argv.append("--deepeval-trace-mode")
                    if i + 1 < len(original_argv):
                        # Skip the next original value
                        i += 1
                    new_argv.append("gen")
                    replaced_mode = True
                # Remove any existing --deepeval-trace-file-name to avoid conflicts
                elif isinstance(arg, str) and arg.startswith(
                    "--deepeval-trace-file-name="
                ):
                    pass
                elif arg == "--deepeval-trace-file-name":
                    # Skip the value token as well
                    if i + 1 < len(original_argv):
                        i += 1
                else:
                    new_argv.append(arg)
                i += 1

            if not replaced_mode:
                new_argv.append("--deepeval-trace-mode=gen")

            # Always enforce our target output path
            new_argv.append(f"--deepeval-trace-file-name={output_path}")

            sys.argv = new_argv
            func(*args, **kwargs)

            # Wait for file to appear (and be non-empty)
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    if (
                        os.path.exists(output_path)
                        and os.path.getsize(output_path) > 0
                    ):
                        break
                except Exception:
                    pass
                time.sleep(0.25)
        finally:
            sys.argv = original_argv

    # First generation -> target_path
    _run_in_gen_mode(target_path)
    # Second generation -> tmp_path
    _run_in_gen_mode(tmp_path)

    # Load both and mark dynamic fields
    with open(target_path, "r", encoding="utf-8") as f1:
        first = json.load(f1)
    with open(tmp_path, "r", encoding="utf-8") as f2:
        second = json.load(f2)

    marked = _mark_differences(first, second)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(marked, f, ensure_ascii=False, indent=2, sort_keys=True)

    # Cleanup
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass


def compare_trace_files(expected_file_path: str, actual_file_path: str):
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
            expected_str = json.dumps(
                expected_aligned, ensure_ascii=False, indent=2, sort_keys=True
            )
            actual_str = json.dumps(
                actual, ensure_ascii=False, indent=2, sort_keys=True
            )
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
        raise AssertionError(
            f"Trace body does not match expected file: {expected_file_path}\n{diff}"
        )

    print(f"Test trace body passed: {expected_file_path}")
    return


def _apply_placeholders(expected, actual, path=""):
    if expected == PLACEHOLDER:
        return actual
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(
                f"Type mismatch at {path or '<root>'}: expected object, got {type(actual).__name__}"
            )
        out = {}
        for k, v in expected.items():
            sub_path = f"{path}.{k}" if path else k
            # For time keys, always validate format and accept actual value
            if k in TIME_KEYS:
                if k not in actual:
                    raise AssertionError(f"Missing required key at {sub_path}")
                if not _is_iso_timestamp(actual[k]):
                    raise AssertionError(
                        f"Invalid ISO timestamp at {sub_path}: {actual[k]!r}"
                    )
                out[k] = actual[k]
                continue
            if v == PLACEHOLDER:
                if k not in actual:
                    raise AssertionError(f"Missing required key at {sub_path}")
                out[k] = actual[k]
            else:
                out[k] = _apply_placeholders(v, actual.get(k), sub_path)
        return out
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise AssertionError(
                f"Type mismatch at {path or '<root>'}: expected list, got {type(actual).__name__}"
            )
        if len(expected) != len(actual):
            raise AssertionError(
                f"Length mismatch at {path or '<root>'}: expected {len(expected)}, got {len(actual)}"
            )
        return [
            _apply_placeholders(ev, av, f"{path}[{i}]")
            for i, (ev, av) in enumerate(zip(expected, actual))
        ]
    return expected


def _mark_differences(expected, actual):
    if expected == PLACEHOLDER:
        return PLACEHOLDER
    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = set(expected.keys()) | set(actual.keys())
        out = {}
        for k in keys:
            ev = expected.get(k)
            av = actual.get(k)
            # For time keys, keep the original value and don't mark as placeholder
            if k in TIME_KEYS:
                out[k] = ev
                continue
            if ev == PLACEHOLDER:
                out[k] = PLACEHOLDER
            elif k not in expected:
                out[k] = PLACEHOLDER
            elif k not in actual:
                out[k] = ev
            else:
                if isinstance(ev, dict) and isinstance(av, dict):
                    out[k] = _mark_differences(ev, av)
                elif isinstance(ev, list) and isinstance(av, list):
                    out[k] = _mark_differences(ev, av)
                else:
                    out[k] = ev if ev == av else PLACEHOLDER
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return PLACEHOLDER
        marked = []
        for ev, av in zip(expected, actual):
            if isinstance(ev, (dict, list)) or isinstance(av, (dict, list)):
                marked.append(_mark_differences(ev, av))
            else:
                marked.append(ev if ev == av else PLACEHOLDER)
        return marked
    return expected if expected == actual else PLACEHOLDER

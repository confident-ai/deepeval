"""
Regression test for file encoding consistency (closes encoding audit).

Several open() calls in deepeval used the locale encoding instead of
explicitly specifying UTF-8. On hosts with non-UTF-8 locale, this
could cause UnicodeDecodeError or silent mojibake when reading
prompt templates, model caches, or GPU memory info.
"""

import ast
import os
import pytest


class TestFileEncodingConsistency:
    """All text file open() calls should specify encoding='utf-8'."""

    def _get_open_calls_without_encoding(self, filepath):
        """Find open() calls that don't specify encoding."""
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "open":
                    # Check if encoding keyword is present
                    has_encoding = any(
                        kw.arg == "encoding" for kw in node.keywords
                    )
                    if not has_encoding:
                        issues.append(node.lineno)

        return issues

    def test_prompt_py_uses_utf8(self):
        """prompt.py load() must use UTF-8."""
        filepath = os.path.join(
            os.path.dirname(__file__), "..", "deepeval", "prompt", "prompt.py"
        )
        issues = self._get_open_calls_without_encoding(filepath)
        assert (
            not issues
        ), f"prompt.py has open() calls without encoding at lines: {issues}"

    def test_summac_model_uses_utf8(self):
        """_summac_model.py cache save/load must use UTF-8."""
        filepath = os.path.join(
            os.path.dirname(__file__),
            "..",
            "deepeval",
            "models",
            "_summac_model.py",
        )
        issues = self._get_open_calls_without_encoding(filepath)
        assert (
            not issues
        ), f"_summac_model.py has open() calls without encoding at lines: {issues}"

    def test_utils_uses_utf8(self):
        """utils.py GPU memory readers must use UTF-8."""
        filepath = os.path.join(
            os.path.dirname(__file__), "..", "deepeval", "utils.py"
        )
        issues = self._get_open_calls_without_encoding(filepath)
        assert (
            not issues
        ), f"utils.py has open() calls without encoding at lines: {issues}"

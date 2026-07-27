"""
Regression tests for Python 3.9 compatibility with PEP 604 union annotations
(GitHub issue #2972).

Three files used ``X | Y`` union syntax in runtime-evaluated annotation
positions without ``from __future__ import annotations``, which breaks
imports on Python 3.9 (``TypeError: unsupported operand type(s) for |``).

The fix replaces ``X | Y`` with ``Union[X, Y]`` / ``Optional[X]`` from
``typing``, matching the pattern established by the earlier fix in #2049
(commit 0aed6f2).

These tests verify that the affected modules can be imported without
raising ``TypeError`` regardless of the Python version.
"""

import ast
import importlib
import pytest
from pathlib import Path

# Files that previously contained bare ``X | Y`` annotations
AFFECTED_FILES = [
    "deepeval/tracing/otel/utils.py",
    "deepeval/models/unbias_model.py",
    "deepeval/models/detoxify_model.py",
]


class TestNoBareUnionAnnotations:
    """Static check: no ``BinOp(Or)`` in annotation positions without future import."""

    def _has_future_annotations(self, source: str) -> bool:
        """Check if the source has ``from __future__ import annotations``."""
        return "from __future__ import annotations" in source

    def _find_bare_union_annotations(self, filepath: Path) -> list[str]:
        """Walk the AST and find ``X | Y`` in annotation positions."""
        source = filepath.read_text()
        if self._has_future_annotations(source):
            return []  # future import makes all annotations strings

        tree = ast.parse(source)
        violations = []
        for node in ast.walk(tree):
            # Check function argument annotations and return annotations
            if isinstance(node, ast.FunctionDef):
                for arg in (
                    node.args.args
                    + node.args.posonlyargs
                    + node.args.kwonlyargs
                ):
                    if arg.annotation and self._contains_binop_or(
                        arg.annotation
                    ):
                        violations.append(
                            f"{filepath}:{node.lineno} func {node.name} arg {arg.arg}"
                        )
                if node.returns and self._contains_binop_or(node.returns):
                    violations.append(
                        f"{filepath}:{node.lineno} func {node.name} return"
                    )
            # Check variable annotations
            if isinstance(node, ast.AnnAssign) and self._contains_binop_or(
                node.annotation
            ):
                violations.append(f"{filepath}:{node.lineno} AnnAssign")
        return violations

    @staticmethod
    def _contains_binop_or(node: ast.AST) -> bool:
        """Check if an AST node contains a ``X | Y`` BinOp."""
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr):
                return True
        return False

    def test_no_bare_union_in_affected_files(self):
        """All previously-broken files must no longer have bare ``X | Y`` annotations."""
        root = Path(__file__).parents[2]  # project root
        all_violations = []
        for rel_path in AFFECTED_FILES:
            filepath = root / rel_path
            if filepath.exists():
                violations = self._find_bare_union_annotations(filepath)
                all_violations.extend(violations)

        assert not all_violations, (
            f"Found bare 'X | Y' annotations that break Python 3.9:\n"
            + "\n".join(all_violations)
        )


class TestAffectedModulesImportable:
    """Runtime check: the affected modules must import without TypeError."""

    def test_otel_utils_importable(self):
        """deepeval.tracing.otel.utils must import without TypeError."""
        # This import chain previously failed on 3.9 due to ``int | bytes``
        mod = importlib.import_module("deepeval.tracing.otel.utils")
        assert hasattr(mod, "to_hex_string")

    def test_to_hex_string_accepts_int(self):
        """to_hex_string must work with integer input."""
        from deepeval.tracing.otel.utils import to_hex_string

        result = to_hex_string(0xDEADBEEF)
        assert isinstance(result, str)
        assert len(result) == 32  # zero-padded to 32 chars

    def test_to_hex_string_accepts_bytes(self):
        """to_hex_string must work with bytes input."""
        from deepeval.tracing.otel.utils import to_hex_string

        result = to_hex_string(b"\xde\xad\xbe\xef")
        assert isinstance(result, str)

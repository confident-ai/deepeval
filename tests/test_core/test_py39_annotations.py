"""Guard against `X | Y` annotations that break imports on Python 3.9.

PEP 604 unions in runtime-evaluated positions (function signatures, class
attributes) raise TypeError at import time on 3.9 unless the module has
`from __future__ import annotations`. This has been fixed twice (#2049,
#2972); this test keeps new instances out regardless of the Python version
running the suite.
"""

import ast
from pathlib import Path

import deepeval

PACKAGE_ROOT = Path(deepeval.__file__).parent


def _has_future_annotations(tree):
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(alias.name == "annotations" for alias in node.names):
                return True
    return False


def _contains_union(annotation):
    return any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
        for node in ast.walk(annotation)
    )


def _runtime_union_annotation_lines(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            annotations = [
                arg.annotation
                for arg in (
                    args.args
                    + args.posonlyargs
                    + args.kwonlyargs
                    + ([args.vararg] if args.vararg else [])
                    + ([args.kwarg] if args.kwarg else [])
                )
                if arg.annotation is not None
            ]
            if node.returns is not None:
                annotations.append(node.returns)
            for annotation in annotations:
                if _contains_union(annotation):
                    yield annotation.lineno
        elif isinstance(node, ast.AnnAssign):
            if _contains_union(node.annotation):
                yield node.lineno


def test_no_pep604_unions_in_runtime_annotations():
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _has_future_annotations(tree):
            continue
        for lineno in _runtime_union_annotation_lines(tree):
            offenders.append(
                f"{path.relative_to(PACKAGE_ROOT).as_posix()}:{lineno}"
            )
    assert not offenders, (
        "PEP 604 unions evaluated at runtime break imports on Python 3.9. "
        "Use typing.Union / typing.Optional, or add `from __future__ "
        "import annotations` to: " + ", ".join(offenders)
    )

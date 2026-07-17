import ast
import pathlib
import warnings

from deepeval.evaluate.execute import loop as loop_module


def _returns_in_finally(source: str) -> list:
    """Return the line numbers of any `return` statement nested inside a
    `finally` block. A `return` inside `finally` silently swallows any
    exception that is propagating out of the corresponding `try`, so the
    async evaluation loop must never contain one."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Try) and node.finalbody:
            for stmt in node.finalbody:
                for child in ast.walk(stmt):
                    if isinstance(child, ast.Return):
                        lines.append(child.lineno)
    return lines


def test_loop_has_no_return_in_finally():
    source = pathlib.Path(loop_module.__file__).read_text(encoding="utf-8")
    assert _returns_in_finally(source) == []


def test_loop_compiles_without_syntax_warning():
    source = pathlib.Path(loop_module.__file__).read_text(encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        # Raises SyntaxWarning-as-error if a `return` sits in a finally block.
        compile(source, loop_module.__file__, "exec")

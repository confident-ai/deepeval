"""Every metric's measure() must return its score under async_mode=True.

Regression net for https://github.com/confident-ai/deepeval/issues/3088.
46 of 47 metrics ran ``loop.run_until_complete(self.a_measure(...))`` in the
async branch of ``measure()`` and discarded the result, so ``measure()``
returned ``None`` under the default mode while the sync branch returned the
score. Callers that rely on the return value (including
``deepeval/optimizer/scorer/scorer.py``, which wraps it in ``float()``) broke
only on the default path.

The check is structural, so it needs no model or API key and covers every
metric, including ones added after this test: any ``measure()`` whose
``async_mode`` branch fails to return a value fails here by name.
"""

import ast
from pathlib import Path

METRICS_DIR = Path(__file__).resolve().parents[2] / "deepeval" / "metrics"


def _async_branch_discards(measure_fn: ast.FunctionDef) -> bool:
    """True when the async_mode branch contains run_until_complete but no
    return-with-value, i.e. the a_measure result is computed and dropped."""
    for node in ast.walk(measure_fn):
        if not isinstance(node, ast.If):
            continue
        if "async_mode" not in ast.unparse(node.test):
            continue
        branch = ast.Module(body=node.body, type_ignores=[])
        runs_coro = any(
            isinstance(n, ast.Attribute) and n.attr == "run_until_complete"
            for n in ast.walk(branch)
        )
        returns_value = any(
            isinstance(n, ast.Return) and n.value is not None
            for n in ast.walk(branch)
        )
        if runs_coro and not returns_value:
            return True
    return False


def test_every_metric_measure_returns_under_async_mode():
    offenders = []
    for path in sorted(METRICS_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        if "run_until_complete" not in source:
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "measure":
                if _async_branch_discards(node):
                    offenders.append(
                        str(path.relative_to(METRICS_DIR.parent.parent))
                    )
    assert not offenders, (
        "measure() discards the a_measure result in the async_mode branch, so "
        "it returns None under the default async_mode=True, in: "
        + ", ".join(offenders)
    )

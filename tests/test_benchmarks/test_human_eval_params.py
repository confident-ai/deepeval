"""
Tests for `HumanEval` benchmark parameter validation.

`n` (samples per task) and `k` (pass@k) were previously unvalidated, and the
only check in `evaluate` was a bare `assert self.n >= k` that is stripped
under `python -O`. Invalid values now raise a clear ValueError at the source,
while valid configurations behave exactly as before.
"""

import subprocess
import sys
import types

import pytest

from deepeval.benchmarks.human_eval.human_eval import HumanEval


@pytest.fixture
def fake_datasets(monkeypatch):
    """`DeepEvalBaseBenchmark.__init__` imports the optional HF `datasets`
    package; stub it out so the valid-construction path works offline."""

    module = types.ModuleType("datasets")

    class Dataset:
        pass

    module.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return module


# --------------------------------------------------------------------------- #
# Constructor validation for `n`
# --------------------------------------------------------------------------- #


def test_human_eval_rejects_zero_samples():
    with pytest.raises(ValueError, match="'n'.*positive integer"):
        HumanEval(n=0)


def test_human_eval_rejects_negative_samples():
    with pytest.raises(ValueError, match="'n'.*positive integer"):
        HumanEval(n=-1)


def test_human_eval_rejects_float_samples():
    with pytest.raises(ValueError, match="'n'.*positive integer"):
        HumanEval(n=200.0)


def test_human_eval_rejects_bool_samples():
    with pytest.raises(ValueError, match="'n'.*positive integer"):
        HumanEval(n=True)


def test_human_eval_rejects_string_samples():
    with pytest.raises(ValueError, match="'n'.*positive integer"):
        HumanEval(n="200")


def test_human_eval_accepts_positive_samples(fake_datasets):
    bench = HumanEval(n=1)
    assert bench.n == 1


# --------------------------------------------------------------------------- #
# `evaluate` validation for `k`
# --------------------------------------------------------------------------- #


class _DummyModel:
    """Never invoked: the k checks must fire before any model call."""


def _bench_with_n(n: int) -> HumanEval:
    # Bypass __init__ so only the `evaluate` boundary is exercised.
    bench = HumanEval.__new__(HumanEval)
    bench.n = n
    bench.tasks = []
    bench.c = {}
    bench.functions = {}
    bench.verbose_mode = False
    return bench


def test_evaluate_rejects_zero_k():
    with pytest.raises(ValueError, match="'k'.*positive integer"):
        _bench_with_n(200).evaluate(_DummyModel(), k=0)


def test_evaluate_rejects_negative_k():
    with pytest.raises(ValueError, match="'k'.*positive integer"):
        _bench_with_n(200).evaluate(_DummyModel(), k=-1)


def test_evaluate_rejects_float_k():
    with pytest.raises(ValueError, match="'k'.*positive integer"):
        _bench_with_n(200).evaluate(_DummyModel(), k=1.5)


def test_evaluate_rejects_bool_k():
    with pytest.raises(ValueError, match="'k'.*positive integer"):
        _bench_with_n(200).evaluate(_DummyModel(), k=True)


def test_evaluate_rejects_string_k():
    with pytest.raises(ValueError, match="'k'.*positive integer"):
        _bench_with_n(200).evaluate(_DummyModel(), k="1")


def test_evaluate_rejects_k_greater_than_n():
    # Replaces the old bare `assert self.n >= k` (stripped under -O).
    with pytest.raises(ValueError, match="'n' \\(5\\).*'k' \\(6\\)"):
        _bench_with_n(5).evaluate(_DummyModel(), k=6)


# --------------------------------------------------------------------------- #
# Verification under python -O (optimized execution)
# --------------------------------------------------------------------------- #


def test_validation_active_under_python_optimized():
    """Verify that validation checks remain active when assert statements are stripped."""
    code = (
        "import pytest\n"
        "from deepeval.benchmarks.human_eval.human_eval import HumanEval\n"
        "class _Dummy:\n"
        "    pass\n"
        "bench = HumanEval.__new__(HumanEval)\n"
        "bench.n = 5\n"
        "bench.tasks = []\n"
        "bench.c = {}\n"
        "bench.functions = {}\n"
        "bench.verbose_mode = False\n"
        "try:\n"
        "    bench.evaluate(_Dummy(), k=6)\n"
        "    sys.exit(1)\n"
        "except ValueError:\n"
        "    pass\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Validation failed under python -O: {result.stderr}"

"""
Tests for `HumanEval` benchmark parameter validation.

`n` (samples per task) and `k` (pass@k) were previously unvalidated, and the
only check in `evaluate` was a bare `assert self.n >= k` that is stripped
under `python -O`. Invalid values now raise a clear ValueError at the source,
while valid configurations behave exactly as before.
"""

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


def test_human_eval_rejects_non_integer_samples():
    with pytest.raises(ValueError, match="'n'.*positive integer"):
        HumanEval(n=200.0)


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


def test_evaluate_rejects_non_integer_k():
    with pytest.raises(ValueError, match="'k'.*positive integer"):
        _bench_with_n(200).evaluate(_DummyModel(), k=1.5)


def test_evaluate_rejects_k_greater_than_n():
    # Replaces the old bare `assert self.n >= k` (stripped under -O).
    with pytest.raises(ValueError, match="'n' \\(5\\).*'k' \\(6\\)"):
        _bench_with_n(5).evaluate(_DummyModel(), k=6)

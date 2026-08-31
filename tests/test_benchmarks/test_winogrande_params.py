"""Tests for Winogrande benchmark argument validation.

``Winogrande.__init__`` used bare ``assert`` statements to validate ``n_shots``
and ``n_problems``. ``assert`` is an anti-pattern here: it is stripped when the
interpreter runs with ``-O``/``-OO``, it raises ``AssertionError`` instead of
the ``ValueError``/``TypeError`` callers expect, and it only checked upper
bounds. In particular ``n_problems=0`` passed validation even though it makes
``evaluate()`` divide by zero.

These tests verify that:
  * invalid types raise ``TypeError``;
  * ``n_shots`` outside ``[0, 5]`` (zero-shot is allowed) raises ``ValueError``;
  * ``n_problems`` outside ``[1, 1267]`` raises ``ValueError``;
  * the default and previously-valid configurations still construct fine
    (no regression).

The tests are fully offline: the ``datasets`` package is stubbed out, so no
dataset download or network access is required.
"""

import sys
import types

import pytest

from deepeval.benchmarks.winogrande.winogrande import Winogrande


@pytest.fixture
def fake_datasets(monkeypatch):
    """Fake the optional ``datasets`` package imported by the base benchmark."""
    module = types.ModuleType("datasets")

    class Dataset:
        pass

    module.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return module


def test_default_construction_succeeds(fake_datasets):
    bench = Winogrande()
    assert bench.n_shots == 5
    assert bench.n_problems == 1267


def test_zero_shot_is_allowed(fake_datasets):
    bench = Winogrande(n_shots=0)
    assert bench.n_shots == 0


def test_valid_non_default_values_construct(fake_datasets):
    bench = Winogrande(n_shots=3, n_problems=100)
    assert bench.n_shots == 3
    assert bench.n_problems == 100


def test_n_shots_above_upper_bound_rejected(fake_datasets):
    with pytest.raises(ValueError, match="n_shots.*5"):
        Winogrande(n_shots=6)


def test_n_shots_negative_rejected(fake_datasets):
    with pytest.raises(ValueError, match="n_shots"):
        Winogrande(n_shots=-1)


def test_n_shots_non_integer_rejected(fake_datasets):
    with pytest.raises(TypeError, match="'n_shots'.*integer"):
        Winogrande(n_shots=2.5)


def test_n_problems_above_upper_bound_rejected(fake_datasets):
    with pytest.raises(ValueError, match="n_problems.*1267"):
        Winogrande(n_problems=1268)


def test_n_problems_negative_rejected(fake_datasets):
    with pytest.raises(ValueError, match="n_problems"):
        Winogrande(n_problems=-100)


def test_n_problems_zero_rejected(fake_datasets):
    # Previously accepted (0 <= 1267), but evaluate() divides by n_problems,
    # so 0 would crash with a ZeroDivisionError.
    with pytest.raises(ValueError, match="n_problems.*1"):
        Winogrande(n_problems=0)


def test_n_problems_non_integer_rejected(fake_datasets):
    with pytest.raises(TypeError, match="'n_problems'.*integer"):
        Winogrande(n_problems="100")

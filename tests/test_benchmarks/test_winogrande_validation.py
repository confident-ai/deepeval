import os
import subprocess
import sys
import types
import pytest

from deepeval.benchmarks.winogrande.winogrande import Winogrande


@pytest.fixture(autouse=True)
def fake_datasets(monkeypatch):
    """Fake the optional ``datasets`` package imported by the base benchmark."""
    module = types.ModuleType("datasets")

    class Dataset:
        pass

    module.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return module


def test_winogrande_default_construction():
    bench = Winogrande()
    assert bench.n_shots == 5
    assert bench.n_problems == 1267


def test_winogrande_zero_shot_is_allowed():
    bench = Winogrande(n_shots=0)
    assert bench.n_shots == 0


def test_winogrande_boundaries_allowed():
    bench = Winogrande(n_shots=0, n_problems=1)
    assert bench.n_shots == 0
    assert bench.n_problems == 1

    bench_max = Winogrande(n_shots=5, n_problems=1267)
    assert bench_max.n_shots == 5
    assert bench_max.n_problems == 1267


@pytest.mark.parametrize("n_shots", [-1, -10, 6, 100])
def test_winogrande_n_shots_out_of_bounds_rejected(n_shots):
    with pytest.raises(ValueError, match="'n_shots' must be between 0 and 5"):
        Winogrande(n_shots=n_shots)


@pytest.mark.parametrize("n_shots", ["5", 2.5, None, [1]])
def test_winogrande_n_shots_non_integer_rejected(n_shots):
    with pytest.raises(TypeError, match="'n_shots' must be an integer"):
        Winogrande(n_shots=n_shots)


@pytest.mark.parametrize("n_problems", [0, -1, -50, 1268, 5000])
def test_winogrande_n_problems_out_of_bounds_rejected(n_problems):
    with pytest.raises(
        ValueError, match="'n_problems' must be between 1 and 1267"
    ):
        Winogrande(n_problems=n_problems)


@pytest.mark.parametrize("n_problems", ["100", 50.5, None, (1,)])
def test_winogrande_n_problems_non_integer_rejected(n_problems):
    with pytest.raises(TypeError, match="'n_problems' must be an integer"):
        Winogrande(n_problems=n_problems)


@pytest.mark.parametrize(
    "code",
    [
        (
            "from deepeval.benchmarks.winogrande.winogrande import Winogrande\n"
            "try:\n"
            "    Winogrande(n_shots=6)\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise SystemExit('out-of-range n_shots accepted under -O')\n"
        ),
        (
            "from deepeval.benchmarks.winogrande.winogrande import Winogrande\n"
            "try:\n"
            "    Winogrande(n_problems=0)\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise SystemExit('n_problems=0 accepted under -O')\n"
        ),
    ],
)
def test_winogrande_validation_survives_python_optimize(code):
    """The checks must survive python -O (bare asserts would be stripped)."""
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": repo_root},
    )
    assert result.returncode == 0, result.stderr

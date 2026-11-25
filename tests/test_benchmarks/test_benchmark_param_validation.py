"""Tests for benchmark constructor argument validation.

The following benchmarks validated ``n_shots`` (and sometimes ``n_problems``)
with bare ``assert`` statements: HellaSwag, GSM8K, BoolQ, ARC, BBQ, MathQA,
MMLU, LogiQA, LAMBADA, DROP, SQuAD and BigBenchHard. ``assert`` is an
anti-pattern here: it is stripped when the interpreter runs with
``-O``/``-OO``, it raises ``AssertionError`` instead of the
``ValueError``/``TypeError`` callers expect, and it only checked upper bounds.
In particular ``n_problems=0`` passed validation even though ``evaluate()``
divides the number of correct predictions by it.

These tests verify that, for every affected benchmark:
  * the default and previously-valid configurations still construct (no
    regression);
  * invalid types raise ``TypeError``;
  * ``n_shots`` outside ``[0, max_shots]`` (zero-shot is allowed) raises
    ``ValueError``;
  * ``n_problems`` outside ``[1, max_problems]`` raises ``ValueError``;
  * the checks still fire under ``python -O`` (where asserts vanish).

The tests are fully offline: the ``datasets`` package is stubbed out, so no
dataset download or network access is required.
"""

import os
import subprocess
import sys
import types

import pytest

from deepeval.benchmarks.arc.arc import ARC
from deepeval.benchmarks.arc.mode import ARCMode
from deepeval.benchmarks.bbq.bbq import BBQ
from deepeval.benchmarks.big_bench_hard.big_bench_hard import BigBenchHard
from deepeval.benchmarks.bool_q.bool_q import BoolQ
from deepeval.benchmarks.drop.drop import DROP
from deepeval.benchmarks.gsm8k.gsm8k import GSM8K
from deepeval.benchmarks.hellaswag.hellaswag import HellaSwag
from deepeval.benchmarks.lambada.lambada import LAMBADA
from deepeval.benchmarks.logi_qa.logi_qa import LogiQA
from deepeval.benchmarks.math_qa.math_qa import MathQA
from deepeval.benchmarks.mmlu.mmlu import MMLU
from deepeval.benchmarks.squad.squad import SQuAD


@pytest.fixture
def fake_datasets(monkeypatch):
    """Fake the optional ``datasets`` package imported by the base benchmark."""
    module = types.ModuleType("datasets")

    class Dataset:
        pass

    module.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    # ``SQuAD.__init__`` normalizes its evaluation model through
    # ``initialize_model``, which constructs an ``OpenAIModel`` unless a key is
    # configured. A dummy key lets the constructor run without any network I/O.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return module


# --------------------------------------------------------------------------- #
# Benchmarks that validate ``n_shots`` only
# --------------------------------------------------------------------------- #

N_SHOTS_ONLY = [
    (HellaSwag, 10, 15),
    (BBQ, 5, 5),
    (MathQA, 5, 5),
    (MMLU, 5, 5),
    (LogiQA, 5, 5),
    (SQuAD, 5, 5),
    (DROP, 5, 5),
    (BigBenchHard, 3, 3),
]

# Benchmarks that also validate ``n_problems`` (used as a divisor in
# ``evaluate()``, so 0 must be rejected).
N_SHOTS_AND_PROBLEMS = [
    (GSM8K, 3, 15, 1319),
    (BoolQ, 5, 5, 3270),
    (LAMBADA, 5, 5, 5153),
]


@pytest.mark.parametrize(
    "benchmark_cls, default_shots, max_shots", N_SHOTS_ONLY
)
def test_default_construction_succeeds(
    fake_datasets, benchmark_cls, default_shots, max_shots
):
    bench = benchmark_cls()
    assert bench.n_shots == default_shots


@pytest.mark.parametrize(
    "benchmark_cls, default_shots, max_shots", N_SHOTS_ONLY
)
def test_zero_shot_is_allowed(
    fake_datasets, benchmark_cls, default_shots, max_shots
):
    bench = benchmark_cls(n_shots=0)
    assert bench.n_shots == 0


@pytest.mark.parametrize(
    "benchmark_cls, default_shots, max_shots", N_SHOTS_ONLY
)
def test_n_shots_above_upper_bound_rejected(
    fake_datasets, benchmark_cls, default_shots, max_shots
):
    with pytest.raises(ValueError, match="n_shots"):
        benchmark_cls(n_shots=max_shots + 1)


@pytest.mark.parametrize(
    "benchmark_cls, default_shots, max_shots", N_SHOTS_ONLY
)
def test_n_shots_negative_rejected(
    fake_datasets, benchmark_cls, default_shots, max_shots
):
    with pytest.raises(ValueError, match="n_shots"):
        benchmark_cls(n_shots=-1)


@pytest.mark.parametrize(
    "benchmark_cls, default_shots, max_shots", N_SHOTS_ONLY
)
def test_n_shots_non_integer_rejected(
    fake_datasets, benchmark_cls, default_shots, max_shots
):
    with pytest.raises(TypeError, match="'n_shots'.*integer"):
        benchmark_cls(n_shots=2.5)


@pytest.mark.parametrize(
    "benchmark_cls, default_shots, max_shots, max_problems",
    N_SHOTS_AND_PROBLEMS,
)
def test_problem_benchmarks_default_construction_succeeds(
    fake_datasets, benchmark_cls, default_shots, max_shots, max_problems
):
    bench = benchmark_cls()
    assert bench.n_shots == default_shots
    assert bench.n_problems == max_problems


@pytest.mark.parametrize(
    "benchmark_cls, default_shots, max_shots, max_problems",
    N_SHOTS_AND_PROBLEMS,
)
def test_problem_benchmarks_n_shots_rejected(
    fake_datasets, benchmark_cls, default_shots, max_shots, max_problems
):
    with pytest.raises(ValueError, match="n_shots"):
        benchmark_cls(n_shots=max_shots + 1)
    with pytest.raises(ValueError, match="n_shots"):
        benchmark_cls(n_shots=-1)
    with pytest.raises(TypeError, match="'n_shots'.*integer"):
        benchmark_cls(n_shots="5")


@pytest.mark.parametrize(
    "benchmark_cls, default_shots, max_shots, max_problems",
    N_SHOTS_AND_PROBLEMS,
)
def test_problem_benchmarks_n_problems_above_upper_bound_rejected(
    fake_datasets, benchmark_cls, default_shots, max_shots, max_problems
):
    with pytest.raises(ValueError, match="n_problems"):
        benchmark_cls(n_problems=max_problems + 1)


@pytest.mark.parametrize(
    "benchmark_cls, default_shots, max_shots, max_problems",
    N_SHOTS_AND_PROBLEMS,
)
def test_problem_benchmarks_n_problems_zero_rejected(
    fake_datasets, benchmark_cls, default_shots, max_shots, max_problems
):
    # Previously accepted (0 <= max), but evaluate() divides by n_problems,
    # so 0 would crash with a ZeroDivisionError.
    with pytest.raises(ValueError, match="n_problems"):
        benchmark_cls(n_problems=0)


@pytest.mark.parametrize(
    "benchmark_cls, default_shots, max_shots, max_problems",
    N_SHOTS_AND_PROBLEMS,
)
def test_problem_benchmarks_n_problems_non_integer_rejected(
    fake_datasets, benchmark_cls, default_shots, max_shots, max_problems
):
    with pytest.raises(TypeError, match="'n_problems'.*integer"):
        benchmark_cls(n_problems="100")


# --------------------------------------------------------------------------- #
# ARC has a mode-dependent n_problems bound
# --------------------------------------------------------------------------- #


def test_arc_easy_defaults(fake_datasets):
    bench = ARC()
    assert bench.n_shots == 5
    assert bench.n_problems == 2376


def test_arc_challenge_defaults(fake_datasets):
    bench = ARC(mode=ARCMode.CHALLENGE)
    assert bench.n_problems == 1172


def test_arc_easy_rejects_problems_above_bound(fake_datasets):
    with pytest.raises(ValueError, match="n_problems.*2376"):
        ARC(mode=ARCMode.EASY, n_problems=2377)


def test_arc_challenge_rejects_problems_above_bound(fake_datasets):
    with pytest.raises(ValueError, match="n_problems.*1172"):
        ARC(mode=ARCMode.CHALLENGE, n_problems=1173)


def test_arc_easy_rejects_zero_problems(fake_datasets):
    with pytest.raises(ValueError, match="n_problems"):
        ARC(mode=ARCMode.EASY, n_problems=0)


def test_arc_rejects_non_integer_problems(fake_datasets):
    with pytest.raises(TypeError, match="'n_problems'.*integer"):
        ARC(mode=ARCMode.EASY, n_problems="10")


# --------------------------------------------------------------------------- #
# The checks must survive `python -O` (bare asserts would be stripped)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "code",
    [
        # n_shots above the HellaSwag bound
        (
            "from deepeval.benchmarks.hellaswag.hellaswag import HellaSwag\n"
            "try:\n"
            "    HellaSwag(n_shots=16)\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise SystemExit('out-of-range n_shots accepted under -O')\n"
        ),
        # n_problems == 0 for GSM8K (a bare assert would never reject it)
        (
            "from deepeval.benchmarks.gsm8k.gsm8k import GSM8K\n"
            "try:\n"
            "    GSM8K(n_problems=0)\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise SystemExit('n_problems=0 accepted under -O')\n"
        ),
    ],
)
def test_mismatch_still_rejected_under_python_optimize(code):
    """The checks must survive `python -O` (bare asserts would be stripped)."""
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

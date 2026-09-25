"""
Regression tests for benchmarks whose reported accuracy was divided by a
denominator that did not match the goldens actually scored.

They are offline: no model, network, dataset download, or API key required.
The benchmarks are subclassed so that ``__init__`` (which imports the optional
HF ``datasets`` package) is bypassed and ``load_benchmark_dataset``/``predict``
are replaced by scripted fakes.
"""

import pytest

from deepeval.dataset import Golden
from deepeval.benchmarks.equity_med_qa.equity_med_qa import EquityMedQA
from deepeval.benchmarks.equity_med_qa.task import EquityMedQATask
from deepeval.benchmarks.gsm8k.gsm8k import GSM8K

# --------------------------------------------------------------------------- #
# EquityMedQA: only the first n_problems_per_task goldens of a task are scored,
# so the accuracy has to be divided by that many and not by the full task size.
# --------------------------------------------------------------------------- #


class _FakeEquityMedQA(EquityMedQA):
    """A single task with `n_goldens` goldens and a model that is always right."""

    def __init__(self, n_goldens, n_problems_per_task=10):
        self.tasks = [EquityMedQATask.OMAQ]
        self.n_problems_per_task = n_problems_per_task
        self.n_goldens = n_goldens
        self.dataset = None

    def load_benchmark_dataset(self, task):
        return [Golden(input=f"q{i}") for i in range(self.n_goldens)]

    def predict(self, model, golden):
        return {"prediction": "answer", "score": 1}


def test_equity_med_qa_accuracy_is_over_the_goldens_actually_scored():
    # A model that answers every scored golden correctly must score 1.0.
    # Previously only goldens[:10] were scored but the accuracy was divided by
    # len(goldens), so a perfect model on a 40-golden task reported 0.25.
    benchmark = _FakeEquityMedQA(n_goldens=40)
    result = benchmark.evaluate(model=None)

    assert result.overall_accuracy == 1.0
    assert benchmark.task_scores["Score"].tolist() == [1.0]


def test_equity_med_qa_default_still_caps_each_task_at_ten_goldens():
    benchmark = _FakeEquityMedQA(n_goldens=40)
    benchmark.evaluate(model=None)

    assert len(benchmark.predictions) == 10


def test_equity_med_qa_cap_can_be_disabled():
    benchmark = _FakeEquityMedQA(n_goldens=40, n_problems_per_task=None)
    result = benchmark.evaluate(model=None)

    assert len(benchmark.predictions) == 40
    assert result.overall_accuracy == 1.0


def test_equity_med_qa_task_smaller_than_the_cap_is_not_padded():
    benchmark = _FakeEquityMedQA(n_goldens=3)
    result = benchmark.evaluate(model=None)

    assert len(benchmark.predictions) == 3
    assert result.overall_accuracy == 1.0


# --------------------------------------------------------------------------- #
# GSM8K: accuracy must be divided by the goldens actually loaded, and
# n_problems must be bounded by the dataset size like its sibling benchmarks.
# --------------------------------------------------------------------------- #


class _FakeGSM8K(GSM8K):
    """`n_available` goldens in the dataset and a model that is always right."""

    def __init__(self, n_problems, n_available):
        self.n_problems = n_problems
        self.n_available = n_available
        self.verbose_mode = False
        self.tasks = []
        self.dataset = None

    def load_benchmark_dataset(self):
        return [
            Golden(input=f"q{i}", expected_output="1")
            for i in range(self.n_available)
        ]

    def predict(self, model, golden):
        return {"prediction": "1", "score": 1}


def test_gsm8k_accuracy_is_over_the_goldens_actually_loaded():
    # A perfect model must score 1.0 even when the dataset yields fewer goldens
    # than n_problems. Previously the denominator was n_problems, so 4 loaded
    # goldens against the default n_problems=1319 reported 0.003.
    benchmark = _FakeGSM8K(n_problems=1319, n_available=4)
    result = benchmark.evaluate(model=None)

    assert result.overall_accuracy == 1.0
    assert len(benchmark.predictions) == 4


def test_gsm8k_accuracy_unchanged_when_enough_goldens_are_available():
    benchmark = _FakeGSM8K(n_problems=5, n_available=100)
    result = benchmark.evaluate(model=None)

    assert result.overall_accuracy == 1.0
    assert len(benchmark.predictions) == 5


@pytest.mark.parametrize("n_problems", [1320, 5000])
def test_gsm8k_rejects_n_problems_larger_than_the_dataset(n_problems):
    # BoolQ, LAMBADA, Winogrande and ARC all bound n_problems this way; without
    # it GSM8K silently deflated the accuracy by len(goldens) / n_problems.
    with pytest.raises(AssertionError):
        GSM8K(n_problems=n_problems)

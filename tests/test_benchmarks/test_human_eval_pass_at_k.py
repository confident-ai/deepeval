"""
Regression tests for HumanEval collapsing pass@k into pass@n.

pass@k is a probability in [0, 1]. `evaluate()` used to fold it through
`if score:`, which is true for every non-zero probability, so a task counted as
a full pass whenever at least one of the n samples passed -- that is pass@n,
regardless of the k that was asked for.

Offline: no model, network, dataset download, or API key required.
"""

import pytest

from deepeval.dataset import Golden
from deepeval.scorer import Scorer
from deepeval.benchmarks.human_eval.human_eval import HumanEval
from deepeval.benchmarks.human_eval.task import HumanEvalTask


class _FakeHumanEval(HumanEval):
    """`n` samples per task of which `c` pass, over `n_tasks` tasks."""

    def __init__(self, n, c, n_tasks=1):
        self.tasks = list(HumanEvalTask)[:n_tasks]
        self.n = n
        self.n_correct = c
        self.verbose_mode = False
        self.scorer = Scorer()
        self.dataset = None

    def load_benchmark_dataset(self, task):
        return Golden(input="prompt", expected_output="assert True")

    def predict(self, model, task, golden, k):
        return {
            "prediction": ["sample"] * self.n,
            "score": self.scorer.pass_at_k(self.n, self.n_correct, k),
        }


def test_pass_at_1_of_one_correct_sample_in_two_hundred_is_not_a_full_pass():
    # 1 of 200 samples passes: pass@1 is 0.005, not 1.0. `if score:` used to
    # round that up to a full pass and report an overall accuracy of 1.0.
    benchmark = _FakeHumanEval(n=200, c=1)
    result = benchmark.evaluate(model=None, k=1)

    assert result.overall_accuracy == pytest.approx(0.005)


@pytest.mark.parametrize("c, k, expected", [(2, 1, 0.02), (10, 1, 0.1)])
def test_overall_accuracy_is_the_mean_pass_at_k(c, k, expected):
    benchmark = _FakeHumanEval(n=100, c=c, n_tasks=4)
    result = benchmark.evaluate(model=None, k=k)

    assert result.overall_accuracy == pytest.approx(expected)
    assert benchmark.task_scores["Score"].tolist() == pytest.approx(
        [expected] * 4
    )


def test_zero_correct_samples_still_scores_zero():
    benchmark = _FakeHumanEval(n=200, c=0)
    result = benchmark.evaluate(model=None, k=1)

    assert result.overall_accuracy == 0.0


@pytest.mark.parametrize("c", [1, 5, 200])
def test_pass_at_n_is_unchanged(c):
    # With k == n, pass@k is exactly 1.0 whenever any sample passes, so runs
    # that were already correct report the same number as before.
    benchmark = _FakeHumanEval(n=200, c=c)
    result = benchmark.evaluate(model=None, k=200)

    assert result.overall_accuracy == 1.0

"""Offline regression tests for EquityMedQA scoring (deepeval#3528).

Two bugs, both verifiable without any model, network, or API key:
1. predict() inverted BiasMetric's score, counting unbiased answers as wrong
   after #3102 switched the metric to the standard direction (1 = unbiased).
2. The scoring loop ran over only the first 10 goldens while the accuracy
   divisor counted all of them.
"""

from unittest.mock import patch

import pytest

from deepeval.benchmarks.equity_med_qa.equity_med_qa import EquityMedQA
from deepeval.dataset import Golden

N_GOLDENS = 20
N_TASKS = 9  # number of EquityMedQATask values


class FakeModel:
    def generate(self, input):
        return "unbiased answer"


class FakeMetric:
    score = 1.0
    success = True

    def __init__(self, *args, **kwargs):
        pass

    def measure(self, *args, **kwargs):
        return 1.0


@pytest.fixture
def emq_with_mocks(monkeypatch):
    # EquityMedQA's constructor requires an API key secret at init time.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-repro")
    emq = EquityMedQA()
    goldens = [Golden(input=f"q{i}") for i in range(N_GOLDENS)]
    patches = [
        patch.object(
            EquityMedQA, "load_benchmark_dataset", return_value=goldens
        ),
        patch(
            "deepeval.benchmarks.equity_med_qa.equity_med_qa.BiasMetric",
            FakeMetric,
        ),
    ]
    for p in patches:
        p.start()
    yield emq, goldens
    for p in patches:
        p.stop()


def test_unbiased_answers_score_as_correct(emq_with_mocks):
    """An all-unbiased run must score 1.0, not 0.0 (the stale inversion
    flipped every unbiased answer to 0)."""
    emq, _ = emq_with_mocks
    result = emq.evaluate(FakeModel())
    assert result.overall_accuracy == 1.0


def test_all_goldens_are_scored(emq_with_mocks):
    """Every golden is scored; the accuracy divisor matches the rows.

    The old `goldens[:10]` slice scored 10 per task while dividing by
    len(goldens), capping accuracy at 0.5 for 20+ golden tasks.
    """
    emq, goldens = emq_with_mocks
    result = emq.evaluate(FakeModel())
    assert len(emq.predictions) == N_GOLDENS * N_TASKS
    assert result.overall_accuracy == 1.0

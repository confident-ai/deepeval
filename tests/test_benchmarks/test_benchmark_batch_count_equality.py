"""
Regression tests for #2982: benchmark `batch_predict` compared the number of
generations against the number of goldens with `is not` (identity) instead of
`!=` (equality).

CPython only caches small integers (-5..256) as singletons, so two equal but
separately-computed lengths compare unequal under `is not` once they exceed
256. Any custom model with a `batch_generate` method evaluated against more
than 256 goldens therefore hit a false `ValueError` even though the counts
matched. These tests drive the schema-less fallback path that custom models
exercise, with > 256 goldens to expose the bug.
"""

import pytest

from deepeval.dataset import Golden
from deepeval.scorer.scorer import Scorer
from deepeval.benchmarks.hellaswag.hellaswag import HellaSwag
from deepeval.benchmarks.hellaswag.task import HellaSwagTask
from deepeval.benchmarks.math_qa.math_qa import MathQA
from deepeval.benchmarks.logi_qa.logi_qa import LogiQA
from deepeval.benchmarks.mmlu.mmlu import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.benchmarks.truthful_qa.truthful_qa import TruthfulQA
from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
from deepeval.benchmarks.big_bench_hard.big_bench_hard import BigBenchHard
from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask

# Above CPython's small-int singleton cache so `is not` would always be True
# for two equal lengths computed independently.
N_GOLDENS = 300


class _SchemaLessBatchModel:
    """A custom model that only supports schema-less `batch_generate`,
    simulating the fallback path real custom models take."""

    def batch_generate(self, prompts, schemas=None):
        if schemas is not None:
            raise TypeError("schema-less generation not supported")
        return [f"answer-{i}" for i in range(len(prompts))]


def _goldens(n: int):
    return [
        Golden(input=f"question {i}", expected_output=f"expected {i}")
        for i in range(n)
    ]


def _benchmark(klass, **attrs):
    # Bypass __init__ (which imports the optional HF `datasets` package); the
    # batch path only needs the attributes set below.
    bench = klass.__new__(klass)
    bench.scorer = Scorer()
    for name, value in attrs.items():
        setattr(bench, name, value)
    return bench


def test_hellaswag_batch_predict_accepts_over_256_goldens():
    bench = _benchmark(HellaSwag, shots_dataset=[], n_shots=0)
    result = bench.batch_predict(
        _SchemaLessBatchModel(), HellaSwagTask.WAKEBOARDING, _goldens(N_GOLDENS)
    )
    assert len(result) == N_GOLDENS


def test_math_qa_batch_predict_accepts_over_256_goldens():
    bench = _benchmark(MathQA, n_shots=0)
    result = bench.batch_predict(_SchemaLessBatchModel(), _goldens(N_GOLDENS))
    assert len(result) == N_GOLDENS


def test_logi_qa_batch_predict_accepts_over_256_goldens():
    bench = _benchmark(LogiQA, n_shots=0)
    result = bench.batch_predict(_SchemaLessBatchModel(), _goldens(N_GOLDENS))
    assert len(result) == N_GOLDENS


def test_mmlu_batch_predict_accepts_over_256_goldens():
    bench = _benchmark(MMLU, shots_dataset=["dummy"], n_shots=0)
    result = bench.batch_predict(
        _SchemaLessBatchModel(), MMLUTask.VIROLOGY, _goldens(N_GOLDENS)
    )
    assert len(result) == N_GOLDENS


def test_truthful_qa_batch_predict_accepts_over_256_goldens():
    bench = _benchmark(TruthfulQA)
    result = bench.batch_predict(
        _SchemaLessBatchModel(), _goldens(N_GOLDENS), TruthfulQAMode.MC1
    )
    assert len(result) == N_GOLDENS


def test_big_bench_hard_batch_predict_accepts_over_256_goldens():
    bench = _benchmark(BigBenchHard, n_shots=0, enable_cot=False)
    result = bench.batch_predict(
        _SchemaLessBatchModel(),
        BigBenchHardTask.HYPERBATON,
        _goldens(N_GOLDENS),
    )
    assert len(result) == N_GOLDENS

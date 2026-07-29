import pytest

from deepeval.scorer import Scorer
from deepeval.dataset import Golden
from deepeval.benchmarks.hellaswag.hellaswag import HellaSwag
from deepeval.benchmarks.math_qa.math_qa import MathQA
from deepeval.benchmarks.logi_qa.logi_qa import LogiQA
from deepeval.benchmarks.big_bench_hard.big_bench_hard import BigBenchHard
from deepeval.benchmarks.truthful_qa.truthful_qa import TruthfulQA
from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
from deepeval.benchmarks.mmlu.mmlu import MMLU
from deepeval.benchmarks.tasks import HellaSwagTask, MMLUTask, BigBenchHardTask

OVER_SMALL_INT_CACHE = 300


def test_is_not_on_equal_lengths_above_small_int_cache_is_unreliable():
    predictions = ["x"] * OVER_SMALL_INT_CACHE
    goldens = ["y"] * OVER_SMALL_INT_CACHE
    assert len(predictions) == len(goldens)
    assert len(predictions) is not len(goldens)


class _FakeNoSchemaBatchModel:
    def get_model_name(self):
        return "fake"

    def batch_generate(self, prompts):
        return ["A"] * len(prompts)


def _make_goldens(n, expected_output="A"):
    return [
        Golden(input=f"question {i}", expected_output=expected_output)
        for i in range(n)
    ]


def test_hellaswag_batch_predict_accepts_large_equal_length_batch():
    bench = HellaSwag.__new__(HellaSwag)
    bench.shots_dataset = []
    bench.n_shots = 0
    bench.scorer = Scorer()

    goldens = _make_goldens(OVER_SMALL_INT_CACHE)
    result = bench.batch_predict(
        _FakeNoSchemaBatchModel(), HellaSwagTask.APPLYING_SUNSCREEN, goldens
    )
    assert len(result) == OVER_SMALL_INT_CACHE


def test_mmlu_batch_predict_accepts_large_equal_length_batch():
    bench = MMLU.__new__(MMLU)
    bench.shots_dataset = ["placeholder"]
    bench.n_shots = 0
    bench.scorer = Scorer()

    goldens = _make_goldens(OVER_SMALL_INT_CACHE)
    result = bench.batch_predict(
        _FakeNoSchemaBatchModel(),
        MMLUTask.HIGH_SCHOOL_EUROPEAN_HISTORY,
        goldens,
    )
    assert len(result) == OVER_SMALL_INT_CACHE


def test_math_qa_batch_predict_accepts_large_equal_length_batch():
    bench = MathQA.__new__(MathQA)
    bench.n_shots = 0
    bench.scorer = Scorer()

    goldens = _make_goldens(OVER_SMALL_INT_CACHE)
    result = bench.batch_predict(_FakeNoSchemaBatchModel(), goldens)
    assert len(result) == OVER_SMALL_INT_CACHE


def test_logi_qa_batch_predict_accepts_large_equal_length_batch():
    bench = LogiQA.__new__(LogiQA)
    bench.n_shots = 0
    bench.scorer = Scorer()

    goldens = _make_goldens(OVER_SMALL_INT_CACHE)
    result = bench.batch_predict(_FakeNoSchemaBatchModel(), goldens)
    assert len(result) == OVER_SMALL_INT_CACHE


def test_big_bench_hard_batch_predict_accepts_large_equal_length_batch():
    bench = BigBenchHard.__new__(BigBenchHard)
    bench.n_shots = 0
    bench.enable_cot = False
    bench.scorer = Scorer()

    goldens = _make_goldens(OVER_SMALL_INT_CACHE)
    result = bench.batch_predict(
        _FakeNoSchemaBatchModel(), BigBenchHardTask.BOOLEAN_EXPRESSIONS, goldens
    )
    assert len(result) == OVER_SMALL_INT_CACHE


def test_truthful_qa_batch_predict_accepts_large_equal_length_batch():
    bench = TruthfulQA.__new__(TruthfulQA)
    bench.scorer = Scorer()

    goldens = _make_goldens(OVER_SMALL_INT_CACHE)
    result = bench.batch_predict(
        _FakeNoSchemaBatchModel(), goldens, TruthfulQAMode.MC1
    )
    assert len(result) == OVER_SMALL_INT_CACHE

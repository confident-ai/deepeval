"""
Regression tests for TruthfulQA MC2 scoring bugs (GitHub issue #2957).

Two defects fixed:
1. **Scale**: MC2 score was a 0-100 percentage while MC1 was 0/1, making
   MC2 accuracy ~100x inflated and unbounded above 1.
2. **Batch inconsistency**: the batch path treated any non-zero score as 1
   (full credit), so partial-credit questions were scored identically to
   perfect ones.  Non-batch summed the raw percentage.

After the fix both paths normalise MC2 to a 0-1 fraction and accumulate
fractionally, so batch_size=None and batch_size=N report the same accuracy.

All tests are fully offline: no model, network, dataset download, or API
key required.
"""

import pytest

from deepeval.scorer.scorer import Scorer
from deepeval.benchmarks.truthful_qa.truthful_qa import TruthfulQA
from deepeval.benchmarks.truthful_qa.task import TruthfulQATask
from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
from deepeval.dataset import Golden

# --------------------------------------------------------------------------- #
# Helper: stub models with controllable correctness
# --------------------------------------------------------------------------- #


class _StubModel:
    """A model that always returns *answer* for MC2 list-of-numbers schema."""

    def __init__(self, answer):
        self._answer = answer

    def get_model_name(self):
        return "stub"

    def generate(self, prompt, schema=None):
        return schema(answer=self._answer)

    def batch_generate(self, prompts, schemas=None):
        return [schema(answer=self._answer) for schema in schemas]


def _make_bench(goldens):
    """Create a TruthfulQA instance with a stub dataset (bypass __init__)."""
    bench = TruthfulQA.__new__(TruthfulQA)
    bench.scorer = Scorer()
    bench.tasks = [TruthfulQATask.LANGUAGE]
    bench.mode = TruthfulQAMode.MC2
    bench.n_problems_per_task = None
    bench.verbose_mode = False
    bench.mc_dataset = None
    bench.load_benchmark_dataset = lambda task, mode: goldens
    return bench


# --------------------------------------------------------------------------- #
# 1. Scaler: truth_identification_score returns 0-100 (unchanged)
# --------------------------------------------------------------------------- #


class TestTruthIdentificationScore:
    """Verify the underlying Scorer method is unchanged (returns 0-100)."""

    def test_perfect_match(self):
        assert Scorer.truth_identification_score("[1, 2]", "[1, 2]") == 100

    def test_zero_match(self):
        assert Scorer.truth_identification_score("[1, 2]", "[3, 4]") == 0

    def test_partial_match(self):
        # 1 of 2 correct → 50%
        assert Scorer.truth_identification_score("[1, 2]", "[1]") == 50

    def test_empty_prediction(self):
        assert Scorer.truth_identification_score("[1, 2]", "") == 0

    def test_empty_target(self):
        assert Scorer.truth_identification_score("", "[1]") == 0


# --------------------------------------------------------------------------- #
# 2. MC2 accuracy is bounded [0, 1] after normalisation
# --------------------------------------------------------------------------- #


class TestMC2ScoreNormalization:
    """The per-question MC2 score used in evaluate() must be 0-1."""

    def test_predict_returns_normalized_score(self):
        """predict() must return a 0-1 fraction for MC2."""
        goldens = [Golden(input="q", expected_output="[1, 2]")]
        bench = _make_bench(goldens)

        # Half right: predicts [1] when correct is [1, 2]
        model = _StubModel(answer=[1])
        result = bench.predict(model, goldens[0], TruthfulQAMode.MC2)
        assert (
            0 <= result["score"] <= 1
        ), f"MC2 score {result['score']} is outside [0, 1]"
        assert result["score"] == pytest.approx(0.5, abs=0.01)

    def test_predict_perfect_model_scores_one(self):
        goldens = [Golden(input="q", expected_output="[1, 2]")]
        bench = _make_bench(goldens)
        model = _StubModel(answer=[1, 2])
        result = bench.predict(model, goldens[0], TruthfulQAMode.MC2)
        assert result["score"] == pytest.approx(1.0)

    def test_predict_zero_match_scores_zero(self):
        goldens = [Golden(input="q", expected_output="[1, 2]")]
        bench = _make_bench(goldens)
        model = _StubModel(answer=[3, 4])
        result = bench.predict(model, goldens[0], TruthfulQAMode.MC2)
        assert result["score"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 3. Batch and non-batch report the same accuracy (the core regression)
# --------------------------------------------------------------------------- #


class TestBatchNonBatchConsistency:
    """batch_size=None and batch_size=N must report identical accuracy."""

    @pytest.fixture
    def half_right_bench(self):
        """4 goldens, model gets 1 of 2 correct answers each time → 0.5."""
        goldens = [
            Golden(input="q", expected_output="[1, 2]") for _ in range(4)
        ]
        return _make_bench(goldens)

    def test_half_right_model_both_paths(self, half_right_bench):
        model = _StubModel(answer=[1])

        result_no_batch = half_right_bench.evaluate(model, batch_size=None)
        result_batch = half_right_bench.evaluate(model, batch_size=2)

        assert result_no_batch.overall_accuracy == pytest.approx(
            0.5, abs=0.01
        ), f"Non-batch accuracy {result_no_batch.overall_accuracy} != 0.5"
        assert result_batch.overall_accuracy == pytest.approx(
            0.5, abs=0.01
        ), f"Batch accuracy {result_batch.overall_accuracy} != 0.5"
        assert result_no_batch.overall_accuracy == pytest.approx(
            result_batch.overall_accuracy
        )

    def test_perfect_model_both_paths(self):
        goldens = [
            Golden(input="q", expected_output="[1, 2]") for _ in range(4)
        ]
        bench = _make_bench(goldens)
        model = _StubModel(answer=[1, 2])

        result_no_batch = bench.evaluate(model, batch_size=None)
        result_batch = bench.evaluate(model, batch_size=2)

        assert result_no_batch.overall_accuracy == pytest.approx(1.0)
        assert result_batch.overall_accuracy == pytest.approx(1.0)

    def test_zero_match_model_both_paths(self):
        goldens = [
            Golden(input="q", expected_output="[1, 2]") for _ in range(4)
        ]
        bench = _make_bench(goldens)
        model = _StubModel(answer=[3, 4])

        result_no_batch = bench.evaluate(model, batch_size=None)
        result_batch = bench.evaluate(model, batch_size=2)

        assert result_no_batch.overall_accuracy == pytest.approx(0.0)
        assert result_batch.overall_accuracy == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 4. MC1 is unaffected by the fix (exact_match_score returns 0/1)
# --------------------------------------------------------------------------- #


class TestMC1Unaffected:
    """MC1 path uses exact_match_score (0/1) — must be unchanged."""

    def test_mc1_predict_returns_binary_score(self):
        goldens = [Golden(input="q", expected_output="1")]
        bench = TruthfulQA.__new__(TruthfulQA)
        bench.scorer = Scorer()
        bench.tasks = [TruthfulQATask.LANGUAGE]
        bench.mode = TruthfulQAMode.MC1
        bench.n_problems_per_task = None
        bench.verbose_mode = False
        bench.mc_dataset = None
        bench.load_benchmark_dataset = lambda task, mode: goldens

        model = _StubModel(answer=1)
        result = bench.predict(model, goldens[0], TruthfulQAMode.MC1)
        assert result["score"] in (0, 1)

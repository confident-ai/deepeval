"""Tests for GEval score normalization with zero-span score ranges."""

from unittest.mock import patch

from deepeval.metrics import GEval
from deepeval.metrics.g_eval.utils import Rubric, get_score_range
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


class _StubLLM(DeepEvalBaseLLM):
    def load_model(self):
        return None

    def generate(self, *a, **kw):
        return ""

    async def a_generate(self, *a, **kw):
        return ""

    def get_model_name(self):
        return "stub"


_TEST_CASE = LLMTestCase(input="hi", actual_output="hello")


def _make_metric(rubric=None, strict_mode=False):
    return GEval(
        name="test",
        evaluation_params=[LLMTestCaseParams.INPUT],
        criteria="test",
        model=_StubLLM(),
        rubric=rubric,
        strict_mode=strict_mode,
        async_mode=False,
    )


def _run(metric, g_score, reason="ok"):
    with patch.object(
        metric, "_generate_evaluation_steps", return_value=["s1"]
    ), patch.object(metric, "_evaluate", return_value=(g_score, reason)):
        return metric.measure(
            _TEST_CASE,
            _show_indicator=False,
            _log_metric_to_confident=False,
        )


class TestScoreNormalizationRealPath:
    def test_normal_range(self):
        m = _make_metric()
        assert _run(m, 5) == 0.5
        assert _run(m, 0) == 0.0
        assert _run(m, 10) == 1.0

    def test_custom_range(self):
        rubric = [Rubric(score_range=(2, 8), expected_outcome="x")]
        m = _make_metric(rubric=rubric)
        assert _run(m, 2) == 0.0
        assert _run(m, 8) == 1.0
        assert _run(m, 5) == 0.5

    def test_zero_span_does_not_crash(self):
        rubric = [Rubric(score_range=(5, 5), expected_outcome="x")]
        m = _make_metric(rubric=rubric)
        assert _run(m, 5) == 5

    def test_strict_mode_returns_int(self):
        m = _make_metric(strict_mode=True)
        assert _run(m, 7) == 7


class TestGetScoreRange:
    def test_no_rubric_returns_default(self):
        assert get_score_range(None) == (0, 10)

    def test_single_rubric(self):
        rubrics = [Rubric(score_range=(3, 7), expected_outcome="mid")]
        assert get_score_range(rubrics) == (3, 7)

    def test_single_point_rubric(self):
        rubrics = [Rubric(score_range=(5, 5), expected_outcome="exact")]
        assert get_score_range(rubrics) == (5, 5)

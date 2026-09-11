import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCase


def test_identical_outputs_score_one():
    """E2: expected == actual"""
    tc = LLMTestCase(input="q", actual_output="hello", expected_output="hello")
    metric = ExactMatchMetric()
    assert metric.measure(tc, _show_indicator=False) == 1.0
    assert metric.success is True


def test_different_outputs_score_zero():
    """E3: expected != actual"""
    tc = LLMTestCase(input="q", actual_output="hello", expected_output="world")
    metric = ExactMatchMetric()
    assert metric.measure(tc, _show_indicator=False) == 0.0
    assert metric.success is False


def test_whitespace_is_stripped_before_comparison():
    """E4: 前后空格在比较前被 strip 掉"""
    tc = LLMTestCase(
        input="q", actual_output=" hello ", expected_output="hello"
    )
    metric = ExactMatchMetric()
    assert metric.measure(tc, _show_indicator=False) == 1.0
    assert metric.success is True


def test_threshold_none_leaves_success_undefined():
    """E5: threshold=None 时 success 是 None，不是 True/False"""
    tc = LLMTestCase(input="q", actual_output="hello", expected_output="hello")
    metric = ExactMatchMetric(threshold=None)
    assert metric.measure(tc, _show_indicator=False) == 1.0
    assert metric.success is None


def test_verbose_mode_populates_verbose_logs():
    """E6: verbose_mode=True 才填 verbose_logs"""
    tc = LLMTestCase(input="q", actual_output="hello", expected_output="hello")

    verbose = ExactMatchMetric(verbose_mode=True)
    verbose.measure(tc, _show_indicator=False)
    assert verbose.verbose_logs is not None
    assert "Score: 1.00" in verbose.verbose_logs

    quiet = ExactMatchMetric(verbose_mode=False)
    quiet.measure(tc, _show_indicator=False)
    assert quiet.verbose_logs is None


async def test_a_measure_returns_same_score_as_measure():
    """a_measure 是 measure 的 async 转发，结果必须一致"""
    tc = LLMTestCase(input="q", actual_output="hello", expected_output="hello")
    metric = ExactMatchMetric()
    assert await metric.a_measure(tc, _show_indicator=False) == 1.0
    assert metric.success is True


def test_missing_expected_output_raises():
    """E1: measure() 里的参数校验，缺 expected_output 直接抛异常"""
    tc = LLMTestCase(input="q", actual_output="hello")
    metric = ExactMatchMetric()
    with pytest.raises(MissingTestCaseParamsError):
        metric.measure(tc, _show_indicator=False)

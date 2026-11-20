import pytest

from deepeval.metrics import PatternMatchMetric
from deepeval.test_case import LLMTestCase


def test_matching_output_scores_one():
    """P5: 整串匹配 -> 1.0"""
    tc = LLMTestCase(input="q", actual_output="hello")
    metric = PatternMatchMetric(pattern="h.llo")
    assert metric.measure(tc, _show_indicator=False) == 1.0
    assert metric.success is True


def test_non_matching_output_scores_zero():
    """P5b: 不匹配 -> 0.0"""
    tc = LLMTestCase(input="q", actual_output="world")
    metric = PatternMatchMetric(pattern="hello")
    assert metric.measure(tc, _show_indicator=False) == 0.0
    assert metric.success is False


def test_pattern_must_match_the_entire_output():
    """P4: 用的是 re.fullmatch 不是 re.search，模式必须覆盖整串"""
    tc = LLMTestCase(input="q", actual_output="say hello now")
    metric = PatternMatchMetric(pattern="hello")
    assert metric.measure(tc, _show_indicator=False) == 0.0
    assert metric.success is False


def test_case_sensitive_by_default():
    """P2: ignore_case 默认 False，大小写不同就不匹配"""
    tc = LLMTestCase(input="q", actual_output="HELLO")
    metric = PatternMatchMetric(pattern="hello")
    assert metric.measure(tc, _show_indicator=False) == 0.0


def test_ignore_case_makes_match_case_insensitive():
    """P3: ignore_case=True 时同样的输入变成匹配"""
    tc = LLMTestCase(input="q", actual_output="HELLO")
    metric = PatternMatchMetric(pattern="hello", ignore_case=True)
    assert metric.measure(tc, _show_indicator=False) == 1.0


def test_expected_output_is_not_required():
    """P6: _required_params 只有 input 和 actual_output"""
    tc = LLMTestCase(input="q", actual_output="hello")
    assert tc.expected_output is None
    metric = PatternMatchMetric(pattern="hello")
    assert metric.measure(tc, _show_indicator=False) == 1.0


def test_invalid_regex_raises_value_error():
    """P1: 坏正则在 __init__ 就转成 ValueError"""
    with pytest.raises(ValueError):
        PatternMatchMetric(pattern="[")


def test_whitespace_is_stripped_before_matching():
    """P4b: actual_output 在匹配前被 strip"""
    tc = LLMTestCase(input="q", actual_output="  hello  ")
    metric = PatternMatchMetric(pattern="hello")
    assert metric.measure(tc, _show_indicator=False) == 1.0


def test_verbose_mode_populates_verbose_logs():
    """P7: verbose_mode 控制 verbose_logs 是否被填"""
    tc = LLMTestCase(input="q", actual_output="hello")

    verbose = PatternMatchMetric(pattern="hello", verbose_mode=True)
    verbose.measure(tc, _show_indicator=False)
    assert verbose.verbose_logs is not None
    assert "Pattern: hello" in verbose.verbose_logs
    assert "Actual: hello" in verbose.verbose_logs

    quiet = PatternMatchMetric(pattern="hello", verbose_mode=False)
    quiet.measure(tc, _show_indicator=False)
    assert quiet.verbose_logs is None


async def test_a_measure_returns_same_score_as_measure():
    """a_measure 是 measure 的 async 转发，结果必须一致"""
    tc = LLMTestCase(input="q", actual_output="hello")
    metric = PatternMatchMetric(pattern="hello")
    assert await metric.a_measure(tc, _show_indicator=False) == 1.0
    assert metric.success is True

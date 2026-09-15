"""Tests for AnswerRelevancyMetric empty actual_output and expected_output validation.

These tests verify that AnswerRelevancyMetric raises MissingTestCaseParamsError
when actual_output or expected_output is missing/empty:
  - None (missing param)
  - "" (empty string)
  - Whitespace-only strings (e.g. "   ", "\n\t")

These tests use DummyModel and do not require OPENAI_API_KEY.
"""

from unittest.mock import patch
import pytest

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics.utils import check_llm_test_case_params
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.errors import MissingTestCaseParamsError
from ..test_core.stubs import DummyModel


def make_metric(*, async_mode: bool = False) -> AnswerRelevancyMetric:
    """Create AnswerRelevancyMetric with DummyModel so no LLM calls are made."""
    with patch(
        "deepeval.metrics.answer_relevancy.answer_relevancy.initialize_model"
    ) as mock_init:
        mock_init.return_value = (DummyModel(), True)
        return AnswerRelevancyMetric(
            async_mode=async_mode,
        )


def make_metric_with_expected_output(
    *, async_mode: bool = False
) -> AnswerRelevancyMetric:
    """Create AnswerRelevancyMetric that requires expected_output."""
    with patch(
        "deepeval.metrics.answer_relevancy.answer_relevancy.initialize_model"
    ) as mock_init:
        mock_init.return_value = (DummyModel(), True)
        metric = AnswerRelevancyMetric(async_mode=async_mode)
        # Override required params to include expected_output
        metric._required_params = [  # pylint: disable=protected-access
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ]
        return metric


def test_answer_relevancy_none_actual_output_raises_sync():
    """Test None actual_output raises MissingTestCaseParamsError."""
    metric = make_metric(async_mode=False)
    tc = LLMTestCase(input="hi", actual_output=None)

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        metric.measure(tc, _show_indicator=False)

    msg = str(exc_info.value).lower()
    assert "actual_output" in msg


def test_answer_relevancy_empty_actual_output_raises_sync():
    """Empty string actual_output should raise MissingTestCaseParamsError (sync)."""
    metric = make_metric(async_mode=False)
    tc = LLMTestCase(input="What if these shoes don't fit?", actual_output="")

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        metric.measure(tc, _show_indicator=False)

    msg = str(exc_info.value).lower()
    assert "cannot be empty" in msg or "actual_output" in msg


def test_answer_relevancy_whitespace_actual_output_raises_sync():
    """Whitespace-only actual_output should raise MissingTestCaseParamsError (sync)."""
    metric = make_metric(async_mode=False)
    tc = LLMTestCase(
        input="What if these shoes don't fit?", actual_output="   "
    )

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        check_llm_test_case_params(
            test_case=tc,
            test_case_params=metric._required_params,  # pylint: disable=protected-access
            input_image_count=None,
            actual_output_image_count=None,
            metric=metric,
            model=metric.model,
            multimodal=tc.multimodal,
        )

    msg = str(exc_info.value).lower()
    assert "cannot be empty" in msg or "actual_output" in msg


def test_answer_relevancy_none_expected_output_raises_sync():
    """None expected_output should raise MissingTestCaseParamsError (sync)."""
    metric = make_metric_with_expected_output(async_mode=False)
    tc = LLMTestCase(input="hi", actual_output="hello", expected_output=None)

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        metric.measure(tc, _show_indicator=False)

    msg = str(exc_info.value).lower()
    assert "expected_output" in msg


def test_answer_relevancy_empty_expected_output_raises_sync():
    """Empty string expected_output should raise MissingTestCaseParamsError (sync)."""
    metric = make_metric_with_expected_output(async_mode=False)
    tc = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="hello",
        expected_output="",
    )

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        metric.measure(tc, _show_indicator=False)

    msg = str(exc_info.value).lower()
    assert "cannot be empty" in msg or "expected_output" in msg


def test_answer_relevancy_whitespace_expected_output_raises_sync():
    """Whitespace-only expected_output should raise MissingTestCaseParamsError (sync)."""
    metric = make_metric_with_expected_output(async_mode=False)
    tc = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="hello",
        expected_output="   ",
    )

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        check_llm_test_case_params(
            test_case=tc,
            test_case_params=metric._required_params,  # pylint: disable=protected-access
            input_image_count=None,
            actual_output_image_count=None,
            metric=metric,
            model=metric.model,
            multimodal=tc.multimodal,
        )

    msg = str(exc_info.value).lower()
    assert "cannot be empty" in msg or "expected_output" in msg


def test_answer_relevancy_both_whitespace_strings_raises_sync():
    """Test whitespace-only actual_output and expected_output."""
    metric = make_metric_with_expected_output(async_mode=False)
    tc = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="   ",
        expected_output="\n\t",
    )

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        check_llm_test_case_params(
            test_case=tc,
            test_case_params=metric._required_params,  # pylint: disable=protected-access
            input_image_count=None,
            actual_output_image_count=None,
            metric=metric,
            model=metric.model,
            multimodal=tc.multimodal,
        )

    # Should fail on actual_output first (based on order in check_llm_test_case_params)
    msg = str(exc_info.value).lower()
    assert "cannot be empty" in msg and (
        "actual_output" in msg or "expected_output" in msg
    )

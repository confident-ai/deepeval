"""Tests for AnswerRelevancyMetric empty actual_output validation.

These tests verify that AnswerRelevancyMetric raises MissingTestCaseParamsError
when actual_output is missing/empty:
  - None (missing param)
  - "" (empty string)

Whitespace-only strings are intentionally not validated because we can't make assumptions
about the value of the actual_output beyond its existence or emptiness.

These tests use DummyModel and do not require OPENAI_API_KEY.
"""

import pytest
from unittest.mock import patch
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics.answer_relevancy.template import AnswerRelevancyTemplate
from deepeval.metrics.utils import check_llm_test_case_params
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.errors import MissingTestCaseParamsError
from tests.test_core.stubs import DummyModel


def make_metric(*, async_mode: bool = False) -> AnswerRelevancyMetric:
    """Create AnswerRelevancyMetric with DummyModel so no LLM calls are made."""
    with patch(
        "deepeval.metrics.answer_relevancy.answer_relevancy.initialize_model"
    ) as mock_init:
        mock_init.return_value = (DummyModel(), True)
        return AnswerRelevancyMetric(
            async_mode=async_mode,
            evaluation_template=AnswerRelevancyTemplate,
        )


def test_answer_relevancy_none_actual_output_raises_sync():
    metric = make_metric(async_mode=False)
    tc = LLMTestCase(input="hi", actual_output=None)

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        metric.measure(
            tc, _show_indicator=False, _log_metric_to_confident=False
        )

    msg = str(exc_info.value).lower()
    assert "actual_output" in msg


def test_answer_relevancy_empty_actual_output_raises_sync():
    """Empty string actual_output should raise MissingTestCaseParamsError (sync)."""
    metric = make_metric(async_mode=False)
    tc = LLMTestCase(input="What if these shoes don't fit?", actual_output="")

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        metric.measure(
            tc, _show_indicator=False, _log_metric_to_confident=False
        )

    msg = str(exc_info.value).lower()
    assert "cannot be empty" in msg or "actual_output" in msg


def test_answer_relevancy_whitespace_actual_output_does_not_raise_validation():
    """Whitespace-only actual_output should NOT raise MissingTestCaseParamsError."""
    metric = make_metric(async_mode=False)
    tc = LLMTestCase(
        input="What if these shoes don't fit?", actual_output="   "
    )

    # Only validate inputs here. Running the full metric would require a real
    # model that supports generate_with_schema.
    check_llm_test_case_params(
        test_case=tc,
        test_case_params=metric._required_params,
        input_image_count=None,
        actual_output_image_count=None,
        metric=metric,
        model=metric.model,
        multimodal=tc.multimodal,
    )

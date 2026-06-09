"""Tests for empty/blank list-content param validation (issue #2248).

`check_llm_test_case_params` already rejects an empty-string `actual_output`
as a missing param. A string-content list param such as `retrieval_context`
that is present but contains only empty/whitespace strings carries no usable
content either, so it must be rejected the same way instead of silently
producing a meaningless score.

An intentionally empty list ([]) and lists containing at least one non-blank
string are left untouched.

These tests use DummyModel and do not require OPENAI_API_KEY.
"""

import pytest
from unittest.mock import patch
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics.utils import check_llm_test_case_params
from deepeval.test_case import LLMTestCase
from deepeval.errors import MissingTestCaseParamsError
from tests.test_core.stubs import DummyModel


def make_metric() -> FaithfulnessMetric:
    """Create FaithfulnessMetric with DummyModel so no LLM calls are made."""
    with patch(
        "deepeval.metrics.faithfulness.faithfulness.initialize_model"
    ) as mock_init:
        mock_init.return_value = (DummyModel(), True)
        return FaithfulnessMetric(async_mode=False)


def _check(metric: FaithfulnessMetric, tc: LLMTestCase) -> None:
    check_llm_test_case_params(
        test_case=tc,
        test_case_params=metric._required_params,
        input_image_count=None,
        actual_output_image_count=None,
        metric=metric,
        model=metric.model,
        multimodal=tc.multimodal,
    )


def test_retrieval_context_only_empty_string_raises():
    """retrieval_context=[''] must raise MissingTestCaseParamsError (issue #2248)."""
    metric = make_metric()
    tc = LLMTestCase(input="x", actual_output="y", retrieval_context=[""])

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        _check(metric, tc)

    msg = str(exc_info.value).lower()
    assert "retrieval_context" in msg


def test_retrieval_context_only_whitespace_strings_raises():
    """A list of whitespace-only strings carries no content and must raise."""
    metric = make_metric()
    tc = LLMTestCase(
        input="x", actual_output="y", retrieval_context=["   ", "\n"]
    )

    with pytest.raises(MissingTestCaseParamsError):
        _check(metric, tc)


def test_empty_retrieval_context_list_does_not_raise():
    """An intentionally empty list ([]) keeps its existing behavior."""
    metric = make_metric()
    tc = LLMTestCase(input="x", actual_output="y", retrieval_context=[])

    # Should not raise; param validation must leave [] untouched.
    _check(metric, tc)


def test_retrieval_context_with_real_content_does_not_raise():
    """A list with at least one non-blank string is valid."""
    metric = make_metric()
    tc = LLMTestCase(
        input="x", actual_output="y", retrieval_context=["", "real context"]
    )

    _check(metric, tc)

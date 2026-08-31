import pytest
from unittest.mock import patch

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics.utils import check_llm_test_case_params
from deepeval.test_case import LLMTestCase
from tests.test_core.stubs import DummyModel


def make_metric() -> FaithfulnessMetric:
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
    metric = make_metric()
    tc = LLMTestCase(input="x", actual_output="y", retrieval_context=[""])

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        _check(metric, tc)

    assert "retrieval_context" in str(exc_info.value).lower()


def test_retrieval_context_only_whitespace_strings_raises():
    metric = make_metric()
    tc = LLMTestCase(
        input="x", actual_output="y", retrieval_context=["   ", "\n"]
    )

    with pytest.raises(MissingTestCaseParamsError):
        _check(metric, tc)


def test_empty_retrieval_context_list_does_not_raise():
    metric = make_metric()
    tc = LLMTestCase(input="x", actual_output="y", retrieval_context=[])

    _check(metric, tc)


def test_retrieval_context_with_real_content_does_not_raise():
    metric = make_metric()
    tc = LLMTestCase(
        input="x", actual_output="y", retrieval_context=["", "real context"]
    )

    _check(metric, tc)

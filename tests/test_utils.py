"""Test module for utils."""

import pytest

from deepeval.metrics import BaseMetric, utils
from deepeval.test_case import LLMTestCaseParams


def test_check_llm_test_case_params_raies_ValueError_for_wrong_type():
    with pytest.raises(ValueError):
        utils.check_llm_test_case_params(
            test_case="test_case",
            test_case_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            metric=BaseMetric(),
        )

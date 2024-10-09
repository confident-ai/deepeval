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


def test_trimAndLoadJson_correctly_parses_with_trailing_comma():
    test_data = [
        '{\n    "verdict": "yes",\n}',
        '{\n    "verdict": "yes",\n}',
    ]
    verdicts = [utils.trimAndLoadJson(v) for v in test_data]

    assert len(verdicts) == 2
    for v in verdicts:
        assert v.get("verdict") == "yes"

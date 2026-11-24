from typing import List

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCaseParams


class BaseRedTeamingMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

import pytest
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
import random
import time


class FakeMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        include_reason=True,
        model="asd",
        strict_mode=True,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.evaluation_model = model
        self.strict_mode = strict_mode

    def measure(self, test_case: LLMTestCase):
        time.sleep(5)
        self.score = random.uniform(0.0, 1.0)
        self.success = self.score >= self.threshold
        if self.include_reason:
            self.reason = "This metric looking good!"

        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Fake Metric"


def test_cache():
    test_case = LLMTestCase(input="input", actual_output="output")
    metric = FakeMetric(
        threshold=0.2, include_reason=False, strict_mode=True, model="omg"
    )
    assert_test(test_case, [metric])


# @pytest.mark.skip("askjdfn")
def test_cache_again():
    test_case = LLMTestCase(input="input 3", actual_output="output 2")
    metric = FakeMetric(
        threshold=0.2, include_reason=False, strict_mode=True, model="omg"
    )
    assert_test(test_case, [metric])

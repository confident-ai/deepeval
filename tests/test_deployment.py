import random
import time

import pytest

import deepeval
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


# Inherit BaseMetric
class FakeMetric(BaseMetric):
    # This metric by default checks if the latency is greater than 10 seconds
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        # Set self.success and self.score in the "measure" method
        self.score = random.uniform(0.5, 1.0)
        self.success = self.score >= self.threshold
        # You can also optionally set a reason for the score returned.
        # This is particularly useful for a score computed using LLMs
        self.reason = "This metric looking good!"
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        self.score = random.uniform(0.5, 1.0)
        self.success = self.score >= self.threshold
        # You can also optionally set a reason for the score returned.
        # This is particularly useful for a score computed using LLMs
        self.reason = "This async metric looking good!"
        random_sleep_time = random.uniform(1, 5)
        time.sleep(random_sleep_time)
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Coherence"


dataset = EvaluationDataset()

# Pull from Confident
dataset.pull(alias="test")


@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    fake_metric = FakeMetric()
    assert_test(test_case, [fake_metric])

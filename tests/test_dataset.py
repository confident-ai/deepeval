import os

import pytest
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import HallucinationMetric
from deepeval import assert_test, evaluate
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import aggregate_metric_pass_rates, TestResult
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric


class FakeMetric1(BaseMetric):
    def __init__(self, threshold: float = 0.5, _success: bool = True):
        self.threshold = threshold
        self.success = _success

    def measure(self, test_case: LLMTestCase):
        self.reason = "This metric looking good!"
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        self.score = 0.5
        self.reason = "This async metric looking good!"
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Fake"


class FakeMetric2(BaseMetric):
    def __init__(self, threshold: float = 0.5, _success: bool = True):
        self.threshold = threshold
        self.success = _success

    def measure(self, test_case: LLMTestCase):
        self.score = 0.5
        self.reason = "This metric looking good!"
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        self.score = 0.5
        self.reason = "This async metric looking good!"
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Fake"


def test_create_dataset():
    dataset = EvaluationDataset()
    module_b_dir = os.path.dirname(os.path.realpath(__file__))

    file_path = os.path.join(module_b_dir, "data", "dataset.csv")

    dataset.add_test_cases_from_csv_file(
        file_path,
        input_col_name="query",
        actual_output_col_name="actual_output",
        expected_output_col_name="expected_output",
        context_col_name="context",
        retrieval_context_col_name="retrieval",
    )
    assert len(dataset.test_cases) == 5, "Test Cases not loaded from CSV"
    file_path = os.path.join(module_b_dir, "data", "dataset.json")
    dataset.add_test_cases_from_json_file(
        file_path,
        input_key_name="query",
        expected_output_key_name="expected_output",
        context_key_name="context",
        retrieval_context_key_name="retrieval",
        actual_output_key_name="actual_output",
    )
    assert len(dataset.test_cases) == 10, "Test Cases not loaded from JSON"

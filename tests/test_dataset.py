import os

import pytest
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import HallucinationMetric
from deepeval import assert_test, evaluate
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import aggregate_metric_pass_rates, TestResult
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric


class StubMetric(BaseMetric):
    def __init__(self, _success: bool, _name: str):
        self.success = _success
        self.name = _name

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return self.name


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
        actual_output_key_name="actual_output",
        expected_output_key_name="expected_output",
        context_key_name="context",
        retrieval_context_key_name="retrieval",
    )
    assert len(dataset.test_cases) == 10, "Test Cases not loaded from JSON"


def test_aggregate_metric_pass_rates():
    test_results = [
        TestResult(
            success=True,
            metrics=[
                StubMetric(_success=True, _name="AnswerRelevancyMetric"),
                StubMetric(_success=True, _name="BiasMetric"),
            ],
            input="some input",
            actual_output="some output",
            expected_output="expected output",
            context=["context"],
            retrieval_context=["retrieval context"],
        ),
        TestResult(
            success=True,
            metrics=[
                StubMetric(_success=False, _name="AnswerRelevancyMetric"),
                StubMetric(_success=True, _name="BiasMetric"),
            ],
            input="another input",
            actual_output="another output",
            expected_output="another expected output",
            context=["another context"],
            retrieval_context=["another retrieval context"],
        ),
    ]

    expected_result = {"AnswerRelevancyMetric": 0.5, "BiasMetric": 1.0}
    result = aggregate_metric_pass_rates(test_results)
    assert (
        result == expected_result
    ), "The aggregate metric pass rates do not match the expected values."

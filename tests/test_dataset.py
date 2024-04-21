import os

import pytest
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import HallucinationMetric
from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase


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


# test_case = LLMTestCase(
#     input="What if these shoes don't fit?",
#     # Replace this with the actual output from your LLM application
#     actual_output="We offer a 30-day full refund at no extra costs.",
#     context=["All customers are eligible for a 30 day full refund at no extra costs."]
# )
# dataset = EvaluationDataset(alias="123", test_cases=[test_case])

# @pytest.mark.parametrize(
#     "test_case",
#     dataset,
# )
# def test_test_dataset(test_case: LLMTestCase):
#     metric = HallucinationMetric(threshold=0.5)
#     assert_test(test_case, [metric])


# dataset = EvaluationDataset()
# dataset.pull("Testa")
# print(dataset.test_cases)
# @pytest.mark.parametrize(
#     "test_case",
#     dataset,
# )
# def test_customer_chatbot(test_case: LLMTestCase):
#     hallucination_metric = HallucinationMetric(threshold=0.3)
#     assert_test(test_case, [hallucination_metric])

# dataset = EvaluationDataset()
# dataset.pull(alias="Evals Dataset")

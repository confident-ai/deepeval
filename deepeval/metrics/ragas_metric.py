"""An implementation of the Ragas metric
"""
import os
from deepeval.metrics.metric import Metric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import run_test
from typing import List


class RagasMetric(Metric):
    """This metric checks if the output is more than 3 letters"""

    def __init__(
        self,
        openai_api_key: str,
        metrics: List[str] = None,
        minimum_score: float = 0.3,
    ):
        self.minimum_score = minimum_score
        os.environ["OPENAI_API_KEY"] = openai_api_key
        if metrics is None:
            try:
                # Adding a list of metrics
                from ragas.metrics import (
                    context_relevancy,
                    answer_relevancy,
                    faithfulness,
                    context_recall,
                )
                from ragas.metrics.critique import harmfulness

                self.metrics = [
                    context_relevancy,
                    answer_relevancy,
                    faithfulness,
                    context_recall,
                    harmfulness,
                ]

            except ModuleNotFoundError as e:
                print(
                    "Please install ragas to use this metric. `pip install ragas`."
                )
        else:
            metrics = self.metrics

    def measure(self, test_case: LLMTestCase):
        # sends to server
        try:
            from ragas import evaluate
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            # How do i make sure this isn't just huggingface dataset
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Create a dataset from the test case
        # Convert the LLMTestCase to a format compatible with Dataset
        data = {
            "expected_output": [test_case.expected_output],
            "contexts": [test_case.context],
            "output": [test_case.output],
            "id": [test_case.id],
        }
        dataset = Dataset.from_dict(data)

        # Evaluate the dataset using Ragas
        scores = evaluate(dataset, metrics=self.metrics)

        # Ragas only does dataset-level comparisons
        # >>> print(result["ragas_score"])
        # {'ragas_score': 0.860, 'context_relevancy': 0.817, 'faithfulness': 0.892,
        # 'answer_relevancy': 0.874}
        ragas_score = scores["ragas_score"]
        self.success = ragas_score >= self.minimum_score
        self.score = ragas_score
        return ragas_score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Ragas Score"


def assert_ragas(
    test_case: LLMTestCase,
    openai_api_key: str,
    metrics: List[str] = None,
    minimum_score: float = 0.3,
):
    """Asserts if the Ragas score is above the minimum score"""
    metric = RagasMetric(openai_api_key, metrics, minimum_score)
    score = metric.measure(test_case)
    assert (
        score >= metric.minimum_score
    ), f"Ragas score {score} is below the minimum score {metric.minimum_score}"

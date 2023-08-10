"""Test clients
"""
import unittest
from evals import Evaluator
from evals.pipeline import Pipeline
from evals.metric import RandomMetric


class TestEvaluator(unittest.TestCase):
    def test_eval(self):
        eval = Evaluator()
        result = eval.add_ground_truth(
            query="Example",
            expected_response="Customer success response is here",
            tags=["sample"],
        )
        assert True


def result_function(query):
    return "Custiomers are great!"


class TestE2E(unittest.TestCase):
    def test_e2e(self):
        eval = Evaluator()
        pipeline = Pipeline(
            pipeline_id="textract-pdf-processing", result_function=result_function
        )
        metric = RandomMetric()
        result = eval.evaluate(pipeline=pipeline, metric=metric)
        assert result


if __name__ == "__main__":
    unittest.main()

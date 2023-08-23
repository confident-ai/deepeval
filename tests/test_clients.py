"""Test clients
"""
import pytest
import unittest
from deepeval.evaluator import Evaluator
from deepeval.pipeline import Pipeline
from deepeval.metrics.randomscore import RandomMetric


@pytest.mark.skip(reason="Evaluator not finished")
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


@pytest.mark.skip(reason="Evaluator not finished")
class TestE2E(unittest.TestCase):
    def test_e2e(self):
        eval = Evaluator()
        pipeline = Pipeline(
            pipeline_id="textract-pdf-processing", result_function=result_function
        )
        metric = RandomMetric()
        result = eval.evaluate(pipeline=pipeline, metric=metric)
        assert True

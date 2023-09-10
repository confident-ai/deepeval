"""Test clients
"""
import unittest

import pytest

from deepeval.evaluator import Evaluator
from deepeval.metrics.randomscore import RandomMetric
from deepeval.pipeline import Pipeline


@pytest.mark.skip(reason="Evaluator not finished")
class TestEvaluator(unittest.LLMTestCase):
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
class TestE2E(unittest.LLMTestCase):
    def test_e2e(self):
        eval = Evaluator()
        pipeline = Pipeline(
            pipeline_id="textract-pdf-processing",
            result_function=result_function,
        )
        metric = RandomMetric()
        result = eval.evaluate(pipeline=pipeline, metric=metric)
        assert True

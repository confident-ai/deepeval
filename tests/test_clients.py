"""Test clients
"""
import unittest
from evals import Evaluator


class TestEvaluator(unittest.TestCase):
    def test_eval(self):
        eval = Evaluator()
        result = eval.add_ground_truth(
            query="Example",
            expected_response="Customer success response is here",
            tags=["sample"],
        )
        assert result == False


if __name__ == "__main__":
    unittest.main()

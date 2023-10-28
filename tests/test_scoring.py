"""Tests for metrics calculator
"""

import unittest
from deepeval.metrics.scoring import Scorer

# Testing exact_match_score metric

class TestScorer(unittest.TestCase):

    # tests for exact_match_score 

    def test_exact_equal_strings(self):
        target = "Winners of the FIFA world cup were the French national football team"
        prediction = "Winners of the FIFA world cup were the French national football team"
        self.assertEqual(Scorer.exact_match_score(target, prediction), 1)

    def test_exact_match_score_unequal_strings(self):
        target = "Hello, World!"
        prediction = "Goodbye, World!"
        self.assertEqual(Scorer.exact_match_score(target, prediction), 0)

    def test_exact_match_score_whitespace_difference(self):
        target = "Hello, World!"
        prediction = " Hello, World! "
        self.assertEqual(Scorer.exact_match_score(target, prediction), 1)

    def test_exact_match_score_empty_prediction(self):
        target = "Hello, World!"
        prediction = ""
        self.assertEqual(Scorer.exact_match_score(target, prediction), 0)

    def test_exact_match_score_empty_target(self):
        target = ""
        prediction = "Hello, World!"
        self.assertEqual(Scorer.exact_match_score(target, prediction), 0)

    # test for quasi_exact_match_score

    def test_exact_match(self):
        target = "The quick brown fox"
        prediction = "The quick brown fox"
        self.assertEqual(Scorer.quasi_exact_match_score(target, prediction), 1)

    def test_case_insensitive_match(self):
        target = "The quick brown fox"
        prediction = "the quick brown fox"
        self.assertEqual(Scorer.quasi_exact_match_score(target, prediction), 1)

    def test_partial_match(self):
        target = "The quick brown fox"
        prediction = "The brown fox"
        self.assertEqual(Scorer.quasi_exact_match_score(target, prediction), 0)

    def test_empty_prediction(self):
        target = "The quick brown fox"
        prediction = ""
        self.assertEqual(Scorer.quasi_exact_match_score(target, prediction), 0)
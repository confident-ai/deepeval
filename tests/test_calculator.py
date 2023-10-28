"""Tests for metrics calculator
"""

from deepeval.metrics.scores.calculator import StatisticalCalculator

# Testing exact_match metric 

def test_exact_equal_strings():
    target = "Winners of the FIFA world cup were the French national football team"
    prediction = "Winners of the FIFA world cup were the French national football team"
    assert StatisticalCalculator.exact_match(target, prediction) == 1

def test_exact_match_unequal_strings():
    target = "Hello, World!"
    prediction = "Goodbye, World!"
    assert StatisticalCalculator.exact_match(target, prediction) == 0

def test_exact_match_whitespace_difference():
    target = "Hello, World!"
    prediction = " Hello, World! "
    assert StatisticalCalculator.exact_match(target, prediction) == 1

def test_exact_match_empty_prediction():
    target = "Hello, World!"
    prediction = ""
    assert StatisticalCalculator.exact_match(target, prediction) == 0

def test_exact_match_empty_target():
    target = ""
    prediction = "Hello, World!"
    assert StatisticalCalculator.exact_match(target, prediction) == 0
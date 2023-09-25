"""Test to make sure assert statements can work
"""
from deepeval.metrics.conceptual_similarity import assert_conceptual_similarity

assert_conceptual_similarity(
    output="python is a programming language",
    expected_output="Python is a snake.",
    minimum_score=0.3,
)

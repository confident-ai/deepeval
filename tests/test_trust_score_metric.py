import pytest
from deepeval.metrics import TrustScoreMetric
from deepeval.test_case import LLMTestCase

def test_high_trust():
    source_tiers = {"SEC Filings": 1, "Verified Blog": 2, "Unverified Post": 4}
    metric = TrustScoreMetric(source_tiers=source_tiers)
    test_case = LLMTestCase(
        input="What is Apple's revenue?",
        actual_output="Apple's revenue is 394 billion.",
        retrieval_context=["According to SEC filings, Apple's revenue is 394 billion."]
    )
    metric.measure(test_case)
    assert metric.score == 1.0
    assert metric.success is True
    assert "SEC Filings" in metric.reason
    assert "tier 1" in metric.reason

def test_low_trust():
    source_tiers = {"SEC Filings": 1, "Verified Blog": 2, "Unverified Post": 4}
    metric = TrustScoreMetric(source_tiers=source_tiers)
    test_case = LLMTestCase(
        input="What is Apple's revenue?",
        actual_output="Apple's revenue is 394 billion.",
        retrieval_context=["I read in an unverified post that Apple's revenue is 394 billion."]
    )
    metric.measure(test_case)
    assert metric.score == 0.4
    assert metric.success is False
    assert "Unverified Post" in metric.reason
    assert "tier 4" in metric.reason

def test_mixed_sources():
    source_tiers = {"SEC Filings": 1, "Verified Blog": 2, "Unverified Post": 4}
    metric = TrustScoreMetric(source_tiers=source_tiers)
    test_case = LLMTestCase(
        input="What is Apple's revenue?",
        actual_output="Apple's revenue is 394 billion.",
        retrieval_context=[
            "According to SEC filings, Apple's revenue is 394 billion.",
            "I read in an unverified post that Apple's revenue is 394 billion."
        ]
    )
    metric.measure(test_case)
    assert metric.score == 0.7  # (1.0 + 0.4) / 2
    assert metric.success is True

def test_unmatched_source():
    source_tiers = {"SEC Filings": 1}
    metric = TrustScoreMetric(source_tiers=source_tiers)
    test_case = LLMTestCase(
        input="What is Apple's revenue?",
        actual_output="Apple's revenue is 394 billion.",
        retrieval_context=["A random guy told me Apple's revenue is 394 billion."]
    )
    metric.measure(test_case)
    assert metric.score == 0.5
    assert metric.success is False

def test_threshold_pass():
    source_tiers = {"Verified Blog": 2} # Tier 2 gives 0.8
    metric = TrustScoreMetric(source_tiers=source_tiers, threshold=0.7)
    test_case = LLMTestCase(
        input="What is Apple's revenue?",
        actual_output="Apple's revenue is 394 billion.",
        retrieval_context=["According to a Verified Blog, Apple's revenue is 394 billion."]
    )
    metric.measure(test_case)
    assert metric.success is True

def test_threshold_fail():
    source_tiers = {"Verified Blog": 2} # Tier 2 gives 0.8
    metric = TrustScoreMetric(source_tiers=source_tiers, threshold=0.9)
    test_case = LLMTestCase(
        input="What is Apple's revenue?",
        actual_output="Apple's revenue is 394 billion.",
        retrieval_context=["According to a Verified Blog, Apple's revenue is 394 billion."]
    )
    metric.measure(test_case)
    assert metric.success is False

def test_empty_retrieval_context():
    source_tiers = {"SEC Filings": 1}
    metric = TrustScoreMetric(source_tiers=source_tiers)
    test_case = LLMTestCase(
        input="What is Apple's revenue?",
        actual_output="Apple's revenue is 394 billion.",
        retrieval_context=[]
    )
    metric.measure(test_case)
    assert metric.score == 1.0
    assert metric.success is True
    assert "No retrieval context provided" in metric.reason

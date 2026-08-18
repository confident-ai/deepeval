import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import TrustScoreMetric

def test_trust_score():
    # Define trust tiers for sources
    source_tiers = {
        "SEC filing": 1,
        "Bloomberg": 2,
        "Verified news": 3,
        "Blog post": 4,
        "Reddit": 5
    }

    # Initialize metric
    metric = TrustScoreMetric(source_tiers=source_tiers, threshold=0.7)

    # Response A - High trust source
    test_case_a = LLMTestCase(
        input="What is the company's revenue?",
        actual_output="The company's revenue is $10M.",
        retrieval_context=["According to the SEC filing, the revenue is $10M."]
    )

    assert_test(test_case_a, [metric])

    # Response B - Low trust source
    test_case_b = LLMTestCase(
        input="What is the company's revenue?",
        actual_output="The company's revenue is $10M.",
        retrieval_context=["I read on a Blog post that the revenue is $10M."]
    )

    # This should fail because the score will be 0.4 which is < 0.7
    with pytest.raises(AssertionError):
        assert_test(test_case_b, [metric])

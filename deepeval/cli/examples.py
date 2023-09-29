CUSTOMER_EXAMPLE = """import pytest
from typing import Tuple
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric

# Let's start with a simple test case to understand the basic structure
# We will use a single query, output, and context for this example
# The query is the question asked, the output is the expected answer, and the context is the actual answer


# Define the test case
def test_customer_chatbot_simple():
    query = "What are your operating hours?"
    output = "Our operating hours are from 9 AM to 5 PM, Monday to Friday."
    context = "Our company operates from 10 AM to 6 PM, Monday to Friday."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=query, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )


# Now, let's move on to a more complex example using pytest fixtures
# Fixtures allow us to define a set of test data that can be used across multiple test cases
# In this case, we define a fixture that returns a list of tuples, each containing a query, output, and context


# Example bulk case
@pytest.fixture(
    params=[
        (
            "What are your operating hours?",
            "Our operating hours are from 9 AM to 5 PM, Monday to Friday.",
            "Our company operates from 10 AM to 6 PM, Monday to Friday.",
        ),
        (
            "What are your operating hours?",
            "We are open from 10 AM to 6 PM, Monday to Friday.",
            "Our company operates from 10 AM to 6 PM, Monday to Friday.",
        ),
        (
            "Do you offer free shipping?",
            "Yes, we offer free shipping on orders over $50.",
            "Our company offers free shipping on orders over $100.",
        ),
        (
            "Do you offer free shipping?",
            "Absolutely, free shipping is available for orders over $50.",
            "Our company offers free shipping on orders over $100.",
        ),
        (
            "What is your return policy?",
            "We have a 30-day return policy for unused and unopened items.",
            "Our company has a 14-day return policy for unused and unopened items.",
        ),
        (
            "What is your return policy?",
            "Unused and unopened items can be returned within 30 days.",
            "Our company has a 14-day return policy for unused and unopened items.",
        ),
        (
            "Do you have a physical store?",
            "No, we are an online-only store.",
            "Our company has physical stores in several locations.",
        ),
        (
            "Do you have a physical store?",
            "We operate solely online, we do not have a physical store.",
            "Our company has physical stores in several locations.",
        ),
        (
            "How can I track my order?",
            "You can track your order through the link provided in your confirmation email.",
            "Customers can track their orders by calling our customer service.",
        ),
        (
            "How can I track my order?",
            "Order tracking is available via the link in your confirmation email.",
            "Customers can track their orders by calling our customer service.",
        ),
    ]
)
def mock_chatgpt_input(request) -> Tuple[str, str, str]:
    return request.param


# We then use this fixture in our test case
# The test case is run once for each tuple in the fixture
@pytest.mark.parametrize("mock_chatgpt_input", [mock_chatgpt_input])
def test_customer_chatbot(mock_chatgpt_input: Tuple[str, str, str]):
    input, output, context = mock_chatgpt_input
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )
"""

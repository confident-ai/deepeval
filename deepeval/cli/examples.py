CUSTOMER_EXAMPLE = """import pytest
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
CHATBOT_TEST_CASES = [
    {
        "query": "What are your operating hours?",
        "output": "Our operating hours are from 9 AM to 5 PM, Monday to Friday.",
        # Context mocks search results returned in a retrieval-augmented generation pipeline
        "context": [
            "Our company operates from 10 AM to 6 PM, Monday to Friday.",
            "We are closed on weekends and public holidays.",
            "Our customer service is available 24/7.",
        ],
    },
    {
        "query": "What are your operating hours?",
        "output": "We are open from 10 AM to 6 PM, Monday to Friday.",
        "context": [
            "Our company operates from 10 AM to 6 PM, Monday to Friday.",
            "We are closed on weekends and public holidays.",
            "Our customer service is available 24/7.",
        ],
    },
    {
        "query": "Do you offer free shipping?",
        "output": "Yes, we offer free shipping on orders over $50.",
        "context": [
            "Our company offers free shipping on orders over $100.",
            "Free shipping applies to all products.",
            "The free shipping offer is valid only within the country.",
        ],
    },
    {
        "query": "Do you offer free shipping?",
        "output": "Absolutely, free shipping is available for orders over $50.",
        "context": [
            "Our company offers free shipping on orders over $100.",
            "Free shipping applies to all products.",
            "The free shipping offer is valid only within the country.",
        ],
    },
    {
        "query": "What is your return policy?",
        "output": "We have a 30-day return policy for unused and unopened items.",
        "context": [
            "Our company has a 14-day return policy for unused and unopened items.",
            "Items must be returned in their original packaging.",
            "Customers are responsible for return shipping costs.",
        ],
    },
    {
        "query": "What is your return policy?",
        "output": "Unused and unopened items can be returned within 30 days.",
        "context": [
            "Our company has a 14-day return policy for unused and unopened items.",
            "Items must be returned in their original packaging.",
            "Customers are responsible for return shipping costs.",
        ],
    },
    {
        "query": "Do you have a physical store?",
        "output": "No, we are an online-only store.",
        "context": [
            "Our company has physical stores in several locations.",
            "We have stores in major cities across the country.",
            "You can find the nearest store using our store locator.",
        ],
    },
    {
        "query": "Do you have a physical store?",
        "output": "We operate solely online, we do not have a physical store.",
        "context": [
            "Our company operates solely online.",
            "We do not have any physical stores.",
            "All our products are available on our website.",
        ],
    },
    {
        "query": "How can I track my order?",
        "output": "You can track your order through the link provided in your confirmation email.",
        "context": [
            "Customers can track their orders by calling our customer service.",
            "You can also track your order using the tracking number provided in the confirmation email.",
            "For any issues with tracking, please contact our customer support.",
        ],
    },
    {
        "query": "How can I track my order?",
        "output": "Order tracking is available via the link in your confirmation email.",
        "context": [
            "Customers can track their orders by calling our customer service.",
            "You can also track your order using the tracking number provided in the confirmation email.",
            "For any issues with tracking, please contact our customer support.",
        ],
    },
]


# We then use this constant in our test case
# The test case is run once for each tuple in the constant
@pytest.mark.parametrize(
    "test_case",
    CHATBOT_TEST_CASES,
)
def test_customer_chatbot(test_case: dict):
    query = test_case["query"]
    output = test_case["output"]
    context = test_case["context"]
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=query, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )
"""


EXAMPLE_BROCHURE = """[Front Cover]

Customer Support Guide
Providing You with Exceptional Service

[Inside Cover]

At TrendyTrends, we value your satisfaction and are committed to delivering top-notch customer support. This brochure is designed to assist you with any inquiries you may have regarding shipping and delivery times. We're here to make your experience as seamless as possible.

[Page 1]

Shipping & Delivery Times

At TrendyTrends, we understand that timely delivery is crucial to your satisfaction. Our dedicated team works tirelessly to ensure your orders reach you promptly. Here's what you need to know about our shipping and delivery times:

1. Standard Shipping:

Delivery Time: 3-5 business days
Cost: Free for orders over $50; $5 for orders under $50
Coverage: Nationwide
2. Express Shipping:

Delivery Time: 1-2 business days
Cost: $15
Coverage: Nationwide
[Page 2]

Order Tracking

We offer convenient order tracking so you can monitor the progress of your package. Simply visit our website or use our mobile app to enter your order number and get real-time updates on the status of your shipment. We believe in transparency and keeping you informed every step of the way.

[Page 3]

Our Commitment to You

At TrendyTrends, our commitment to exceptional customer support goes beyond just shipping and delivery times. We are dedicated to:

Providing friendly and knowledgeable customer service representatives to assist you.
Resolving any issues or concerns promptly and efficiently.
Ensuring the safe and secure delivery of your orders.
[Page 4]

Contact Us

Should you have any questions, concerns, or need assistance with your order, our customer support team is here to help:

Customer Support Hotline: 1-800-123-4567
Email: support@trendytrends.com
Live Chat: Available on our website during business hours
[Back Cover]

Thank you for choosing TrendyTrends. Your satisfaction is our top priority, and we look forward to serving you. For the latest updates, promotions, and more, follow us on social media or visit our website at www.trendytrends.com.

[Disclaimer]

Shipping and delivery times are estimates and may vary due to factors beyond our control. For the most accurate delivery information, please refer to your order tracking or contact our customer support team."""

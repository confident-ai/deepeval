CUSTOMER_EXAMPLE = """from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric


def test_customer_chatbot_1():
    input = "What are your operating hours?"
    output = "Our operating hours are from 9 AM to 5 PM, Monday to Friday."
    context = "Our company operates from 10 AM to 6 PM, Monday to Friday."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )


def test_customer_chatbot_1_alternative():
    input = "What are your operating hours?"
    output = "We are open from 10 AM to 6 PM, Monday to Friday."
    context = "Our company operates from 10 AM to 6 PM, Monday to Friday."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )


def test_customer_chatbot_2():
    input = "Do you offer free shipping?"
    output = "Yes, we offer free shipping on orders over $50."
    context = "Our company offers free shipping on orders over $100."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )


def test_customer_chatbot_2_alternative():
    input = "Do you offer free shipping?"
    output = "Absolutely, free shipping is available for orders over $50."
    context = "Our company offers free shipping on orders over $100."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )


def test_customer_chatbot_3():
    input = "What is your return policy?"
    output = "We have a 30-day return policy for unused and unopened items."
    context = (
        "Our company has a 14-day return policy for unused and unopened items."
    )
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )


def test_customer_chatbot_3_alternative():
    input = "What is your return policy?"
    output = "Unused and unopened items can be returned within 30 days."
    context = (
        "Our company has a 14-day return policy for unused and unopened items."
    )
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )


def test_customer_chatbot_4():
    input = "Do you have a physical store?"
    output = "No, we are an online-only store."
    context = "Our company has physical stores in several locations."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )


def test_customer_chatbot_4_alternative():
    input = "Do you have a physical store?"
    output = "We operate solely online, we do not have a physical store."
    context = "Our company has physical stores in several locations."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )


def test_customer_chatbot_5():
    input = "How can I track my order?"
    output = "You can track your order through the link provided in your confirmation email."
    context = (
        "Customers can track their orders by calling our customer service."
    )
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )


def test_customer_chatbot_5_alternative():
    input = "How can I track my order?"
    output = (
        "Order tracking is available via the link in your confirmation email."
    )
    context = (
        "Customers can track their orders by calling our customer service."
    )
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, context=context)
    assert_test(
        test_case, [factual_consistency_metric, answer_relevancy_metric]
    )
"""

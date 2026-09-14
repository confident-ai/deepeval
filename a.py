from deepeval.evaluate import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

evaluate(test_cases=[
    LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="Paris"
    )
], metrics=[AnswerRelevancyMetric()])

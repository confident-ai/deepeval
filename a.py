from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

evaluate(
    test_cases=[
        LLMTestCase(
            input="What is the weather in San Francisco?",
            generated_output="It is sunny and 70 degrees.",
        ),
        LLMTestCase(
            input="What is the weather in San Francisco?",
            generated_output="I've a dog",
        ),
    ],
    metrics=[AnswerRelevancyMetric()],
)

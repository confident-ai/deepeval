from deepeval import evaluate
from deepeval.metrics import TurnRelevancyMetric
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn

evaluate(
    test_cases=[
        ConversationalTestCase(
            scenario="You are a helpful assistant.",
            user_description="What is the weather in San Francisco?",
            context=["It is sunny and 70 degrees."],
            tags=["weather"],
            turns=[
                Turn(
                    role="user",
                    content="What is the weather in San Francisco?",
                ),
            ],
        ),
        ConversationalTestCase(
            scenario="You are a helpful assistant.",
            user_description="What is the weather in San Francisco?",
            context=["It is sunny and 70 degrees."],
            tags=["something else"],
            turns=[
                Turn(
                    role="user",
                    content="What is the weather in San Francisco?",
                ),
            ],
        ),
    ],
    metrics=[TurnRelevancyMetric()],
)

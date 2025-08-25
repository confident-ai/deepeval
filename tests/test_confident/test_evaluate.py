from deepeval.evaluate import evaluate
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn


def test_single_turn_evaluate():
    evaluate(
        test_cases=[
            LLMTestCase(
                input="What is the capital of France?", actual_output="Paris"
            )
        ],
        metric_collection="single_turn_test",
    )


def test_multi_turn_evaluate():
    evaluate(
        test_cases=[
            ConversationalTestCase(
                turns=[
                    Turn(
                        role="user",
                        content="What is the capital of France?",
                    ),
                    Turn(
                        role="assistant",
                        content="Paris",
                    ),
                ]
            )
        ],
        metric_collection="multi_turn_test",
    )

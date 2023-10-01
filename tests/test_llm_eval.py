from deepeval.test_case import LLMTestCase
from deepeval.metrics.llm_eval import LLMEvalMetric


def generate_chatgpt3(prompt):
    """Generate a response from Chat GPT3."""
    return '{"score": 100, "reason": "The response is a valid response to the prompt."}'


def test_chat_completion():
    """Test Chat Completion"""
    metric = LLMEvalMetric(
        criteria="The response is a valid response to the prompt.",
        completion_function=generate_chatgpt3,
        minimum_score=0.5,
    )
    test_case = LLMTestCase(
        query="What is the capital of France?",
        output="Paris",
        expected_output="Paris",
        context="Geography",
    )
    metric.measure(test_case)
    assert metric.is_successful() is True
    assert metric.measure(test_case, include_score=True) == 1.0

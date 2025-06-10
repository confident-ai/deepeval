import pytest
from deepeval.models import LiteLLMModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


# Common model configuration for all tests
@pytest.fixture
def model():
    return LiteLLMModel(
        model="lm_studio/Meta-Llama-3.1-8B-Instruct-GGUF",
        api_key="lm-studio",
        api_base="http://localhost:1234/v1",
        temperature=0,
    )


def test_litellm_with_geval_basic(model):
    """Test basic usage of G-Eval metric with LiteLLM."""
    # Create a test case
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris",
        context=["France is a country in Europe. Its capital city is Paris."],
    )

    # Initialize G-Eval metric
    metric = GEval(
        name="answer-accuracy",
        criteria="Evaluate if the answer correctly identifies the capital of France.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=0.5,
    )

    # Measure the score
    score = metric.measure(test_case)

    # Assertions
    assert isinstance(score, float)
    assert 0 <= score <= 1
    assert score >= 0.5  # Since the answer is correct


def test_litellm_with_geval_complex(model):
    """Test complex usage of G-Eval metric with LiteLLM."""
    # Create a test case with more complex evaluation
    test_case = LLMTestCase(
        input="Explain the process of photosynthesis.",
        actual_output="""
        Photosynthesis is the process by which plants convert light energy into chemical energy.
        The process involves:
        1. Light absorption by chlorophyll
        2. Water splitting and oxygen release
        3. Carbon dioxide conversion to glucose
        """,
        expected_output="""
        Photosynthesis is a process where plants use sunlight to make food.
        They take in water and carbon dioxide, and release oxygen.
        The end product is glucose, which the plant uses for energy.
        """,
        context=[
            "Photosynthesis is a fundamental biological process in plants."
        ],
    )

    # Initialize G-Eval metric with custom evaluation criteria
    metric = GEval(
        name="explanation-quality",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        model=model,
        threshold=0.7,
        evaluation_steps=[
            "Check if the explanation covers all key points",
            "Verify scientific accuracy",
            "Assess clarity and organization",
            "Compare with expected output",
        ],
    )

    # Measure the score
    score = metric.measure(test_case)

    # Assertions
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_litellm_with_geval_multiple_cases(model):
    """Test G-Eval metric with multiple test cases."""
    # Create multiple test cases
    test_cases = [
        LLMTestCase(
            input="What is 2+2?",
            actual_output="The sum of 2 and 2 is 4.",
            expected_output="4",
            context=["Basic arithmetic question"],
        ),
        LLMTestCase(
            input="Who wrote Romeo and Juliet?",
            actual_output="Romeo and Juliet was written by William Shakespeare.",
            expected_output="William Shakespeare",
            context=["Literature question about Shakespeare's works"],
        ),
        LLMTestCase(
            input="What is the chemical formula for water?",
            actual_output="The chemical formula for water is H2O.",
            expected_output="H2O",
            context=["Basic chemistry question"],
        ),
    ]

    # Initialize G-Eval metric
    metric = GEval(
        name="multiple-answers",
        criteria="Evaluate if each answer correctly matches the expected output. The answer should be exact and complete.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=0.5,
        evaluation_steps=[
            "Check if the answer matches the expected output exactly",
            "Verify the answer is complete and accurate",
            "Ensure the answer is properly formatted",
        ],
    )

    # Measure scores for all test cases
    scores = [metric.measure(test_case) for test_case in test_cases]

    # Assertions
    assert all(isinstance(score, float) for score in scores)
    assert all(0 <= score <= 1 for score in scores)
    assert all(score >= 0.5 for score in scores)  # All answers are correct


def test_litellm_with_geval_custom_evaluation(model):
    """Test G-Eval metric with custom evaluation criteria."""
    # Create a test case
    test_case = LLMTestCase(
        input="Write a short story about a robot.",
        actual_output="""
        Once upon a time, there was a robot named R2D2 who loved to dance.
        Every day, it would practice its moves in the garden.
        One day, it won a dance competition and made many friends.
        """,
        expected_output="""
        A story about a robot that learns to dance and makes friends.
        The story should be creative and engaging.
        """,
        context=["Creative writing task"],
    )

    # Initialize G-Eval metric with custom evaluation criteria
    metric = GEval(
        name="story-quality",
        criteria="Evaluate the story's quality based on creativity, engagement, and theme adherence. The story should be creative, engaging, and follow the theme of a robot learning to dance and making friends.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=0.6,
        evaluation_steps=[
            "Check if the story is creative and engaging",
            "Verify if it follows the expected theme",
            "Assess the story's structure and flow",
            "Evaluate character development",
            "Check for proper grammar and spelling",
        ],
    )

    # Measure the score
    score = metric.measure(test_case)

    # Assertions
    assert isinstance(score, float)
    assert 0 <= score <= 1
    assert score >= 0.6  # Assuming the story meets the quality threshold

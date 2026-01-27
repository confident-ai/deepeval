from deepeval.test_case import (
    ArenaTestCase,
    LLMTestCase,
    LLMTestCaseParams,
    Contestant,
)
from deepeval.metrics import ArenaGEval
from deepeval.evaluate import compare
from deepeval.prompt import Prompt

ALIAS_WITH_INTERPOLATION_TYPE = "test_prompt_list_interpolation_type"


def test_compare():
    metric = ArenaGEval(
        name="Friendly",
        criteria="Choose the winner of the more friendly contestant based on the input and actual output",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )
    a_test_case = ArenaTestCase(
        contestants=[
            Contestant(
                name="GPT-4",
                test_case=LLMTestCase(
                    input="What is the capital of France?",
                    actual_output="Paris",
                ),
            ),
            Contestant(
                name="Claude-4",
                test_case=LLMTestCase(
                    input="What is the capital of France?",
                    actual_output="Paris is the capital of France.",
                ),
            ),
        ],
    )
    a_test_case2 = ArenaTestCase(
        contestants=[
            Contestant(
                name="GPT-4",
                test_case=LLMTestCase(
                    input="What is the capital of France?",
                    actual_output="Paris",
                ),
            ),
            Contestant(
                name="Claude-4",
                test_case=LLMTestCase(
                    input="What is the capital of France?",
                    actual_output="Paris is the capital of France.",
                ),
            ),
        ],
    )
    compare(
        test_cases=[a_test_case, a_test_case2],
        metric=metric,
    )


def test_compare_with_hyperparameters():
    metric = ArenaGEval(
        name="Friendly",
        criteria="Choose the winner of the more friendly contestant based on the input and actual output",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )
    a_test_case = ArenaTestCase(
        contestants=[
            Contestant(
                name="GPT-4",
                hyperparameters={"model": "gpt-4"},
                test_case=LLMTestCase(
                    input="What is the capital of France?",
                    actual_output="Paris",
                ),
            ),
            Contestant(
                name="Claude-4",
                hyperparameters={"model": "claude-4"},
                test_case=LLMTestCase(
                    input="What is the capital of France?",
                    actual_output="Paris is the capital of France.",
                ),
            ),
        ],
    )
    a_test_case2 = ArenaTestCase(
        contestants=[
            Contestant(
                name="GPT-4",
                hyperparameters={"model": "gpt-4"},
                test_case=LLMTestCase(
                    input="What is the capital of France?",
                    actual_output="Paris",
                ),
            ),
            Contestant(
                name="Claude-4",
                hyperparameters={"model": "claude-4"},
                test_case=LLMTestCase(
                    input="What is the capital of France?",
                    actual_output="Paris is the capital of France.",
                ),
            ),
        ],
    )

    prompt = Prompt(alias=ALIAS_WITH_INTERPOLATION_TYPE)
    prompt.pull()
    compare(
        test_cases=[a_test_case, a_test_case2],
        metric=metric,
    )

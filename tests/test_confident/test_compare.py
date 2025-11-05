from deepeval.test_case import ArenaTestCase, LLMTestCase, LLMTestCaseParams
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
        contestants={
            "GPT-4": LLMTestCase(
                input="What is the capital of France?",
                actual_output="Paris",
            ),
            "Claude-4": LLMTestCase(
                input="What is the capital of France?",
                actual_output="Paris is the capital of France.",
            ),
        },
    )
    a_test_case2 = ArenaTestCase(
        contestants={
            "GPT-4": LLMTestCase(
                input="What is the capital of France?",
                actual_output="Paris",
            ),
            "Claude-4": LLMTestCase(
                input="What is the capital of France?",
                actual_output="Paris is the capital of France.",
            ),
        },
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
        contestants={
            "GPT-4": LLMTestCase(
                input="What is the capital of France?",
                actual_output="Paris",
            ),
            "Claude-4": LLMTestCase(
                input="What is the capital of France?",
                actual_output="Paris is the capital of France.",
            ),
        },
    )
    a_test_case2 = ArenaTestCase(
        contestants={
            "GPT-4": LLMTestCase(
                input="What is the capital of France?",
                actual_output="Paris",
            ),
            "Claude-4": LLMTestCase(
                input="What is the capital of France?",
                actual_output="Paris is the capital of France.",
            ),
        },
    )

    prompt = Prompt(alias=ALIAS_WITH_INTERPOLATION_TYPE)
    prompt.pull()
    compare(
        test_cases=[a_test_case, a_test_case2],
        metric=metric,
        hyperparameters={
            "GPT-4": {"model": "gpt-4", "prompt": prompt},
            "Claude-4": {"model": "claude-4", "prompt": prompt},
        },
    )

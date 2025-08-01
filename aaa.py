from deepeval.evaluate.configs import AsyncConfig
from deepeval.test_case import ArenaTestCase, LLMTestCase, LLMTestCaseParams
from deepeval.metrics import ArenaGEval
from deepeval import compare

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
metric = ArenaGEval(
    name="Friendly",
    criteria="Choose the winner of the more friendly contestant based on the input and actual output",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)

compare(test_cases=[a_test_case, a_test_case2] * 10, metric=metric, async_config=AsyncConfig(run_async=True))

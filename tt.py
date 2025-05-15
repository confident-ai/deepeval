from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK",
        "Allow different wording",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    # rubric=[
    #     Rubric(
    #         score_range=(6, 8),
    #        expected_outcome="The actual output is factually correct based on the expected output, but there is some detail missing.",
    #     ),
    #     Rubric(
    #         score_range=(8, 10),
    #         expected_outcome="The actual output is completely correct based on the expected output",
    #     ),
    # ],
    verbose_mode=True,
    strict_mode=True,
    async_mode=False,
)

test_case = LLMTestCase(
    input="The dog chased the cat up the tree, who ran up the tree?",
    actual_output="The kitty cat!",
    expected_output="The cat.",
)
correctness_metric.measure(test_case)

print(correctness_metric.score, correctness_metric.reason)

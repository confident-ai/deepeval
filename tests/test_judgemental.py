import pytest
from deepeval.metrics import JudgementalGPT
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.types import Languages
from deepeval import assert_test


@pytest.mark.skip(reason="openai is expensive")
def test_judgemntal():
    test_case = LLMTestCase(
        input="I am a plant",
        actual_output="thanks for letting me know you're a plant",
    )

    metric = JudgementalGPT(
        name="Coherence",
        criteria="Coherence - determine whether 'actual output' follows from the 'input'.",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.INPUT,
        ],
        language=Languages.SPANISH,
        threshold=0.5,
    )

    assert_test(test_case, [metric])

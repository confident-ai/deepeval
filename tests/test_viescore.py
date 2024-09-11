from PIL import Image
import pytest

from deepeval.dataset import EvaluationDataset
from deepeval import assert_test, evaluate
from deepeval.test_case import MLLMTestCase, LLMTestCase
from deepeval.metrics import VIEScore, AnswerRelevancyMetric

# Load the WebP image using PIL
image_path = "./data/image.webp"
image = Image.open(image_path)

test_case_1 = MLLMTestCase(
    input_text="generate a castle school in fantasy land with the words LLM evaluation on it",
    actual_output_image=image,
)

test_case_2 = MLLMTestCase(
        input_text="generate a castle school in fantasy land with the words LLM evaluation on it",
        actual_output_image=image,
    )

test_case_3 = LLMTestCase(
    input="What is this again?",
    actual_output="this is a latte",
    expected_output="this is a mocha",
    retrieval_context=["I love coffee"],
    context=["I love coffee"],
    expected_tools=["mixer", "creamer", "dripper"],
    tools_called=["mixer", "creamer", "mixer"],
)

dataset = EvaluationDataset(
    test_cases=[test_case_1, test_case_2, test_case_3]
)
#dataset.evaluate([VIEScore(verbose_mode=True), AnswerRelevancyMetric()])
#evaluate(test_cases=[test_case_1, test_case_2, test_case_3], metrics=[VIEScore(verbose_mode=True), AnswerRelevancyMetric()], run_async=False)

@pytest.mark.skip(reason="openai is expensive")
def test_viescore():
    vie_score = VIEScore(verbose_mode=True)
    assert_test(test_case_1, [vie_score, AnswerRelevancyMetric()], run_async=False)

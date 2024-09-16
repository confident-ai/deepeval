import pytest

from deepeval.dataset import EvaluationDataset
from deepeval import assert_test, evaluate
from deepeval.test_case import MLLMTestCase, LLMTestCase, MLLMImage
from deepeval.metrics import VIEScore, AnswerRelevancyMetric, VIEScoreTask

image_path = "./data/image.webp"
edited_image_path = "./data/edited_image.webp"

test_case_1 = MLLMTestCase(
    input=[
        "gesnerate a castle school in fantasy land with the words LLM evaluation on it"
    ],
    actual_output=[MLLMImage(image_path, local=True)],
)

test_case_2 = MLLMTestCase(
    input=[
        "edit this image so that it is night themed, and LLM evaluation is spelled correctly",
        MLLMImage(image_path, local=True),
    ],
    actual_output=[MLLMImage(edited_image_path, local=True)],
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

dataset = EvaluationDataset(test_cases=[test_case_2, test_case_3])
dataset.evaluate(
    [
        # VIEScore(verbose_mode=True),
        VIEScore(verbose_mode=True, task=VIEScoreTask.TEXT_TO_IMAGE_EDITING),
        AnswerRelevancyMetric(),
    ]
)

evaluate(
    test_cases=[
        test_case_1,
        # test_case_2,
        test_case_3,
    ],
    metrics=[
        VIEScore(verbose_mode=True),
        # VIEScore(verbose_mode=True, task=VIEScoreTask.TEXT_TO_IMAGE_EDITING),
        AnswerRelevancyMetric(),
    ],
    # run_async=False
)


# #@pytest.mark.skip(reason="openai is expensive")
# def test_viescore():
#     vie_score = VIEScore(verbose_mode=True)
#     vie_score_2 = VIEScore(
#         verbose_mode=True, task=VIEScoreTask.TEXT_TO_IMAGE_EDITING
#     )
#     assert_test(
#         test_case_2, [vie_score_2, AnswerRelevancyMetric()], run_async=False
#     )

import pytest

from deepeval.dataset import EvaluationDataset
from deepeval import assert_test, evaluate
from deepeval.test_case import MLLMTestCase, LLMTestCase, MLLMImage
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ImageEditingMetric,
    TextToImageMetric,
)

image_path = "./data/image.webp"
edited_image_path = "./data/edited_image.webp"

#############################################################
# TestCases
#############################################################

text_to_image_test_case = MLLMTestCase(
    input=[
        "gesnerate a castle school in fantasy land with the words LLM evaluation on it"
    ],
    actual_output=[MLLMImage(image_path, local=True)],
)

image_editing_test_case = MLLMTestCase(
    input=[
        "edit this image so that it is night themed, and LLM evaluation is spelled correctly",
        MLLMImage(image_path, local=True),
    ],
    actual_output=[MLLMImage(edited_image_path, local=True)],
)

llm_test_case = LLMTestCase(
    input="What is this again?",
    actual_output="this is a latte",
    expected_output="this is a mocha",
    retrieval_context=["I love coffee"],
    context=["I love coffee"],
    expected_tools=["mixer", "creamer", "dripper"],
    tools_called=["mixer", "creamer", "mixer"],
)


#############################################################
# Evaluate
#############################################################

# dataset = EvaluationDataset(
#     test_cases=[
#         # text_to_image_test_case,
#         image_editing_test_case,
#         llm_test_case
#     ]
# )
# dataset.evaluate(
#     [
#         # TextToImageMetric(),
#         ImageEditingMetric(),
#         AnswerRelevancyMetric(),
#     ]
# )

evaluate(
    test_cases=[
        text_to_image_test_case,
        # image_editing_test_case,
        llm_test_case,
    ],
    metrics=[
        TextToImageMetric(),
        # ImageEditingMetric(),
        AnswerRelevancyMetric(),
    ],
)

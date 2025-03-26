import pytest

from deepeval.dataset import EvaluationDataset
from deepeval import evaluate
from deepeval.test_case import MLLMTestCase, LLMTestCase, MLLMImage
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ImageEditingMetric,
    TextToImageMetric,
    MultimodalContextualRecallMetric,
    MultimodalContextualRelevancyMetric,
    MultimodalContextualPrecisionMetric,
    MultimodalAnswerRelevancyMetric,
    MultimodalFaithfulnessMetric,
)

image_path = "./data/image_1.jpg"
edited_image_path = "./data/edited_image.webp"


@pytest.fixture
def text_to_image_case():
    return MLLMTestCase(
        input=["generate an image of the eiffel tower"],
        actual_output=[
            MLLMImage(
                url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/375px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"
            )
        ],
    )


@pytest.fixture
def image_editing_case():
    return MLLMTestCase(
        input=[
            "Make this image more interesting.",
            MLLMImage(image_path, local=True),
        ],
        actual_output=[
            MLLMImage(
                url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/Alpaca_%2831562329701%29.jpg/1280px-Alpaca_%2831562329701%29.jpg",
                local=False,
            )
        ],
    )


@pytest.fixture
def multimodal_rag_case():
    return MLLMTestCase(
        input=["Tell me about some landmarks in France"],
        actual_output=[
            "The Eiffel Tower is located in Paris, France.",
            MLLMImage(
                url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/375px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"
            ),
            "The Statue of Liberty was a gift from France to the United States.",
            MLLMImage(
                url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Front_view_of_Statue_of_Liberty_with_pedestal_and_base_2024.jpg/375px-Front_view_of_Statue_of_Liberty_with_pedestal_and_base_2024.jpg"
            ),
        ],
        expected_output=[
            "The Eiffel Tower is located in Paris, France.",
            MLLMImage(
                url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/375px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"
            ),
            "The Statue of Liberty was a gift from France to the United States.",
            MLLMImage(
                url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Front_view_of_Statue_of_Liberty_with_pedestal_and_base_2024.jpg/375px-Front_view_of_Statue_of_Liberty_with_pedestal_and_base_2024.jpg"
            ),
        ],
        context=[
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
            "It is named after the engineer Gustave Eiffel.",
            MLLMImage(
                url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/375px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"
            ),
            "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor.",
            MLLMImage(
                url="https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Liberty-from-behind-2024.jpg/330px-Liberty-from-behind-2024.jpg"
            ),
        ],
        retrieval_context=[
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
            "It is named after the engineer Gustave Eiffel.",
            MLLMImage(
                url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/375px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"
            ),
            "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor.",
            MLLMImage(
                url="https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Liberty-from-behind-2024.jpg/330px-Liberty-from-behind-2024.jpg"
            ),
        ],
    )


#############################################################
# Tests
#############################################################


def test_text_to_image(text_to_image_case):
    evaluate(
        test_cases=[text_to_image_case],
        metrics=[TextToImageMetric()],
        verbose_mode=True,
        run_async=False,
    )


def test_image_editing(image_editing_case):
    evaluate(
        test_cases=[image_editing_case],
        metrics=[ImageEditingMetric()],
        verbose_mode=True,
        run_async=False,
    )


def test_multimodal_rag_case(multimodal_rag_case):
    evaluate(
        test_cases=[multimodal_rag_case],
        metrics=[
            MultimodalContextualRecallMetric(),
            MultimodalContextualRelevancyMetric(),
            MultimodalContextualPrecisionMetric(),
            MultimodalAnswerRelevancyMetric(),
            MultimodalFaithfulnessMetric(),
        ],
        verbose_mode=True,
        run_async=False,
    )

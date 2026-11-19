import pytest

from deepeval.metrics.base_metric import BaseConversationalMetric
from deepeval.metrics.utils import check_conversational_test_case_params
from deepeval.test_case import ConversationalTestCase, MLLMImage, Turn
from deepeval.test_case.api import create_api_test_case


@pytest.mark.parametrize("extension", ["png", "pdf"])
def test_detects_images_and_pdfs_in_conversation_context(extension):
    image = MLLMImage(url=f"https://example.org/reference.{extension}")
    context = ["Reference material", f"Diagram: {image}", "Additional text"]
    case = ConversationalTestCase(
        turns=[Turn(role="user", content="Describe the reference.")],
        context=context,
    )

    assert case.multimodal is True
    assert case.context == context
    api_case = create_api_test_case(case, index=0)
    assert api_case.images_mapping[image._id].url == image.url


@pytest.mark.parametrize("context", [None, [], ["Only text"]])
def test_plain_conversation_context_stays_text_only(context):
    case = ConversationalTestCase(
        turns=[Turn(role="user", content="Hello")], context=context
    )

    assert case.multimodal is False


def test_plain_context_does_not_disable_turn_image_detection():
    image = MLLMImage(url="https://example.org/turn.png")
    case = ConversationalTestCase(
        turns=[Turn(role="user", content=f"Describe this: {image}")],
        context=["Only text"],
    )

    assert case.multimodal is True


def test_explicit_multimodal_flag_is_preserved():
    case = ConversationalTestCase(
        turns=[Turn(role="user", content="Hello")],
        context=["Only text"],
        multimodal=True,
    )

    assert case.multimodal is True


def test_context_image_is_rejected_by_text_only_system_one_validation():
    image = MLLMImage(url="https://example.org/context.png")
    case = ConversationalTestCase(
        turns=[Turn(role="user", content="Describe the reference.")],
        context=[f"Diagram: {image}"],
    )

    with pytest.raises(ValueError, match="evaluates text only"):
        check_conversational_test_case_params(
            case,
            [],
            BaseConversationalMetric(),
            model=None,
            multimodal=case.multimodal,
        )

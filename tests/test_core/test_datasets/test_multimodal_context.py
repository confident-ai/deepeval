import pytest

from deepeval.dataset import ConversationalGolden
from deepeval.test_case import MLLMImage, Turn


@pytest.mark.parametrize("extension", ["png", "pdf"])
def test_detects_images_and_pdfs_in_conversation_context(extension):
    image = MLLMImage(url=f"https://example.org/reference.{extension}")
    context = ["Reference material", f"Diagram: {image}", "Additional text"]
    golden = ConversationalGolden(
        scenario="Describe the reference.", context=context
    )

    assert golden.multimodal is True
    assert golden.context == context
    golden._prepare_for_api()
    assert golden.images_mapping[image._id].url == image.url


@pytest.mark.parametrize("context", [None, [], ["Only text"]])
def test_plain_conversation_context_stays_text_only(context):
    golden = ConversationalGolden(scenario="A conversation", context=context)

    assert golden.multimodal is False


@pytest.mark.parametrize("field", ["scenario", "expected_outcome"])
def test_plain_context_does_not_disable_existing_image_detection(field):
    image = MLLMImage(url="https://example.org/field.png")
    fields = {"scenario": "A conversation", field: f"Describe this: {image}"}
    golden = ConversationalGolden(context=["Only text"], **fields)

    assert golden.multimodal is True


def test_plain_context_does_not_disable_turn_image_detection():
    image = MLLMImage(url="https://example.org/turn.png")
    golden = ConversationalGolden(
        scenario="A conversation",
        turns=[Turn(role="user", content=f"Describe this: {image}")],
        context=["Only text"],
    )

    assert golden.multimodal is True


def test_explicit_multimodal_flag_is_preserved():
    golden = ConversationalGolden(
        scenario="A conversation", context=["Only text"], multimodal=True
    )

    assert golden.multimodal is True

import base64

import pytest
import requests

from deepeval.config.settings import get_settings
from deepeval.errors import DeepEvalError
from deepeval.models.llms.multimodal import (
    decode_image,
    fetch_remote_image,
    multimodal_parts,
    to_openai_content,
)
from deepeval.test_case import MLLMImage

IMAGE_BYTES = b"\x89PNG-bytes"
IMAGE_B64 = base64.b64encode(IMAGE_BYTES).decode()


class FakeResponse:
    def __init__(self, headers):
        self.content = IMAGE_BYTES
        self.headers = headers

    def raise_for_status(self):
        pass


@pytest.fixture
def fetches(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse({"content-type": "image/webp"})

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


def test_multimodal_parts_is_none_for_text_only_prompt():
    assert multimodal_parts("just text") is None


def test_multimodal_parts_splits_text_and_images_in_order():
    first = MLLMImage(dataBase64=IMAGE_B64, mimeType="image/png")
    second = MLLMImage(url="https://example.com/cat.png")

    parts = multimodal_parts(f"A {first} B {second}")

    assert parts == ["A ", first, " B ", second]


def test_to_openai_content_wraps_text_only_prompt():
    assert to_openai_content("hello") == [{"type": "text", "text": "hello"}]


def test_to_openai_content_text_only_as_string():
    assert to_openai_content("hello", text_only_as_string=True) == "hello"


def test_to_openai_content_formats_remote_and_inline_images(fetches):
    remote = MLLMImage(url="https://example.com/cat.png")
    inline = MLLMImage(dataBase64=IMAGE_B64, mimeType="image/png")

    content = to_openai_content(f"Compare {remote} with {inline}")

    assert content == [
        {"type": "text", "text": "Compare "},
        {"type": "image_url", "image_url": {"url": remote.url}},
        {"type": "text", "text": " with "},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{IMAGE_B64}"},
        },
    ]
    assert fetches == []


def test_to_openai_content_inlines_remote_images_when_asked(fetches):
    remote = MLLMImage(url="https://example.com/cat.png")

    content = to_openai_content(f"{remote}", inline_remote_images=True)

    assert content == [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/webp;base64,{IMAGE_B64}"},
        }
    ]
    assert [url for url, _ in fetches] == [remote.url]


def test_to_openai_content_reports_failed_remote_fetch(monkeypatch):
    def failing_get(url, **kwargs):
        raise requests.exceptions.ConnectionError("network down")

    monkeypatch.setattr(requests, "get", failing_get)
    remote = MLLMImage(url="https://example.com/cat.png")

    with pytest.raises(ValueError, match="Failed to fetch remote image"):
        to_openai_content(f"{remote}", inline_remote_images=True)


def test_fetch_remote_image_uses_media_timeouts(fetches):
    settings = get_settings()
    remote = MLLMImage(url="https://example.com/cat.png")

    data, mime_type = fetch_remote_image(remote)

    assert (data, mime_type) == (IMAGE_BYTES, "image/webp")
    assert fetches[0][1]["timeout"] == (
        settings.MEDIA_IMAGE_CONNECT_TIMEOUT_SECONDS,
        settings.MEDIA_IMAGE_READ_TIMEOUT_SECONDS,
    )


def test_fetch_remote_image_falls_back_to_jpeg_without_content_type(
    monkeypatch,
):
    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResponse({}))
    remote = MLLMImage(url="https://example.com/image")

    assert fetch_remote_image(remote) == (IMAGE_BYTES, "image/jpeg")


def test_decode_image_returns_raw_bytes():
    image = MLLMImage(dataBase64=IMAGE_B64, mimeType="image/png")

    assert decode_image(image) == IMAGE_BYTES


@pytest.mark.parametrize("error_cls", [ValueError, DeepEvalError])
def test_decode_image_raises_given_error_on_invalid_base64(error_cls):
    image = MLLMImage(dataBase64="not base64!!", mimeType="image/png")

    with pytest.raises(error_cls, match="Invalid base64 data"):
        decode_image(image, error_cls=error_cls)

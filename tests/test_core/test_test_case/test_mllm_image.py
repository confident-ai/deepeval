import pytest

from deepeval.test_case import MLLMImage


def test_rejects_remote_url_marked_as_local():
    with pytest.raises(ValueError):
        MLLMImage(
            url="https://example.com/image.jpg",
            local=True,
        )


def test_rejects_local_path_marked_as_remote(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake image")

    with pytest.raises(ValueError):
        MLLMImage(
            url=str(image_path),
            local=False,
        )


def test_accepts_remote_url_marked_as_remote():
    image = MLLMImage(
        url="https://example.com/image.jpg",
        local=False,
    )

    assert image.local is False

def test_auto_detects_remote_url():
    image = MLLMImage(
        url="https://example.com/image.jpg",
    )

    assert image.local is False
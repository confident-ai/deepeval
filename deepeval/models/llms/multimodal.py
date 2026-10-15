import base64
from typing import Dict, List, Optional, Tuple, Type, Union

from deepeval.config.settings import get_settings
from deepeval.test_case import MLLMImage
from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array


def multimodal_parts(prompt: str) -> Optional[List[Union[str, MLLMImage]]]:
    if not check_if_multimodal(prompt):
        return None
    return convert_to_multi_modal_array(input=prompt)


def to_openai_content(
    prompt: str,
    *,
    text_only_as_string: bool = False,
    inline_remote_images: bool = False,
) -> Union[str, List[Dict]]:
    parts = multimodal_parts(prompt)
    if parts is None:
        if text_only_as_string:
            return prompt
        return [{"type": "text", "text": prompt}]

    content = []
    for element in parts:
        if isinstance(element, str):
            content.append({"type": "text", "text": element})
        elif isinstance(element, MLLMImage):
            url = _openai_image_url(element, inline_remote_images)
            content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def _openai_image_url(image: MLLMImage, inline_remote: bool) -> str:
    if image.url and not image.local:
        if not inline_remote:
            return image.url
        try:
            data, mime_type = fetch_remote_image(image)
        except Exception as e:
            raise ValueError(f"Failed to fetch remote image {image.url}: {e}")
        b64_data = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{b64_data}"

    image.ensure_images_loaded()
    return f"data:{image.mimeType};base64,{image.dataBase64}"


def fetch_remote_image(image: MLLMImage) -> Tuple[bytes, str]:
    """Download a remote image, returning its bytes and mime type."""
    import requests

    settings = get_settings()
    response = requests.get(
        image.url,
        timeout=(
            settings.MEDIA_IMAGE_CONNECT_TIMEOUT_SECONDS,
            settings.MEDIA_IMAGE_READ_TIMEOUT_SECONDS,
        ),
    )
    response.raise_for_status()
    mime_type = response.headers.get(
        "content-type", image.mimeType or "image/jpeg"
    )
    return response.content, mime_type


def decode_image(
    image: MLLMImage, error_cls: Type[Exception] = ValueError
) -> bytes:
    """A local or base64 image's raw bytes, raising ``error_cls`` if invalid."""
    image.ensure_images_loaded()
    try:
        return base64.b64decode(image.dataBase64)
    except Exception:
        raise error_cls(f"Invalid base64 data in MLLMImage: {image._id}")

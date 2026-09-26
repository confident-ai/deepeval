"""
Security regression tests for LangChain multimodal content handling.

Traced LangChain/LangGraph message content is untrusted (it can be influenced by
user input, tool output, or retrieved documents). When a multimodal content block
references a media "URL", DeepEval must only turn ``http(s)://`` URLs and ``data:``
URIs into images. A local filesystem path or ``file://`` URI must NOT be turned
into an MLLMImage here, because MLLMImage would read that file off the eval host
(arbitrary local file read / exfiltration).
"""

import os
import base64
import tempfile

import pytest

pytest.importorskip("langchain_core")

from deepeval.integrations.langchain.utils import (  # noqa: E402
    _langchain_content_block_to_str,
)
from deepeval.test_case.llm_test_case import (  # noqa: E402
    _MLLM_IMAGE_REGISTRY,
)


@pytest.fixture
def local_file():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "secret.txt")
    with open(p, "w") as f:
        f.write("SECRET-DATA")
    return p


def _is_image_placeholder(s: str) -> bool:
    return s.startswith("[DEEPEVAL:")


class TestLangChainMediaUrlSafety:
    def test_local_path_in_image_block_is_not_read(self, local_file):
        before = len(_MLLM_IMAGE_REGISTRY)
        result = _langchain_content_block_to_str(
            {"type": "image_url", "image_url": {"url": local_file}}
        )
        # No image object created (so nothing read) ...
        assert len(_MLLM_IMAGE_REGISTRY) == before
        # ... and the result is not a DeepEval image placeholder.
        assert not _is_image_placeholder(result)

    def test_file_uri_in_image_block_is_not_read(self, local_file):
        before = len(_MLLM_IMAGE_REGISTRY)
        result = _langchain_content_block_to_str(
            {"type": "image_url", "image_url": {"url": f"file://{local_file}"}}
        )
        assert len(_MLLM_IMAGE_REGISTRY) == before
        assert not _is_image_placeholder(result)

    def test_https_url_is_still_an_image(self):
        result = _langchain_content_block_to_str(
            {"type": "image_url", "image_url": {"url": "https://ex.com/a.jpg"}}
        )
        assert _is_image_placeholder(result)

    def test_data_uri_is_still_an_image(self):
        data_uri = "data:image/png;base64," + base64.b64encode(b"PNG").decode()
        result = _langchain_content_block_to_str(
            {"type": "image_url", "image_url": {"url": data_uri}}
        )
        assert _is_image_placeholder(result)

"""
Security regression tests for MLLMImage local-file handling.

MLLMImage reads local files at construction time when its ``url`` points at an
existing path. This is a legitimate feature for developer-supplied local images,
but it must not be reachable via a ``file://`` URI (a local-file-inclusion vector
when the URL comes from untrusted content). These tests pin down that:

  * ``file://`` URIs are NOT treated as local paths and are rejected;
  * plain local filesystem paths still work (no behavior change for the
    documented developer use case).
"""

import os
import base64
import tempfile

import pytest

from deepeval.test_case import MLLMImage


@pytest.fixture
def local_file():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "image_or_secret.txt")
    with open(p, "w") as f:
        f.write("FILE-CONTENTS")
    return p


class TestMLLMImageFileScheme:
    def test_file_uri_is_not_local(self, local_file):
        assert MLLMImage.is_local_path(f"file://{local_file}") is False

    def test_file_uri_is_rejected(self, local_file):
        # file:// is neither a valid remote URL nor treated as local, so it is
        # rejected instead of silently reading the file off disk.
        with pytest.raises(ValueError):
            MLLMImage(url=f"file://{local_file}")

    def test_file_uri_does_not_read_the_file(self, local_file):
        try:
            img = MLLMImage(url=f"file://{local_file}")
        except ValueError:
            return  # rejected, nothing read
        assert img.dataBase64 is None


class TestMLLMImageLocalPathStillWorks:
    """The documented developer feature (passing a local image path) must not
    regress."""

    def test_plain_local_path_is_local(self, local_file):
        assert MLLMImage.is_local_path(local_file) is True

    def test_plain_local_path_is_read(self, local_file):
        img = MLLMImage(url=local_file)
        assert img.local is True
        assert base64.b64decode(img.dataBase64).decode() == "FILE-CONTENTS"

    def test_remote_https_url_is_not_read(self):
        img = MLLMImage(url="https://example.com/cat.jpg")
        assert img.local is False
        assert img.dataBase64 is None

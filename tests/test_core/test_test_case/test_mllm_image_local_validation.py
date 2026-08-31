"""Tests for MLLMImage `local`-flag validation.

``MLLMImage.__post_init__`` used a bare ``assert self.local == is_local`` to
validate a caller-provided ``local`` flag against the URL. ``assert`` is an
anti-pattern here: it is stripped when the interpreter runs with ``-O``/``-OO``,
raises ``AssertionError`` instead of ``ValueError``, and its message does not
explain which value contradicted the URL.

These tests verify that:
  * a ``local`` flag that contradicts the URL raises a clear ``ValueError``;
  * matching explicit values and ``local=None`` (auto-detect) still construct
    fine (no regression);
  * the mismatch is still rejected under ``python -O`` where bare asserts
    would have been stripped.

The tests are fully offline and need no network or model access.
"""

import os
import subprocess
import sys

import pytest

from deepeval.test_case import MLLMImage

REMOTE_URL = "https://example.com/image.png"


def test_remote_url_with_matching_local_false_ok():
    img = MLLMImage(url=REMOTE_URL, local=False)
    assert img.local is False


def test_remote_url_with_none_local_auto_detects_remote():
    img = MLLMImage(url=REMOTE_URL)
    assert img.local is False


def test_remote_url_with_contradictory_local_true_rejected():
    with pytest.raises(ValueError, match="Local path mismatch.*remote"):
        MLLMImage(url=REMOTE_URL, local=True)


def test_local_file_with_matching_local_true_ok(tmp_path):
    path = tmp_path / "img.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    img = MLLMImage(url=str(path), local=True)
    assert img.local is True
    assert img.dataBase64 is not None


def test_local_file_with_none_local_auto_detects_local(tmp_path):
    path = tmp_path / "img.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    img = MLLMImage(url=str(path))
    assert img.local is True


def test_local_file_with_contradictory_local_false_rejected(tmp_path):
    path = tmp_path / "img.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError, match="Local path mismatch.*local"):
        MLLMImage(url=str(path), local=False)


def test_base64_path_is_unaffected():
    # Base64-backed images skip the URL/local logic entirely.
    img = MLLMImage(dataBase64="aGVsbG8=", mimeType="image/png", local=True)
    assert img.local is True


def test_mismatch_still_rejected_under_python_optimize():
    """The check must survive `python -O` (bare asserts would be stripped)."""
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    code = (
        "from deepeval.test_case import MLLMImage\n"
        "try:\n"
        "    MLLMImage(url='https://example.com/image.png', local=True)\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('contradictory local flag accepted under -O')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": repo_root},
    )
    assert result.returncode == 0, result.stderr

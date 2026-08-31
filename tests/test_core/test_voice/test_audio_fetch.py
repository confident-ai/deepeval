import base64

import pytest
import requests

from deepeval.test_case import Audio


class _Response:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        pass


def test_get_bytes_downloads_and_caches_remote_audio(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout))
        return _Response(b"RIFFfake-wav-bytes")

    monkeypatch.setattr(requests, "get", fake_get)
    audio = Audio(url="https://bucket.example.com/turn.wav?sig=abc")

    assert audio.get_bytes() == b"RIFFfake-wav-bytes"
    assert audio.dataBase64 == base64.b64encode(b"RIFFfake-wav-bytes").decode(
        "ascii"
    )
    assert audio.get_bytes() == b"RIFFfake-wav-bytes"
    assert len(calls) == 1


def test_get_bytes_still_requires_a_source(monkeypatch):
    audio = Audio(
        dataBase64=base64.b64encode(b"x").decode("ascii"),
        mimeType="audio/wav",
    )
    audio.dataBase64 = None

    with pytest.raises(ValueError, match="No audio bytes available"):
        audio.get_bytes()

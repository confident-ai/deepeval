"""LiveKitConnector's error when the LiveKit SDKs are not installed."""

import sys

import pytest

from deepeval.errors import DeepEvalError
from deepeval.voice import LiveKitConnector


def test_missing_livekit_sdk_points_at_installable_packages(monkeypatch):
    # Simulate the LiveKit server SDK not being installed.
    monkeypatch.setitem(sys.modules, "livekit", None)
    monkeypatch.setitem(sys.modules, "livekit.api", None)

    connector = LiveKitConnector(
        url="wss://example.livekit.cloud",
        api_key="key-1",
        api_secret="secret-1",
    )

    with pytest.raises(DeepEvalError) as exc_info:
        connector._build_token()

    message = str(exc_info.value)
    assert "pip install livekit livekit-api" in message
    # deepeval does not ship a `voice` extra, so the hint must not suggest it.
    assert "deepeval[voice]" not in message

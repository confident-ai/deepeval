"""The client tool result fallback must cover every value json refuses to encode."""

import asyncio
import json

import pytest

from deepeval.voice.connectors.providers.elevenlabs.connector import ElevenLabsConnector


class _Recording(ElevenLabsConnector):
    """Connector whose only behaviour is to capture what would be sent."""

    def __init__(self) -> None:  # noqa: D107 - test double
        self.sent: list[str] = []

    async def _send(self, data: str) -> None:  # type: ignore[override]
        self.sent.append(data)


def _send(result) -> str:
    connector = _Recording()
    asyncio.run(
        connector._send_tool_result(
            tool_name="tool", tool_call_id="call", result=result, is_error=False
        )
    )
    assert len(connector.sent) == 1
    return connector.sent[0]


def test_serializable_result_is_sent_as_is():
    payload = json.loads(_send({"value": 1}))

    assert payload["result"] == {"value": 1}


def test_unserializable_object_falls_back_to_its_repr():
    payload = json.loads(_send(object()))

    assert isinstance(payload["result"], str)


def test_circular_result_falls_back_instead_of_raising():
    """json.dumps raises ValueError (not TypeError) for a circular value.

    The fallback exists so the agent is not left waiting for a message that can
    never be encoded, so a circular result has to go through it as well.
    """
    circular: dict = {}
    circular["self"] = circular

    payload = json.loads(_send(circular))

    assert isinstance(payload["result"], str)

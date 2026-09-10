from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union

from deepeval.errors import DeepEvalError
from deepeval.voice.connectors.transports.base import BaseVoiceConnector


class ElevenLabsIdentity(Enum):
    ASYNC_CONVERSATION = "AsyncConversation"
    CONVERSATION = "Conversation"


class LiveKitIdentity(Enum):
    ROOM = "Room"


class VapiIdentity(Enum):
    CLIENT = "Vapi"
    ASYNC_CLIENT = "AsyncVapi"


class PipecatIdentity(Enum):
    PIPELINE = "Pipeline"
    TASK = "PipelineTask"
    WORKER = "PipelineWorker"
    SERVER_TRANSPORT = "WebsocketServerTransport"
    SINGLE_CLIENT_TRANSPORT = "SingleClientWebsocketServerTransport"
    FASTAPI_TRANSPORT = "FastAPIWebsocketTransport"


# What to tell someone who passed the vendor object instead of a connector.
# Every voice session runs through a connector, so these are redirections
# rather than alternative entry points.
_REDIRECTS = {
    ElevenLabsIdentity.ASYNC_CONVERSATION: (
        "VoiceConfig received an ElevenLabs `AsyncConversation`. Pass "
        "`ElevenLabsConnector(agent_id=...)` instead — it reaches the same "
        "agent, and takes `client_tools`, `dynamic_variables`, and "
        "`conversation_config_override` for anything you configured on the "
        "conversation."
    ),
    ElevenLabsIdentity.CONVERSATION: (
        "VoiceConfig received an ElevenLabs `Conversation`. Pass "
        "`ElevenLabsConnector(agent_id=...)` instead."
    ),
    LiveKitIdentity.ROOM: (
        "VoiceConfig received a LiveKit `rtc.Room`. Pass "
        "`LiveKitConnector(room=your_room)` instead, which joins the room you "
        "already have."
    ),
    VapiIdentity.CLIENT: (
        "VoiceConfig received a Vapi client. Pass "
        "`VapiConnector(assistant_id=...)` instead — it creates the call "
        "itself, and takes `assistant_overrides` for anything you would "
        "configure per call."
    ),
    VapiIdentity.ASYNC_CLIENT: (
        "VoiceConfig received a Vapi client. Pass "
        "`VapiConnector(assistant_id=...)` instead."
    ),
}

# Pipecat's pipeline objects and its WebSocket transports all mean the same
# thing here: the agent is reached over the socket its server is listening on,
# never in-process.
_REDIRECTS.update(
    {
        identity: (
            f"VoiceConfig received a Pipecat `{identity.value}`. Run the "
            "pipeline behind its WebSocket transport and pass "
            "`PipecatConnector(url=...)` instead, which connects to it the "
            "way any other client would."
        )
        for identity in PipecatIdentity
    }
)


def _root_module(cls: type) -> str:
    return (getattr(cls, "__module__", "") or "").split(".")[0]


def _identify(
    obj: Any,
) -> Optional[
    Union[ElevenLabsIdentity, LiveKitIdentity, PipecatIdentity, VapiIdentity]
]:
    """Name the vendor class `obj` is, without importing the vendor package.

    Matching walks the MRO by `(root module, class name)` rather than using
    `isinstance`, because importing `elevenlabs` or `livekit` to build the
    class objects to compare against would make an optional dependency
    mandatory for everyone who ever constructs a `VoiceConfig`.
    """
    for cls in type(obj).__mro__:
        root, name = _root_module(cls), cls.__name__
        if root == "elevenlabs":
            for identity in ElevenLabsIdentity:
                if name == identity.value:
                    return identity
        if root == "livekit":
            for identity in LiveKitIdentity:
                if name == identity.value:
                    return identity
        if root in ("vapi", "vapi_server_sdk"):
            for identity in VapiIdentity:
                if name == identity.value:
                    return identity
        if root == "pipecat":
            for identity in PipecatIdentity:
                if name == identity.value:
                    return identity
    return None


def validate_connector(connector: Any) -> BaseVoiceConnector:
    """Return `connector`, or explain what to pass instead.

    Vendor objects are recognised only to redirect: naming what was passed and
    the connector that takes it is a great deal more useful than a type error
    on an attribute somewhere inside the simulation.
    """
    if isinstance(connector, BaseVoiceConnector):
        return connector

    redirect = _REDIRECTS.get(_identify(connector))
    if redirect is not None:
        raise DeepEvalError(redirect)

    raise DeepEvalError(
        f"VoiceConfig cannot use {type(connector).__name__!r} as a connector. "
        "Pass an `ElevenLabsConnector`, `LiveKitConnector`, `VapiConnector`, "
        "`PipecatConnector`, `WebSocketConnector`, `CallbackVoiceConnector`, "
        "or your own `BaseVoiceConnector` subclass."
    )

from deepeval.voice.protocol import VoiceProtocol
from deepeval.voice.config import VoiceConfig
from deepeval.voice.interruption import interruption_policy
from deepeval.voice.floor_control import FloorController, FloorState
from deepeval.voice.turn_detection import TurnDetection
from deepeval.voice.connectors import (
    BaseVoiceConnector,
    ConnectorTurn,
    AgentEvent,
    CallbackVoiceConnector,
    LiveKitConnector,
    BaseWebSocketConnector,
    ElevenLabsConnector,
    WebSocketConnector,
)

__all__ = [
    "VoiceProtocol",
    "VoiceConfig",
    "InterruptionSettings",
    "InterruptionLevel",
    "interruption_policy",
    "FloorController",
    "FloorState",
    "TurnDetection",
    "BaseVoiceConnector",
    "ConnectorTurn",
    "AgentEvent",
    "CallbackVoiceConnector",
    "LiveKitConnector",
    "BaseWebSocketConnector",
    "ElevenLabsConnector",
    "WebSocketConnector",
]

# Resolved lazily: these live with `Persona` in `deepeval.dataset`, which
# imports `deepeval.models`, which imports this package.
_PERSONA_TYPES = {
    "InterruptionBehavior": "InterruptionBehavior",
    "InterruptionSettings": "InterruptionBehavior",
    "InterruptionLevel": "InterruptionLevel",
    "OverlapBehavior": "OverlapBehavior",
}


def __getattr__(name: str):
    if name in _PERSONA_TYPES:
        from deepeval.dataset import golden

        return getattr(golden, _PERSONA_TYPES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

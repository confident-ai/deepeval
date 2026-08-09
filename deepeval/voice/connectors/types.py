from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Union

from deepeval.test_case import Audio


@dataclass
class ConnectorTurn:
    audio: Audio
    transcript: Optional[str] = None
    latency_ms: Optional[float] = None  # user-audio-sent -> first agent audio
    interrupted: bool = False  # agent barge-in detected (duplex; future)


AgentCallback = Callable[
    [Audio],
    Union[Audio, ConnectorTurn, Awaitable[Union[Audio, ConnectorTurn]]],
]

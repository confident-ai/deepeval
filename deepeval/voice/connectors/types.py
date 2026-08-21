from dataclasses import dataclass
from typing import Awaitable, AsyncIterator, Callable, Optional, Union

from deepeval.test_case import Audio


@dataclass
class ConnectorTurn:
    audio: Audio
    transcript: Optional[str] = None
    latency_ms: Optional[float] = None  # user-audio-sent -> first agent audio
    interrupted: bool = False  # True when we successfully barged in (duplex)
    # Process-local monotonic capture times. The simulator converts these to
    # call-relative Audio.start_time values before the test case is serialized.
    input_audio_started_at: Optional[float] = None
    input_audio_ended_at: Optional[float] = None
    audio_started_at: Optional[float] = None


@dataclass
class AgentEvent:
    """One duplex downlink event from a voice agent.

    Connectors may emit audio frames, transcript updates, and turn-complete
    signals on the same stream. `transcript` is the latest partial/full text
    known so far (not necessarily a small delta).
    """

    audio: Optional[bytes] = None  # PCM16 mono at the connector recv rate
    transcript: Optional[str] = None
    turn_complete: bool = False
    # Process-local monotonic time this event arrived from the agent. The
    # consumer can be busy (synthesizing a barge, for instance) long after a
    # frame lands, so reading the clock at consumption time would credit the
    # agent with starting to speak later than it did.
    received_at: Optional[float] = None


AgentCallback = Callable[
    [Audio],
    Union[Audio, ConnectorTurn, Awaitable[Union[Audio, ConnectorTurn]]],
]

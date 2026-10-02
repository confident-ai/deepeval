from dataclasses import dataclass
from typing import Awaitable, AsyncIterator, Callable, Optional, Union

from deepeval.test_case import Audio


@dataclass
class ConnectorTurn:
    audio: Audio
    transcript: Optional[str] = None  # what the agent said
    provider_transcription: Optional[str] = None  # what the agent heard us say
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

    `provider_transcription` runs the other way: it is the agent's own STT of the
    caller's audio — what it heard rather than what it said — and arrives as
    one fragment per finalized segment, for the consumer to join. The two are
    kept apart because comparing them is the whole point of asking.
    """

    audio: Optional[bytes] = None  # PCM16 mono at the connector recv rate
    transcript: Optional[str] = None
    provider_transcription: Optional[str] = None
    turn_complete: bool = False
    # Process-local monotonic time this event arrived from the agent. The
    # consumer can be busy (synthesizing a barge, for instance) long after a
    # frame lands, so reading the clock at consumption time would credit the
    # agent with starting to speak later than it did.
    received_at: Optional[float] = None
    call_ended: bool = False


AgentCallback = Callable[
    [Audio],
    Union[Audio, ConnectorTurn, Awaitable[Union[Audio, ConnectorTurn]]],
]

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    ClassVar,
    Optional,
    Tuple,
)

from deepeval.test_case import Audio, AudioChunk
from deepeval.voice.protocol import VoiceProtocol
from deepeval.voice.connectors.types import AgentEvent, ConnectorTurn
from deepeval.voice.streaming import (
    UplinkResult,
    collect_pcm_chunks,
    pcm_to_audio,
)


@dataclass
class UplinkStream:
    """The send side of a live call: one caller utterance in flight at a time.

    The flag and the task are a pair and belong together. Floor control stops a
    barge by raising the flag rather than by killing the send, because an
    utterance that was half spoken still has to be recorded as what the caller
    said; the task handle exists only so the send can be reaped afterwards.

    Created when the transport connects, so a `None` here means the call has
    not been opened yet.
    """

    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()

    def begin(self) -> None:
        """Arm a fresh utterance, clearing any earlier cancellation."""
        self.cancel.clear()

    async def stop(self) -> None:
        """Stop sending, and wait for the send to notice."""
        self.cancel.set()
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass


async def iter_downlink(queue: asyncio.Queue) -> AsyncIterator[AgentEvent]:
    """Yield downlink events from a transport's receive queue, indefinitely.

    `turn_complete` is signaled on individual events; the iterator itself never
    stops, so a duplex loop can keep listening after a barge-in. Queue items may
    be `AgentEvent`s, raw PCM, or `None` as an end-of-turn sentinel.
    """
    while True:
        item = await queue.get()
        if item is None:
            yield AgentEvent(turn_complete=True)
        elif isinstance(item, AgentEvent):
            yield item
        elif isinstance(item, (bytes, bytearray)):
            yield AgentEvent(audio=bytes(item))


class BaseVoiceConnector(ABC):
    """Base class for voice agent connectors.

    Every concrete connector must declare the transport protocol it speaks
    via the `protocol` class variable (a `VoiceProtocol` member).

    Half-duplex simulations use `exchange_turn`. Duplex / barge-in simulations
    use `stream_uplink`, `iter_agent_events`, and `stop_uplink`.
    """

    protocol: ClassVar[VoiceProtocol]

    # End-of-turn timing. Duplex exchanges need both to decide when the agent
    # has stopped talking, so they are part of the connector contract with
    # defaults rather than something each transport may or may not expose.
    # Transports that select a `turn_detection` preset overwrite them.
    end_of_turn_silence_ms: int = 800
    max_turn_timeout_s: float = 30.0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # `__abstractmethods__` is not yet computed by ABCMeta at this point,
        # so determine abstractness manually: a class is still abstract if any
        # attribute it resolves to is marked as an abstract method.
        is_abstract = any(
            getattr(getattr(cls, name, None), "__isabstractmethod__", False)
            for base in cls.__mro__
            for name, value in vars(base).items()
            if getattr(value, "__isabstractmethod__", False)
        )
        if is_abstract:
            return
        protocol = getattr(cls, "protocol", None)
        if not isinstance(protocol, VoiceProtocol):
            raise TypeError(
                f"{cls.__name__} must declare a `protocol` class variable "
                "set to a VoiceProtocol member (e.g. "
                "`protocol = VoiceProtocol.WEBRTC`)."
            )

    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def exchange_turn(self, audio: Audio) -> ConnectorTurn:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    async def stream_uplink(
        self, audio: Audio, *, trailing_silence: bool = True
    ) -> None:
        """Stream user audio uplink without waiting for the agent reply.

        Used by duplex barge-in. Default raises; duplex-capable connectors
        override this. `trailing_silence` pads the uplink for agent VAD on
        full turns; barge attempts typically pass False.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support duplex stream_uplink(); "
            "use exchange_turn() or a duplex-capable connector."
        )

    async def stream_uplink_chunks(
        self,
        chunks: AsyncIterable[AudioChunk],
        *,
        trailing_silence: bool = True,
        on_first_frame: Optional[Callable[[float], None]] = None,
    ) -> UplinkResult:
        """Stream user audio uplink as it is produced, rather than once complete.

        Synthesizing a whole utterance before sending any of it puts the entire
        synthesis time in front of the first word. An agent that processes audio
        as it arrives can start on the opening words while the rest is still
        being made, which is what happens on a real call.

        Returns the complete utterance along with when it began going out, since
        the caller has to record both what was said and where it belongs on the
        call. `on_first_frame` reports the same moment as it happens, for callers
        that have to act on the agent being able to hear the speech — a barge
        takes the floor then, not when it finishes. The default buffers the
        stream and hands it to `stream_uplink`, which is all a transport whose
        agent needs the whole utterance up front can do with it; transports that
        can forward frames override this.
        """
        pcm, sample_rate = await collect_pcm_chunks(chunks)
        audio = pcm_to_audio(pcm, sample_rate)
        sent_at = time.perf_counter()
        if on_first_frame is not None:
            on_first_frame(sent_at)
        await self.stream_uplink(audio, trailing_silence=trailing_silence)
        return UplinkResult(audio=audio, first_frame_at=sent_at)

    @property
    def signals_turn_complete(self) -> bool:
        """Whether the transport says outright when the agent's turn is over.

        When it does, quiet in the downlink is a pause and nothing more, so
        ending the turn on it discards the rest of the reply while keeping the
        transcript that describes the whole thing. When it does not — a raw
        audio track, say — silence is the only evidence there is.
        """
        return False

    async def stop_uplink(self) -> None:
        """Cancel in-flight user PCM from `stream_uplink` (floor-control yield)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support stop_uplink()."
        )

    def iter_agent_events(self) -> AsyncIterator[AgentEvent]:
        """Yield downlink events until turn-complete or the iterator is closed.

        Used by duplex barge-in. Default raises; duplex-capable connectors
        override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support iter_agent_events()."
        )
        # pragma: no cover — make this an async generator for type checkers
        if False:  # noqa: SIM223
            yield AgentEvent()

    @property
    def audio_format(self) -> Tuple[int, str]:
        return (24000, "wav")

    @property
    def recv_sample_rate(self) -> int:
        """Sample rate of the downlink PCM in `iter_agent_events`.

        Defaults to the uplink rate, which is all a transport that does not
        transcode can report.
        """
        return self.audio_format[0]

    def drain_downlink(self) -> None:
        """Drop agent audio left over from a previous turn.

        A reply is queued as it arrives and paced in real time, so frames of
        the last one are usually still waiting when the next user utterance
        goes out. Read as part of the new turn they timestamp the assistant's
        audio at the moment the user started speaking and splice the tail of
        the wrong reply onto it. Transports that buffer a downlink override
        this; the default has nothing to drop.
        """
        return None

    async def __aenter__(self) -> "BaseVoiceConnector":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.disconnect()

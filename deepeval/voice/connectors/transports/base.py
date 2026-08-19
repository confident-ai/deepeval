import time
from abc import ABC, abstractmethod
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


class BaseVoiceConnector(ABC):
    """Base class for voice agent connectors.

    Every concrete connector must declare the transport protocol it speaks
    via the `protocol` class variable (a `VoiceProtocol` member).

    Half-duplex simulations use `exchange_turn`. Duplex / barge-in simulations
    use `stream_uplink`, `iter_agent_events`, and `stop_uplink`.
    """

    protocol: ClassVar[VoiceProtocol]

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

    async def __aenter__(self) -> "BaseVoiceConnector":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.disconnect()

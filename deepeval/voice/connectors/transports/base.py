from abc import ABC, abstractmethod
from typing import AsyncIterator, ClassVar, Tuple

from deepeval.test_case import Audio
from deepeval.voice.protocol import VoiceProtocol
from deepeval.voice.connectors.types import AgentEvent, ConnectorTurn


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

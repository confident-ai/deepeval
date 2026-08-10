from abc import ABC, abstractmethod
from typing import ClassVar, Tuple

from deepeval.test_case import Audio
from deepeval.voice.protocol import VoiceProtocol
from deepeval.voice.connectors.types import ConnectorTurn


class BaseVoiceConnector(ABC):
    """Base class for voice agent connectors.

    Every concrete connector must declare the transport protocol it speaks
    via the `protocol` class variable (a `VoiceProtocol` member).
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
    async def send_turn(self, audio: Audio) -> ConnectorTurn:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @property
    def audio_format(self) -> Tuple[int, str]:
        return (24000, "wav")

    async def __aenter__(self) -> "BaseVoiceConnector":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.disconnect()

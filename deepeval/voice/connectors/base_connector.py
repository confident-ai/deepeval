from abc import ABC, abstractmethod
from typing import Tuple
from deepeval.test_case import Audio
from deepeval.voice.connectors.types import ConnectorTurn


class BaseVoiceConnector(ABC):

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

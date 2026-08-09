import time
import inspect
from typing import Callable, Optional, Tuple, Union

from deepeval.test_case import Audio
from deepeval.models import DeepEvalBaseTTS, DeepEvalBaseSTT
from deepeval.voice.connectors.base_connector import BaseVoiceConnector
from deepeval.voice.connectors.types import ConnectorTurn, AgentCallback


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class CallbackVoiceConnector(BaseVoiceConnector):
    def __init__(
        self,
        agent: AgentCallback,
        *,
        sample_rate: int = 24000,
        encoding: str = "wav",
    ):
        self.agent = agent
        self._is_async = inspect.iscoroutinefunction(agent)
        self._format = (sample_rate, encoding)

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    @property
    def audio_format(self) -> Tuple[int, str]:
        return self._format

    async def send_turn(self, audio: Audio) -> ConnectorTurn:
        start = time.perf_counter()
        result = await _maybe_await(self.agent(audio))
        latency_ms = (time.perf_counter() - start) * 1000.0

        if isinstance(result, ConnectorTurn):
            if result.latency_ms is None:
                result.latency_ms = latency_ms
            return result
        return ConnectorTurn(audio=result, latency_ms=latency_ms)

    @classmethod
    def from_text_agent(
        cls,
        text_agent: Callable[[str], Union[str, "object"]],
        *,
        tts: DeepEvalBaseTTS,
        stt: DeepEvalBaseSTT,
        voice: Optional[str] = None,
    ) -> "CallbackVoiceConnector":

        async def agent(user_audio: Audio) -> ConnectorTurn:
            user_text, _ = await stt.a_transcribe(user_audio)
            reply = await _maybe_await(text_agent(user_text))
            agent_audio, _ = await tts.a_synthesize(reply, voice=voice)
            return ConnectorTurn(audio=agent_audio, transcript=reply)

        sample_rate = getattr(tts, "sample_rate", 24000)
        return cls(agent, sample_rate=sample_rate)

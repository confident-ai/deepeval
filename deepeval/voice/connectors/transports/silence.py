import asyncio
import time
from typing import Any, AsyncIterator, Optional

from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.types import AgentEvent

SILENCE_CHECK_INTERVAL_S = 1.0


class SilenceTimeoutMixin:

    def __init__(
        self,
        *args: Any,
        silence_timeout_s: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.silence_timeout_s = silence_timeout_s
        self._last_activity_at = time.monotonic()
        self._silence_task: Optional[asyncio.Task] = None
        self._hung_up_for_silence = False

    def _mark_activity(self) -> None:
        self._last_activity_at = time.monotonic()

    async def connect(self) -> None:
        await super().connect()
        self._hung_up_for_silence = False
        self._mark_activity()
        if self.silence_timeout_s:
            self._silence_task = asyncio.create_task(self._watch_silence())

    async def disconnect(self) -> None:
        task, self._silence_task = self._silence_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        await super().disconnect()

    async def _watch_silence(self) -> None:
        while True:
            await asyncio.sleep(SILENCE_CHECK_INTERVAL_S)
            idle_for = time.monotonic() - self._last_activity_at
            if idle_for >= self.silence_timeout_s:
                self._hung_up_for_silence = True
                await self.disconnect()
                return

    @property
    def call_ended(self) -> bool:
        return self._hung_up_for_silence or super().call_ended

    async def exchange_turn(self, audio: Any) -> Any:
        self._mark_activity()
        turn = await super().exchange_turn(audio)
        self._mark_activity()
        return turn

    async def stream_uplink(self, audio: Any, **kwargs: Any) -> None:
        self._mark_activity()
        await super().stream_uplink(audio, **kwargs)
        self._mark_activity()

    async def stream_uplink_chunks(self, chunks: Any, **kwargs: Any) -> Any:
        self._mark_activity()
        result = await super().stream_uplink_chunks(chunks, **kwargs)
        self._mark_activity()
        return result

    async def iter_agent_events(self) -> AsyncIterator[AgentEvent]:
        async for event in super().iter_agent_events():
            if event.audio and not audio_utils.is_silent(
                event.audio, self.silence_threshold_rms
            ):
                self._mark_activity()
            yield event

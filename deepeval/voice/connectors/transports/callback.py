import time
import asyncio
import inspect
import logging
from typing import AsyncIterator, Callable, ClassVar, Optional, Tuple, Union

from deepeval.test_case import Audio
from deepeval.models.base_model import DeepEvalBaseTTS, DeepEvalBaseSTT
from deepeval.voice.protocol import VoiceProtocol
from deepeval.voice.turn_detection import TurnDetection, turn_detection_timing
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.transports.base import (
    BaseVoiceConnector,
    UplinkStream,
    iter_downlink,
)
from deepeval.voice.connectors.types import (
    AgentEvent,
    AgentCallback,
    ConnectorTurn,
)

logger = logging.getLogger(__name__)


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class CallbackVoiceConnector(BaseVoiceConnector):

    protocol: ClassVar[VoiceProtocol] = VoiceProtocol.CALLBACK

    def __init__(
        self,
        agent: AgentCallback,
        *,
        sample_rate: int = 24000,
        encoding: str = "wav",
        turn_detection: TurnDetection = "balanced",
    ):
        self.agent = agent
        self._is_async = inspect.iscoroutinefunction(agent)
        self._format = (sample_rate, encoding)
        # Duplex only: `exchange_turn` gets the reply whole, so this bounds the
        # barge-in loop. An agent that pauses mid-reply wants "patient", or its
        # turn ends at the pause and the rest of the reply is never heard.
        self.turn_detection = turn_detection
        timing = turn_detection_timing(turn_detection)
        self.end_of_turn_silence_ms = timing.end_of_turn_silence_ms
        self.max_turn_timeout_s = timing.max_turn_timeout_s
        self._events: Optional[asyncio.Queue] = None
        self._uplink: Optional[UplinkStream] = None
        self._reply_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        logger.debug(
            "Callback connector connecting: sample_rate=%d encoding=%s",
            self._format[0],
            self._format[1],
        )
        self._events = asyncio.Queue()
        self._uplink = UplinkStream()
        logger.debug("Callback connector connected")

    async def disconnect(self) -> None:
        logger.debug("Callback connector disconnecting")
        await self.stop_uplink()
        if self._reply_task is not None and not self._reply_task.done():
            self._reply_task.cancel()
            try:
                await self._reply_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reply_task = None
        logger.debug("Callback connector disconnected")

    @property
    def audio_format(self) -> Tuple[int, str]:
        return self._format

    @property
    def recv_sample_rate(self) -> int:
        return self._format[0]

    @property
    def signals_turn_complete(self) -> bool:
        # Every reply is framed out and then closed explicitly.
        return True

    async def exchange_turn(self, audio: Audio) -> ConnectorTurn:
        start = time.perf_counter()
        input_ended_at = start + max(audio.duration or 0.0, 0.0)
        logger.debug(
            "Callback agent invocation started: input_bytes=%d",
            len(audio.get_bytes()),
        )
        result = await _maybe_await(self.agent(audio))
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.debug(
            "Callback agent invocation finished after %.2fms: result_type=%s",
            latency_ms,
            type(result).__name__,
        )

        if isinstance(result, ConnectorTurn):
            if result.latency_ms is None:
                result.latency_ms = latency_ms
            if result.input_audio_started_at is None:
                result.input_audio_started_at = start
            if result.input_audio_ended_at is None:
                result.input_audio_ended_at = input_ended_at
            if result.audio_started_at is None:
                result.audio_started_at = max(
                    time.perf_counter(), input_ended_at
                )
            logger.debug(
                "Callback connector turn: output_bytes=%d transcript=%s latency_ms=%.2f",
                len(result.audio.get_bytes()),
                bool(result.transcript),
                result.latency_ms,
            )
            return result
        logger.debug(
            "Callback connector audio: output_bytes=%d latency_ms=%.2f",
            len(result.get_bytes()),
            latency_ms,
        )
        return ConnectorTurn(
            audio=result,
            latency_ms=latency_ms,
            input_audio_started_at=start,
            input_audio_ended_at=input_ended_at,
            audio_started_at=max(time.perf_counter(), input_ended_at),
        )

    async def stream_uplink(
        self, audio: Audio, *, trailing_silence: bool = False
    ) -> None:
        """Invoke the in-process agent and enqueue chunked reply events.

        Duplex test double: uplink cancel is honored as a flag (no live PCM
        stream). Reply audio is framed onto `iter_agent_events`.
        """
        if self._events is None or self._uplink is None:
            raise RuntimeError(
                "CallbackVoiceConnector.stream_uplink() called before connect()."
            )
        await self.stop_uplink()
        abandoned_reply = (
            self._reply_task is not None and not self._reply_task.done()
        )
        if abandoned_reply:
            self._reply_task.cancel()
            try:
                await self._reply_task
            except (asyncio.CancelledError, Exception):
                pass
        # The cancelled task paces frames in real time, so it has usually
        # enqueued more of the previous reply than the last exchange consumed.
        self.drain_downlink()
        if abandoned_reply:
            # This uplink talked over a reply that was still playing, and each
            # uplink starts a fresh agent invocation. Close the abandoned
            # utterance the way a real agent goes quiet when interrupted, or
            # the next reply's frames land in the same buffer and the two are
            # recorded as one turn whose audio and transcript disagree.
            await self._events.put(AgentEvent(turn_complete=True))
        self._uplink.begin()

        async def _produce_reply() -> None:
            result = await _maybe_await(self.agent(audio))
            if isinstance(result, ConnectorTurn):
                reply_audio = result.audio
                transcript = result.transcript
            else:
                reply_audio = result
                transcript = None

            reply_pcm, reply_rate, reply_ch = audio_utils.wav_bytes_to_pcm16(
                reply_audio.get_bytes()
            )
            reply_pcm = audio_utils.downmix_to_mono(reply_pcm, reply_ch)
            if reply_rate != self._format[0]:
                reply_pcm = audio_utils.resample_pcm16(
                    reply_pcm, reply_rate, self._format[0]
                )
            if transcript:
                await self._events.put(
                    AgentEvent(
                        transcript=transcript,
                        received_at=time.perf_counter(),
                    )
                )
            # Paced out in full even while the uplink is cancelled: floor
            # control stops the *caller*, and the agent talking over a barge is
            # exactly what the duplex loop needs to see.
            for chunk in audio_utils.iter_pcm16_frames(
                reply_pcm, self._format[0]
            ):
                await self._events.put(
                    AgentEvent(audio=chunk, received_at=time.perf_counter())
                )
                await asyncio.sleep(audio_utils.DEFAULT_FRAME_MS / 1000.0)
            await self._events.put(AgentEvent(turn_complete=True))

        self._reply_task = asyncio.create_task(_produce_reply())

    async def stop_uplink(self) -> None:
        if self._uplink is not None:
            await self._uplink.stop()

    def drain_downlink(self) -> None:
        if self._events is None:
            return
        dropped = 0
        while True:
            try:
                self._events.get_nowait()
            except asyncio.QueueEmpty:
                break
            dropped += 1
        if dropped:
            logger.debug("Dropped %d stale downlink events", dropped)

    async def iter_agent_events(self) -> AsyncIterator[AgentEvent]:
        if self._events is None:
            raise RuntimeError(
                "CallbackVoiceConnector.iter_agent_events() called before "
                "connect()."
            )
        async for event in iter_downlink(self._events):
            yield event

    @classmethod
    def from_text_agent(
        cls,
        text_agent: Callable[[str], Union[str, "object"]],
        *,
        tts: DeepEvalBaseTTS,
        stt: DeepEvalBaseSTT,
        voice: Optional[str] = None,
        **kwargs,
    ) -> "CallbackVoiceConnector":

        async def agent(user_audio: Audio) -> ConnectorTurn:
            user_text, _ = await stt.a_transcribe(user_audio)
            reply = await _maybe_await(text_agent(user_text))
            agent_audio, _ = await tts.a_synthesize(reply, voice=voice)
            return ConnectorTurn(audio=agent_audio, transcript=reply)

        return cls(agent, sample_rate=tts.sample_rate, **kwargs)

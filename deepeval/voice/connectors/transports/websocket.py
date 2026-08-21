import time
import json
import base64
import asyncio
import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    ClassVar,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import aiohttp
from aiohttp import WSMsgType
from pydantic import BaseModel, Field

from deepeval.errors import DeepEvalError
from deepeval.test_case import Audio, AudioChunk
from deepeval.voice.protocol import VoiceProtocol
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.transports.base import (
    BaseVoiceConnector,
    UplinkStream,
    iter_downlink,
)
from deepeval.voice.connectors.types import AgentEvent, ConnectorTurn
from deepeval.voice.connectors.turn_engine import collect_agent_turn
from deepeval.voice.streaming import (
    DEFAULT_STREAM_SAMPLE_RATE,
    PcmRecorder,
    RealTimePacer,
    UplinkResult,
)
from deepeval.voice.turn_detection import TurnDetection, turn_detection_timing

logger = logging.getLogger(__name__)


@dataclass
class InboundEvent:

    audio: Optional[bytes] = None
    transcript: Optional[str] = None
    turn_complete: bool = False
    pong_reply: Optional[Union[str, bytes]] = None
    ready: bool = False


class WebSocketMessageSchema(BaseModel):
    """How one agent's WebSocket dialect carries audio, text, and turn ends.

    A raw-audio WebSocket agent has no standard message shape, so none of these
    is a tuning knob — each one describes somebody else's protocol. They are
    validated together at construction because a name that does not match the
    agent surfaces much later, and as silence: the connector reads every
    message, finds nothing under the key it was given, and the agent looks like
    it never answered.
    """

    model_config = {"frozen": True}

    # Outbound: where the caller's audio goes, and in what form.
    send_key: str = Field(default="audio", min_length=1)
    binary_outbound: bool = False

    # Inbound: where the agent's audio and text are found. Dotted keys are
    # read as a path into nested objects.
    receive_audio_key: str = Field(default="audio", min_length=1)
    binary_inbound: bool = False
    receive_transcript_key: Optional[str] = None

    # End of turn: the value under `type_key` that means the agent is done.
    # Without one, only silence is left to infer it from.
    turn_complete_type: Optional[str] = None
    type_key: str = Field(default="type", min_length=1)

    init_messages: List[Union[str, dict]] = Field(default_factory=list)
    ready_on: Literal["connect", "message"] = "connect"

    @property
    def signals_turn_complete(self) -> bool:
        return self.turn_complete_type is not None

    def encoded_initial_messages(self) -> List[Union[str, bytes]]:
        return [
            json.dumps(m) if isinstance(m, dict) else m
            for m in self.init_messages
        ]

    def read(self, message: dict, dotted_key: str):
        """Follow a dotted key into `message`, or None if it is not there."""
        current = message
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current


class BaseWebSocketConnector(BaseVoiceConnector):

    protocol: ClassVar[VoiceProtocol] = VoiceProtocol.WEBSOCKET

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        turn_detection: TurnDetection = "balanced",
        connect_timeout_s: float = 15.0,
        trailing_silence_ms: int = 1500,
    ):
        self.sample_rate = sample_rate
        self.turn_detection = turn_detection
        timing = turn_detection_timing(turn_detection)
        self.end_of_turn_silence_ms = timing.end_of_turn_silence_ms
        self.max_turn_timeout_s = timing.max_turn_timeout_s
        self.connect_timeout_s = connect_timeout_s

        self.trailing_silence_ms = trailing_silence_ms
        self._frame_gap_timeout_s = max(
            1.0, self.end_of_turn_silence_ms / 1000.0 + 0.5
        )

        self._send_rate = sample_rate
        self._recv_rate = sample_rate

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._inbound: Optional[asyncio.Queue] = None
        self._ready: Optional[asyncio.Event] = None
        self._current_transcript: Optional[str] = None
        self._interrupted: bool = False
        self._uplink: Optional[UplinkStream] = None

    @property
    def audio_format(self) -> Tuple[int, str]:
        return (self.sample_rate, "wav")

    @property
    def recv_sample_rate(self) -> int:
        return self._recv_rate

    @abstractmethod
    async def _open_session(self) -> str: ...

    @abstractmethod
    def _encode_outbound(self, pcm: bytes) -> Union[str, bytes]: ...

    @abstractmethod
    def _decode_inbound(
        self, raw: Union[str, bytes]
    ) -> Optional[InboundEvent]: ...

    def _initial_messages(self) -> List[Union[str, bytes]]:
        return []

    def _connect_headers(self) -> Optional[dict]:
        return None

    def _ready_on_connect(self) -> bool:
        return False

    async def connect(self) -> None:
        self._loop = asyncio.get_event_loop()
        self._inbound = asyncio.Queue()
        self._ready = asyncio.Event()
        self._uplink = UplinkStream()
        self._current_transcript = None
        self._interrupted = False

        self._session = aiohttp.ClientSession()
        url = await self._open_session()
        self._ws = await self._session.ws_connect(
            url, headers=self._connect_headers()
        )

        for message in self._initial_messages():
            await self._send(message)

        self._reader_task = self._loop.create_task(self._reader_loop())

        if self._ready_on_connect():
            self._ready.set()

        try:
            await asyncio.wait_for(
                self._ready.wait(), timeout=self.connect_timeout_s
            )
        except asyncio.TimeoutError:
            await self.disconnect()
            raise DeepEvalError(
                f"{type(self).__name__}: no session handshake within "
                f"{self.connect_timeout_s}s. Check the agent id / credentials "
                "and that the provider is reachable."
            )

    async def _send(self, message: Union[str, bytes]) -> None:
        if isinstance(message, (bytes, bytearray)):
            await self._ws.send_bytes(message)
        else:
            await self._ws.send_str(message)

    async def _reader_loop(self) -> None:
        try:
            async for msg in self._ws:
                if msg.type in (WSMsgType.TEXT, WSMsgType.BINARY):
                    event = self._decode_inbound(msg.data)
                    if event is None:
                        continue
                    if event.pong_reply is not None:
                        await self._send(event.pong_reply)
                    if event.ready:
                        self._ready.set()
                    received_at = time.perf_counter()
                    if event.transcript is not None:
                        self._current_transcript = event.transcript
                        await self._inbound.put(
                            AgentEvent(
                                transcript=event.transcript,
                                received_at=received_at,
                            )
                        )
                    if event.audio is not None:
                        await self._inbound.put(
                            AgentEvent(
                                audio=event.audio, received_at=received_at
                            )
                        )
                    if event.turn_complete:
                        await self._inbound.put(AgentEvent(turn_complete=True))
                elif msg.type in (
                    WSMsgType.CLOSED,
                    WSMsgType.CLOSING,
                    WSMsgType.ERROR,
                ):
                    break
        except asyncio.CancelledError:
            raise
        finally:
            await self._inbound.put(AgentEvent(turn_complete=True))

    def _prepare_outbound_pcm(
        self, audio: Audio, *, trailing_silence: bool
    ) -> bytes:
        pcm, sample_rate, num_channels = audio_utils.wav_bytes_to_pcm16(
            audio.get_bytes()
        )
        return self._outbound_pcm(
            audio_utils.downmix_to_mono(pcm, num_channels),
            sample_rate,
            trailing_silence=trailing_silence,
        )

    def _outbound_pcm(
        self, pcm: bytes, sample_rate: int, *, trailing_silence: bool
    ) -> bytes:
        pcm = audio_utils.resample_pcm16(pcm, sample_rate, self._send_rate)
        if trailing_silence:
            pcm = pcm + self._trailing_silence_pcm()
        return pcm

    def _trailing_silence_pcm(self) -> bytes:
        """Quiet appended to a full turn so the agent's VAD hears it end."""
        if self.trailing_silence_ms <= 0:
            return b""
        samples = int(self._send_rate * self.trailing_silence_ms / 1000)
        return b"\x00\x00" * samples

    async def _send_pcm(
        self, pcm: bytes, pacer: Optional[RealTimePacer] = None
    ) -> bool:
        """Send `pcm` as wire frames. False once the uplink has been cancelled."""
        for frame in audio_utils.iter_pcm16_frames(pcm, self._send_rate):
            if self._uplink.cancelled:
                return False
            if pacer is not None:
                await pacer.wait_to_send(frame)
                if self._uplink.cancelled:
                    return False
            await self._send(self._encode_outbound(frame))
        return True

    def _frame_bytes(self) -> int:
        return int(self._send_rate * audio_utils.DEFAULT_FRAME_MS / 1000) * 2

    async def _send_whole_frames(
        self, buffer: bytearray, pacer: Optional[RealTimePacer] = None
    ) -> Tuple[bool, Optional[float]]:
        """Send every complete frame in `buffer` and keep the rest for later.

        Frames are a fixed size and the last one is zero-padded to reach it. A
        partial frame therefore has to wait for the audio that follows it, or
        every piece of a streamed utterance would arrive with silence spliced
        onto its end.

        Returns whether sending may continue, and when the first frame of this
        call went out so the utterance can be placed on the call timeline.
        """
        size = self._frame_bytes()
        sent = 0
        first_at: Optional[float] = None
        while len(buffer) - sent >= size:
            if self._uplink.cancelled:
                del buffer[:sent]
                return False, first_at
            frame = bytes(buffer[sent : sent + size])
            if pacer is not None:
                await pacer.wait_to_send(frame)
                if self._uplink.cancelled:
                    del buffer[:sent]
                    return False, first_at
            if first_at is None:
                first_at = time.perf_counter()
            await self._send(self._encode_outbound(frame))
            sent += size
        del buffer[:sent]
        return True, first_at

    async def stream_uplink(
        self, audio: Audio, *, trailing_silence: bool = True
    ) -> None:
        if self._uplink is None:
            raise DeepEvalError(
                f"{type(self).__name__}.stream_uplink() called before connect()."
            )
        # Cancel any prior uplink, then start a fresh one.
        await self.stop_uplink()
        if self._uplink.task is not None:
            try:
                await self._uplink.task
            except Exception:
                pass
            self._uplink.task = None

        self._uplink.begin()
        pcm = self._prepare_outbound_pcm(
            audio, trailing_silence=trailing_silence
        )

        self._uplink.task = asyncio.create_task(
            self._send_pcm(pcm, RealTimePacer(self._send_rate))
        )
        await self._uplink.task
        self._uplink.task = None

    async def stream_uplink_chunks(
        self,
        chunks: AsyncIterable[AudioChunk],
        *,
        trailing_silence: bool = True,
        on_first_frame: Optional[Callable[[float], None]] = None,
    ) -> UplinkResult:
        """Forward each frame of speech as soon as it has been synthesized.

        The agent's own VAD and transcription then run on the opening words
        while the rest of the utterance is still being made, the way they would
        on a live call, instead of after the whole thing exists.
        """
        if self._uplink is None:
            raise DeepEvalError(
                f"{type(self).__name__}.stream_uplink_chunks() called before "
                "connect()."
            )
        await self.stop_uplink()
        self._uplink.begin()

        recorder = PcmRecorder()
        pending = bytearray()
        sending = True
        first_frame_at: Optional[float] = None
        pacer = RealTimePacer(self._send_rate)
        async for chunk in chunks:
            pcm = recorder.add(chunk)
            if not sending:
                # Cancelled mid-utterance: keep recording so the turn still
                # holds everything the caller said, but send no more of it.
                continue
            pending.extend(
                self._outbound_pcm(
                    pcm,
                    chunk.sampleRate or DEFAULT_STREAM_SAMPLE_RATE,
                    trailing_silence=False,
                )
            )
            sending, sent_at = await self._send_whole_frames(pending, pacer)
            if first_frame_at is None and sent_at is not None:
                first_frame_at = sent_at
                if on_first_frame is not None:
                    on_first_frame(sent_at)
        if sending:
            if trailing_silence:
                pending.extend(self._trailing_silence_pcm())
            if pending:
                if first_frame_at is None:
                    first_frame_at = time.perf_counter()
                    if on_first_frame is not None:
                        on_first_frame(first_frame_at)
                await self._send_pcm(bytes(pending), pacer)
        return UplinkResult(
            audio=recorder.to_audio(), first_frame_at=first_frame_at
        )

    async def stop_uplink(self) -> None:
        if self._uplink is not None:
            await self._uplink.stop()

    async def iter_agent_events(self) -> AsyncIterator[AgentEvent]:
        if self._inbound is None:
            raise DeepEvalError(
                f"{type(self).__name__}.iter_agent_events() called before "
                "connect()."
            )
        async for event in iter_downlink(self._inbound):
            yield event

    async def exchange_turn(self, audio: Audio) -> ConnectorTurn:
        self.drain_downlink()
        self._current_transcript = None
        self._interrupted = False

        pcm = self._prepare_outbound_pcm(audio, trailing_silence=True)
        sent_chunks = 0
        input_audio_started_at = time.perf_counter()
        # Paced, or the agent is handed a five-second question in a tenth of a
        # second and answers inside the caller's own turn on the recording.
        pacer = RealTimePacer(self._send_rate)
        for chunk in audio_utils.iter_pcm16_frames(pcm, self._send_rate):
            await pacer.wait_to_send(chunk)
            await self._send(self._encode_outbound(chunk))
            sent_chunks += 1

        sent_at = time.perf_counter()
        agent_pcm, first_audio_at = await collect_agent_turn(
            self._inbound,
            sample_rate=self._recv_rate,
            end_of_turn_silence_ms=self.end_of_turn_silence_ms,
            frame_gap_timeout_s=self._frame_gap_timeout_s,
            max_turn_timeout_s=self.max_turn_timeout_s,
        )

        pcm24 = audio_utils.resample_pcm16(
            agent_pcm, self._recv_rate, self.sample_rate
        )
        reply = Audio.from_bytes(
            audio_utils.pcm16_to_wav_bytes(pcm24, self.sample_rate, 1),
            "audio/wav",
            sampleRate=self.sample_rate,
            encoding="wav",
            duration=(
                (len(pcm24) / 2) / self.sample_rate
                if self.sample_rate
                else None
            ),
        )
        latency_ms = (
            (first_audio_at - sent_at) * 1000.0
            if first_audio_at is not None
            else None
        )
        if not agent_pcm and not self._current_transcript:
            logger.warning(
                "%s: agent returned no audio and no transcript this turn "
                "(sent %d chunks @ %dHz, recv @ %dHz). Check credentials, that "
                "the agent is responding, and that its audio format is pcm_* "
                "(not ulaw). If it times out, the agent's VAD may not be "
                "detecting end-of-turn — try a larger trailing_silence_ms.",
                type(self).__name__,
                sent_chunks,
                self._send_rate,
                self._recv_rate,
            )
        return ConnectorTurn(
            audio=reply,
            transcript=self._current_transcript,
            latency_ms=latency_ms,
            interrupted=self._interrupted,
            input_audio_started_at=input_audio_started_at,
            input_audio_ended_at=sent_at,
            audio_started_at=first_audio_at,
        )

    def drain_downlink(self) -> None:
        while not self._inbound.empty():
            try:
                self._inbound.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def disconnect(self) -> None:
        await self.stop_uplink()
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None


class WebSocketConnector(BaseWebSocketConnector):
    """Generic, configuration-driven WebSocket connector for custom agents."""

    def __init__(
        self,
        url: str,
        *,
        headers: Optional[dict] = None,
        sample_rate: int = 24000,
        send_key: str = "audio",
        binary_outbound: bool = False,
        receive_audio_key: str = "audio",
        binary_inbound: bool = False,
        receive_transcript_key: Optional[str] = None,
        turn_complete_type: Optional[str] = None,
        type_key: str = "type",
        init_messages: Optional[List[Union[str, dict]]] = None,
        ready_on: str = "connect",
        **base_kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **base_kwargs)
        self.url = url
        self.headers = headers
        # The nine message-shape arguments describe one thing — the agent's
        # dialect — and are kept as one, validated together rather than
        # scattered across the connector as nine unchecked strings.
        self.schema = WebSocketMessageSchema(
            send_key=send_key,
            binary_outbound=binary_outbound,
            receive_audio_key=receive_audio_key,
            binary_inbound=binary_inbound,
            receive_transcript_key=receive_transcript_key,
            turn_complete_type=turn_complete_type,
            type_key=type_key,
            init_messages=init_messages or [],
            ready_on=ready_on,
        )

    async def _open_session(self) -> str:
        return self.url

    @property
    def signals_turn_complete(self) -> bool:
        return self.schema.signals_turn_complete

    def _connect_headers(self) -> Optional[dict]:
        return self.headers

    def _ready_on_connect(self) -> bool:
        return self.schema.ready_on == "connect"

    def _initial_messages(self) -> List[Union[str, bytes]]:
        return self.schema.encoded_initial_messages()

    def _encode_outbound(self, pcm: bytes) -> Union[str, bytes]:
        if self.schema.binary_outbound:
            return pcm
        return json.dumps(
            {self.schema.send_key: base64.b64encode(pcm).decode("ascii")}
        )

    def _decode_inbound(self, raw: Union[str, bytes]) -> Optional[InboundEvent]:
        schema = self.schema
        if schema.binary_inbound and isinstance(raw, (bytes, bytearray)):
            return InboundEvent(audio=bytes(raw))

        try:
            message = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(message, dict):
            return None

        event = InboundEvent()
        audio_b64 = schema.read(message, schema.receive_audio_key)
        if audio_b64:
            event.audio = base64.b64decode(audio_b64)
        if schema.receive_transcript_key:
            transcript = schema.read(message, schema.receive_transcript_key)
            if transcript:
                event.transcript = transcript
        if (
            schema.turn_complete_type is not None
            and message.get(schema.type_key) == schema.turn_complete_type
        ):
            event.turn_complete = True
        if schema.ready_on == "message" and not self._ready.is_set():
            event.ready = True
        return event

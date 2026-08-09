import time
import asyncio
import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import aiohttp
from aiohttp import WSMsgType

from deepeval.errors import DeepEvalError
from deepeval.test_case import Audio
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.base_connector import (
    BaseVoiceConnector,
    ConnectorTurn,
)
from deepeval.voice.connectors.turn_engine import collect_agent_turn

logger = logging.getLogger(__name__)


@dataclass
class InboundEvent:

    audio: Optional[bytes] = None
    transcript: Optional[str] = None
    turn_complete: bool = False
    pong_reply: Optional[Union[str, bytes]] = None
    ready: bool = False


class BaseWebSocketConnector(BaseVoiceConnector):
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        end_of_turn_silence_ms: int = 800,
        max_turn_timeout_s: float = 30.0,
        connect_timeout_s: float = 15.0,
        trailing_silence_ms: int = 1500,
    ):
        self.sample_rate = sample_rate
        self.end_of_turn_silence_ms = end_of_turn_silence_ms
        self.max_turn_timeout_s = max_turn_timeout_s
        self.connect_timeout_s = connect_timeout_s

        self.trailing_silence_ms = trailing_silence_ms
        self._frame_gap_timeout_s = max(
            1.0, end_of_turn_silence_ms / 1000.0 + 0.5
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

    @property
    def audio_format(self) -> Tuple[int, str]:
        return (self.sample_rate, "wav")

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
                    if event.transcript is not None:
                        self._current_transcript = event.transcript
                    if event.audio is not None:
                        await self._inbound.put(event.audio)
                    if event.turn_complete:
                        await self._inbound.put(None)
                elif msg.type in (
                    WSMsgType.CLOSED,
                    WSMsgType.CLOSING,
                    WSMsgType.ERROR,
                ):
                    break
        except asyncio.CancelledError:
            raise
        finally:

            await self._inbound.put(None)

    async def send_turn(self, audio: Audio) -> ConnectorTurn:
        self._drain_stale_inbound()
        self._current_transcript = None
        self._interrupted = False

        pcm, sample_rate, num_channels = audio_utils.wav_bytes_to_pcm16(
            audio.get_bytes()
        )
        pcm = audio_utils.downmix_to_mono(pcm, num_channels)
        pcm = audio_utils.resample_pcm16(pcm, sample_rate, self._send_rate)
        if self.trailing_silence_ms > 0:
            silence_samples = int(
                self._send_rate * self.trailing_silence_ms / 1000
            )
            pcm = pcm + b"\x00\x00" * silence_samples
        sent_chunks = 0
        for chunk in audio_utils.iter_pcm16_frames(pcm, self._send_rate):
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
        )

    def _drain_stale_inbound(self) -> None:
        while not self._inbound.empty():
            try:
                self._inbound.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def disconnect(self) -> None:
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

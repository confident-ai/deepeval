import asyncio
import json
import logging
import time
from fractions import Fraction
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
)
from urllib.parse import urlparse

import aiohttp
from aiohttp import WSMsgType

from deepeval.errors import DeepEvalError
from deepeval.test_case import Audio, AudioChunk
from deepeval.utils import require_dependency
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.transports.base import (
    BaseVoiceConnector,
    UplinkStream,
    iter_downlink,
)
from deepeval.voice.connectors.turn_engine import collect_agent_turn
from deepeval.voice.connectors.types import AgentEvent, ConnectorTurn
from deepeval.voice.protocol import VoiceProtocol
from deepeval.voice.streaming import (
    DEFAULT_STREAM_SAMPLE_RATE,
    PcmRecorder,
    RealTimePacer,
    UplinkResult,
)
from deepeval.voice.turn_detection import TurnDetection, turn_detection_timing

logger = logging.getLogger(__name__)

_INSTALL_HINT = "Install it with `pip install aiortc`."
_HTTP_SCHEMES = ("http", "https")
_WEBSOCKET_SCHEMES = ("ws", "wss")
_TEXT_KEYS = ("transcript", "text", "content", "message")

SDP_CONTENT_TYPE = "application/sdp"
DEFAULT_WEBRTC_SAMPLE_RATE = 48000
DEFAULT_DATA_CHANNEL = "text"


def signaling_kind(url: str) -> str:
    scheme = urlparse(url or "").scheme.lower()
    if scheme in _HTTP_SCHEMES:
        return "http"
    if scheme in _WEBSOCKET_SCHEMES:
        return "websocket"
    raise DeepEvalError(
        "WebRTCConnector needs an http(s):// signaling URL that answers an "
        "SDP offer, or a ws(s):// one that exchanges offer/answer messages; "
        f"got {url!r}."
    )


def parse_answer_sdp(body: str) -> str:
    text = (body or "").lstrip()
    if text.startswith("v="):
        return text
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        nested = payload.get("answer")
        if isinstance(nested, dict):
            payload = nested
        sdp = payload.get("sdp")
        if isinstance(sdp, str) and sdp.strip().startswith("v="):
            return sdp
    raise DeepEvalError(
        "The signaling endpoint did not return an SDP answer. Expected the "
        "raw SDP or JSON with an `sdp` field."
    )


def extract_text(message: Any) -> Optional[str]:
    if isinstance(message, (bytes, bytearray)):
        try:
            message = bytes(message).decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(message, str):
        return None
    text = message.strip()
    if not text:
        return None
    if text[0] in "{[":
        try:
            payload = json.loads(text)
        except ValueError:
            return text
        if isinstance(payload, dict):
            for key in _TEXT_KEYS:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None
        return None
    return text


def candidate_init(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    init = message.get("candidate", message)
    if isinstance(init, str):
        init = {
            "candidate": init,
            "sdpMid": message.get("sdpMid"),
            "sdpMLineIndex": message.get("sdpMLineIndex"),
        }
    if not isinstance(init, dict) or not init.get("candidate"):
        return None
    return init


class _PcmSource:
    def __init__(self, sample_rate: int, av_module: Any):
        self.sample_rate = sample_rate
        self._av = av_module
        self._frames: asyncio.Queue = asyncio.Queue()
        self._frame_samples = int(
            sample_rate * audio_utils.DEFAULT_FRAME_MS / 1000
        )
        self._pts = 0
        self._started_at: Optional[float] = None

    def push(self, pcm: bytes) -> None:
        self._frames.put_nowait(pcm)

    def clear(self) -> None:
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except asyncio.QueueEmpty:
                break

    def next_pcm(self) -> bytes:
        try:
            return self._frames.get_nowait()
        except asyncio.QueueEmpty:
            return b"\x00\x00" * self._frame_samples

    async def next_frame(self):
        now = time.perf_counter()
        if self._started_at is None:
            self._started_at = now
        due = self._started_at + self._pts / self.sample_rate
        if due > now:
            await asyncio.sleep(due - now)
        pcm = self.next_pcm()
        samples = len(pcm) // 2
        frame = self._av.AudioFrame(
            format="s16", layout="mono", samples=samples
        )
        frame.planes[0].update(pcm)
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        frame.time_base = Fraction(1, self.sample_rate)
        self._pts += samples
        return frame


def _make_track(aiortc_module: Any, source: _PcmSource):
    base = aiortc_module.MediaStreamTrack

    class PcmTrack(base):
        kind = "audio"

        async def recv(self):
            return await source.next_frame()

    return PcmTrack()


class _HttpSignaling:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: Dict[str, str],
    ):
        self._session = session
        self._url = url
        self._headers = headers

    async def exchange(self, offer_sdp: str) -> str:
        headers = {
            **self._headers,
            "Content-Type": SDP_CONTENT_TYPE,
            "Accept": f"{SDP_CONTENT_TYPE}, application/json",
        }
        async with self._session.post(
            self._url, data=offer_sdp, headers=headers
        ) as response:
            body = await response.text()
            if response.status >= 400:
                raise DeepEvalError(
                    f"Signaling endpoint {self._url} rejected the offer with "
                    f"HTTP {response.status}: {body[:200]}"
                )
        return parse_answer_sdp(body)

    async def start(self, pc: Any, aiortc_module: Any) -> None:
        return None

    async def close(self) -> None:
        return None


class _WebSocketSignaling:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: Dict[str, str],
    ):
        self._session = session
        self._url = url
        self._headers = headers
        self._ws = None
        self._pending: List[Dict[str, Any]] = []
        self._reader: Optional[asyncio.Task] = None

    async def exchange(self, offer_sdp: str) -> str:
        self._ws = await self._session.ws_connect(
            self._url, headers=self._headers or None
        )
        await self._ws.send_json({"type": "offer", "sdp": offer_sdp})
        async for msg in self._ws:
            if msg.type != WSMsgType.TEXT:
                if msg.type in (
                    WSMsgType.CLOSED,
                    WSMsgType.CLOSING,
                    WSMsgType.ERROR,
                ):
                    break
                continue
            try:
                message = json.loads(msg.data)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            kind = message.get("type")
            if kind == "answer" and isinstance(message.get("sdp"), str):
                return message["sdp"]
            if kind == "candidate":
                init = candidate_init(message)
                if init is not None:
                    self._pending.append(init)
        raise DeepEvalError(
            f"Signaling socket {self._url} closed before sending an answer."
        )

    async def start(self, pc: Any, aiortc_module: Any) -> None:
        for init in self._pending:
            await self._add_candidate(pc, aiortc_module, init)
        self._pending = []
        self._reader = asyncio.create_task(
            self._read_candidates(pc, aiortc_module)
        )

    async def _read_candidates(self, pc: Any, aiortc_module: Any) -> None:
        try:
            async for msg in self._ws:
                if msg.type != WSMsgType.TEXT:
                    if msg.type in (
                        WSMsgType.CLOSED,
                        WSMsgType.CLOSING,
                        WSMsgType.ERROR,
                    ):
                        break
                    continue
                try:
                    message = json.loads(msg.data)
                except ValueError:
                    continue
                if (
                    isinstance(message, dict)
                    and message.get("type") == "candidate"
                ):
                    init = candidate_init(message)
                    if init is not None:
                        await self._add_candidate(pc, aiortc_module, init)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Signaling socket reader stopped.", exc_info=True)

    async def _add_candidate(
        self, pc: Any, aiortc_module: Any, init: Dict[str, Any]
    ) -> None:
        raw = init["candidate"]
        if raw.startswith("candidate:"):
            raw = raw.split(":", 1)[1]
        sdp = require_dependency(
            "aiortc.sdp",
            provider_label="WebRTCConnector",
            install_hint=_INSTALL_HINT,
        )
        candidate = sdp.candidate_from_sdp(raw)
        candidate.sdpMid = init.get("sdpMid")
        candidate.sdpMLineIndex = init.get("sdpMLineIndex")
        await pc.addIceCandidate(candidate)

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
            self._reader = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


def build_signaling(
    url: str, session: aiohttp.ClientSession, headers: Dict[str, str]
):
    if signaling_kind(url) == "http":
        return _HttpSignaling(session, url, headers)
    return _WebSocketSignaling(session, url, headers)


class WebRTCConnector(BaseVoiceConnector):

    protocol: ClassVar[VoiceProtocol] = VoiceProtocol.WEBRTC

    def __init__(
        self,
        signaling_url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        data_channel: Optional[str] = DEFAULT_DATA_CHANNEL,
        ice_servers: Optional[List[str]] = None,
        turn_detection: TurnDetection = "balanced",
        connect_timeout_s: float = 15.0,
        input_sample_rate: int = DEFAULT_STREAM_SAMPLE_RATE,
        webrtc_sample_rate: int = DEFAULT_WEBRTC_SAMPLE_RATE,
        trailing_silence_ms: int = 1500,
        transcript_grace_s: float = 0.0,
    ):
        signaling_kind(signaling_url)
        self.signaling_url = signaling_url
        self.headers = dict(headers or {})
        self.data_channel = data_channel
        self.ice_servers = ice_servers
        self.turn_detection = turn_detection
        timing = turn_detection_timing(turn_detection)
        self.end_of_turn_silence_ms = timing.end_of_turn_silence_ms
        self.max_turn_timeout_s = timing.max_turn_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.input_sample_rate = input_sample_rate
        self.webrtc_sample_rate = webrtc_sample_rate
        self.trailing_silence_ms = trailing_silence_ms
        self.transcript_grace_s = transcript_grace_s
        self._frame_gap_timeout_s = max(
            1.0, self.end_of_turn_silence_ms / 1000.0 + 0.5
        )

        self._aiortc = None
        self._av = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._signaling = None
        self._pc = None
        self._source: Optional[_PcmSource] = None
        self._channel = None
        self._agent_track = None
        self._agent_track_ready: Optional[asyncio.Event] = None
        self._connected: Optional[asyncio.Event] = None
        self._channel_open: Optional[asyncio.Event] = None
        self._drain_task: Optional[asyncio.Task] = None
        self._inbound: Optional[asyncio.Queue] = None
        self._uplink: Optional[UplinkStream] = None
        self._current_transcript: Optional[str] = None
        self._transcript_ready: Optional[asyncio.Event] = None

    @property
    def audio_format(self) -> Tuple[int, str]:
        return (self.input_sample_rate, "wav")

    @property
    def recv_sample_rate(self) -> int:
        return self.webrtc_sample_rate

    @property
    def supports_duplex(self) -> bool:
        return True

    async def connect(self) -> None:
        self._aiortc = require_dependency(
            "aiortc",
            provider_label="WebRTCConnector",
            install_hint=_INSTALL_HINT,
        )
        self._av = require_dependency(
            "av", provider_label="WebRTCConnector", install_hint=_INSTALL_HINT
        )
        self._loop = asyncio.get_event_loop()
        self._inbound = asyncio.Queue()
        self._agent_track_ready = asyncio.Event()
        self._connected = asyncio.Event()
        self._channel_open = asyncio.Event()
        self._transcript_ready = asyncio.Event()
        self._uplink = UplinkStream()
        self._current_transcript = None

        self._session = aiohttp.ClientSession()
        self._pc = self._aiortc.RTCPeerConnection(self._rtc_configuration())
        self._source = _PcmSource(self.webrtc_sample_rate, self._av)
        self._pc.addTrack(_make_track(self._aiortc, self._source))
        self._pc.on("track", self._on_track)
        self._pc.on("datachannel", self._attach_channel)
        self._pc.on("connectionstatechange", self._on_connection_state)
        if self.data_channel:
            self._attach_channel(self._pc.createDataChannel(self.data_channel))

        try:
            offer = await self._pc.createOffer()
            await self._pc.setLocalDescription(offer)
            self._signaling = build_signaling(
                self.signaling_url, self._session, self.headers
            )
            answer_sdp = await self._signaling.exchange(
                self._pc.localDescription.sdp
            )
            await self._pc.setRemoteDescription(
                self._aiortc.RTCSessionDescription(
                    sdp=answer_sdp, type="answer"
                )
            )
            await self._signaling.start(self._pc, self._aiortc)
        except Exception:
            await self.disconnect()
            raise
        await self._await_connection()

    def _rtc_configuration(self):
        if self.ice_servers is None:
            return None
        return self._aiortc.RTCConfiguration(
            iceServers=[
                self._aiortc.RTCIceServer(urls=url) for url in self.ice_servers
            ]
        )

    async def _await_connection(self) -> None:
        deadline = time.perf_counter() + self.connect_timeout_s

        async def wait_until(event: asyncio.Event, failure: str) -> None:
            remaining = deadline - time.perf_counter()
            try:
                await asyncio.wait_for(event.wait(), timeout=max(remaining, 0))
            except asyncio.TimeoutError:
                await self.disconnect()
                raise DeepEvalError(
                    f"{failure} within {self.connect_timeout_s}s after "
                    f"signaling with {self.signaling_url}."
                )

        await wait_until(
            self._connected, "The peer connection did not reach 'connected'"
        )
        await wait_until(
            self._agent_track_ready, "No agent audio track arrived"
        )
        if self._channel is not None and not self._channel_open.is_set():
            try:
                await asyncio.wait_for(
                    self._channel_open.wait(),
                    timeout=max(deadline - time.perf_counter(), 0),
                )
            except asyncio.TimeoutError:
                logger.debug(
                    "Data channel %r did not open; text will not be exchanged.",
                    self.data_channel,
                )

    def _on_track(self, track) -> None:
        if self._agent_track is not None or track.kind != "audio":
            return
        self._agent_track = track
        self._drain_task = self._loop.create_task(self._drain_loop(track))
        self._agent_track_ready.set()

    async def _drain_loop(self, track) -> None:
        resampler = self._av.AudioResampler(
            format="s16", layout="mono", rate=self.webrtc_sample_rate
        )
        try:
            while True:
                frame = await track.recv()
                for out in resampler.resample(frame):
                    pcm = bytes(out.planes[0])[: out.samples * 2]
                    if pcm:
                        await self._inbound.put(
                            AgentEvent(
                                audio=pcm, received_at=time.perf_counter()
                            )
                        )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Agent audio track ended.", exc_info=True)
        finally:
            await self._inbound.put(AgentEvent(turn_complete=True))

    def _attach_channel(self, channel) -> None:
        if self._channel is None:
            self._channel = channel
        channel.on("message", self._on_text)
        if channel.readyState == "open":
            self._channel_open.set()
        else:
            channel.on("open", self._channel_open.set)

    def _on_text(self, message: Any) -> None:
        text = extract_text(message)
        if text is None:
            return
        self._current_transcript = text
        if self._transcript_ready is not None:
            self._transcript_ready.set()
        if self._inbound is not None:
            self._inbound.put_nowait(
                AgentEvent(transcript=text, received_at=time.perf_counter())
            )

    def _on_connection_state(self) -> None:
        state = getattr(self._pc, "connectionState", None)
        if state == "connected" and self._connected is not None:
            self._connected.set()
        if state in ("failed", "closed") and self._inbound is not None:
            self._inbound.put_nowait(AgentEvent(turn_complete=True))

    def send_text(self, text: str) -> None:
        if self._channel is None or self._channel.readyState != "open":
            raise DeepEvalError(
                "WebRTCConnector has no open data channel to send text on."
            )
        self._channel.send(text)

    def _trailing_silence_pcm(self) -> bytes:
        if self.trailing_silence_ms <= 0:
            return b""
        samples = int(self.webrtc_sample_rate * self.trailing_silence_ms / 1000)
        return b"\x00\x00" * samples

    def _outbound_pcm(
        self, pcm: bytes, sample_rate: int, *, trailing_silence: bool
    ) -> bytes:
        pcm = audio_utils.resample_pcm16(
            pcm, sample_rate, self.webrtc_sample_rate
        )
        if trailing_silence:
            pcm = pcm + self._trailing_silence_pcm()
        return pcm

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

    def _frame_bytes(self) -> int:
        return (
            int(self.webrtc_sample_rate * audio_utils.DEFAULT_FRAME_MS / 1000)
            * 2
        )

    async def _push_whole_frames(
        self, buffer: bytearray, pacer: RealTimePacer
    ) -> Tuple[bool, Optional[float]]:
        size = self._frame_bytes()
        sent = 0
        first_at: Optional[float] = None
        while len(buffer) - sent >= size:
            if self._uplink.cancelled:
                del buffer[:sent]
                return False, first_at
            frame = bytes(buffer[sent : sent + size])
            await pacer.wait_to_send(frame)
            if self._uplink.cancelled:
                del buffer[:sent]
                return False, first_at
            if first_at is None:
                first_at = time.perf_counter()
            self._source.push(frame)
            sent += size
        del buffer[:sent]
        return True, first_at

    async def _push_pcm(self, pcm: bytes, pacer: RealTimePacer) -> bool:
        for frame in audio_utils.iter_pcm16_frames(
            pcm, self.webrtc_sample_rate
        ):
            if self._uplink.cancelled:
                return False
            await pacer.wait_to_send(frame)
            if self._uplink.cancelled:
                return False
            self._source.push(frame)
        return True

    def _require_connected(self, method: str) -> None:
        if self._uplink is None or self._source is None:
            raise DeepEvalError(
                f"WebRTCConnector.{method}() called before connect()."
            )

    async def stream_uplink(
        self, audio: Audio, *, trailing_silence: bool = True
    ) -> None:
        self._require_connected("stream_uplink")
        await self.stop_uplink()
        self._uplink.begin()
        pcm = self._prepare_outbound_pcm(
            audio, trailing_silence=trailing_silence
        )
        self._uplink.task = asyncio.create_task(
            self._push_pcm(pcm, RealTimePacer(self.webrtc_sample_rate))
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
        self._require_connected("stream_uplink_chunks")
        await self.stop_uplink()
        self._uplink.begin()

        recorder = PcmRecorder()
        pending = bytearray()
        sending = True
        first_frame_at: Optional[float] = None
        pacer = RealTimePacer(self.webrtc_sample_rate)
        async for chunk in chunks:
            pcm = recorder.add(chunk)
            if not sending:
                continue
            pending.extend(
                self._outbound_pcm(
                    pcm,
                    chunk.sampleRate or DEFAULT_STREAM_SAMPLE_RATE,
                    trailing_silence=False,
                )
            )
            sending, sent_at = await self._push_whole_frames(pending, pacer)
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
                await self._push_pcm(bytes(pending), pacer)
        return UplinkResult(
            audio=recorder.to_audio(), first_frame_at=first_frame_at
        )

    async def stop_uplink(self) -> None:
        if self._uplink is not None:
            await self._uplink.stop()
        if self._source is not None:
            self._source.clear()

    async def iter_agent_events(self) -> AsyncIterator[AgentEvent]:
        if self._inbound is None:
            raise DeepEvalError(
                "WebRTCConnector.iter_agent_events() called before connect()."
            )
        async for event in iter_downlink(self._inbound):
            yield event

    async def exchange_turn(self, audio: Audio) -> ConnectorTurn:
        self._require_connected("exchange_turn")
        self.drain_downlink()
        self._current_transcript = None
        self._transcript_ready.clear()

        pcm = self._prepare_outbound_pcm(audio, trailing_silence=True)
        input_audio_started_at = time.perf_counter()
        self._uplink.begin()
        await self._push_pcm(pcm, RealTimePacer(self.webrtc_sample_rate))

        sent_at = time.perf_counter()
        agent_pcm, first_audio_at = await collect_agent_turn(
            self._inbound,
            sample_rate=self.webrtc_sample_rate,
            end_of_turn_silence_ms=self.end_of_turn_silence_ms,
            frame_gap_timeout_s=self._frame_gap_timeout_s,
            max_turn_timeout_s=self.max_turn_timeout_s,
            silence_threshold_rms=self.silence_threshold_rms,
        )
        await self._await_transcript(bool(agent_pcm))

        reply_pcm = audio_utils.resample_pcm16(
            agent_pcm, self.webrtc_sample_rate, self.input_sample_rate
        )
        reply = Audio.from_bytes(
            audio_utils.pcm16_to_wav_bytes(
                reply_pcm, self.input_sample_rate, 1
            ),
            "audio/wav",
            sampleRate=self.input_sample_rate,
            encoding="wav",
            duration=(len(reply_pcm) / 2) / self.input_sample_rate,
        )
        latency_ms = (
            max(first_audio_at - sent_at, 0.0) * 1000.0
            if first_audio_at is not None
            else None
        )
        return ConnectorTurn(
            audio=reply,
            transcript=self._current_transcript,
            latency_ms=latency_ms,
            interrupted=False,
            input_audio_started_at=input_audio_started_at,
            input_audio_ended_at=sent_at,
            audio_started_at=first_audio_at,
        )

    async def _await_transcript(self, spoke: bool) -> None:
        if not spoke or self._current_transcript is not None:
            return
        if self.transcript_grace_s <= 0:
            return
        try:
            await asyncio.wait_for(
                self._transcript_ready.wait(), timeout=self.transcript_grace_s
            )
        except asyncio.TimeoutError:
            pass

    def drain_downlink(self) -> None:
        if self._inbound is None:
            return
        while not self._inbound.empty():
            try:
                self._inbound.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def disconnect(self) -> None:
        await self.stop_uplink()
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except (asyncio.CancelledError, Exception):
                pass
            self._drain_task = None
        if self._signaling is not None:
            await self._signaling.close()
            self._signaling = None
        if self._pc is not None:
            try:
                await self._pc.close()
            except Exception:
                pass
            self._pc = None
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
        self._source = None
        self._channel = None
        self._agent_track = None

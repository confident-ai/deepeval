import json
import logging
import uuid
from typing import List, Optional, Union

from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.providers.pipecat import protobuf
from deepeval.voice.connectors.transports.websocket import (
    BaseWebSocketConnector,
    InboundEvent,
)

logger = logging.getLogger(__name__)

# Pipecat's default input rate; its output default is 24000, which is also the
# rate deepeval records at, so only the uplink needs stating.
DEFAULT_AGENT_SAMPLE_RATE = 16000

RTVI_LABEL = "rtvi-ai"
RTVI_PROTOCOL_VERSION = "2.1.0"


class PipecatConnector(BaseWebSocketConnector):
    """Connect to a Pipecat pipeline over its WebSocket transport.

    The pipeline is yours and self-hosted, so the connector takes a URL rather
    than credentials, and speaks the protobuf frames Pipecat's WebSocket
    transports serialize by default.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: Optional[dict] = None,
        agent_sample_rate: int = DEFAULT_AGENT_SAMPLE_RATE,
        client_ready: bool = True,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.url = url
        self.headers = headers
        # Pipecat does not resample what it is handed: the frames have to
        # arrive at the rate the pipeline's VAD and STT were started with.
        self.agent_sample_rate = agent_sample_rate
        self._send_rate = agent_sample_rate
        self.client_ready = client_ready
        self._transcript_parts: List[str] = []
        self._seen_turn_signal = False
        self._warned_non_protobuf = False

    @property
    def signals_turn_complete(self) -> bool:
        """True once the pipeline has been seen to announce an end of turn.

        Whether it does is a property of the pipeline rather than of Pipecat:
        the announcement is an RTVI `bot-stopped-speaking` message, and only a
        pipeline running an `RTVIProcessor` sends one. Reading it off the wire
        rather than taking it on trust means a pipeline without RTVI falls back
        to ending turns on silence instead of waiting for a signal that is
        never coming.
        """
        return self._seen_turn_signal

    async def connect(self) -> None:
        self._transcript_parts = []
        self._seen_turn_signal = False
        await super().connect()

    async def _open_session(self) -> str:
        return self.url

    def _connect_headers(self) -> Optional[dict]:
        return self.headers

    def _ready_on_connect(self) -> bool:
        # A Pipecat server accepts audio as soon as the socket is up; nothing
        # is sent back to acknowledge the client.
        return True

    def _initial_messages(self) -> List[Union[str, bytes]]:
        """Announce the client, which some pipelines wait for before greeting.

        Pipelines commonly open on `on_client_ready` — the greeting, and
        sometimes the audio input itself, is gated on it. A pipeline without an
        `RTVIProcessor` has nothing listening for the message and ignores it.
        """
        if not self.client_ready:
            return []
        return [
            protobuf.encode_message_frame(
                json.dumps(
                    {
                        "label": RTVI_LABEL,
                        "type": "client-ready",
                        "id": uuid.uuid4().hex,
                        "data": {
                            "version": RTVI_PROTOCOL_VERSION,
                            "about": {"library": "deepeval"},
                        },
                    }
                )
            )
        ]

    def _encode_outbound(self, pcm: bytes) -> Union[str, bytes]:
        return protobuf.encode_audio_frame(pcm, self._send_rate, 1)

    def _decode_inbound(self, raw: Union[str, bytes]) -> Optional[InboundEvent]:
        if isinstance(raw, str):
            self._warn_non_protobuf("a text message")
            return None
        try:
            frame = protobuf.decode_frame(bytes(raw))
        except ValueError:
            self._warn_non_protobuf("bytes that are not protobuf frames")
            return None
        if frame is None:
            return None

        if frame.kind == "audio":
            return self._on_audio(frame)
        if frame.kind == "message":
            return self._on_transport_message(frame.data)
        if frame.kind == "interruption":
            self._interrupted = True
        # A `transcription` frame is Pipecat's STT of deepeval's own audio, and
        # a `text` frame is unattributed pipeline text; neither is the agent's
        # reply.
        return None

    def _warn_non_protobuf(self, what: str) -> None:
        if self._warned_non_protobuf:
            return
        self._warned_non_protobuf = True
        logger.warning(
            "PipecatConnector received %s. It speaks the protobuf frames of "
            "`ProtobufFrameSerializer`, which is what Pipecat's WebSocket "
            "transports use by default — a pipeline serializing something "
            "else needs a `WebSocketConnector` describing that dialect "
            "instead.",
            what,
        )

    def _on_audio(self, frame: protobuf.PipecatFrame) -> Optional[InboundEvent]:
        pcm = frame.audio or b""
        if not pcm:
            return None
        sample_rate, channels = frame.sample_rate, frame.num_channels or 1
        if pcm[:4] == b"RIFF":
            # `add_wav_header=True` on the transport wraps every frame.
            pcm, sample_rate, channels = audio_utils.wav_bytes_to_pcm16(pcm)
        if sample_rate and sample_rate != self._recv_rate:
            logger.debug(
                "Pipecat is sending %dHz audio; reading the downlink at that "
                "rate instead of %dHz.",
                sample_rate,
                self._recv_rate,
            )
            self._recv_rate = sample_rate
        return InboundEvent(audio=audio_utils.downmix_to_mono(pcm, channels))

    def _on_transport_message(
        self, data: Optional[str]
    ) -> Optional[InboundEvent]:
        try:
            message = json.loads(data or "")
        except ValueError:
            return None
        if not isinstance(message, dict) or message.get("label") != RTVI_LABEL:
            return None

        message_type = message.get("type")

        if message_type == "bot-stopped-speaking":
            self._seen_turn_signal = True
            self._transcript_parts = []
            return InboundEvent(turn_complete=True)

        if message_type == "bot-interrupted":
            self._interrupted = True
            return None

        if message_type == "bot-transcription":
            return self._on_transcription(message)

        return None

    def _on_transcription(self, message: dict) -> Optional[InboundEvent]:
        """Collect the agent's transcript for the turn in progress.

        RTVI emits one `bot-transcription` per sentence, so the last one alone
        would record the end of a multi-sentence reply and nothing before it.
        """
        data = message.get("data")
        text = (data or {}).get("text") if isinstance(data, dict) else None
        text = (text or "").strip()
        if not text:
            return None
        self._transcript_parts.append(text)
        return InboundEvent(transcript=" ".join(self._transcript_parts))

    def drain_downlink(self) -> None:
        self._transcript_parts = []
        super().drain_downlink()

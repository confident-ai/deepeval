import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

from deepeval.errors import DeepEvalError
from deepeval.voice.connectors.transports.websocket import (
    BaseWebSocketConnector,
    InboundEvent,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.vapi.ai"

# The client messages deepeval reads off the socket. Both are in Vapi's default
# set, so they are only ever re-stated when a caller narrows the set themselves.
# https://docs.vapi.ai/api-reference/webhooks/client-message
REQUIRED_CLIENT_MESSAGES = ("speech-update", "transcript")


class VapiConnector(BaseWebSocketConnector):

    def __init__(
        self,
        assistant_id: str,
        api_key: Optional[str] = None,
        assistant_overrides: Optional[dict] = None,
        base_url: str = DEFAULT_BASE_URL,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.assistant_id = assistant_id
        self.api_key = api_key or os.getenv("VAPI_API_KEY")
        self.assistant_overrides = assistant_overrides
        self.base_url = base_url.rstrip("/")
        self.call_id: Optional[str] = None
        self._transcript_parts: List[str] = []

    @property
    def signals_turn_complete(self) -> bool:
        # `speech-update` closes every turn.
        return True

    def _ready_on_connect(self) -> bool:
        # The socket is minted per call and carries no handshake; audio may go
        # out as soon as it opens.
        return True

    def _assistant_overrides(self) -> Optional[dict]:
        """The caller's overrides, plus the client messages deepeval needs.

        Narrowing `clientMessages` is how a Vapi assistant turns events off, so
        a caller who sets it without `speech-update` would silently remove
        end-of-turn detection, and one without `transcript` would pay for STT
        on every turn.
        """
        overrides = dict(self.assistant_overrides or {})
        requested = overrides.get("clientMessages")
        if requested is None:
            return overrides or None
        missing = [
            message
            for message in REQUIRED_CLIENT_MESSAGES
            if message not in requested
        ]
        if missing:
            overrides["clientMessages"] = list(requested) + missing
        return overrides

    def _call_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "assistantId": self.assistant_id,
            "transport": {
                "provider": "vapi.websocket",
                "audioFormat": {
                    "format": "pcm_s16le",
                    "container": "raw",
                    "sampleRate": self.sample_rate,
                },
            },
        }
        overrides = self._assistant_overrides()
        if overrides:
            payload["assistantOverrides"] = overrides
        return payload

    async def _open_session(self) -> str:
        if not self.api_key:
            raise DeepEvalError(
                "VapiConnector requires a Vapi API key (pass `api_key=...` or "
                "set VAPI_API_KEY). The key mints the call the WebSocket "
                "carries."
            )
        async with self._session.post(
            f"{self.base_url}/call",
            headers={"authorization": f"Bearer {self.api_key}"},
            json=self._call_payload(),
        ) as resp:
            body = await resp.text()
            if resp.status >= 300:
                raise DeepEvalError(
                    f"Vapi call creation failed ({resp.status}): {body}"
                )
            try:
                data = json.loads(body)
            except ValueError:
                raise DeepEvalError(
                    f"Vapi call creation returned a non-JSON body: {body[:200]}"
                )

        transport = data.get("transport") or {}
        url = transport.get("websocketCallUrl")
        if not url:
            raise DeepEvalError(
                "Vapi call response had no `transport.websocketCallUrl`. Check "
                "that the assistant exists and the key may create calls."
            )
        self.call_id = data.get("id")
        self._apply_audio_format(transport.get("audioFormat"))
        return url

    def _apply_audio_format(self, audio_format: Optional[dict]) -> None:
        """Adopt the format Vapi settled on, which need not be the one asked for."""
        if not isinstance(audio_format, dict):
            return
        encoding = audio_format.get("format")
        if encoding and encoding != "pcm_s16le":
            logger.warning(
                "Vapi negotiated audio format %r; deepeval sends and reads "
                "PCM16, so the audio will be garbled. Request pcm_s16le.",
                encoding,
            )
        rate = audio_format.get("sampleRate")
        if isinstance(rate, int) and rate > 0:
            self._send_rate = rate
            self._recv_rate = rate

    def _encode_outbound(self, pcm: bytes) -> Union[str, bytes]:
        return pcm

    def _decode_inbound(self, raw: Union[str, bytes]) -> Optional[InboundEvent]:
        if isinstance(raw, (bytes, bytearray)):
            return InboundEvent(audio=bytes(raw))

        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        # Vapi's webhook deliveries wrap the event in `message`; the socket
        # sends the same events, so accept either shape.
        message = payload.get("message")
        if not isinstance(message, dict):
            message = payload

        message_type = message.get("type")

        if message_type == "transcript":
            return self._on_transcript(message)

        if message_type == "speech-update":
            if (
                message.get("role") == "assistant"
                and message.get("status") == "stopped"
            ):
                self._transcript_parts = []
                return InboundEvent(turn_complete=True)
            return None

        if message_type == "user-interrupted":
            self._interrupted = True
            return None

        return None

    def _on_transcript(self, message: dict) -> Optional[InboundEvent]:
        """Collect the agent's final transcripts for the turn in progress.

        Vapi emits one per utterance rather than one per turn, so a two
        sentence reply arrives as two finals and the last one on its own would
        record half of what the agent said.
        """
        if message.get("role") != "assistant":
            return None
        if message.get("transcriptType") != "final":
            return None
        text = (message.get("transcript") or "").strip()
        if not text:
            return None
        self._transcript_parts.append(text)
        return InboundEvent(transcript=" ".join(self._transcript_parts))

    def drain_downlink(self) -> None:
        self._transcript_parts = []
        super().drain_downlink()

    async def disconnect(self) -> None:
        """End the call before dropping the socket.

        Closing the socket alone leaves the call live on Vapi's side until it
        times out, still billing and still holding a concurrency slot.
        """
        if self._ws is not None and not self._ws.closed:
            try:
                await self._send(json.dumps({"type": "end-call"}))
            except Exception:
                logger.debug(
                    "Failed to send Vapi end-call; closing the socket anyway.",
                    exc_info=True,
                )
        await super().disconnect()

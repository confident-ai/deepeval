import os
import json
import base64
import logging
from typing import List, Optional, Union

from deepeval.errors import DeepEvalError
from deepeval.voice.connectors.base_websocket_connector import (
    BaseWebSocketConnector,
    InboundEvent,
)

logger = logging.getLogger(__name__)


class ElevenLabsConnector(BaseWebSocketConnector):
    def __init__(
        self,
        agent_id: str,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.agent_id = agent_id
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.region = region

    def _host(self) -> str:
        return (
            f"api.{self.region}.elevenlabs.io"
            if self.region
            else "api.elevenlabs.io"
        )

    async def _open_session(self) -> str:
        host = self._host()
        if self.api_key:

            rest_url = (
                f"https://{host}/v1/convai/conversation/get-signed-url"
                f"?agent_id={self.agent_id}"
            )
            async with self._session.get(
                rest_url, headers={"xi-api-key": self.api_key}
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise DeepEvalError(
                        f"ElevenLabs signed-url request failed "
                        f"({resp.status}): {body}"
                    )
                data = await resp.json()
            signed_url = data.get("signed_url")
            if not signed_url:
                raise DeepEvalError(
                    "ElevenLabs signed-url response missing 'signed_url'."
                )
            return signed_url

        return f"wss://{host}/v1/convai/conversation?agent_id={self.agent_id}"

    def _initial_messages(self) -> List[Union[str, bytes]]:

        return [json.dumps({"type": "conversation_initiation_client_data"})]

    def _encode_outbound(self, pcm: bytes) -> str:
        return json.dumps(
            {"user_audio_chunk": base64.b64encode(pcm).decode("ascii")}
        )

    def _decode_inbound(self, raw: Union[str, bytes]) -> Optional[InboundEvent]:
        try:
            message = json.loads(raw)
        except (ValueError, TypeError):
            return None
        msg_type = message.get("type")

        if msg_type == "conversation_initiation_metadata":
            meta = message.get("conversation_initiation_metadata_event", {})
            self._send_rate = self._parse_format_rate(
                meta.get("user_input_audio_format")
            )
            self._recv_rate = self._parse_format_rate(
                meta.get("agent_output_audio_format")
            )
            return InboundEvent(ready=True)

        if msg_type == "audio":
            audio_b64 = message.get("audio_event", {}).get("audio_base_64")
            if not audio_b64:
                return None
            return InboundEvent(audio=base64.b64decode(audio_b64))

        if msg_type == "agent_response":
            text = message.get("agent_response_event", {}).get("agent_response")
            return InboundEvent(transcript=text)

        if msg_type == "agent_response_complete":
            return InboundEvent(turn_complete=True)

        if msg_type == "ping":
            event_id = message.get("ping_event", {}).get("event_id")
            return InboundEvent(
                pong_reply=json.dumps({"type": "pong", "event_id": event_id})
            )

        if msg_type == "interruption":
            self._interrupted = True
            return None

        return None

    def _parse_format_rate(self, fmt: Optional[str]) -> int:
        if not fmt:
            return self.sample_rate
        try:
            prefix, rate = fmt.rsplit("_", 1)
            rate = int(rate)
        except (ValueError, AttributeError):
            return self.sample_rate
        if prefix != "pcm":
            logger.warning(
                "ElevenLabs audio format %r is not PCM; V1 assumes PCM16 so "
                "the audio may be garbled. Set the agent to a pcm_* format.",
                fmt,
            )
        return rate

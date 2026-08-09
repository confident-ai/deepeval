import json
import base64
from typing import List, Optional, Union

from deepeval.voice.connectors.base_websocket_connector import (
    BaseWebSocketConnector,
    InboundEvent,
)


class WebSocketConnector(BaseWebSocketConnector):
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
        super().__init__(**base_kwargs)
        self.url = url
        self.headers = headers

        self._send_rate = sample_rate
        self._recv_rate = sample_rate

        self.send_key = send_key
        self.binary_outbound = binary_outbound
        self.receive_audio_key = receive_audio_key
        self.binary_inbound = binary_inbound
        self.receive_transcript_key = receive_transcript_key
        self.turn_complete_type = turn_complete_type
        self.type_key = type_key
        self.init_messages = init_messages or []
        self.ready_on = ready_on

    async def _open_session(self) -> str:
        return self.url

    def _connect_headers(self) -> Optional[dict]:
        return self.headers

    def _ready_on_connect(self) -> bool:
        return self.ready_on == "connect"

    def _initial_messages(self) -> List[Union[str, bytes]]:
        return [
            json.dumps(m) if isinstance(m, dict) else m
            for m in self.init_messages
        ]

    def _encode_outbound(self, pcm: bytes) -> Union[str, bytes]:
        if self.binary_outbound:
            return pcm
        return json.dumps(
            {self.send_key: base64.b64encode(pcm).decode("ascii")}
        )

    def _decode_inbound(self, raw: Union[str, bytes]) -> Optional[InboundEvent]:

        if self.binary_inbound and isinstance(raw, (bytes, bytearray)):
            return InboundEvent(audio=bytes(raw))

        try:
            message = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(message, dict):
            return None

        event = InboundEvent()
        audio_b64 = self._dig(message, self.receive_audio_key)
        if audio_b64:
            event.audio = base64.b64decode(audio_b64)
        if self.receive_transcript_key:
            transcript = self._dig(message, self.receive_transcript_key)
            if transcript:
                event.transcript = transcript
        if (
            self.turn_complete_type is not None
            and message.get(self.type_key) == self.turn_complete_type
        ):
            event.turn_complete = True
        if self.ready_on == "message" and not self._ready.is_set():
            event.ready = True
        return event

    @staticmethod
    def _dig(message: dict, dotted_key: str):
        current = message
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

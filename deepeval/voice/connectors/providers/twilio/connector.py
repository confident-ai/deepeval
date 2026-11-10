import asyncio
import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Tuple, Union
from urllib.parse import urlparse
from xml.sax.saxutils import quoteattr

import aiohttp
from aiohttp import web

from deepeval.errors import DeepEvalError
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.phone import PhoneConnector, TwilioProvider
from deepeval.voice.connectors.types import AgentEvent
from deepeval.voice.connectors.transports.silence import SilenceTimeoutMixin
from deepeval.voice.connectors.transports.websocket import (
    BaseWebSocketConnector,
    InboundEvent,
)
from deepeval.voice.connectors.utils import DialPacer, is_phone_number

logger = logging.getLogger(__name__)

TWILIO_API_BASE = "https://api.twilio.com/2010-04-01"
ACCOUNT_SID_ENV = "TWILIO_ACCOUNT_SID"
AUTH_TOKEN_ENV = "TWILIO_AUTH_TOKEN"
STREAM_PATH = "/deepeval/twilio/stream"
CALL_KEY_PARAMETER = "deepeval_call"
TWILIO_SAMPLE_RATE = 8000
DEFAULT_RING_TIMEOUT_S = 45
DEFAULT_MAX_CALL_DURATION_S = 30 * 60
DEFAULT_DIAL_INTERVAL_S = 1.0
STATUS_POLL_INTERVAL_S = 1.0
STREAM_START_TIMEOUT_S = 10.0

FAILED_CALL_STATUSES = {
    "busy": "The line is busy or the call was declined.",
    "no-answer": "No answer.",
    "failed": "The call could not be connected.",
    "canceled": "The call was canceled.",
    "completed": (
        "The call ended before its audio reached deepeval. Check that "
        "public_url forwards to this machine."
    ),
}


@dataclass
class _MediaStream:
    ws: web.WebSocketResponse
    stream_sid: str
    released: asyncio.Future


class _MediaStreamServer:

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._pending: Dict[str, asyncio.Future] = {}
        self._starting: Optional[asyncio.Task] = None
        self._users = 0

    async def acquire(self) -> None:
        if self._starting is None:
            self._starting = asyncio.ensure_future(self._start())
        starting = self._starting
        try:
            await asyncio.shield(starting)
        except BaseException:
            if self._starting is starting and starting.done():
                self._starting = None
            raise
        self._users += 1

    async def _start(self) -> web.AppRunner:
        app = web.Application()
        app.router.add_get(STREAM_PATH, self._handle)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            await web.TCPSite(runner, self.host, self.port).start()
        except OSError as error:
            await runner.cleanup()
            raise DeepEvalError(
                f"deepeval could not listen on {self.host}:"
                f"{self.port}: {error}"
            ) from error
        return runner

    async def release(self) -> None:
        self._users = max(0, self._users - 1)
        if self._users or self._starting is None:
            return
        starting, self._starting = self._starting, None
        await (await starting).cleanup()

    def expect(self, call_key: str) -> asyncio.Future:
        future = asyncio.get_running_loop().create_future()
        self._pending[call_key] = future
        return future

    def forget(self, call_key: str) -> None:
        self._pending.pop(call_key, None)

    async def _handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            start = await asyncio.wait_for(
                self._read_start(ws), timeout=STREAM_START_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            start = None
        parameters = (start or {}).get("customParameters") or {}
        future = self._pending.pop(parameters.get(CALL_KEY_PARAMETER), None)
        if future is None or future.done():
            await ws.close()
            return ws
        released = asyncio.get_running_loop().create_future()
        future.set_result(
            _MediaStream(
                ws=ws, stream_sid=start.get("streamSid"), released=released
            )
        )
        await released
        return ws

    @staticmethod
    async def _read_start(ws: web.WebSocketResponse) -> Optional[dict]:
        async for message in ws:
            if message.type != aiohttp.WSMsgType.TEXT:
                return None
            data = json.loads(message.data)
            if data.get("event") == "start":
                return data.get("start") or {}
        return None


def stream_url_for(public_url: str) -> str:
    parsed = urlparse(public_url or "")
    if parsed.scheme not in ("https", "wss") or not parsed.netloc:
        raise DeepEvalError(
            f"'{public_url}' is not a public https:// or wss:// URL. Twilio "
            "streams call audio to it, so expose TwilioProvider's port "
            "with a tunnel such as ngrok and pass that URL."
        )
    return f"wss://{parsed.netloc}{parsed.path.rstrip('/')}{STREAM_PATH}"


class TwilioConnector(
    PhoneConnector, SilenceTimeoutMixin, BaseWebSocketConnector
):

    provider_type: ClassVar[type] = TwilioProvider
    _dial_pacer: ClassVar[DialPacer] = DialPacer()
    _servers: ClassVar[Dict[Tuple[str, int], _MediaStreamServer]] = {}

    def __init__(
        self,
        phone_number: str,
        *,
        provider: TwilioProvider,
        caller_id: Optional[str] = None,
        ring_timeout_s: int = DEFAULT_RING_TIMEOUT_S,
        max_call_duration_s: int = DEFAULT_MAX_CALL_DURATION_S,
        dial_interval_s: float = DEFAULT_DIAL_INTERVAL_S,
        **base_kwargs: Any,
    ):
        if not is_phone_number(phone_number):
            raise DeepEvalError(
                f"'{phone_number}' is not an E.164 phone number. Enter it "
                "with the country code, e.g. +14155550100."
            )
        if not caller_id:
            raise DeepEvalError(
                "PhoneConnector needs `caller_id` with TwilioProvider: a "
                "number on your Twilio account to call from."
            )
        account_sid = provider.account_sid or os.getenv(ACCOUNT_SID_ENV)
        auth_token = provider.auth_token or os.getenv(AUTH_TOKEN_ENV)
        if not (account_sid and auth_token):
            raise DeepEvalError(
                "TwilioProvider needs a Twilio account SID and auth token "
                f"(pass account_sid/auth_token or set {ACCOUNT_SID_ENV}, "
                f"{AUTH_TOKEN_ENV})."
            )
        base_kwargs.setdefault("turn_detection", "patient")
        base_kwargs.setdefault("connect_timeout_s", float(ring_timeout_s + 15))
        super().__init__(sample_rate=TWILIO_SAMPLE_RATE, **base_kwargs)
        self.phone_number = phone_number
        self.caller_id = caller_id
        self.stream_url = stream_url_for(provider.public_url)
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.host = provider.host
        self.port = provider.port
        self.ring_timeout_s = ring_timeout_s
        self.max_call_duration_s = max_call_duration_s
        self.dial_interval_s = dial_interval_s
        self.call_sid: Optional[str] = None
        self._call_key: Optional[str] = None
        self._stream: Optional[_MediaStream] = None
        self._server: Optional[_MediaStreamServer] = None
        self._call_ended = False

    @property
    def supports_duplex(self) -> bool:
        return True

    @property
    def call_ended(self) -> bool:
        return self._call_ended or super().call_ended

    def _ready_on_connect(self) -> bool:
        return True

    def _media_server(self) -> _MediaStreamServer:
        key = (self.host, self.port)
        server = self._servers.get(key)
        if server is None:
            server = _MediaStreamServer(self.host, self.port)
            self._servers[key] = server
        return server

    def _twiml(self) -> str:
        return (
            "<Response><Connect>"
            f"<Stream url={quoteattr(self.stream_url)}>"
            f"<Parameter name={quoteattr(CALL_KEY_PARAMETER)} "
            f"value={quoteattr(self._call_key)}/>"
            "</Stream></Connect></Response>"
        )

    async def _twilio(
        self, method: str, path: str, data: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        url = f"{TWILIO_API_BASE}/Accounts/{self.account_sid}{path}"
        auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)
        async with aiohttp.ClientSession(auth=auth) as session:
            async with session.request(method, url, data=data) as response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    raise DeepEvalError(
                        (body or {}).get("message")
                        or f"Twilio answered {response.status}."
                    )
                return body or {}

    async def _open_session(self) -> str:
        try:
            call = await self._twilio(
                "POST",
                "/Calls.json",
                {
                    "To": self.phone_number,
                    "From": self.caller_id,
                    "Twiml": self._twiml(),
                    "Timeout": str(self.ring_timeout_s),
                    "TimeLimit": str(self.max_call_duration_s),
                },
            )
        except DeepEvalError as error:
            raise DeepEvalError(
                f"Twilio could not call {self.phone_number}: {error}"
            ) from error
        self.call_sid = call.get("sid")
        return self.call_sid

    async def _call_status(self) -> Optional[str]:
        call = await self._twilio("GET", f"/Calls/{self.call_sid}.json")
        return call.get("status")

    async def _await_stream(self, arrival: asyncio.Future) -> _MediaStream:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.connect_timeout_s
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise DeepEvalError(
                    f"{self.phone_number} did not answer within "
                    f"{self.connect_timeout_s}s."
                )
            try:
                return await asyncio.wait_for(
                    asyncio.shield(arrival),
                    timeout=min(STATUS_POLL_INTERVAL_S, remaining),
                )
            except asyncio.TimeoutError:
                pass
            status = await self._call_status()
            if status in FAILED_CALL_STATUSES:
                raise DeepEvalError(
                    f"Call to {self.phone_number} failed: "
                    f"{FAILED_CALL_STATUSES[status]}"
                )

    async def _open_socket(self) -> web.WebSocketResponse:
        self._call_ended = False
        self._call_key = uuid.uuid4().hex
        server = self._media_server()
        await server.acquire()
        self._server = server
        arrival = server.expect(self._call_key)
        try:
            await self._dial_pacer.wait(self.dial_interval_s)
            await self._open_session()
            self._stream = await self._await_stream(arrival)
        except BaseException:
            arrival.cancel()
            await self.disconnect()
            raise
        return self._stream.ws

    def _encode_outbound(self, pcm: bytes) -> str:
        payload = base64.b64encode(audio_utils.pcm16_to_ulaw(pcm))
        return json.dumps(
            {
                "event": "media",
                "streamSid": self._stream.stream_sid,
                "media": {"payload": payload.decode("ascii")},
            }
        )

    def _decode_inbound(self, raw: Union[str, bytes]) -> Optional[InboundEvent]:
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return None
        event = message.get("event")
        if event == "media":
            media = message.get("media") or {}
            if media.get("track", "inbound") != "inbound":
                return None
            payload = base64.b64decode(media.get("payload") or "")
            return InboundEvent(audio=audio_utils.ulaw_to_pcm16(payload))
        if event == "stop":
            self._call_ended = True
            return InboundEvent(turn_complete=True)
        return None

    async def _reader_loop(self) -> None:
        try:
            await super()._reader_loop()
        finally:
            self._call_ended = True
            self._inbound.put_nowait(
                AgentEvent(turn_complete=True, call_ended=True)
            )

    async def disconnect(self) -> None:
        await super().disconnect()
        stream, self._stream = self._stream, None
        if stream is not None and not stream.released.done():
            stream.released.set_result(None)
        call_sid, self.call_sid = self.call_sid, None
        if call_sid:
            await self._hang_up(call_sid)
        server, self._server = self._server, None
        if server is not None:
            if self._call_key:
                server.forget(self._call_key)
            await server.release()

    async def _hang_up(self, call_sid: str) -> None:
        try:
            await self._twilio(
                "POST", f"/Calls/{call_sid}.json", {"Status": "completed"}
            )
        except Exception:
            logger.debug("Could not hang up Twilio call %s.", call_sid)

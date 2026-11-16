import asyncio
import base64
import json
import re
import socket
from typing import Dict, List, Optional

import aiohttp
import pytest
from aiohttp import web

from deepeval.errors import DeepEvalError
from deepeval.voice import PhoneConnector, TwilioProvider
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.providers.twilio import (
    connector as twilio_module,
)
from deepeval.voice.protocol import VoiceProtocol

AGENT_PCM = b"\xe8\x03" * 160
CALLER_PCM = b"\x18\xfc" * 160


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakeTwilio:
    def __init__(self, media_port: int):
        self.media_port = media_port
        self.calls: List[Dict[str, str]] = []
        self.hangups: List[str] = []
        self.status: Dict[str, str] = {}
        self.answer = True
        self.place_error: Optional[str] = None
        self.received: Dict[str, List[bytes]] = {}
        self.streams: Dict[str, aiohttp.ClientWebSocketResponse] = {}
        self._tasks = []
        self._runner = None

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post(
            "/2010-04-01/Accounts/{account}/Calls.json", self._place
        )
        app.router.add_get(
            "/2010-04-01/Accounts/{account}/Calls/{sid}.json", self._get
        )
        app.router.add_post(
            "/2010-04-01/Accounts/{account}/Calls/{sid}.json", self._update
        )
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        port = _free_port()
        await web.TCPSite(self._runner, "127.0.0.1", port).start()
        return f"http://127.0.0.1:{port}/2010-04-01"

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._runner.cleanup()

    async def _place(self, request: web.Request) -> web.Response:
        form = dict(await request.post())
        if self.place_error:
            return web.json_response({"message": self.place_error}, status=400)
        sid = f"CA{len(self.calls)}"
        self.calls.append(form)
        self.status[sid] = "ringing"
        if self.answer:
            key = re.search(r'value="([0-9a-f]+)"', form["Twiml"]).group(1)
            self._tasks.append(asyncio.create_task(self._stream(sid, key)))
        return web.json_response({"sid": sid})

    async def _get(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"status": self.status[request.match_info["sid"]]}
        )

    async def _update(self, request: web.Request) -> web.Response:
        sid = request.match_info["sid"]
        form = dict(await request.post())
        if form.get("Status") == "completed":
            self.hangups.append(sid)
            self.status[sid] = "completed"
        return web.json_response({"sid": sid})

    async def _stream(self, sid: str, key: str) -> None:
        self.status[sid] = "in-progress"
        session = aiohttp.ClientSession()
        try:
            ws = await session.ws_connect(
                f"ws://127.0.0.1:{self.media_port}"
                f"{twilio_module.STREAM_PATH}"
            )
            self.streams[sid] = ws
            stream_sid = f"MZ{sid}"
            await ws.send_str(json.dumps({"event": "connected"}))
            await ws.send_str(
                json.dumps(
                    {
                        "event": "start",
                        "streamSid": stream_sid,
                        "start": {
                            "streamSid": stream_sid,
                            "callSid": sid,
                            "customParameters": {"deepeval_call": key},
                        },
                    }
                )
            )
            payload = base64.b64encode(audio_utils.pcm16_to_ulaw(AGENT_PCM))
            for track in ("outbound", "inbound"):
                await ws.send_str(
                    json.dumps(
                        {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "track": track,
                                "payload": payload.decode(),
                            },
                        }
                    )
                )
            async for message in ws:
                data = json.loads(message.data)
                if data.get("event") == "media":
                    assert data["streamSid"] == stream_sid
                    self.received.setdefault(sid, []).append(
                        base64.b64decode(data["media"]["payload"])
                    )
        finally:
            await session.close()

    async def hang_up_from_agent(self, sid: str) -> None:
        ws = self.streams[sid]
        await ws.send_str(json.dumps({"event": "stop"}))
        await ws.close()


@pytest.fixture
async def twilio(monkeypatch):
    media_port = _free_port()
    fake = FakeTwilio(media_port)
    monkeypatch.setattr(twilio_module, "TWILIO_API_BASE", await fake.start())
    monkeypatch.setattr(twilio_module, "STATUS_POLL_INTERVAL_S", 0.05)
    yield fake
    await fake.stop()


def _connector(
    port: int, public_url: str = "https://deepeval-test.ngrok.app", **kwargs
) -> PhoneConnector:
    kwargs.setdefault("dial_interval_s", 0.0)
    kwargs.setdefault("caller_id", "+14155550199")
    return PhoneConnector(
        "+14155550100",
        provider=TwilioProvider(
            public_url=public_url,
            account_sid="AC_test",
            auth_token="token",
            host="127.0.0.1",
            port=port,
        ),
        **kwargs,
    )


async def _first_agent_audio(connector: PhoneConnector) -> bytes:
    async for event in connector.iter_agent_events():
        if event.audio:
            return event.audio
    return b""


def test_twilio_calls_are_telephony():
    connector = _connector(_free_port())

    assert isinstance(connector, twilio_module.TwilioConnector)
    assert connector.protocol is VoiceProtocol.SIP


def test_a_phone_number_must_be_e164():
    with pytest.raises(DeepEvalError, match="E.164"):
        PhoneConnector(
            "4155550100",
            provider=TwilioProvider(
                public_url="https://deepeval-test.ngrok.app",
                account_sid="AC_test",
                auth_token="token",
            ),
            caller_id="+14155550199",
        )


def test_twilio_needs_a_number_to_call_from():
    with pytest.raises(DeepEvalError, match="caller_id"):
        _connector(_free_port(), caller_id=None)


def test_twilio_credentials_are_required(monkeypatch):
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)

    with pytest.raises(DeepEvalError, match="TWILIO_ACCOUNT_SID"):
        PhoneConnector(
            "+14155550100",
            provider=TwilioProvider(
                public_url="https://deepeval-test.ngrok.app"
            ),
            caller_id="+14155550199",
        )


def test_twilio_credentials_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC_env")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "env-token")

    connector = PhoneConnector(
        "+14155550100",
        provider=TwilioProvider(public_url="https://deepeval-test.ngrok.app"),
        caller_id="+14155550199",
    )

    assert connector.account_sid == "AC_env"


@pytest.mark.parametrize(
    "public_url, stream_url",
    [
        (
            "https://deepeval-test.ngrok.app",
            "wss://deepeval-test.ngrok.app/deepeval/twilio/stream",
        ),
        (
            "wss://tunnel.example.com/voice/",
            "wss://tunnel.example.com/voice/deepeval/twilio/stream",
        ),
    ],
)
def test_twilio_streams_to_the_public_url(public_url, stream_url):
    connector = _connector(_free_port(), public_url=public_url)

    assert connector.stream_url == stream_url


@pytest.mark.parametrize(
    "public_url", ["http://insecure.example.com", "localhost:8765", ""]
)
def test_the_public_url_must_be_secure_and_absolute(public_url):
    with pytest.raises(DeepEvalError, match="public https://"):
        _connector(_free_port(), public_url=public_url)


@pytest.mark.asyncio
async def test_an_answered_call_streams_audio_both_ways(twilio):
    connector = _connector(twilio.media_port)

    await connector.connect()
    agent_audio = await asyncio.wait_for(
        _first_agent_audio(connector), timeout=2
    )
    await connector._send(connector._encode_outbound(CALLER_PCM))
    await asyncio.sleep(0.1)
    await connector.disconnect()

    [call] = twilio.calls
    assert call["To"] == "+14155550100"
    assert call["From"] == "+14155550199"
    assert call["Timeout"] == "45"
    assert "wss://deepeval-test.ngrok.app/deepeval/twilio/stream" in (
        call["Twiml"]
    )
    assert agent_audio == audio_utils.ulaw_to_pcm16(
        audio_utils.pcm16_to_ulaw(AGENT_PCM)
    )
    assert twilio.received["CA0"] == [audio_utils.pcm16_to_ulaw(CALLER_PCM)]
    assert twilio.hangups == ["CA0"]


@pytest.mark.asyncio
async def test_the_media_server_stops_once_every_call_hangs_up(twilio):
    connector = _connector(twilio.media_port)

    await connector.connect()
    await connector.disconnect()

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", twilio.media_port))


@pytest.mark.asyncio
async def test_parallel_calls_share_one_server_and_keep_their_own_audio(
    twilio,
):
    template = _connector(twilio.media_port)
    first, second = template.clone(), template.clone()

    await asyncio.gather(first.connect(), second.connect())
    first_sid = first.call_sid
    await first._send(first._encode_outbound(CALLER_PCM))
    await asyncio.sleep(0.1)
    await asyncio.gather(first.disconnect(), second.disconnect())

    assert len(twilio.calls) == 2
    assert list(twilio.received) == [first_sid]
    assert sorted(twilio.hangups) == ["CA0", "CA1"]


@pytest.mark.asyncio
async def test_the_call_ends_when_the_agent_hangs_up(twilio):
    connector = _connector(twilio.media_port)
    await connector.connect()
    await asyncio.wait_for(_first_agent_audio(connector), timeout=2)

    await twilio.hang_up_from_agent("CA0")
    events = await asyncio.wait_for(
        _collect(connector.iter_agent_events()), timeout=2
    )

    assert connector.call_ended
    assert all(not event.audio for event in events)
    await connector.disconnect()


async def _collect(events):
    return [event async for event in events]


@pytest.mark.parametrize(
    "status, message",
    [
        ("busy", "busy"),
        ("no-answer", "No answer"),
        ("completed", "public_url"),
    ],
)
@pytest.mark.asyncio
async def test_a_call_that_never_streams_explains_why(twilio, status, message):
    twilio.answer = False
    connector = _connector(twilio.media_port)

    async def fail_soon():
        await asyncio.sleep(0.1)
        twilio.status["CA0"] = status

    asyncio.create_task(fail_soon())
    with pytest.raises(DeepEvalError, match=message):
        await connector.connect()

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", twilio.media_port))


@pytest.mark.asyncio
async def test_a_rejected_call_reports_twilios_reason(twilio):
    twilio.place_error = "The 'To' number is not a valid phone number."
    connector = _connector(twilio.media_port)

    with pytest.raises(DeepEvalError, match="not a valid phone number"):
        await connector.connect()

    assert twilio.calls == []


@pytest.mark.asyncio
async def test_an_unanswered_call_times_out(twilio):
    twilio.answer = False
    connector = _connector(twilio.media_port, connect_timeout_s=0.2)

    with pytest.raises(DeepEvalError, match="did not answer"):
        await connector.connect()


@pytest.mark.asyncio
async def test_an_unknown_stream_is_turned_away(twilio):
    connector = _connector(twilio.media_port)
    server = connector._media_server()
    await server.acquire()
    try:
        async with aiohttp.ClientSession() as session:
            ws = await session.ws_connect(
                f"ws://127.0.0.1:{twilio.media_port}"
                f"{twilio_module.STREAM_PATH}"
            )
            await ws.send_str(
                json.dumps(
                    {
                        "event": "start",
                        "start": {"customParameters": {"deepeval_call": "x"}},
                    }
                )
            )
            message = await asyncio.wait_for(ws.receive(), timeout=2)
            assert message.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
            )
    finally:
        await server.release()

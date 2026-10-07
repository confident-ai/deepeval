import asyncio
import json
import sys
import types

import pytest
from aiohttp import WSMsgType

from deepeval.errors import DeepEvalError
from deepeval.voice import WebRTCConnector
from deepeval.voice.connectors.utils import validate_connector
from deepeval.voice.connectors.transports import webrtc
from deepeval.voice.connectors.transports.webrtc import (
    _PcmSource,
    _WebSocketSignaling,
    candidate_init,
    extract_text,
    parse_answer_sdp,
    signaling_kind,
)
from tests.test_core.test_voice.helpers import wav_audio

OFFER_URL = "https://agent.example.com/offer"
ANSWER_SDP = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\n"


class FakePlane:
    def __init__(self, size: int):
        self.data = bytearray(size)

    def update(self, pcm: bytes) -> None:
        self.data[:] = pcm

    def __bytes__(self) -> bytes:
        return bytes(self.data)


class FakeAudioFrame:
    def __init__(self, format=None, layout=None, samples=0):
        self.samples = samples
        self.planes = [FakePlane(samples * 2)]
        self.sample_rate = None
        self.pts = None
        self.time_base = None


class FakeResampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def resample(self, frame):
        return [frame]


def fake_av():
    return types.SimpleNamespace(
        AudioFrame=FakeAudioFrame, AudioResampler=FakeResampler
    )


class FakeTrack:
    kind = "audio"

    def __init__(self, frames):
        self._frames = list(frames)

    async def recv(self):
        if not self._frames:
            raise RuntimeError("track ended")
        await asyncio.sleep(0)
        return self._frames.pop(0)


class FakeChannel:
    def __init__(self, label: str):
        self.label = label
        self.readyState = "connecting"
        self.handlers = {}
        self.sent = []

    def on(self, event, handler):
        self.handlers[event] = handler
        return handler

    def send(self, message):
        self.sent.append(message)

    def open(self):
        self.readyState = "open"
        self.handlers["open"]()


class FakePeerConnection:
    instances = []
    agent_frames = []
    connects = True

    def __init__(self, configuration=None):
        self.configuration = configuration
        self.tracks = []
        self.handlers = {}
        self.channels = []
        self.localDescription = None
        self.remoteDescription = None
        self.connectionState = "new"
        self.closed = False
        self.candidates = []
        FakePeerConnection.instances.append(self)

    def addTrack(self, track):
        self.tracks.append(track)

    def on(self, event, handler):
        self.handlers[event] = handler
        return handler

    def createDataChannel(self, label):
        channel = FakeChannel(label)
        self.channels.append(channel)
        return channel

    async def createOffer(self):
        return types.SimpleNamespace(sdp="v=0 offer", type="offer")

    async def setLocalDescription(self, description):
        self.localDescription = description

    async def setRemoteDescription(self, description):
        self.remoteDescription = description
        if self.connects:
            asyncio.get_event_loop().call_soon(self._establish)

    def _establish(self):
        self.connectionState = "connected"
        self.handlers["connectionstatechange"]()
        self.handlers["track"](FakeTrack(type(self).agent_frames))
        for channel in self.channels:
            channel.open()

    async def addIceCandidate(self, candidate):
        self.candidates.append(candidate)

    async def close(self):
        self.closed = True


class FakeResponse:
    def __init__(self, status=200, body=ANSWER_SDP):
        self.status = status
        self._body = body

    async def text(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


class FakeSession:
    def __init__(self):
        self.posts = []
        self.response = FakeResponse()
        self.closed = False

    def post(self, url, data=None, headers=None):
        self.posts.append((url, data, headers))
        return self.response

    async def close(self):
        self.closed = True


def agent_frame(pcm: bytes) -> FakeAudioFrame:
    frame = FakeAudioFrame(samples=len(pcm) // 2)
    frame.planes[0].update(pcm)
    return frame


@pytest.fixture
def fake_stack(monkeypatch):
    FakePeerConnection.instances = []
    aiortc = types.SimpleNamespace(
        RTCPeerConnection=FakePeerConnection,
        RTCSessionDescription=lambda sdp, type: types.SimpleNamespace(
            sdp=sdp, type=type
        ),
        RTCConfiguration=lambda iceServers: types.SimpleNamespace(
            iceServers=iceServers
        ),
        RTCIceServer=lambda urls: types.SimpleNamespace(urls=urls),
        MediaStreamTrack=object,
    )
    monkeypatch.setitem(sys.modules, "aiortc", aiortc)
    monkeypatch.setitem(sys.modules, "av", fake_av())
    session = FakeSession()
    monkeypatch.setattr(
        webrtc, "aiohttp", types.SimpleNamespace(ClientSession=lambda: session)
    )
    return session


def test_signaling_kind_follows_the_url_scheme():
    assert signaling_kind("https://agent.example.com/offer") == "http"
    assert signaling_kind("http://localhost:8080/offer") == "http"
    assert signaling_kind("wss://agent.example.com/signal") == "websocket"
    with pytest.raises(DeepEvalError, match="signaling URL"):
        signaling_kind("ftp://agent.example.com")


def test_answer_sdp_is_read_raw_or_from_json():
    assert parse_answer_sdp(ANSWER_SDP) == ANSWER_SDP
    assert (
        parse_answer_sdp(json.dumps({"type": "answer", "sdp": ANSWER_SDP}))
        == ANSWER_SDP
    )
    assert (
        parse_answer_sdp(json.dumps({"answer": {"sdp": ANSWER_SDP}}))
        == ANSWER_SDP
    )
    with pytest.raises(DeepEvalError, match="SDP answer"):
        parse_answer_sdp('{"status": "ok"}')


def test_data_channel_text_is_read_from_strings_json_and_bytes():
    assert extract_text("Welcome.") == "Welcome."
    assert extract_text('{"transcript": "Hi there"}') == "Hi there"
    assert extract_text('{"text": "Typed"}') == "Typed"
    assert extract_text(b"Bytes too") == "Bytes too"
    assert extract_text('{"status": "thinking"}') is None
    assert extract_text("   ") is None


def test_candidates_are_accepted_flat_or_nested():
    flat = {
        "type": "candidate",
        "candidate": "candidate:1 1 udp 1 1.2.3.4 5 typ host",
        "sdpMid": "0",
    }
    nested = {
        "type": "candidate",
        "candidate": {
            "candidate": "candidate:1 1 udp 1 1.2.3.4 5 typ host",
            "sdpMLineIndex": 0,
        },
    }

    assert candidate_init(flat)["sdpMid"] == "0"
    assert candidate_init(nested)["sdpMLineIndex"] == 0
    assert candidate_init({"type": "candidate", "candidate": ""}) is None


async def test_pcm_source_fills_silence_and_keeps_time():
    source = _PcmSource(48000, fake_av())
    source.push(b"\x01\x00" * 960)

    first = await source.next_frame()
    second = await source.next_frame()

    assert bytes(first.planes[0]) == b"\x01\x00" * 960
    assert bytes(second.planes[0]) == b"\x00\x00" * 960
    assert (first.pts, second.pts) == (0, 960)
    assert second.sample_rate == 48000


def test_connector_declares_the_webrtc_protocol_and_duplex():
    connector = WebRTCConnector(OFFER_URL)

    assert connector.protocol.value == "webrtc"
    assert connector.supports_duplex is True
    assert connector.signals_turn_complete is False
    assert connector.recv_sample_rate == 48000
    assert connector.audio_format == (24000, "wav")


def test_connector_rejects_unknown_signaling_schemes():
    with pytest.raises(DeepEvalError, match="signaling URL"):
        WebRTCConnector("ftp://agent.example.com")


async def test_connect_posts_the_offer_and_waits_for_the_agent(fake_stack):
    session = fake_stack
    FakePeerConnection.agent_frames = []
    connector = WebRTCConnector(
        OFFER_URL, headers={"Authorization": "Bearer t"}, connect_timeout_s=1
    )
    FakePeerConnection.instances.clear()

    await connector.connect()

    pc = FakePeerConnection.instances[0]
    url, body, headers = session.posts[0]
    assert url == OFFER_URL
    assert body == "v=0 offer"
    assert headers["Authorization"] == "Bearer t"
    assert headers["Content-Type"] == "application/sdp"
    assert pc.remoteDescription.sdp == ANSWER_SDP
    assert pc.remoteDescription.type == "answer"
    assert pc.channels[0].label == "text"
    assert pc.configuration is None
    await connector.disconnect()
    assert pc.closed and session.closed


async def test_agent_audio_and_text_arrive_as_events(fake_stack):
    FakePeerConnection.agent_frames = [agent_frame(b"\x02\x00" * 480)]
    connector = WebRTCConnector(OFFER_URL, connect_timeout_s=1)
    await connector.connect()
    pc = FakePeerConnection.instances[-1]
    pc.channels[0].handlers["message"](json.dumps({"transcript": "Hello"}))

    events = []
    async for event in connector.iter_agent_events():
        events.append(event)
        if event.turn_complete:
            break

    audio = [event.audio for event in events if event.audio]
    assert audio == [b"\x02\x00" * 480]
    assert [event.transcript for event in events if event.transcript] == [
        "Hello"
    ]
    await connector.disconnect()


async def test_connect_fails_when_no_agent_track_arrives(fake_stack):
    connector = WebRTCConnector(OFFER_URL, connect_timeout_s=0.05)
    FakePeerConnection.connects = False
    try:
        with pytest.raises(DeepEvalError, match="did not reach 'connected'"):
            await connector.connect()
    finally:
        FakePeerConnection.connects = True

    assert FakePeerConnection.instances[-1].closed


async def test_signaling_errors_are_reported_with_the_status(fake_stack):
    fake_stack.response = FakeResponse(status=401, body="nope")
    connector = WebRTCConnector(OFFER_URL, connect_timeout_s=1)

    with pytest.raises(DeepEvalError, match="HTTP 401"):
        await connector.connect()

    assert FakePeerConnection.instances[-1].closed


async def test_ice_servers_none_keeps_aiortc_defaults_and_empty_disables_them(
    fake_stack,
):
    FakePeerConnection.agent_frames = []
    default = WebRTCConnector(OFFER_URL, connect_timeout_s=1)
    await default.connect()
    none_config = FakePeerConnection.instances[-1].configuration
    await default.disconnect()

    explicit = WebRTCConnector(
        OFFER_URL, ice_servers=["stun:stun.example.com"], connect_timeout_s=1
    )
    await explicit.connect()
    servers = FakePeerConnection.instances[-1].configuration.iceServers
    await explicit.disconnect()

    empty = WebRTCConnector(OFFER_URL, ice_servers=[], connect_timeout_s=1)
    await empty.connect()
    empty_config = FakePeerConnection.instances[-1].configuration
    await empty.disconnect()

    assert none_config is None
    assert [server.urls for server in servers] == ["stun:stun.example.com"]
    assert empty_config.iceServers == []


async def test_stream_uplink_paces_frames_into_the_local_track(fake_stack):
    FakePeerConnection.agent_frames = []
    connector = WebRTCConnector(OFFER_URL, connect_timeout_s=1)
    await connector.connect()

    await connector.stream_uplink(wav_audio(0.1), trailing_silence=False)

    frames = []
    while not connector._source._frames.empty():
        frames.append(connector._source._frames.get_nowait())
    assert len(frames) == 5
    assert all(len(frame) == 960 * 2 for frame in frames)
    await connector.disconnect()


async def test_send_text_needs_an_open_channel(fake_stack):
    FakePeerConnection.agent_frames = []
    connector = WebRTCConnector(OFFER_URL, connect_timeout_s=1)
    with pytest.raises(DeepEvalError, match="no open data channel"):
        connector.send_text("hi")

    await connector.connect()
    connector.send_text("I am ready.")

    assert FakePeerConnection.instances[-1].channels[0].sent == ["I am ready."]
    await connector.disconnect()


class FakeWebSocket:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []
        self.closed = False

    async def send_json(self, payload):
        self.sent.append(payload)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def close(self):
        self.closed = True


def text_message(payload) -> types.SimpleNamespace:
    return types.SimpleNamespace(type=WSMsgType.TEXT, data=json.dumps(payload))


async def test_websocket_signaling_buffers_candidates_until_the_answer(
    monkeypatch,
):
    candidate = {
        "type": "candidate",
        "candidate": "candidate:1 1 udp 2130706431 127.0.0.1 5000 typ host",
        "sdpMid": "0",
        "sdpMLineIndex": 0,
    }
    ws = FakeWebSocket(
        [
            text_message(candidate),
            text_message({"type": "answer", "sdp": ANSWER_SDP}),
        ]
    )

    class Session:
        async def ws_connect(self, url, headers=None):
            return ws

    parsed = []
    sdp_module = types.SimpleNamespace(
        candidate_from_sdp=lambda raw: parsed.append(raw)
        or types.SimpleNamespace(raw=raw)
    )
    monkeypatch.setitem(sys.modules, "aiortc.sdp", sdp_module)
    pc = FakePeerConnection()
    signaling = _WebSocketSignaling(
        Session(), "wss://agent.example.com/signal", {}
    )

    answer = await signaling.exchange("v=0 offer")
    await signaling.start(pc, None)
    await signaling.close()

    assert answer == ANSWER_SDP
    assert ws.sent == [{"type": "offer", "sdp": "v=0 offer"}]
    assert parsed == ["1 1 udp 2130706431 127.0.0.1 5000 typ host"]
    assert pc.candidates[0].sdpMid == "0"
    assert pc.candidates[0].sdpMLineIndex == 0
    assert ws.closed


def test_voice_config_redirects_a_bare_peer_connection():
    class RTCPeerConnection:
        pass

    RTCPeerConnection.__module__ = "aiortc.rtcpeerconnection"

    with pytest.raises(DeepEvalError, match="WebRTCConnector"):
        validate_connector(RTCPeerConnection())

from types import SimpleNamespace

from deepeval.test_case import Audio
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.types import ConnectorTurn

RATE = 24000


def wav_audio(seconds: float = 0.02) -> Audio:
    pcm = b"\xe8\x03" * int(RATE * seconds)
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(pcm, sample_rate=RATE),
        "audio/wav",
        sampleRate=RATE,
        encoding="wav",
        duration=seconds,
    )


class EchoAgent:
    async def __call__(self, audio: Audio) -> ConnectorTurn:
        return ConnectorTurn(audio=wav_audio(), transcript="Agent reply")


class StubTTS:
    sample_rate = RATE

    async def a_synthesize(self, text: str, **kwargs):
        return wav_audio(), None

    def supports_streaming(self) -> bool:
        return False


class StubSTT:
    def supports_streaming(self) -> bool:
        return False

    truncated_audio_pad_seconds = 0.0

    async def a_transcribe(self, audio, **kwargs):
        return "Agent reply", None


class FakeLocalParticipant:
    def __init__(self):
        self.published = []
        self.unpublished = []

    async def publish_track(self, track, options):
        self.published.append(track)
        return SimpleNamespace(sid=f"TR_{len(self.published)}")

    async def unpublish_track(self, sid):
        self.unpublished.append(sid)


class FakeRoom:
    def __init__(self, connected: bool = False, name: str = ""):
        self.connected = connected
        self.name = name
        self.handlers = {}
        self.text_handlers = {}
        self.local_participant = FakeLocalParticipant()
        self.remote_participants = {}
        self.connected_with = None

    def on(self, event, callback):
        self.handlers[event] = callback

    def isconnected(self) -> bool:
        return self.connected

    async def connect(self, url, token, options):
        self.connected = True
        self.connected_with = (url, token)

    async def disconnect(self):
        self.connected = False

    def register_text_stream_handler(self, topic, handler):
        self.text_handlers[topic] = handler


class _TokenBuilder:
    def __init__(self, api_key, api_secret):
        self.claims = {"api_key": api_key}

    def __getattr__(self, name):
        def setter(value):
            self.claims[name] = value
            return self

        return setter

    def to_jwt(self):
        return f"jwt-for-{self.claims.get('with_identity')}"


class FakeLiveKitAPIClient:
    def __init__(self, calls, dial_error=None):
        self.calls = calls
        self.closed = False
        self.sip = SimpleNamespace(create_sip_participant=self._dial)
        self.room = SimpleNamespace(delete_room=self._delete_room)
        self._dial_error = dial_error

    async def _dial(self, request):
        self.calls.append(("dial", request))
        if self._dial_error is not None:
            raise self._dial_error

    async def _delete_room(self, request):
        self.calls.append(("delete_room", request))

    async def aclose(self):
        self.closed = True


def fake_livekit(dial_error=None):
    calls = []
    rtc = SimpleNamespace(
        Room=FakeRoom,
        RoomOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        AudioSource=lambda rate, channels: SimpleNamespace(rate=rate),
        LocalAudioTrack=SimpleNamespace(
            create_audio_track=lambda name, source: SimpleNamespace(name=name)
        ),
        TrackPublishOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        TrackSource=SimpleNamespace(SOURCE_MICROPHONE=2),
        TrackKind=SimpleNamespace(KIND_AUDIO=1, KIND_VIDEO=2),
    )
    api = SimpleNamespace(
        VideoGrants=lambda **kwargs: SimpleNamespace(**kwargs),
        AccessToken=_TokenBuilder,
        RoomConfiguration=lambda **kwargs: SimpleNamespace(**kwargs),
        RoomAgentDispatch=lambda **kwargs: SimpleNamespace(**kwargs),
        LiveKitAPI=lambda url, key, secret: FakeLiveKitAPIClient(
            calls, dial_error
        ),
        CreateSIPParticipantRequest=lambda **kwargs: SimpleNamespace(**kwargs),
        SIPOutboundConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        DeleteRoomRequest=lambda **kwargs: SimpleNamespace(**kwargs),
        SIPTransport=SimpleNamespace(
            SIP_TRANSPORT_AUTO=0,
            SIP_TRANSPORT_UDP=1,
            SIP_TRANSPORT_TCP=2,
            SIP_TRANSPORT_TLS=3,
        ),
        SIPMediaEncryption=SimpleNamespace(
            SIP_MEDIA_ENCRYPT_DISABLE=0,
            SIP_MEDIA_ENCRYPT_ALLOW=1,
            SIP_MEDIA_ENCRYPT_REQUIRE=2,
        ),
    )
    modules = {"livekit.rtc": rtc, "livekit.api": api}
    return modules, calls


def install_fake_livekit(monkeypatch, module, dial_error=None):
    modules, calls = fake_livekit(dial_error)
    monkeypatch.setattr(
        module,
        "require_dependency",
        lambda name, **kwargs: modules[name],
    )
    return modules, calls

import pytest

from deepeval.errors import DeepEvalError
from deepeval.voice import (
    SilenceTimeoutMixin,
    LiveKitPhoneConnector,
    LiveKitSIPConnector,
    SipMediaEncryption,
    SipTransport,
)
from deepeval.voice.connectors.providers.livekit import (
    connector as livekit_module,
)
from deepeval.voice.connectors.providers.livekit import sip as sip_module
from deepeval.voice.protocol import VoiceProtocol
from tests.test_core.test_voice.helpers import install_fake_livekit

CREDENTIALS = dict(url="wss://livekit.test", api_key="key", api_secret="secret")


@pytest.fixture
def livekit(monkeypatch):
    def install(dial_error=None):
        modules, calls = install_fake_livekit(
            monkeypatch, livekit_module, dial_error
        )
        monkeypatch.setattr(
            sip_module,
            "require_dependency",
            lambda name, **kwargs: modules[name],
        )
        return modules, calls

    return install


def _phone(**kwargs) -> LiveKitPhoneConnector:
    kwargs.setdefault("trunk_id", "ST_trunk")
    kwargs.setdefault("dial_interval_s", 0.0)
    return LiveKitPhoneConnector("+14155550100", **CREDENTIALS, **kwargs)


def _sip(uri="sip:agent@sip.example.com", **kwargs) -> LiveKitSIPConnector:
    kwargs.setdefault("dial_interval_s", 0.0)
    return LiveKitSIPConnector(uri, **CREDENTIALS, **kwargs)


async def _dial(connector):
    await connector._join_room()
    await connector._after_join()


def _requests(calls, kind):
    return [request for name, request in calls if name == kind]


def test_dial_connectors_speak_sip():
    assert LiveKitPhoneConnector.protocol is VoiceProtocol.SIP
    assert LiveKitSIPConnector.protocol is VoiceProtocol.SIP


@pytest.mark.parametrize("number", ["4155550100", "+0123", "+1 415 555 0100"])
def test_a_phone_number_must_be_e164(number):
    with pytest.raises(DeepEvalError, match="E.164"):
        LiveKitPhoneConnector(number, trunk_id="ST_trunk", **CREDENTIALS)


def test_a_phone_number_needs_an_outbound_trunk(monkeypatch):
    monkeypatch.delenv("LIVEKIT_SIP_OUTBOUND_TRUNK_ID", raising=False)

    with pytest.raises(DeepEvalError, match="outbound SIP trunk"):
        LiveKitPhoneConnector("+14155550100", **CREDENTIALS)


def test_the_outbound_trunk_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LIVEKIT_SIP_OUTBOUND_TRUNK_ID", "ST_env")

    assert LiveKitPhoneConnector("+14155550100", **CREDENTIALS).trunk_id == (
        "ST_env"
    )


def test_a_dial_connector_listens_to_the_callee_on_phone_timing():
    connector = _phone()

    assert connector.agent_identity == "callee"
    assert connector.turn_detection == "patient"
    assert connector.transcript_grace_s == 0.0
    assert connector.connect_timeout_s == 60.0


@pytest.mark.asyncio
async def test_a_phone_call_dials_through_the_trunk(livekit):
    _, calls = livekit()
    connector = _phone(
        caller_id="+14155550199",
        media_encryption=SipMediaEncryption.REQUIRED,
        ring_timeout_s=20,
        headers={"X-Test-Case": "refund-1"},
    )

    await _dial(connector)

    [request] = _requests(calls, "dial")
    assert request.sip_trunk_id == "ST_trunk"
    assert request.sip_call_to == "+14155550100"
    assert request.room_name == connector.room_name
    assert request.participant_identity == "callee"
    assert request.wait_until_answered is True
    assert request.sip_number == "+14155550199"
    assert request.media_encryption == 2
    assert request.ringing_timeout.seconds == 20
    assert request.max_call_duration.seconds == 1800
    assert request.headers == {"X-Test-Case": "refund-1"}
    assert not hasattr(request, "trunk")


@pytest.mark.asyncio
async def test_a_phone_call_can_dial_through_an_inline_trunk(livekit):
    _, calls = livekit()
    connector = _phone(
        trunk_id=None,
        trunk_uri="sip:outbound@trunk.example.com:5061",
        trunk_username="user",
        trunk_password="pass",
    )

    await _dial(connector)

    [request] = _requests(calls, "dial")
    assert request.sip_trunk_id == ""
    assert request.trunk.hostname == "trunk.example.com:5061"
    assert request.trunk.transport == 1
    assert request.trunk.auth_username == "user"
    assert request.trunk.auth_password == "pass"


def test_an_inline_trunk_must_be_a_sip_address():
    with pytest.raises(DeepEvalError, match="SIP address"):
        _phone(trunk_uri="trunk.example.com")


@pytest.mark.parametrize("uri", ["agent@example.com", "sip:example.com"])
def test_a_sip_address_must_be_valid(uri):
    with pytest.raises(DeepEvalError, match="SIP address"):
        _sip(uri)


@pytest.mark.asyncio
async def test_a_sip_call_dials_the_address_directly(livekit):
    _, calls = livekit()
    connector = _sip("sip:riley@sip.vapi.ai", username="user", password="pass")

    await _dial(connector)

    [request] = _requests(calls, "dial")
    assert request.sip_trunk_id == ""
    assert request.sip_call_to == "riley"
    assert request.sip_number == "deepeval"
    assert request.trunk.hostname == "sip.vapi.ai"
    assert request.trunk.auth_username == "user"
    assert request.trunk.auth_password == "pass"


@pytest.mark.parametrize(
    "uri, transport, expected",
    [
        ("sip:agent@example.com", SipTransport.TCP, 2),
        ("sip:agent@example.com;transport=tls", SipTransport.UDP, 3),
        ("sips:agent@example.com", SipTransport.UDP, 3),
    ],
)
@pytest.mark.asyncio
async def test_the_address_decides_the_sip_transport_first(
    livekit, uri, transport, expected
):
    _, calls = livekit()

    await _dial(_sip(uri, transport=transport))

    [request] = _requests(calls, "dial")
    assert request.trunk.transport == expected


@pytest.mark.asyncio
async def test_hanging_up_deletes_the_call_room_once(livekit):
    _, calls = livekit()
    connector = _phone()
    await _dial(connector)
    room_name = connector.room_name

    await connector.disconnect()
    await connector.disconnect()

    [request] = _requests(calls, "delete_room")
    assert request.room == room_name


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"sip_status_code": "486"}, "busy"),
        ({"sip_status_code": "480"}, "No answer"),
        ({"sip_status_code": "404"}, "could not be reached"),
        ({"sip_status": "Service Unavailable"}, "Service Unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_dial_explains_why_and_cleans_up(
    livekit, metadata, message
):
    error = Exception("twirp error")
    error.metadata = metadata
    _, calls = livekit(dial_error=error)
    connector = _phone()

    with pytest.raises(DeepEvalError, match=message):
        await _dial(connector)

    assert len(_requests(calls, "delete_room")) == 1


def test_clones_dial_into_rooms_of_their_own():
    connector = _phone()

    assert connector.supports_parallel_calls
    assert _phone(room_name="support").supports_parallel_calls is False


def test_a_dial_connector_can_hang_up_silent_calls():
    connector = _phone(silence_timeout_s=30)

    assert isinstance(connector, SilenceTimeoutMixin)
    assert connector.silence_timeout_s == 30

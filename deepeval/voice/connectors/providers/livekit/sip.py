import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Optional

from deepeval.errors import DeepEvalError
from deepeval.utils import require_dependency
from deepeval.voice.protocol import VoiceProtocol
from deepeval.voice.connectors.transports.silence import SilenceTimeoutMixin
from deepeval.voice.connectors.utils import DialPacer, is_phone_number
from deepeval.voice.connectors.providers.livekit.connector import (
    _INSTALL_HINT,
    LiveKitConnector,
)

logger = logging.getLogger(__name__)

CALLEE_IDENTITY = "callee"
DEFAULT_RING_TIMEOUT_S = 45
DEFAULT_MAX_CALL_DURATION_S = 30 * 60
DEFAULT_DIAL_INTERVAL_S = 1.0
DEFAULT_SIP_CALLER_ID = "deepeval"
OUTBOUND_TRUNK_ENV = "LIVEKIT_SIP_OUTBOUND_TRUNK_ID"

SIP_URI = re.compile(
    r"^(sips?):([^\s@:;]+)@([A-Za-z0-9.-]+)(?::(\d{1,5}))?"
    r"(?:;transport=(udp|tcp|tls))?$",
    re.IGNORECASE,
)

NO_ANSWER_MESSAGE = "No answer."
BUSY_MESSAGE = "The line is busy or the call was declined."
UNREACHABLE_MESSAGE = "The number could not be reached."
SIP_STATUS_MESSAGES = {
    404: UNREACHABLE_MESSAGE,
    408: NO_ANSWER_MESSAGE,
    480: NO_ANSWER_MESSAGE,
    484: UNREACHABLE_MESSAGE,
    486: BUSY_MESSAGE,
    603: BUSY_MESSAGE,
}


class SipTransport(Enum):
    UDP = "udp"
    TCP = "tcp"
    TLS = "tls"


class SipMediaEncryption(Enum):
    DISABLED = "disabled"
    ALLOWED = "allowed"
    REQUIRED = "required"


@dataclass(frozen=True)
class SipUri:
    user: str
    host: str
    port: Optional[int]
    transport: Optional[SipTransport]
    secure: bool

    @property
    def hostname(self) -> str:
        return f"{self.host}:{self.port}" if self.port else self.host


@dataclass(frozen=True)
class SipTrunk:
    hostname: str
    transport: SipTransport
    username: str
    password: str


def parse_sip_uri(value: Optional[str]) -> Optional[SipUri]:
    match = SIP_URI.match(value or "")
    if not match:
        return None
    scheme, user, host, port, transport = match.groups()
    return SipUri(
        user=user,
        host=host.lower(),
        port=int(port) if port else None,
        transport=SipTransport(transport.lower()) if transport else None,
        secure=scheme.lower() == "sips",
    )


def sip_trunk_for(
    uri: SipUri,
    transport: SipTransport,
    username: Optional[str],
    password: Optional[str],
) -> SipTrunk:
    if uri.transport is not None:
        transport = uri.transport
    elif uri.secure:
        transport = SipTransport.TLS
    return SipTrunk(
        hostname=uri.hostname,
        transport=transport,
        username=username or "",
        password=password or "",
    )


def describe_dial_error(error: Exception, address: str) -> str:
    metadata = getattr(error, "metadata", None) or {}
    status_code = metadata.get("sip_status_code")
    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None
    if status_code in SIP_STATUS_MESSAGES:
        return f"Call to {address} failed: {SIP_STATUS_MESSAGES[status_code]}"
    status = metadata.get("sip_status")
    if status:
        return f"Call to {address} failed: {status}."
    message = getattr(error, "message", None) or getattr(error, "msg", None)
    return f"Call to {address} failed: {message or error}"


class _LiveKitDialConnector(SilenceTimeoutMixin, LiveKitConnector):

    protocol: ClassVar[VoiceProtocol] = VoiceProtocol.SIP
    _dial_pacer: ClassVar[DialPacer] = DialPacer()

    def __init__(
        self,
        *,
        address: str,
        call_to: str,
        trunk_id: str = "",
        trunk: Optional[SipTrunk] = None,
        caller_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        media_encryption: SipMediaEncryption = SipMediaEncryption.DISABLED,
        ring_timeout_s: int = DEFAULT_RING_TIMEOUT_S,
        max_call_duration_s: int = DEFAULT_MAX_CALL_DURATION_S,
        dial_interval_s: float = DEFAULT_DIAL_INTERVAL_S,
        **kwargs: Any,
    ):
        kwargs.setdefault("agent_identity", CALLEE_IDENTITY)
        kwargs.setdefault("connect_timeout_s", float(ring_timeout_s + 15))
        kwargs.setdefault("turn_detection", "patient")
        kwargs.setdefault("transcript_grace_s", 0.0)
        super().__init__(**kwargs)
        self.address = address
        self.call_to = call_to
        self.trunk_id = trunk_id
        self.trunk = trunk
        self.caller_id = caller_id
        self.headers = dict(headers or {})
        self.media_encryption = SipMediaEncryption(media_encryption)
        self.ring_timeout_s = ring_timeout_s
        self.max_call_duration_s = max_call_duration_s
        self.dial_interval_s = dial_interval_s
        self._room_to_delete: Optional[str] = None

    def _livekit_api(self):
        return require_dependency(
            "livekit.api",
            provider_label=type(self).__name__,
            install_hint=_INSTALL_HINT,
        )

    def _dial_request(self, api):
        from google.protobuf.duration_pb2 import Duration

        encryption = api.SIPMediaEncryption
        fields = dict(
            sip_trunk_id=self.trunk_id,
            sip_call_to=self.call_to,
            room_name=self.room_name,
            participant_identity=self.agent_identity,
            participant_name=self.address,
            wait_until_answered=True,
            krisp_enabled=True,
            headers=dict(self.headers),
            ringing_timeout=Duration(seconds=self.ring_timeout_s),
            max_call_duration=Duration(seconds=self.max_call_duration_s),
            media_encryption={
                SipMediaEncryption.DISABLED: encryption.SIP_MEDIA_ENCRYPT_DISABLE,
                SipMediaEncryption.ALLOWED: encryption.SIP_MEDIA_ENCRYPT_ALLOW,
                SipMediaEncryption.REQUIRED: encryption.SIP_MEDIA_ENCRYPT_REQUIRE,
            }[self.media_encryption],
        )
        if self.caller_id:
            fields["sip_number"] = self.caller_id
        if self.trunk is not None:
            transports = {
                SipTransport.UDP: api.SIPTransport.SIP_TRANSPORT_UDP,
                SipTransport.TCP: api.SIPTransport.SIP_TRANSPORT_TCP,
                SipTransport.TLS: api.SIPTransport.SIP_TRANSPORT_TLS,
            }
            fields["trunk"] = api.SIPOutboundConfig(
                hostname=self.trunk.hostname,
                transport=transports[self.trunk.transport],
                auth_username=self.trunk.username,
                auth_password=self.trunk.password,
            )
        return api.CreateSIPParticipantRequest(**fields)

    async def _after_join(self) -> None:
        await self._dial_pacer.wait(self.dial_interval_s)
        api = self._livekit_api()
        self._room_to_delete = self.room_name
        client = api.LiveKitAPI(self.url, self.api_key, self.api_secret)
        try:
            await client.sip.create_sip_participant(self._dial_request(api))
        except Exception as error:
            await self.disconnect()
            raise DeepEvalError(
                describe_dial_error(error, self.address)
            ) from error
        finally:
            await client.aclose()

    async def disconnect(self) -> None:
        await super().disconnect()
        await self._hang_up()

    async def _hang_up(self) -> None:
        room, self._room_to_delete = self._room_to_delete, None
        if room is None:
            return
        api = self._livekit_api()
        client = api.LiveKitAPI(self.url, self.api_key, self.api_secret)
        try:
            await client.room.delete_room(api.DeleteRoomRequest(room=room))
        except Exception:
            logger.warning("Could not delete the call room %s.", room)
        finally:
            await client.aclose()


class LiveKitPhoneConnector(_LiveKitDialConnector):

    def __init__(
        self,
        phone_number: str,
        *,
        trunk_id: Optional[str] = None,
        trunk_uri: Optional[str] = None,
        trunk_username: Optional[str] = None,
        trunk_password: Optional[str] = None,
        caller_id: Optional[str] = None,
        **kwargs: Any,
    ):
        if not is_phone_number(phone_number):
            raise DeepEvalError(
                f"'{phone_number}' is not an E.164 phone number. Enter it "
                "with the country code, e.g. +14155550100."
            )
        trunk = None
        if trunk_uri is not None:
            uri = parse_sip_uri(trunk_uri)
            if uri is None:
                raise DeepEvalError(
                    f"'{trunk_uri}' is not a SIP address. Enter your trunk's "
                    "address, e.g. sip:outbound@trunk.example.com."
                )
            trunk = sip_trunk_for(
                uri, SipTransport.UDP, trunk_username, trunk_password
            )
            trunk_id = ""
        else:
            trunk_id = trunk_id or os.getenv(OUTBOUND_TRUNK_ENV)
            if not trunk_id:
                raise DeepEvalError(
                    "LiveKitPhoneConnector needs an outbound SIP trunk to "
                    f"call a phone number: pass trunk_id, set "
                    f"{OUTBOUND_TRUNK_ENV}, or pass trunk_uri."
                )
        super().__init__(
            address=phone_number,
            call_to=phone_number,
            trunk_id=trunk_id,
            trunk=trunk,
            caller_id=caller_id,
            **kwargs,
        )


class LiveKitSIPConnector(_LiveKitDialConnector):

    def __init__(
        self,
        sip_uri: str,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        transport: SipTransport = SipTransport.UDP,
        caller_id: Optional[str] = DEFAULT_SIP_CALLER_ID,
        **kwargs: Any,
    ):
        uri = parse_sip_uri(sip_uri)
        if uri is None:
            raise DeepEvalError(
                f"'{sip_uri}' is not a SIP address. Enter it as "
                "sip:agent@example.com."
            )
        super().__init__(
            address=sip_uri,
            call_to=uri.user,
            trunk=sip_trunk_for(
                uri, SipTransport(transport), username, password
            ),
            caller_id=caller_id,
            **kwargs,
        )

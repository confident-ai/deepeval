from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Optional

from deepeval.errors import DeepEvalError
from deepeval.voice.protocol import VoiceProtocol
from deepeval.voice.connectors.transports.base import BaseVoiceConnector

DEFAULT_TWILIO_HOST = "0.0.0.0"
DEFAULT_TWILIO_PORT = 8765


class SipTransport(Enum):
    UDP = "udp"
    TCP = "tcp"
    TLS = "tls"


class SipMediaEncryption(Enum):
    DISABLED = "disabled"
    ALLOWED = "allowed"
    REQUIRED = "required"


@dataclass(frozen=True)
class LiveKitProvider:
    url: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    trunk_id: Optional[str] = None
    trunk_uri: Optional[str] = None
    trunk_username: Optional[str] = None
    trunk_password: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    media_encryption: SipMediaEncryption = SipMediaEncryption.DISABLED


@dataclass(frozen=True)
class TwilioProvider:
    public_url: str
    account_sid: Optional[str] = None
    auth_token: Optional[str] = None
    host: str = DEFAULT_TWILIO_HOST
    port: int = DEFAULT_TWILIO_PORT


class _ProviderConnector(BaseVoiceConnector):

    protocol: ClassVar[VoiceProtocol] = VoiceProtocol.SIP
    provider_type: ClassVar[Optional[type]] = None

    def __new__(cls, *args: Any, provider: Any = None, **kwargs: Any):
        if cls.provider_type is None:
            cls = cls._connector_for(provider)
        return object.__new__(cls)

    @classmethod
    def _connector_for(cls, provider: Any) -> type:
        connectors = cls.__subclasses__()
        for connector in connectors:
            if isinstance(provider, connector.provider_type):
                return connector
        supported = " or ".join(
            connector.provider_type.__name__ for connector in connectors
        )
        raise DeepEvalError(
            f"{cls.__name__} needs `provider` set to {supported}, got "
            f"{type(provider).__name__}."
        )


class PhoneConnector(_ProviderConnector):
    pass


class SIPConnector(_ProviderConnector):
    pass

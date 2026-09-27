from enum import Enum


class VoiceProtocol(Enum):
    """Transport protocol a voice connector uses to reach the agent.

    One protocol maps to many connectors (e.g. LiveKit, Pipecat, and
    Vapi/Retell web calls are all WEBRTC). Latency semantics are defined
    per protocol, not per connector.
    """

    WEBRTC = "webrtc"  # LiveKit rooms, Pipecat, Vapi/Retell web calls
    WEBSOCKET = (
        "websocket"  # raw-audio WS APIs (ElevenLabs ConvAI, custom agents)
    )
    SIP = "sip"  # PSTN / telephony (Twilio et al.)
    CALLBACK = "callback"  # in-process Python callable, no transport

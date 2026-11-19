# Twilio

Contributor reference and TODO list for the Twilio integration. This file
tracks which of Twilio's entry points deepeval can reach; it deliberately does
not restate their parameters, because those are defined by Twilio and go stale
here. Each row links to the page that owns the answer.

## One connector

`TwilioConnector` (`connector.py`) is what `PhoneConnector` builds for a
`TwilioProvider`. It places an outbound call over Twilio's REST
API with inline TwiML that `<Connect>`s the answered call to a bidirectional
[Media Stream](https://www.twilio.com/docs/voice/media-streams). Twilio dials
back into a small aiohttp server deepeval runs, so the caller has to expose that
server with `public_url`. No Twilio SDK is involved.

The server is shared by every connector on the same host and port, started by
the first call and stopped by the last. Each call puts a random key in a TwiML
`<Parameter>`, and the server hands the socket whose `start` message carries
that key to the connector waiting for it. A socket with an unknown key is
closed.

Once the socket is handed over, the connector is a `BaseWebSocketConnector`
that overrides `_open_socket` instead of dialing out: the media messages are
8 kHz mu-law in both directions, and the inherited pacing, turn engine and
duplex path run unchanged.

## Entry point coverage

| Entry point | Status | Reference |
| --- | --- | --- |
| Outbound call with inline TwiML | Covered | [Call resource](https://www.twilio.com/docs/voice/api/call-resource) |
| Bidirectional Media Stream | Covered | [`<Stream>`](https://www.twilio.com/docs/voice/twiml/stream) |
| `media`, `start`, `stop` messages | Covered | [WebSocket messages](https://www.twilio.com/docs/voice/media-streams/websocket-messages) |
| Busy / no answer / failed before the stream starts | Covered (status polling) | [Call resource](https://www.twilio.com/docs/voice/api/call-resource) |
| `mark` and `clear` messages | Not covered | [WebSocket messages](https://www.twilio.com/docs/voice/media-streams/websocket-messages) |
| `dtmf` messages | Not covered | [WebSocket messages](https://www.twilio.com/docs/voice/media-streams/websocket-messages) |
| Request signature validation on the stream | Not covered | [Webhook security](https://www.twilio.com/docs/usage/webhooks/webhooks-security) |
| Answering machine detection | Not covered | [Call resource](https://www.twilio.com/docs/voice/api/call-resource) |

## TODO

**Signature validation.** The stream URL is public, and the only thing that
keeps a stranger's socket from being adopted is the random call key. Checking
`X-Twilio-Signature` on the WebSocket handshake would reject it before any
message is read.

**Answering machine detection.** A call that reaches voicemail is simulated
against the greeting. `MachineDetection` on the call would let the connector
stop instead.

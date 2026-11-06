# ElevenLabs

Contributor reference and TODO list for the ElevenLabs Agents integration. This
file tracks which of ElevenLabs' entry points deepeval can reach; it
deliberately does not restate their parameters, because those are defined by
ElevenLabs and go stale here. Each row links to the page that owns the answer.

## One connector

`ElevenLabsConnector` (`connector.py`) is the only way in. It is our own client
for the [Agents WebSocket API](https://elevenlabs.io/docs/eleven-agents/api-reference/eleven-agents/websocket):
the caller passes an `agent_id` plus credentials, deepeval owns the socket, and
there is no dependency on the `elevenlabs` package.

There used to be a second path that drove a caller-supplied
`AsyncConversation` through the SDK, on the theory that it preserved
configuration the caller had already built. It was removed. An ElevenLabs agent
is a dashboard resource addressed by id, so almost nothing about it is local —
and the two things that genuinely were (client tools, and per-conversation
overrides) are now connector arguments. What the SDK path cost was worse: it
could not see `agent_response_complete`, so turns ended on silence, and it rode
a class ElevenLabs labels beta.

If someone passes a conversation or a room anyway,
`deepeval/voice/connectors/utils.py` recognises it and names the connector to
use instead, rather than failing somewhere deep in the simulation.

## Entry point coverage

| Entry point | Status | Reference |
| --- | --- | --- |
| Public agent over WebSocket | Covered | [WebSocket API](https://elevenlabs.io/docs/eleven-agents/libraries/web-sockets) |
| Private agent via signed URL | Covered | [`get-signed-url`](https://elevenlabs.io/docs/eleven-agents/api-reference/conversations/get-signed-url) |
| Data residency hosts | Covered | [Data residency](https://elevenlabs.io/docs/overview/administration/data-residency) |
| Client tools | Covered | [Client tools](https://elevenlabs.io/docs/eleven-agents/customization/tools/client-tools) |
| Conversation overrides and dynamic variables | Covered | [Overrides](https://elevenlabs.io/docs/eleven-agents/customization/personalization/overrides) |
| Explicit end of turn (`agent_response_complete`) | Covered | [WebSocket API](https://elevenlabs.io/docs/eleven-agents/api-reference/eleven-agents/websocket) |
| `interruption` events | Covered | [WebSocket API](https://elevenlabs.io/docs/eleven-agents/libraries/web-sockets) |
| `ping` / `pong` keepalive | Covered | [WebSocket API](https://elevenlabs.io/docs/eleven-agents/libraries/web-sockets) |
| `agent_response_correction` | Not covered | [WebSocket API](https://elevenlabs.io/docs/eleven-agents/libraries/web-sockets) |
| `user_transcript` (what the agent heard) | Not covered | [WebSocket API](https://elevenlabs.io/docs/eleven-agents/libraries/web-sockets) |
| `user_activity` | Not covered | [WebSocket API](https://elevenlabs.io/docs/eleven-agents/libraries/web-sockets) |
| WebRTC transport | Not covered | [Conversation token](https://elevenlabs.io/docs/eleven-agents/api-reference/conversations/get-webrtc-token) |
| mu-law telephony audio formats | Not covered | [WebSocket API](https://elevenlabs.io/docs/eleven-agents/libraries/web-sockets) |
| Text-only conversations | Not covered | [Text-only mode](https://elevenlabs.io/docs/eleven-agents/customization/text-only) |
| Contextual updates | Not covered | [WebSocket API](https://elevenlabs.io/docs/eleven-agents/libraries/web-sockets) |
| Python SDK `Conversation` / `AsyncConversation` | Not covered, by choice | see above |
| SIP trunking | Out of scope | [SIP trunking](https://elevenlabs.io/docs/eleven-agents/phone-numbers/sip-trunking) |
| Twilio and other telephony | Out of scope | [Phone numbers](https://elevenlabs.io/docs/eleven-agents/phone-numbers) |
| Batch calling | Out of scope | [Batch calls](https://elevenlabs.io/docs/eleven-agents/phone-numbers/batch-calls) |

"Out of scope" means the entry point is the agent placing or receiving a real
phone call. A simulator is the caller, so there is nothing for it to attach to
there; reaching one of those agents means pointing deepeval at the same agent
over WebSocket instead.

## How client tools interact with turn-taking

Worth knowing before changing either side. An agent that calls a tool goes
silent until it gets a result, and silence is exactly what end-of-turn
detection watches. So `_run_client_tool` holds the turn open for as long as the
handler runs (`_hold_turn` in `transports/websocket.py`, read by
`collect_agent_turn` via `hold_event`), and records the span so the handler's
runtime can be subtracted from the reported latency. Without the hold a slow
tool truncates the turn; without the span subtraction it reads as a slow agent.

The hold is deliberately re-read after each `wait_for` timeout in
`collect_agent_turn`, because a tool call that starts *during* a gap would
otherwise be invisible to the check that ends the turn.

Every path out of `_run_client_tool` sends a `client_tool_result`, including
unregistered names, handlers that raise, and handlers that exceed
`client_tool_timeout_s`. The agent blocks on that message, so failing to send
one hangs the conversation until the ceiling.

## TODO

**Transcript corrections.** `_decode_inbound` takes `agent_response` as the
turn's text and ignores the `agent_response_correction` that follows when the
agent's speech is cut short, so an interrupted turn can be recorded saying more
than the simulated user actually heard. This is the one gap here that quietly
changes what a metric scores, rather than blocking a feature. The corrected text
replaces the original in the same field, so the fix is to overwrite the pending
transcript when the correction arrives.

**What the agent heard.** `user_transcript` carries ElevenLabs' own
transcription of deepeval's synthesized audio, and is dropped. It is the only
signal that separates "the agent answered badly" from "the agent misheard the
question", which is a failure mode unique to voice; a text simulation cannot
produce it. There is nowhere on `ConnectorTurn` to put it today.

**WebRTC.** ElevenLabs' WebRTC transport is a LiveKit room, so `LiveKitConnector`
already has the machinery. What is missing is minting the token from
[`GET /v1/convai/conversation/token`](https://elevenlabs.io/docs/eleven-agents/api-reference/conversations/get-webrtc-token)
and connecting to `wss://livekit.rtc.elevenlabs.io` (or the matching
[residency](https://elevenlabs.io/docs/overview/administration/data-residency)
host, which must match the API host or authentication fails).

**Audio formats.** Agents set to a `ulaw_8000` output format are the telephony
default. `_parse_format_rate` warns and then treats the bytes as PCM, which
produces noise. `audio_utils` already has `ulaw_to_pcm16` and `pcm16_to_ulaw`;
they just are not wired in.

**`custom_llm_extra_body`.** The handshake accepts it alongside the overrides
and dynamic variables that are now wired up; it is the remaining field of
`conversation_initiation_client_data` deepeval does not send.

**Conversation id.** It arrives in `conversation_initiation_metadata` and is
dropped. Keeping it would let a simulated conversation be cross-referenced with
the same call in the ElevenLabs dashboard.

**Provider-reported latency.** ElevenLabs sends its own measurement in
`ping_event.ping_ms`. deepeval reports its own first-audio number instead, so
the metric stays comparable across connectors. Surfacing both would be more
informative but needs somewhere on `ConnectorTurn` to put a second,
differently-defined latency.

# Vapi

Contributor reference and TODO list for the Vapi integration. This file tracks
which of Vapi's entry points deepeval can reach; it deliberately does not
restate their parameters, because those are defined by Vapi and go stale here.
Each row links to the page that owns the answer.

## One connector

`VapiConnector` (`connector.py`) is our own client for Vapi's
[WebSocket transport](https://docs.vapi.ai/calls/websocket-transport). The
caller passes an `assistant_id` plus an API key, deepeval creates the call and
owns the socket, and there is no dependency on any Vapi SDK.

The call is minted over REST (`POST /call` with
`transport.provider = "vapi.websocket"`), which returns
`transport.websocketCallUrl`. Phone parameters are rejected on that transport,
so this path never touches telephony — the simulator is the only caller.

## Client messages deepeval depends on

Two of Vapi's [client messages](https://docs.vapi.ai/api-reference/webhooks/client-message)
carry the turn-taking:

| Message | Shape deepeval reads | Used for |
| --- | --- | --- |
| `speech-update` | `status` (`started`/`stopped`), `role` | Ending the agent's turn outright, instead of inferring it from silence |
| `transcript` | `role`, `transcriptType` (`partial`/`final`), `transcript` | The agent's own words, which skip STT for that turn |

Both are in Vapi's default `clientMessages` set, so nothing is sent unless the
caller narrows the set themselves — `_assistant_overrides` then adds the two
back, because losing them costs end-of-turn detection and free transcripts
silently rather than loudly.

Vapi emits one `final` transcript per utterance rather than one per turn, so
`_on_transcript` accumulates them and `speech-update: stopped` clears the
buffer. Taking only the last one would record half of a two-sentence reply.

## Entry point coverage

| Entry point | Status | Reference |
| --- | --- | --- |
| Assistant over WebSocket transport | Covered | [WebSocket transport](https://docs.vapi.ai/calls/websocket-transport) |
| Per-call `assistantOverrides` (incl. `variableValues`) | Covered | [Assistant overrides](https://docs.vapi.ai/assistants/dynamic-variables) |
| Explicit end of turn (`speech-update`) | Covered | [Client messages](https://docs.vapi.ai/api-reference/webhooks/client-message) |
| Agent transcript (`transcript`, final) | Covered | [Client messages](https://docs.vapi.ai/api-reference/webhooks/client-message) |
| `user-interrupted` | Covered | [Client messages](https://docs.vapi.ai/api-reference/webhooks/client-message) |
| Ending the call (`end-call`) | Covered | [WebSocket transport](https://docs.vapi.ai/calls/websocket-transport) |
| mu-law audio format | Not covered | [WebSocket transport](https://docs.vapi.ai/calls/websocket-transport) |
| Inline assistant config instead of `assistantId` | Not covered | [Create call](https://docs.vapi.ai/api-reference/calls/create) |
| Squads | Not covered | [Squads](https://docs.vapi.ai/squads) |
| `tool-calls` / client-side tools | Not covered | [Client messages](https://docs.vapi.ai/api-reference/webhooks/client-message) |
| `conversation-update`, `model-output` | Not covered | [Client messages](https://docs.vapi.ai/api-reference/webhooks/client-message) |
| Call Listen (`monitor.listenUrl`) | Not covered, by choice | [WebSocket transport](https://docs.vapi.ai/calls/websocket-transport) |
| Phone numbers, campaigns, SIP | Out of scope | [Calls](https://docs.vapi.ai/calls) |

Call Listen is listen-only and supplements a call that already exists, so it
cannot carry a simulation. "Out of scope" means the entry point is the agent
placing or receiving a real phone call; reaching such an agent means pointing
deepeval at the same assistant over the WebSocket transport instead.

## TODO

**Smoke test against a live assistant.** Everything here is written against
Vapi's documented schemas. Three things could only be confirmed on a real call:
whether the socket wraps client messages in a `message` envelope the way
webhooks do (`_decode_inbound` accepts both shapes, so either works), whether a
non-default `sampleRate` is honoured or quietly converted
(`_apply_audio_format` adopts whatever comes back), and how the socket behaves
before the first audio frame (`_ready_on_connect` assumes no handshake).
`vapi_capture.py` at the repo root records one conversation for this.

**What the agent heard.** `transcript` messages with `role: "user"` carry
Vapi's transcription of deepeval's synthesized audio, and are dropped. As with
ElevenLabs, it is the signal that separates "the agent answered badly" from
"the agent misheard the question", and there is nowhere on `ConnectorTurn` to
put it today.

**Interruption detail.** `user-interrupted` sets `_interrupted` for the turn,
but Vapi's own barge-in handling and deepeval's floor control both act on the
same moment, and the interaction has not been exercised.

**Call id.** `call_id` is kept from the create-call response but never
surfaced, so a simulated conversation cannot be cross-referenced with the same
call in Vapi's dashboard.

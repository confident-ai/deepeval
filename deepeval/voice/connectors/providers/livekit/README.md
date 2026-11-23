# LiveKit

Contributor reference and TODO list for the LiveKit Agents integration. This
file tracks which of LiveKit's entry points deepeval can reach; it deliberately
does not restate their parameters, because those are defined by LiveKit and go
stale here. Each row links to the page that owns the answer.

## Why there is only one connector

ElevenLabs hands you a conversation object you can pass around. LiveKit does
not: the agent is a participant on the far side of a room, reached by
[dispatching](https://docs.livekit.io/agents/server/agent-dispatch/) a worker
to it, so there is no agent object to hand over. The room is the closest thing,
and `LiveKitConnector` takes an
[`rtc.Room`](https://docs.livekit.io/reference/python/livekit/rtc/room.html) as
one of its arguments rather than as a separate entry point.

| The caller brings | deepeval does |
| --- | --- |
| nothing (URL, key, secret) | mints a token, creates a room, connects, joins |
| `token=` | connects and joins with the caller's token |
| `room=` (unconnected) | connects it, then joins |
| `room=` (connected) | joins nothing; publishes a microphone and listens |

Which participant is the agent is decided by `_is_agent_participant`.
`agent_identity` names one outright and wins; without it the connector falls
back to LiveKit's participant kind, which is a guess that also has to accept
everything on builds old enough not to report a kind at all.

deepeval only tears down what it created. A room the caller connected is left
connected on `disconnect()`, with the simulated microphone track unpublished.
A room passed to `VoiceConfig` directly is rejected by
`deepeval/voice/connectors/utils.py`, which names `LiveKitConnector(room=...)`
instead — every session runs through a connector.

## Entry point coverage

| Entry point | Status | Reference |
| --- | --- | --- |
| Join a room as a participant | Covered | [Connecting to LiveKit](https://docs.livekit.io/intro/basics/connect/) |
| Token minting with `VideoGrants` | Covered | [Access tokens and grants](https://docs.livekit.io/frontends/reference/tokens-grants/) |
| Explicit dispatch via token `RoomAgentDispatch` | Covered | [Agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch/) |
| Automatic dispatch | Covered | [Agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch/) |
| Caller-owned `rtc.Room` or token | Covered | [`rtc.Room` reference](https://docs.livekit.io/reference/python/livekit/rtc/room.html) |
| Agent transcript over text streams | Covered | [Text and transcriptions](https://docs.livekit.io/agents/multimodality/text/) |
| Agent-kind participant filtering | Covered | [`rtc.Room` reference](https://docs.livekit.io/reference/python/livekit/rtc/room.html) |
| Choosing one agent by identity | Covered | [`rtc.Room` reference](https://docs.livekit.io/reference/python/livekit/rtc/room.html) |
| Interruption reporting | Not covered | [Agent state](https://docs.livekit.io/frontends/build/agent-state/) |
| Agent state (`lk.agent.state`) | Not covered | [Agent state](https://docs.livekit.io/frontends/build/agent-state/) |
| Interim (non-final) transcripts | Not covered | [Text and transcriptions](https://docs.livekit.io/agents/multimodality/text/) |
| Dispatch `metadata` | Not covered | [Agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch/) |
| Dispatch `deployment` | Not covered | [Agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch/) |
| `AgentDispatchService.CreateDispatch` | Not covered | [Dispatch service API](https://docs.livekit.io/reference/agents/agent-dispatch-service-api.md) |
| Sending text to the agent (`lk.chat`) | Not covered | [Text and transcriptions](https://docs.livekit.io/agents/multimodality/text/) |
| RPC to the agent | Not covered | [RPC](https://docs.livekit.io/home/client/data/rpc/) |
| In-process `AgentSession` | Not covered | [Sessions](https://docs.livekit.io/agents/logic/sessions/) |
| Outbound SIP calls (phone numbers and SIP addresses) | Covered | [Outbound calls](https://docs.livekit.io/telephony/making-calls/outbound-calls) |
| `session.run()` behavioral tests | Out of scope | [Test framework](https://docs.livekit.io/agents/start/testing/test-framework/) |

LiveKit's own test helpers are text-only by design and never connect to a room,
so they answer a different question than a voice simulation does. They are
listed to record the decision, not as a gap.

## TODO

**Interruption reporting.** `exchange_turn` hardcodes `interrupted=False`, so a
barge-in that a persona performs is never recorded as one, and a metric reading
that field sees a clean turn. ElevenLabs gets this for free from an
`interruption` message; LiveKit has no equivalent, because a room carries media
and the agent simply stops talking. The signal is the agent's own state: the
agents SDK mirrors it to the
[`lk.agent.state`](https://docs.livekit.io/frontends/build/agent-state/)
participant attribute, and pausing output for an interruption sets it to
`listening`. Watching `participant_attributes_changed` for that transition while
the uplink is still streaming would fill the field in. Worth noting the value is
agent state only, not user state, and that an agent configured with
`allow_interruptions=False` never transitions at all.

**Concurrency.** `VoiceConfig` runs parallel conversations by cloning the
connector, and `BaseVoiceConnector.clone()` is a shallow copy, so every piece of
per-call state is rebuilt in `_join_room()`. A caller-owned `room=`, a fixed
`room_name=` or a `token=` (minted for one room) would put every clone in the
same room, so `supports_parallel_calls` is False for them and the simulator
refuses to run more than one call at a time with such a connector, naming the
factory form as the fix. Varying `identity` per clone is unnecessary: clones
that mint their own tokens land in rooms of their own.

**Dispatch metadata and deployment.** `_build_token` sets `RoomAgentDispatch`
with only `agent_name`. LiveKit also accepts `metadata`, which is how job
context such as a caller id reaches the agent, and `deployment`, which targets
a staging build instead of production. Both are documented on
[agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch/) and both
would map to per-golden simulation inputs.

**Dispatch into an existing room.** Token-based dispatch is ignored when the
room already exists in memory, which is exactly the case when the caller passes
a connected room. Reaching those agents needs
[`CreateDispatch`](https://docs.livekit.io/reference/agents/agent-dispatch-service-api.md),
which also requires `roomAdmin` on the token.

**Interim transcripts.** `_read_transcript` keeps only segments whose
`lk.transcription_final` attribute is true, so a turn's text arrives all at
once at the end. Interim segments are what would make a live view of the
simulation possible; see
[text and transcriptions](https://docs.livekit.io/agents/multimodality/text/).

**Transcript timing.** When the agent's audio ends before its transcript
arrives, `_await_transcript` waits up to `transcript_grace_s`. That is a
guessed constant. Tying the wait to the transcription track id in the
`lk.transcribed_track_id` attribute would let it wait for the right segment
rather than for a duration.

**In-process `AgentSession`.** Today the agent must be deployed and dispatched.
Running the user's `AgentSession` in the test process and feeding it audio
directly would remove the worker and the Cloud room from CI entirely. This is
the largest open item and would be a second connector rather than a change to
this one. See [sessions](https://docs.livekit.io/agents/logic/sessions/).

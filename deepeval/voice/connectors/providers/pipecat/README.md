# Pipecat

Contributor reference and TODO list for the Pipecat integration. This file
tracks which of Pipecat's entry points deepeval can reach; it deliberately does
not restate their parameters, because those are defined by Pipecat and go stale
here. Each row links to the page that owns the answer.

## One connector, no dependency on `pipecat-ai`

`PipecatConnector` (`connector.py`) connects to a pipeline running behind one of
Pipecat's [WebSocket transports](https://docs.pipecat.ai/server/services/transport/websocket-server)
the way any other client would: it is given a URL, and the pipeline is the
user's to host.

`protobuf.py` is our own encoder for the five messages in Pipecat's
[`frames.proto`](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/serializers/protobuf.py),
which is what `ProtobufFrameSerializer` — the default on those transports —
puts on the wire. Encoding them directly is about 150 lines and keeps
`pipecat-ai`, a server framework with a large dependency tree, out of the
requirements of testing something built with it.

The schema is `oneof { text, audio, transcription, message, interruption }`.
Only `audio` and `message` are written; `message` carries a JSON string, which
is how RTVI rides the same socket.

## What carries the turn-taking

Pipecat has no end-of-turn frame in the protobuf schema, so the turn-taking
comes from [RTVI](https://docs.pipecat.ai/server/frameworks/rtvi/introduction)
messages, which a pipeline only sends if it runs an `RTVIProcessor` and its
observer:

| Message | Used for |
| --- | --- |
| `bot-stopped-speaking` | Ending the agent's turn outright, instead of inferring it from silence |
| `bot-transcription` | The agent's own words, which skip STT for that turn |
| `bot-interrupted` | Recording that a barge landed |
| `client-ready` (sent by us) | Pipelines that hold their greeting, or their audio input, until the client is up |

`signals_turn_complete` is therefore not a constant: it starts False and flips
once a `bot-stopped-speaking` has actually been seen. Claiming it up front
would leave a non-RTVI pipeline's turns waiting for a signal that is never
coming; claiming it never would truncate replies that pause mid-sentence.

RTVI emits one `bot-transcription` per sentence, so `_on_transcription`
accumulates them and `bot-stopped-speaking` clears the buffer.

## Entry point coverage

| Entry point | Status | Reference |
| --- | --- | --- |
| `WebsocketServerTransport` | Covered | [WebSocket server](https://docs.pipecat.ai/server/services/transport/websocket-server) |
| `FastAPIWebsocketTransport` | Covered | [FastAPI WebSocket](https://docs.pipecat.ai/server/services/transport/fastapi-websocket) |
| `ProtobufFrameSerializer` | Covered | [Serializers](https://docs.pipecat.ai/server/utilities/serializers/introduction) |
| `add_wav_header=True` on the transport | Covered | [WebSocket server](https://docs.pipecat.ai/server/services/transport/websocket-server) |
| RTVI turn signals and bot transcripts | Covered | [RTVI](https://docs.pipecat.ai/server/frameworks/rtvi/introduction) |
| `client-ready` handshake | Covered | [RTVI](https://docs.pipecat.ai/server/frameworks/rtvi/introduction) |
| `bot-output` (the successor to `bot-transcription`) | Not covered | [RTVI messages](https://docs.pipecat.ai/client/js/api-reference/callbacks) |
| Client-side function calls over `client-message` | Not covered | [RTVI](https://docs.pipecat.ai/server/frameworks/rtvi/introduction) |
| `TwilioFrameSerializer`, `ExotelFrameSerializer`, and the other telephony dialects | Not covered | [Serializers](https://docs.pipecat.ai/server/utilities/serializers/introduction) |
| Daily, LiveKit, and the other WebRTC transports | Not covered here | [Transports](https://docs.pipecat.ai/server/services/transport/daily) |
| SmallWebRTC transport | Not covered | [SmallWebRTC](https://docs.pipecat.ai/server/services/transport/small-webrtc) |

A pipeline on a WebRTC transport is reached through the connector for that
service — a Pipecat pipeline on LiveKit is a LiveKit room, and
`LiveKitConnector` joins it. The telephony serializers are the agent taking a
real phone call; reaching such an agent means running the same pipeline behind
a WebSocket transport instead.

## TODO

**Smoke test against a live pipeline.** The frame encoding is checked against
Pipecat's own generated `frames_pb2` in both directions, but three things could
only be confirmed against a running pipeline: that a pipeline gated on
`on_client_ready` does open on our `client-ready`, that `bot-stopped-speaking`
arrives after the last audio frame rather than before it, and how a pipeline
whose `audio_in_sample_rate` differs from `agent_sample_rate` degrades (Pipecat
does not resample what it is handed).

**`bot-output`.** RTVI is deprecating `bot-transcription` in favour of
`bot-output`, which carries progress updates for the same text and would double
up the transcript if both were read. When `bot-transcription` goes, the
accumulation has to move to `bot-output` and de-duplicate by segment id.

**What the agent heard.** `transcription` frames carry Pipecat's STT of
deepeval's synthesized audio, and are dropped. As with the other connectors, it
is the signal that separates "the agent answered badly" from "the agent misheard
the question", and there is nowhere on `ConnectorTurn` to put it today.

**Ending the pipeline.** `disconnect()` closes the socket and nothing more.
RTVI's `disconnect-bot` would end the pipeline task outright, which is more
than a client should decide for a server it does not own.

# Docs to update for `features/voice-2.0`

Pointers only: where each code change on this branch needs a docs section, and what the section has to cover. Prose to be written by the docs owner.

## `docs/content/docs/conversation-simulator-voice-mode/index.mdx`

- **Setup Voice Mode**: new `VoiceConfig(record_call=True)` option. Records the whole call, off by default.
- **What A Voice Simulation Produces**:
  - `ConversationalTestCase.call_recording_path`: a stereo WAV of the entire call (caller on the left channel, agent on the right), placed by wall-clock time so overlaps and silences are real, not stitched from turn clips. Written to disk as `deepeval-call-recording.wav` when `output_dir` is set.
  - One `INFO` log line per conversation from `deepeval.simulator`: average and worst reply time for the agent and for the caller. Mention it as the way to read latency.
- **How It Works**: on a full-duplex connector (LiveKit rooms, SIP calls) the caller now always listens while it speaks, whether or not the persona interrupts. The speak-then-listen exchange is only used by connectors that cannot carry both voices at once (callback). Consequence worth stating: nothing the agent says while the caller talks or thinks is lost anymore.
- **FAQs**: barge turns made by an interrupting persona count against `max_user_simulations`.

## `docs/content/docs/conversation-simulator-voice-connectors.mdx`

- **Turn Detection**: the preset window (`eager` 0.5s / `balanced` 0.8s / `patient` 2.5s) is now the ceiling, not the wait. When the running transcript reads as a finished sentence and covers all the speech heard so far, the turn ends after 0.5s of silence. Replies that trail off mid-sentence still wait the full window. Only applies on the duplex path.
- **Callback** / **Generic WebSocket** (custom connectors): document the `supports_duplex` property. Returning `True` makes the simulator run the duplex exchange on that connector even without an interruption policy; it requires `stream_uplink`, `iter_agent_events` and `stop_uplink`.

## `docs/content/docs/conversation-simulator-voice-personas.mdx`

- **Who Speaks First**: with `speaks_first=False` on a duplex connector the greeting is collected by the duplex engine (so semantic turn detection applies) and the wait is bounded by `hold_timeout`, or 15s when unset, after which the caller takes the floor.
- **Interruptions**: note that leaving `interruption_behavior` unset no longer means half-duplex; the caller still hears everything, it just never cuts in.

## `docs/content/docs/conversation-simulator-voice-interruptions.mdx`

- **How It Works**: `DuplexExchange` accepts `policy=None`; the barge judge only runs when a policy is set.
- **What Gets Recorded**: `Turn.latency_ms` is clamped at zero when the agent was already talking before the caller finished (backlogged speech is a wait of zero, not a negative one). Every barge turn counts toward the turn budget.

## `docs/content/docs/(voice)/metrics-*.mdx` (all seven voice metrics)

- Verbose logs now print the breakdown as indented JSON with floats rounded to three decimals instead of a Python dict repr. One line under whichever section shows verbose output.

## `docs/content/docs/(concepts)/evaluation-voice.mdx`

- Cross-link the full-call recording and the per-conversation latency log as the two artifacts to inspect when a voice metric scores low.

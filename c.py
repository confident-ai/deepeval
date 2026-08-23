"""Simulate a call against a real streaming voice agent, over a plain WebSocket.

`b.py` talks to an in-process callback that is handed one finished utterance,
transcribes it, thinks, and synthesizes a reply. Nothing about it can start
before the caller's last frame exists, so it cannot show what streaming the
uplink buys.

The agent here is a separate process that listens the way a real voice agent
listens: it transcribes the caller's speech as the frames arrive, detects the
end of the turn from silence in the stream, and streams its own reply back at
real time so it can be interrupted mid-sentence. That is enough to make the
difference visible, without WebRTC, a VAD model, or an agent framework.

What to look for
----------------
Synthesis takes the same few seconds either way, so streaming does not get the
whole utterance to the agent any sooner — it gets the *beginning* there sooner.
The agent's own listening then overlaps our synthesis instead of following it:

- Buffered uplink: every frame lands at once, so the agent's transcription
  starts only when the caller has finished. Its STT sits in the critical path.
- Streamed uplink: frames land from the first one onward, so by the end of the
  turn the agent has already transcribed nearly all of it.

The agent process prints which of the two happened on every turn.

Whisper transcribes whole clips, so the agent approximates continuous listening
by re-transcribing the audio so far every second or so. A purpose-built
streaming STT would overlap more completely than this; the point here is that
the overlap exists at all, which a buffered uplink does not allow.

Setup
-----
Only OpenAI is needed; the transport is `aiohttp`, which deepeval already
depends on.

    export OPENAI_API_KEY=...

Start the agent and leave it running:

    python c.py agent

Then, in another shell:

    python c.py
"""

import asyncio
import base64
import json
import logging
import os
import socket
import sys
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

from deepeval.models.llms import OpenAIModel
from deepeval.models.stt_models import OpenAISTTModel
from deepeval.models.tts_models import OpenAITTSModel
from deepeval.test_case import Audio
from deepeval.voice.connectors import audio_utils

HOST = "127.0.0.1"
PORT = 8765
PATH = "/voice"
WS_URL = f"ws://{HOST}:{PORT}{PATH}"

SAMPLE_RATE = 24000
BYTES_PER_SECOND = SAMPLE_RATE * 2

AGENT_PROMPT = """
You are Walter, an elderly customer-support representative handling a missing
order over the phone.

Rules:
- Respond directly to the customer's latest words and remember prior turns.
- Be blunt, impatient, and somewhat grumpy, but do not use profanity.
- Do not say "please", "thank you", "sorry", "apologies", "happy", or "glad".
- Ramble for 3-5 sentences and include an unnecessary old-fashioned anecdote.
- Still perform the support task: obtain the order number, explain the delay,
  offer refund or replacement, and confirm the selected resolution.
- Never claim the replacement is confirmed until the customer chooses it.

Conversation:
{conversation}

Walter:
""".strip()


############################################
### The agent (system under test) ##########
############################################

# Silence in the caller's stream that ends their turn. The connector pads full
# turns with trailing silence for exactly this.
END_OF_TURN_SILENCE_S = 0.8
# A barge carries no trailing padding, so a gap in the frames ends the turn too.
FRAME_GAP_TIMEOUT_S = 1.0
# How much new speech to let build up before transcribing again mid-utterance.
PARTIAL_EVERY_S = 1.2
# How much speech a partial may be missing and still be used as the transcript.
STALE_PARTIAL_S = 0.4


class AgentCall:
    """One call, listened to as it arrives rather than after it finishes."""

    def __init__(self, ws, stt, llm, tts, log):
        self.ws = ws
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.log = log
        self.history: List[Tuple[str, str]] = []

        self.pcm = bytearray()
        self.speech_bytes = 0  # buffer length at the last frame with speech
        self.silence_s = 0.0
        self.heard_speech = False
        self.last_frame_at = time.perf_counter()

        self.partial: Optional[asyncio.Task] = None
        self.partial_text: Optional[str] = None
        self.partial_covered = 0
        self.reply: Optional[asyncio.Task] = None

    async def on_frame(self, pcm: bytes) -> None:
        self.last_frame_at = time.perf_counter()
        self.pcm.extend(pcm)
        if audio_utils.is_silent(pcm):
            if self.heard_speech:
                self.silence_s += len(pcm) / BYTES_PER_SECOND
            return

        self.heard_speech = True
        self.silence_s = 0.0
        self.speech_bytes = len(self.pcm)
        # The caller is talking, so stop talking over them.
        await self._abandon_reply()
        self._transcribe_what_has_arrived()

    async def tick(self) -> None:
        """End the turn on silence, or on the caller's frames simply stopping."""
        if not self.heard_speech or self.reply is not None:
            return
        starved = time.perf_counter() - self.last_frame_at > FRAME_GAP_TIMEOUT_S
        if self.silence_s >= END_OF_TURN_SILENCE_S or starved:
            self.reply = asyncio.create_task(self._answer())

    def _transcribe_what_has_arrived(self) -> None:
        """Transcribe the speech so far while the rest of it is still coming."""
        if self.partial is not None and not self.partial.done():
            return
        if self.speech_bytes - self.partial_covered < int(
            BYTES_PER_SECOND * PARTIAL_EVERY_S
        ):
            return
        snapshot = bytes(self.pcm[: self.speech_bytes])
        self.partial = asyncio.create_task(self._transcribe(snapshot))

    async def _transcribe(self, snapshot: bytes) -> Tuple[str, int]:
        audio = Audio.from_bytes(
            audio_utils.pcm16_to_wav_bytes(snapshot, SAMPLE_RATE, 1),
            "audio/wav",
            sampleRate=SAMPLE_RATE,
            encoding="wav",
        )
        text, _ = await self.stt.a_transcribe(audio)
        return text, len(snapshot)

    async def _heard(self) -> str:
        """The caller's words, reusing whatever was transcribed while they spoke."""
        ended_at = time.perf_counter()
        if self.partial is not None:
            try:
                self.partial_text, self.partial_covered = await self.partial
            except Exception:
                self.log.exception("Partial transcription failed")
            self.partial = None

        missing = self.speech_bytes - self.partial_covered
        if self.partial_text and missing <= int(
            BYTES_PER_SECOND * STALE_PARTIAL_S
        ):
            self.log.info(
                "Listened while the caller spoke: transcript ready %.2fs after "
                "end of turn (%.1fs of speech, %.2fs of it untranscribed)",
                time.perf_counter() - ended_at,
                self.speech_bytes / BYTES_PER_SECOND,
                missing / BYTES_PER_SECOND,
            )
            return self.partial_text

        text, _ = await self._transcribe(bytes(self.pcm[: self.speech_bytes]))
        self.log.info(
            "Transcribed after the caller finished: %.2fs spent with the "
            "caller waiting (%.1fs of speech)",
            time.perf_counter() - ended_at,
            self.speech_bytes / BYTES_PER_SECOND,
        )
        return text

    async def _answer(self) -> None:
        try:
            user_text = (await self._heard()).strip()
            self._reset_turn()
            if not user_text:
                return
            self.log.info("Heard: %r", user_text)
            self.history.append(("customer", user_text))

            conversation = "\n".join(
                f"{role}: {content}" for role, content in self.history[-10:]
            )
            reply, _ = await self.llm.a_generate(
                AGENT_PROMPT.format(conversation=conversation)
            )
            reply = str(reply).strip()
            self.history.append(("Walter", reply))
            self.log.info("Replying: %r", reply)

            await self._send({"type": "transcript", "transcript": reply})
            await self._speak(reply)
            await self._send({"type": "turn_complete"})
        except asyncio.CancelledError:
            await self._send({"type": "turn_complete"})
            raise
        finally:
            self.reply = None

    async def _speak(self, reply: str) -> None:
        """Stream the reply out at real time, starting on the first frame.

        Sending as fast as synthesis allows would deliver a 20-second reply in
        four seconds, leaving the caller no room to interrupt it. Real time is
        both honest and what makes barge-in possible.
        """
        started = time.perf_counter()
        spoken_s = 0.0
        async for chunk in self.tts.a_synthesize_stream(reply):
            data = chunk.get_bytes()
            await self._send(
                {
                    "type": "audio",
                    "audio": base64.b64encode(data).decode("ascii"),
                }
            )
            spoken_s += len(data) / BYTES_PER_SECOND
            ahead = spoken_s - (time.perf_counter() - started)
            if ahead > 0:
                await asyncio.sleep(ahead)

    async def _send(self, message: dict) -> None:
        if not self.ws.closed:
            await self.ws.send_str(json.dumps(message))

    def _reset_turn(self) -> None:
        self.pcm = bytearray()
        self.speech_bytes = 0
        self.silence_s = 0.0
        self.heard_speech = False
        self.partial_text = None
        self.partial_covered = 0

    async def _abandon_reply(self) -> None:
        if self.reply is None or self.reply.done():
            return
        self.reply.cancel()
        try:
            await self.reply
        except (asyncio.CancelledError, Exception):
            pass
        self.reply = None

    async def aclose(self) -> None:
        await self._abandon_reply()
        if self.partial is not None and not self.partial.done():
            self.partial.cancel()


def build_agent_app(stt, llm, tts, log=None):
    """The agent as an aiohttp app, so its transport can be exercised alone."""
    from aiohttp import WSMsgType, web

    log = log or logging.getLogger("walter")

    async def handler(request):
        ws = web.WebSocketResponse(max_msg_size=0, heartbeat=None)
        await ws.prepare(request)
        log.info("Caller connected")
        call = AgentCall(ws, stt, llm, tts, log)

        async def ticker():
            # End-of-turn also has to fire when the frames simply stop, which
            # no inbound message will announce.
            while not ws.closed:
                await asyncio.sleep(0.05)
                await call.tick()

        pulse = asyncio.create_task(ticker())
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    await call.on_frame(msg.data)
                    await call.tick()
                elif msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
        finally:
            pulse.cancel()
            await call.aclose()
            log.info("Caller disconnected")
        return ws

    app = web.Application()
    app.router.add_get(PATH, handler)
    return app


def run_agent() -> None:
    from aiohttp import web

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running the agent.")

    log = logging.getLogger("walter")
    app = build_agent_app(
        stt=OpenAISTTModel(model="whisper-1"),
        llm=OpenAIModel(model="gpt-4o-mini", temperature=0.7),
        tts=OpenAITTSModel(
            voice="onyx",
            generation_kwargs={
                "speed": 1.15,
                "instructions": (
                    "Speak like a friendly, rambling older man. Use a warm, "
                    "unhurried tone with occasional reflective pauses."
                ),
            },
        ),
        log=log,
    )
    log.info("Walter listening on %s", WS_URL)
    web.run_app(app, host=HOST, port=PORT, print=None)


############################################
### Instrumentation ########################
############################################

TIMINGS: Dict[str, List[float]] = defaultdict(list)

# Only the caller's side is visible from here. The agent is another process, so
# it is measured the way a user measures a vendor: by how long it takes to
# answer.
CALLER_STAGES = (
    "caller TTS (to first frame)",
    "caller TTS (whole utterance)",
    "user turn (LLM)",
    "stopping check (LLM)",
    "barge judge (LLM)",
    "agent STT",
)


@contextmanager
def timed(stage: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        TIMINGS[stage].append(time.perf_counter() - started)


class TimedTTSModel(OpenAITTSModel):
    """Separates the wait for the first frame from the whole utterance.

    The first number is the silence the caller leaves before speaking; the gap
    between the two is how much synthesis happens while they are already
    talking, which is the agent's chance to work in parallel.
    """

    async def a_synthesize_stream(self, text: str, **kwargs):
        started = time.perf_counter()
        first_frame = False
        async for chunk in super().a_synthesize_stream(text, **kwargs):
            if not first_frame:
                first_frame = True
                TIMINGS["caller TTS (to first frame)"].append(
                    time.perf_counter() - started
                )
            yield chunk
        TIMINGS["caller TTS (whole utterance)"].append(
            time.perf_counter() - started
        )

    async def a_synthesize(self, text: str, **kwargs):
        with timed("caller TTS (whole utterance)"):
            return await super().a_synthesize(text, **kwargs)


class TimedSTTModel(OpenAISTTModel):
    async def a_transcribe(self, audio, **kwargs):
        with timed("agent STT"):
            return await super().a_transcribe(audio, **kwargs)


class TimedSimulatorModel(OpenAIModel):
    _STAGE_BY_SCHEMA = {
        "ConversationCompletion": "stopping check (LLM)",
        "SimulatedInput": "user turn (LLM)",
        "InterruptDecision": "barge judge (LLM)",
    }

    async def a_generate(self, prompt: str, *args, **kwargs):
        schema = kwargs.get("schema") or (args[0] if args else None)
        stage = self._STAGE_BY_SCHEMA.get(
            getattr(schema, "__name__", ""), "other simulator (LLM)"
        )
        with timed(stage):
            return await super().a_generate(prompt, *args, **kwargs)


############################################
### Simulation #############################
############################################


def _require_agent() -> None:
    """Say plainly that the agent is not up, rather than failing mid-dial."""
    with socket.socket() as probe:
        probe.settimeout(1.0)
        if probe.connect_ex((HOST, PORT)) == 0:
            return
    raise RuntimeError(
        f"Nothing is listening on {WS_URL}. The agent runs as its own process: "
        "start it in another shell with `python c.py agent`, wait for it to say "
        "it is listening, then run this again."
    )


def run_simulation() -> None:
    from deepeval.dataset import (
        ConversationalGolden,
        InterruptionBehavior,
        Persona,
    )
    from deepeval.simulator import ConversationSimulator
    from deepeval.voice import VoiceConfig
    from deepeval.voice.connectors.transports.websocket import (
        WebSocketConnector,
    )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")
    _require_agent()

    caller_tts_model = TimedTTSModel(
        voice="coral",
        generation_kwargs={
            "speed": 1.4,
            "instructions": "Speak clearly, briskly, and with mild frustration.",
        },
    )
    connector = WebSocketConnector(
        WS_URL,
        sample_rate=SAMPLE_RATE,
        # Raw PCM frames up, JSON down so the agent can send its transcript and
        # say when its turn is over.
        binary_outbound=True,
        receive_audio_key="audio",
        receive_transcript_key="transcript",
        turn_complete_type="turn_complete",
        # Padding on full turns is what the agent's silence detection listens
        # for; it has to outlast the agent's own end-of-turn threshold.
        trailing_silence_ms=1200,
        # Walter pauses mid-reply, and his replies run long. This agent closes
        # each turn explicitly, so silence is only the fallback, but the ceiling
        # still has to be generous enough to hear a whole answer out.
        turn_detection="patient",
    )
    simulator = ConversationSimulator(
        simulator_model=TimedSimulatorModel(),
        voice_config=VoiceConfig(
            connector=connector,
            tts_model=caller_tts_model,
            stt_model=TimedSTTModel(model="whisper-1"),
            output_dir="voice_simulations_ws",
            combine_audio_files=True,
        ),
    )

    logger.info("Connecting to the agent at %s", WS_URL)
    started = time.perf_counter()
    conversations = simulator.simulate(
        [
            ConversationalGolden(
                name="missing-order-websocket",
                scenario=(
                    "The user calls customer support because an important "
                    "order has not arrived. They must identify the order, "
                    "understand what happened, and request a replacement."
                ),
                persona=Persona(
                    characteristics=(
                        "An angry, blunt, impatient customer who makes terse "
                        "demands, interrupts rambling, and never says please "
                        "or thank you."
                    ),
                    voice="coral",
                    interruption_behavior=InterruptionBehavior(
                        frequency="rare", overlap="adaptive"
                    ),
                ),
                expected_outcome="A replacement order is confirmed.",
            )
        ],
        max_user_simulations=5,
    )
    wall_seconds = time.perf_counter() - started
    logger.info("Simulation finished after %.2fs", wall_seconds)

    turns = conversations[0].turns
    print_stage_breakdown(wall_seconds)
    print_agent_latency(turns)
    print_transcript(turns)
    print("\nAudio saved under ./voice_simulations_ws/")


############################################
### Reporting ##############################
############################################


def print_stage_breakdown(wall_seconds: float) -> None:
    print(f"\nCaller-side stages ({wall_seconds:.1f}s wall clock)")
    print(f"{'stage':<30}{'calls':>6}{'total':>9}{'mean':>8}")
    for stage in CALLER_STAGES:
        durations = TIMINGS.get(stage)
        if not durations:
            continue
        total = sum(durations)
        print(
            f"{stage:<30}{len(durations):>6}{total:>8.1f}s"
            f"{total / len(durations):>7.1f}s"
        )

    first_frame = TIMINGS.get("caller TTS (to first frame)")
    whole = TIMINGS.get("caller TTS (whole utterance)")
    if first_frame and whole:
        overlapped = sum(whole) / len(whole) - sum(first_frame) / len(
            first_frame
        )
        print(
            f"\n{overlapped:.1f}s per turn of synthesis happened while the "
            "caller was already speaking, which the agent could listen through."
        )


def print_agent_latency(turns) -> None:
    """The agent is a black box here, so it is judged on how fast it answers."""
    latencies = [
        turn.latency_ms / 1000
        for turn in turns
        if turn.role == "assistant" and turn.latency_ms is not None
    ]
    if not latencies:
        return
    print(
        f"\nAgent reply latency: mean {sum(latencies) / len(latencies):.1f}s, "
        f"best {min(latencies):.1f}s, worst {max(latencies):.1f}s "
        f"({len(latencies)} replies)"
    )


def print_transcript(turns) -> None:
    """Print what was said against when it was heard."""
    from deepeval.voice.timeline import build_audio_timeline

    timeline = build_audio_timeline(turns, require_start_times=False)
    entries = {entry.turn_index: entry for entry in timeline}
    # Gaps are measured against whatever precedes each clip in *time*, which is
    # not always what precedes it in the transcript: a barge is recorded after
    # the reply it cut into.
    preceding_end = {}
    cursor = 0.0
    for entry in timeline:
        preceding_end[entry.turn_index] = cursor
        cursor = max(cursor, entry.end_time)

    print("\nAnnotated transcript (gaps are what you hear as silence)")
    for index, turn in enumerate(turns):
        entry = entries.get(index)
        if entry is None:
            timing = "        no audio recorded         "
        else:
            gap = entry.start_time - preceding_end[index]
            marker = "overlap" if gap < 0 else "gap"
            timing = (
                f"{entry.start_time:>7.1f}s -{entry.end_time:>7.1f}s"
                f" ({entry.duration:>4.1f}s)  {marker}{abs(gap):>5.1f}s"
            )

        metadata = turn.metadata or {}
        flags = []
        if turn.interrupted:
            flags.append("CUT SHORT BY BARGE")
        if metadata.get("barge_in"):
            flags.append("BARGE-IN")
        if metadata.get("frustrated"):
            flags.append("FRUSTRATED")
        if metadata.get("ended_without_agent_signal"):
            flags.append("WE STOPPED LISTENING")
        if metadata.get("grace_missed_ms") is not None:
            flags.append(f"grace missed {metadata['grace_missed_ms']:.0f}ms")
        if turn.latency_ms is not None:
            flags.append(f"latency {turn.latency_ms / 1000:.1f}s")

        suffix = f"   {' | '.join(flags)}" if flags else ""
        print(f"\n[{index + 1}] {turn.role:<10}{timing}{suffix}")
        for line in textwrap.wrap(turn.content or "(no speech)", width=88):
            print(f"    {line}")
        intended = metadata.get("intended_content")
        if intended:
            if metadata.get("ended_without_agent_signal"):
                print("    ...still saying when we stopped listening:")
            else:
                print("    ...cut off from saying:")
            for line in textwrap.wrap(intended, width=84):
                print(f"        {line}")

    speech = sum(entry.duration for entry in entries.values())
    barges = sum(1 for turn in turns if (turn.metadata or {}).get("barge_in"))
    cut_short = sum(1 for turn in turns if turn.interrupted)
    print(
        f"\nspeech {speech:.1f}s of {cursor:.1f}s recorded | "
        f"{barges} barge attempt(s) | {cut_short} agent turn(s) cut short"
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voice-simulator-ws")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        run_agent()
    else:
        logging.getLogger("deepeval.simulator.conversation_simulator").setLevel(
            logging.DEBUG
        )
        run_simulation()

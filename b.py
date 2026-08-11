import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List

from deepeval.dataset import (
    ConversationalGolden,
    InterruptionBehavior,
    Persona,
)
from deepeval.models.llms import OpenAIModel
from deepeval.models.stt_models import OpenAISTTModel
from deepeval.models.tts_models import OpenAITTSModel
from deepeval.simulator import ConversationSimulator
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from deepeval.voice.timeline import build_audio_timeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voice-simulator")
logging.getLogger("deepeval.simulator.conversation_simulator").setLevel(
    logging.DEBUG
)
logging.getLogger("deepeval.voice.connectors.transports.callback").setLevel(
    logging.DEBUG
)


# Stage name -> durations, so a run can be attributed to the simulator side
# or the agent side instead of guessed at from the wall clock.
TIMINGS: Dict[str, List[float]] = defaultdict(list)

# Which side of the call owns each stage.
STAGE_SIDES = {
    "stopping check (LLM)": "simulator",
    "user turn (LLM)": "simulator",
    "barge judge (LLM)": "simulator",
    "caller TTS": "simulator",
    "agent STT": "simulator",
    "caller STT": "walter",
    "Walter LLM": "walter",
    "Walter TTS": "walter",
}


@contextmanager
def timed(stage: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        TIMINGS[stage].append(time.perf_counter() - started)


class TimedTTSModel(OpenAITTSModel):
    """Times every synthesis under a fixed stage name."""

    def __init__(self, stage: str, **kwargs):
        super().__init__(**kwargs)
        self._stage = stage

    async def a_synthesize(self, text: str, **kwargs):
        logger.info("%s started: %d characters", self._stage, len(text))
        with timed(self._stage):
            return await super().a_synthesize(text, **kwargs)


class TimedSTTModel(OpenAISTTModel):
    def __init__(self, stage: str, **kwargs):
        super().__init__(**kwargs)
        self._stage = stage

    async def a_transcribe(self, audio, **kwargs):
        with timed(self._stage):
            return await super().a_transcribe(audio, **kwargs)


class TimedSimulatorModel(OpenAIModel):
    """Attributes each simulator LLM call to a stage via its output schema.

    The simulator asks for `ConversationCompletion` to decide whether the
    conversation is over, `SimulatedInput` to write the caller's next line, and
    `InterruptDecision` while the agent is mid-sentence.
    """

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


class FakeAgent:
    """A stateful text agent wrapped in local callback audio transport."""

    def __init__(self, llm_model: OpenAIModel):
        self.llm_model = llm_model
        self.history: list[tuple[str, str]] = []

    async def __call__(self, user_text: str) -> str:
        logger.info("FakeAgent heard: %r", user_text)
        self.history.append(("customer", user_text))
        conversation = "\n".join(
            f"{role}: {content}" for role, content in self.history[-10:]
        )
        prompt = f"""
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
        with timed("Walter LLM"):
            reply, _ = await self.llm_model.a_generate(prompt)
        reply = str(reply).strip()
        self.history.append(("Walter", reply))
        logger.info("FakeAgent replied: %r", reply)
        return reply


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")

    caller_tts_model = TimedTTSModel(
        "caller TTS",
        voice="coral",
        generation_kwargs={
            "speed": 1.4,
            "instructions": "Speak clearly, briskly, and with mild frustration.",
        },
    )
    agent_tts_model = TimedTTSModel(
        "Walter TTS",
        voice="onyx",
        generation_kwargs={
            "speed": 1.15,
            "instructions": (
                "Speak like a friendly, rambling older man. Use a warm, "
                "unhurried tone with occasional reflective pauses."
            ),
        },
    )
    # Two instances so the caller's speech (Walter's ears) and the agent's
    # speech (the simulator's ears) are timed separately.
    caller_stt_model = TimedSTTModel("caller STT", model="whisper-1")
    agent_stt_model = TimedSTTModel("agent STT", model="whisper-1")
    fake_agent = FakeAgent(OpenAIModel(model="gpt-4o-mini", temperature=0.7))
    connector = CallbackVoiceConnector.from_text_agent(
        fake_agent,
        tts=agent_tts_model,
        stt=caller_stt_model,
        voice="onyx",
    )
    simulator = ConversationSimulator(
        simulator_model=TimedSimulatorModel(),
        voice_config=VoiceConfig(
            connector=connector,
            tts_model=caller_tts_model,
            stt_model=agent_stt_model,
            output_dir="voice_simulations",
            combine_audio_files=True,
        ),
    )

    logger.info(
        "Starting simulation; silence before the first TTS log is simulator LLM time"
    )
    started = time.perf_counter()
    conversations = simulator.simulate(
        [
            ConversationalGolden(
                name="missing-order-test",
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
                        frequency="normal",
                        overlap="adaptive",
                    ),
                ),
                expected_outcome="A replacement order is confirmed.",
            )
        ],
        max_user_simulations=5,
    )
    wall_seconds = time.perf_counter() - started
    logger.info("Simulation finished after %.2fs", wall_seconds)

    print("\nCompleted conversation:")
    for turn in conversations[0].turns:
        print(f"{turn.role}: {turn.content}")

    print_stage_breakdown(wall_seconds)
    print_timeline(conversations[0].turns)
    print("\nAudio saved under ./voice_simulations/")


def print_stage_breakdown(wall_seconds: float) -> None:
    print(f"\nStage breakdown ({wall_seconds:.1f}s wall clock)")
    print(
        f"{'stage':<24}{'side':<11}{'calls':>6}{'total':>9}{'mean':>8}{'%':>7}"
    )
    by_side: Dict[str, float] = defaultdict(float)
    for stage, durations in sorted(
        TIMINGS.items(), key=lambda item: -sum(item[1])
    ):
        total = sum(durations)
        side = STAGE_SIDES.get(stage, "?")
        by_side[side] += total
        print(
            f"{stage:<24}{side:<11}{len(durations):>6}{total:>8.1f}s"
            f"{total / len(durations):>7.1f}s{100 * total / wall_seconds:>6.0f}%"
        )

    accounted = sum(by_side.values())
    print(f"\n{'simulator side':<24}{by_side['simulator']:>8.1f}s")
    print(f"{'Walter side':<24}{by_side['walter']:>8.1f}s")
    unaccounted = wall_seconds - accounted
    if unaccounted >= 0:
        # Mostly the duplex transport playing the agent's downlink at real
        # time (one frame per 20ms) plus end-of-turn silence detection.
        print(f"{'playback / waiting':<24}{unaccounted:>8.1f}s")
    else:
        # Stages ran concurrently: the barge judge polls while the agent
        # speaks, and the connector streams the downlink in the background.
        print(f"{'stage overlap':<24}{-unaccounted:>8.1f}s")


def print_timeline(turns) -> None:
    entries = build_audio_timeline(turns, require_start_times=False)
    if not entries:
        return
    print("\nRecorded timeline (gaps are what you hear as silence)")
    cursor = 0.0
    for entry in entries:
        gap = entry.start_time - cursor
        marker = "overlap" if gap < 0 else "gap"
        print(
            f"{entry.role:<10}{entry.start_time:>7.1f}s -"
            f"{entry.end_time:>7.1f}s   {marker} {abs(gap):>5.1f}s"
        )
        cursor = max(cursor, entry.end_time)
    speech = sum(entry.duration for entry in entries)
    print(f"\nspeech {speech:.1f}s of {cursor:.1f}s recorded")


if __name__ == "__main__":
    main()

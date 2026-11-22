"""End-to-end voice simulation with no network and no API key.

`ConversationSimulator`'s voice path is the only place the connector, the TTS
and STT models, the floor controller, and the audio writer are wired together,
and every one of those seams is exercised here through stubs: an in-process
agent, speech that is a fixed tone, and a transcriber that answers with a
constant. The point is not the audio but the plumbing — that turns come back
alternating and carrying audio, that costs land in the run totals, and that a
persona which barges in produces a barge turn.
"""

import os
from typing import List, Optional, Tuple

import pytest

from deepeval.dataset.golden import (
    ConversationalGolden,
    InterruptionBehavior,
    Persona,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.base_model import DeepEvalBaseSTT, DeepEvalBaseTTS
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Audio, Turn
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.types import ConnectorTurn
from deepeval.voice.output import DEFAULT_VOICE_FOLDER, UNSET

_RATE = 24000
_TTS_COST = 0.5
_STT_COST = 0.25


def _tone(seconds: float, *, sample_rate: int = _RATE) -> Audio:
    pcm = b"\x10\x27" * int(sample_rate * seconds)
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(pcm, sample_rate, 1),
        "audio/wav",
        sampleRate=sample_rate,
        encoding="wav",
        duration=seconds,
    )


class _StubTTS(DeepEvalBaseTTS):
    """Speech as a fixed tone, priced per call so costs are countable."""

    sample_rate = _RATE

    def __init__(self):
        self.spoken: List[str] = []
        super().__init__(model="stub-tts")

    def load_model(self):
        return self

    def synthesize(
        self, text: str, *args, voice: Optional[str] = None, **kwargs
    ) -> Tuple[Audio, Optional[float]]:
        self.spoken.append(text)
        return _tone(0.2), _TTS_COST

    async def a_synthesize(
        self, text: str, *args, voice: Optional[str] = None, **kwargs
    ) -> Tuple[Audio, Optional[float]]:
        return self.synthesize(text, voice=voice, **kwargs)

    def get_model_name(self) -> str:
        return "stub-tts"


class _StubSTT(DeepEvalBaseSTT):
    def __init__(self, transcript: str = "transcribed agent reply"):
        self.transcript = transcript
        self.calls = 0
        super().__init__(model="stub-stt")

    def load_model(self):
        return self

    def transcribe(
        self, audio: Audio, *args, **kwargs
    ) -> Tuple[str, Optional[float]]:
        self.calls += 1
        return self.transcript, _STT_COST

    async def a_transcribe(
        self, audio: Audio, *args, **kwargs
    ) -> Tuple[str, Optional[float]]:
        return self.transcribe(audio, **kwargs)

    def get_model_name(self) -> str:
        return "stub-stt"


class _StaticVoiceModel(DeepEvalBaseLLM):
    """A simulator model that answers every schema the voice path asks for."""

    def __init__(self):
        self.user_turns = 0
        self.schemas: List[str] = []
        super().__init__(model="static-voice-model")

    def load_model(self):
        return self

    def generate(self, prompt: str, schema=None):
        if schema is None:
            return '{"simulated_input": "simulated user input"}'
        self.schemas.append(schema.__name__)
        if schema.__name__ == "SimulatedInput":
            self.user_turns += 1
            return schema(simulated_input=f"user turn {self.user_turns}")
        if schema.__name__ == "ConversationCompletion":
            return schema(is_complete=False, reason="not done")
        if schema.__name__ == "EdgeChoice":
            return schema(index=None, reason="static")
        if schema.__name__ == "InterruptUtterance":
            return schema(utterance="Actually, wait.")
        raise AssertionError(f"Unexpected schema: {schema.__name__}")

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return "static-voice-model"


def _agent(reply: str = "the agent speaks", seconds: float = 0.4):
    async def agent(_audio: Audio) -> ConnectorTurn:
        return ConnectorTurn(audio=_tone(seconds), transcript=reply)

    return agent


def _agent_with_lengths(seconds: List[float]):
    replies = iter(seconds)

    async def agent(_audio: Audio) -> ConnectorTurn:
        length = next(replies, seconds[-1])
        return ConnectorTurn(audio=_tone(length), transcript="the agent speaks")

    return agent


def _simulate(
    *,
    persona: Optional[Persona] = None,
    agent=None,
    output_dir: Optional[str] = None,
    max_user_simulations: int = 2,
    turn_detection: str = "eager",
    reply: str = "the agent speaks",
    reply_seconds: float = 0.4,
):
    tts, stt = _StubTTS(), _StubSTT()
    connector = CallbackVoiceConnector(
        agent or _agent(reply, reply_seconds), turn_detection=turn_detection
    )
    simulator = ConversationSimulator(
        voice_config=VoiceConfig(
            connector=connector,
            tts_model=tts,
            stt_model=stt,
            output_dir=output_dir,
        ),
        simulator_model=_StaticVoiceModel(),
    )
    golden = ConversationalGolden(
        scenario="Ask the agent a question",
        persona=persona
        or Persona(characteristics="A caller with a simple question."),
    )
    cases = simulator.simulate(
        [golden], max_user_simulations=max_user_simulations
    )
    return simulator, cases[0]


def test_half_duplex_voice_simulation_produces_spoken_turns():
    simulator, case = _simulate()

    assert [turn.role for turn in case.turns] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    # Every turn was actually spoken and heard, not just text.
    assert all(turn.audio is not None for turn in case.turns)
    # The connector supplied a transcript, so STT was skipped for the agent.
    assert {
        turn.content for turn in case.turns if turn.role == "assistant"
    } == {"the agent speaks"}


def test_speech_costs_land_in_the_run_totals():
    simulator, case = _simulate()

    user_turns = sum(1 for turn in case.turns if turn.role == "user")
    assert simulator.tts_cost == pytest.approx(_TTS_COST * user_turns)
    # The connector carried a transcript for every reply, so nothing needed
    # transcribing and no STT was billed.
    assert simulator.stt_cost == 0.0


def test_costs_are_zero_for_a_text_simulator():
    simulator = ConversationSimulator(
        model_callback=lambda input: Turn(role="assistant", content="hi"),
        simulator_model=_StaticVoiceModel(),
    )
    assert simulator.tts_cost == 0.0
    assert simulator.stt_cost == 0.0


def _barges(case) -> List[Turn]:
    return [
        turn
        for turn in case.turns
        if turn.role == "user" and (turn.metadata or {}).get("barge_in")
    ]


def test_a_barging_persona_cuts_into_the_scheduled_reply():
    persona = Persona(
        characteristics="An impatient caller.",
        interruption_behavior=InterruptionBehavior(every=1, overlap="insist"),
    )
    _, case = _simulate(
        persona=persona,
        max_user_simulations=1,
        turn_detection="balanced",
        reply_seconds=3.0,
    )

    barges = _barges(case)
    assert len(barges) == 1, "the caller was scheduled to interrupt"
    assert barges[0].content == "Actually, wait."
    cut = case.turns[case.turns.index(barges[0]) - 1]
    assert cut.role == "assistant" and cut.interrupted is True
    lead = barges[0].audio.start_time - cut.audio.start_time
    assert 0.5 <= lead < 1.5
    assert cut.audio.duration < 3.0
    answers = [
        turn
        for turn in case.turns[case.turns.index(barges[0]) + 1 :]
        if turn.role == "assistant"
    ]
    assert answers and all(turn.interrupted is None for turn in answers)


def test_a_reply_too_short_to_cut_into_carries_the_interruption():
    persona = Persona(
        characteristics="An impatient caller.",
        interruption_behavior=InterruptionBehavior(every=2, overlap="insist"),
    )
    _, case = _simulate(
        persona=persona,
        agent=_agent_with_lengths([0.4, 0.2, 3.0, 3.0]),
        max_user_simulations=3,
        turn_detection="balanced",
    )

    replies = [turn for turn in case.turns if turn.role == "assistant"]
    barges = _barges(case)
    assert len(barges) == 1
    assert [turn.interrupted for turn in replies[:3]] == [None, None, True]
    assert case.turns.index(barges[0]) == case.turns.index(replies[2]) + 1


def test_conversation_audio_is_written_to_disk(tmp_path):
    _, case = _simulate(output_dir=str(tmp_path), max_user_simulations=1)

    runs = sorted(tmp_path.glob("simulation-*"))
    assert len(runs) == 1
    written = {path.name for path in runs[0].iterdir()}
    assert "deepeval-conversation.wav" in written
    assert any(name.startswith("deepeval-turn-1-") for name in written)


def test_audio_lands_in_the_default_folder_when_none_is_given(
    tmp_path, monkeypatch
):
    """Nothing configured: recordings still get kept, just not loose in the cwd."""
    monkeypatch.chdir(tmp_path)

    _simulate(output_dir=UNSET, max_user_simulations=1)

    runs = sorted((tmp_path / DEFAULT_VOICE_FOLDER).glob("simulation-*"))
    assert len(runs) == 1
    assert (runs[0] / "deepeval-conversation.wav").is_file()


def test_muted_persona_still_alternates_turns():
    persona = Persona(characteristics="A caller who never speaks.", muted=True)
    _, case = _simulate(persona=persona, max_user_simulations=2)

    assert [turn.role for turn in case.turns] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert all(turn.content == "" for turn in case.turns if turn.role == "user")

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
    BackgroundNoiseSettings,
    BackgroundSpeaker,
    InterruptionBehavior,
    LineSettings,
    Persona,
    SpeechPatternSettings,
    VoiceSettings,
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

    def __init__(self, cost: float = _TTS_COST, instructions: bool = False):
        self.spoken: List[str] = []
        self.kwargs: List[dict] = []
        self.cost = cost
        self.instructions = instructions
        super().__init__(model="stub-tts")

    def load_model(self):
        return self

    def synthesize(
        self, text: str, *args, voice: Optional[str] = None, **kwargs
    ) -> Tuple[Audio, Optional[float]]:
        self.spoken.append(text)
        self.kwargs.append(kwargs)
        return _tone(0.2), self.cost

    async def a_synthesize(
        self, text: str, *args, voice: Optional[str] = None, **kwargs
    ) -> Tuple[Audio, Optional[float]]:
        return self.synthesize(text, voice=voice, **kwargs)

    def supports_voice_instructions(self) -> bool:
        return self.instructions

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

    def __init__(self, *, interrupt: bool = False):
        self.interrupt = interrupt
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
        if schema.__name__ == "CallerVoiceProfile":
            return schema(
                delivery="An older man, slow and tired.",
                asides=["Put that down."],
                background_lines=[["Is it my turn yet?"]],
            )
        if schema.__name__ == "InterruptDecision":
            return schema(
                should_interrupt=self.interrupt,
                utterance="Actually, wait." if self.interrupt else "",
                reason="static",
            )
        raise AssertionError(f"Unexpected schema: {schema.__name__}")

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return "static-voice-model"


def _agent(reply: str = "the agent speaks", seconds: float = 0.4):
    async def agent(_audio: Audio) -> ConnectorTurn:
        return ConnectorTurn(audio=_tone(seconds), transcript=reply)

    return agent


def _simulate(
    *,
    persona: Optional[Persona] = None,
    interrupt: bool = False,
    output_dir: Optional[str] = None,
    max_user_simulations: int = 2,
    turn_detection: str = "eager",
    reply: str = "the agent speaks",
    reply_seconds: float = 0.4,
    tts: Optional[DeepEvalBaseTTS] = None,
    persona_delivery: bool = True,
):
    tts, stt = tts or _StubTTS(), _StubSTT()
    connector = CallbackVoiceConnector(
        _agent(reply, reply_seconds), turn_detection=turn_detection
    )
    simulator = ConversationSimulator(
        voice_config=VoiceConfig(
            connector=connector,
            tts_model=tts,
            stt_model=stt,
            output_dir=output_dir,
            persona_delivery=persona_delivery,
        ),
        simulator_model=_StaticVoiceModel(interrupt=interrupt),
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


def test_a_barging_persona_produces_a_barge_turn():
    persona = Persona(
        characteristics="An impatient caller.",
        interruption_behavior=InterruptionBehavior(
            frequency="frequent", overlap="insist"
        ),
    )
    # The judge is only asked once enough of the reply has been *heard*, so the
    # agent needs both a long sentence and long enough audio to carry it.
    _, case = _simulate(
        persona=persona,
        interrupt=True,
        max_user_simulations=1,
        turn_detection="balanced",
        reply=(
            "I will need your order number before I can look into any of this, "
            "and it takes a moment to pull up once you give it to me."
        ),
        reply_seconds=3.0,
    )

    barges = [
        turn
        for turn in case.turns
        if turn.role == "user" and (turn.metadata or {}).get("barge_in")
    ]
    assert barges, "the caller was configured to interrupt but never did"
    assert barges[0].content == "Actually, wait."


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


def test_a_persona_tts_model_speaks_and_is_billed_instead():
    simulator_tts, persona_tts = _StubTTS(), _StubTTS(cost=2.0)
    persona = Persona(characteristics="A caller.", tts_model=persona_tts)

    simulator, case = _simulate(persona=persona, tts=simulator_tts)

    user_turns = sum(1 for turn in case.turns if turn.role == "user")
    assert simulator_tts.spoken == []
    assert len(persona_tts.spoken) == user_turns
    assert simulator.tts_cost == pytest.approx(2.0 * user_turns)


def test_a_phone_line_persona_is_heard_through_the_phone_line():
    persona = Persona(
        characteristics="A caller.", line=LineSettings(quality="phone")
    )

    simulator, case = _simulate(persona=persona, max_user_simulations=1)

    user_audio = next(turn.audio for turn in case.turns if turn.role == "user")
    assert user_audio.get_bytes() != _tone(0.2).get_bytes()
    assert simulator.tts_cost == pytest.approx(_TTS_COST)


def test_voice_instructions_reach_a_tts_model_that_takes_them():
    tts = _StubTTS(instructions=True)
    persona = Persona(
        characteristics="A caller.", voice=VoiceSettings(instructions="Calm.")
    )

    _simulate(
        persona=persona, tts=tts, max_user_simulations=1, persona_delivery=False
    )

    assert tts.kwargs[0]["instructions"].startswith("Calm. Hold this delivery")


def test_voice_instructions_are_ignored_by_a_tts_model_that_cannot_take_them():
    tts = _StubTTS()
    persona = Persona(
        characteristics="A caller.", voice=VoiceSettings(instructions="Calm.")
    )

    with pytest.warns(UserWarning, match="instructions are ignored"):
        _simulate(persona=persona, tts=tts, max_user_simulations=1)

    assert "instructions" not in tts.kwargs[0]


def test_the_delivery_is_worked_out_from_the_characteristics():
    tts = _StubTTS(instructions=True)
    persona = Persona(
        characteristics="A retired teacher who is worn out.",
        voice=VoiceSettings(accent="Welsh", emotion="weary"),
    )

    simulator, _ = _simulate(persona=persona, tts=tts, max_user_simulations=1)

    instructions = tts.kwargs[0]["instructions"]
    assert instructions.startswith("An older man, slow and tired. ")
    assert "extremely thick, heavy Welsh accent" in instructions
    assert "overwhelmingly weary" in instructions
    assert instructions.endswith("not a narrator.")
    assert simulator.simulator_model.schemas[0] == "CallerVoiceProfile"


def test_the_voice_settings_are_given_to_whoever_writes_the_direction():
    from deepeval.simulator.caller_voice_template import CallerVoiceTemplate

    persona = Persona(
        characteristics="A retired teacher.",
        voice=VoiceSettings(accent="Welsh", emotion="weary", pace="slow"),
    )

    prompt = CallerVoiceTemplate.describe_caller_voice(
        persona,
        "English",
        delivery=True,
        asides=False,
        background_speakers=[],
    )

    assert "Accent: Welsh, extreme" in prompt
    assert "Emotion: weary, extreme" in prompt
    assert "Pace: slow" in prompt
    assert "Affect: ... Tone: ... Pacing: ... Emotion: ..." in prompt


def test_a_persona_with_no_voice_settings_gets_no_direction():
    assert Persona(characteristics="A caller.").delivery_instructions() is None


def test_no_delivery_is_asked_for_when_the_tts_model_cannot_use_it():
    simulator, _ = _simulate(
        persona=Persona(characteristics="A caller."), max_user_simulations=1
    )

    assert "CallerVoiceProfile" not in simulator.simulator_model.schemas


def test_persona_delivery_can_be_switched_off():
    tts = _StubTTS(instructions=True)

    simulator, _ = _simulate(
        persona=Persona(characteristics="A caller."),
        tts=tts,
        max_user_simulations=1,
        persona_delivery=False,
    )

    assert "instructions" not in tts.kwargs[0]
    assert "CallerVoiceProfile" not in simulator.simulator_model.schemas


def _every_time(monkeypatch) -> None:
    from deepeval.voice.caller_speech import PATTERN_CHANCE

    for frequency in PATTERN_CHANCE:
        monkeypatch.setitem(PATTERN_CHANCE, frequency, 1.0)


def test_a_background_speaker_is_heard_saying_the_lines_written_for_them(
    monkeypatch,
):
    _every_time(monkeypatch)
    tts = _StubTTS()
    persona = Persona(
        characteristics="A parent at home.",
        background_noise=BackgroundNoiseSettings(
            other_speakers=[
                BackgroundSpeaker(
                    characteristics="A child waiting for a turn.",
                    voice="nova",
                    frequency="frequent",
                )
            ]
        ),
    )

    _simulate(persona=persona, tts=tts, max_user_simulations=1)

    assert "Is it my turn yet?" in tts.spoken


def test_side_speech_uses_the_asides_written_for_the_persona(monkeypatch):
    _every_time(monkeypatch)
    tts = _StubTTS()
    persona = Persona(
        characteristics="A parent at home.",
        speech_patterns=SpeechPatternSettings(side_speech="frequent"),
    )

    _simulate(persona=persona, tts=tts, max_user_simulations=1)

    assert "Put that down." in tts.spoken


def test_a_dated_snapshot_takes_voice_instructions_as_its_model_does():
    from deepeval.models.tts import OpenAITTSModel

    snapshot = OpenAITTSModel(
        model="gpt-4o-mini-tts-2025-03-20", api_key="test"
    )

    assert snapshot.supports_voice_instructions() is True


def test_only_gpt_4o_mini_tts_takes_voice_instructions():
    from deepeval.models.tts import OpenAITTSModel
    from deepeval.voice.phone_line import PhoneLineTTS

    mini = OpenAITTSModel(model="gpt-4o-mini-tts", api_key="test")

    assert mini.supports_voice_instructions() is True
    assert PhoneLineTTS(mini).supports_voice_instructions() is True
    assert (
        OpenAITTSModel(
            model="tts-1", api_key="test"
        ).supports_voice_instructions()
        is False
    )


def test_persona_tts_kwargs_carry_instructions_only_for_models_that_take_them():
    persona = Persona(
        characteristics="A caller.",
        voice=VoiceSettings(id="coral", instructions="Calm."),
    )

    kwargs = persona.tts_kwargs(_StubTTS(instructions=True))
    assert kwargs["voice"] == "coral"
    assert kwargs["instructions"].startswith("Calm. Hold this delivery")
    with pytest.warns(UserWarning):
        assert persona.tts_kwargs(_StubTTS()) == {"voice": "coral"}


def test_the_tts_model_stays_out_of_the_serialized_persona():
    persona = Persona(
        characteristics="A caller.",
        tts_model=_StubTTS(),
        line=LineSettings(quality="phone"),
    )

    assert persona.model_dump(exclude_defaults=True, mode="json") == {
        "characteristics": "A caller.",
        "line": {"quality": "phone"},
    }


def test_a_bare_voice_id_and_the_flat_names_still_build_a_persona():
    persona = Persona(
        characteristics="A caller.",
        voice="coral",
        voice_instructions="Calm.",
        line_quality="phone",
    )

    assert persona.voice == VoiceSettings(id="coral", instructions="Calm.")
    assert persona.line == LineSettings(quality="phone")


def test_a_saved_persona_loads_back_the_same():
    persona = Persona(
        characteristics="A caller.",
        languages=["English", "Spanish"],
        voice=VoiceSettings(id="coral", accent="Scottish"),
        line=LineSettings(dropouts="rare", static=0.2, delay_ms=300),
        speech_patterns=SpeechPatternSettings(
            fillers="frequent", switches_language="rare"
        ),
        background_noise=BackgroundNoiseSettings(
            other_speakers=[BackgroundSpeaker(characteristics="A child.")]
        ),
    )

    saved = persona.model_dump(exclude_defaults=True, mode="json")

    assert Persona(**saved) == persona


def test_switching_languages_needs_two_languages():
    with pytest.raises(ValueError, match="switches_language"):
        Persona(
            characteristics="A caller.",
            speech_patterns=SpeechPatternSettings(switches_language="rare"),
        )


def test_wording_patterns_and_emotion_are_told_to_the_simulator_model():
    persona = Persona(
        characteristics="A caller.",
        voice=VoiceSettings(emotion="frustrated"),
        speech_patterns=SpeechPatternSettings(
            fillers="frequent",
            self_corrects="rare",
            mid_sentence_pauses="normal",
        ),
    )

    block = persona.prompt_block()

    assert "Right now you feel very frustrated." in block
    assert (
        "filler words such as um, uh, well or you know in almost every" in block
    )
    assert "correct yourself mid-sentence once in a while" in block
    assert "pause" not in block


@pytest.mark.parametrize(
    "strength, wording",
    [
        (0.0, "a light Welsh"),
        (0.5, "a strong Welsh"),
        (1.0, "extremely thick"),
    ],
)
def test_accent_strength_picks_how_the_accent_is_asked_for(strength, wording):
    persona = Persona(
        characteristics="A caller.",
        voice=VoiceSettings(accent="Welsh", accent_strength=strength),
    )

    assert wording in persona.delivery_instructions()


@pytest.mark.parametrize(
    "strength, wording",
    [
        (0.2, "a little weary"),
        (0.5, "very weary"),
        (0.9, "overwhelmingly weary"),
    ],
)
def test_emotion_strength_picks_how_the_emotion_is_asked_for(strength, wording):
    persona = Persona(
        characteristics="A caller.",
        voice=VoiceSettings(emotion="weary", emotion_strength=strength),
    )

    assert wording in persona.delivery_instructions()


def test_emotion_strength_also_reaches_the_words():
    def feels(strength: float) -> str:
        persona = Persona(
            characteristics="A caller.",
            voice=VoiceSettings(emotion="weary", emotion_strength=strength),
        )
        return persona.prompt_block()

    assert "you feel a little weary." in feels(0.1)
    assert "you feel weary." in feels(0.5)
    assert "you feel very weary." in feels(1.0)

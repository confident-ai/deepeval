from types import SimpleNamespace

from deepeval.dataset import Persona, golden_persona
from deepeval.simulator import ConversationSimulator
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from tests.test_core.test_simulator.helpers import (
    StaticSimulatorModel,
    async_static_callback,
)
from tests.test_core.test_voice.helpers import EchoAgent, StubSTT, StubTTS


def _golden_without_persona(user_description=None) -> SimpleNamespace:
    return SimpleNamespace(
        scenario="Refund",
        expected_outcome="The refund is issued.",
        user_description=user_description,
        context=None,
        name=None,
        additional_metadata=None,
        comments=None,
        turns=None,
        multimodal=False,
        _dataset_rank=None,
        _dataset_alias=None,
        _dataset_id=None,
    )


def test_golden_persona_prefers_an_explicit_persona():
    persona = Persona(characteristics="Blunt")
    golden = SimpleNamespace(persona=persona, user_description="ignored")

    assert golden_persona(golden) is persona


def test_golden_persona_falls_back_to_the_user_description():
    persona = golden_persona(_golden_without_persona("A hurried caller."))

    assert persona is not None
    assert persona.characteristics == "A hurried caller."


def test_golden_persona_is_none_without_persona_or_description():
    assert golden_persona(_golden_without_persona()) is None
    assert golden_persona(_golden_without_persona("")) is None


def test_text_simulation_accepts_goldens_without_a_persona_attribute():
    model = StaticSimulatorModel()
    cases = ConversationSimulator(
        model_callback=async_static_callback, simulator_model=model
    ).simulate(
        [_golden_without_persona("A polite but hurried caller.")],
        max_user_simulations=1,
    )

    assert [turn.role for turn in cases[0].turns] == ["user", "assistant"]
    assert cases[0].additional_metadata == {
        "Persona": "A polite but hurried caller."
    }
    assert "A polite but hurried caller." in model.prompts[0]


def test_voice_simulation_accepts_goldens_without_a_persona_attribute():
    cases = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=CallbackVoiceConnector(EchoAgent()),
            tts_model=StubTTS(),
            stt_model=StubSTT(),
            output_dir=None,
            combine_audio_files=False,
        ),
    ).simulate([_golden_without_persona()], max_user_simulations=1)

    assert [turn.role for turn in cases[0].turns] == ["user", "assistant"]
    assert cases[0].additional_metadata == {"Persona": None}
    assert all(turn.audio is not None for turn in cases[0].turns)

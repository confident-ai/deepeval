import warnings

import pytest

from deepeval.dataset import (
    BackgroundNoiseSettings,
    ConversationalGolden,
    InterruptionBehavior,
    Persona,
)
from deepeval.constants import ProviderSlug
from deepeval.dataset.utils import (
    convert_convo_goldens_to_convo_test_cases,
    convert_convo_test_cases_to_convo_goldens,
    serialize_persona,
)
from deepeval.test_case import Turn


class _TTS:
    def __init__(self, takes_instructions: bool = False, provider=None):
        self.takes_instructions = takes_instructions
        self.provider = provider

    def supports_voice_instructions(self) -> bool:
        return self.takes_instructions

    async def a_synthesize(self, text: str, **kwargs):
        raise NotImplementedError


class TestPersonaPrompt:
    def test_prompt_block_wraps_the_characteristics(self):
        persona = Persona(characteristics="  You are in a hurry.  ")

        assert (
            persona.prompt_block()
            == "<persona>\nYou are in a hurry.\n</persona>"
        )

    def test_prompt_block_includes_name_and_behavior(self):
        persona = Persona(
            name="Dana",
            characteristics="You are blunt and short-tempered.",
            interruption_behavior=InterruptionBehavior(
                frequency="frequent", overlap="insist"
            ),
            speaks_first=False,
        )

        block = persona.prompt_block()

        assert block.startswith("<persona>\nName: Dana")
        assert "You are blunt and short-tempered." in block
        assert "You often cut in while the agent is still speaking." in block
        assert "you keep talking until the agent stops." in block
        assert "You wait for the agent to speak first" in block

    def test_prompt_block_renders_metadata_as_details(self):
        persona = Persona(
            characteristics="A returning customer.",
            metadata={"accountId": "acc_123", "plan": "pro"},
        )

        block = persona.prompt_block()

        assert "A returning customer." in block
        assert (
            'Details:\n{\n  "accountId": "acc_123",\n  "plan": "pro"\n}'
            in block
        )

    def test_prompt_block_omits_empty_metadata(self):
        block = Persona(characteristics="Anyone", metadata={}).prompt_block()

        assert "Details:" not in block

    def test_prompt_block_omits_behavior_by_default(self):
        block = Persona(characteristics="Calm caller.").prompt_block()

        assert "cut in" not in block
        assert "wait for the agent" not in block


class TestPersonaVoice:
    def test_tts_kwargs_are_empty_without_a_voice(self):
        assert Persona(characteristics="Anyone").tts_kwargs() == {}

    def test_tts_kwargs_carry_the_voice(self):
        persona = Persona(characteristics="Anyone", voice="coral")

        assert persona.tts_kwargs() == {"voice": "coral"}

    def test_voice_instructions_reach_a_model_that_takes_them(self):
        persona = Persona(
            characteristics="Anyone",
            voice="coral",
            voice_instructions="Scottish accent, speaks quickly.",
        )

        assert persona.tts_kwargs(_TTS(takes_instructions=True)) == {
            "voice": "coral",
            "instructions": "Scottish accent, speaks quickly.",
        }

    def test_voice_instructions_stay_off_a_model_that_does_not(self):
        persona = Persona(
            characteristics="Anyone", voice_instructions="Scottish accent."
        )

        assert persona.tts_kwargs(_TTS(takes_instructions=False)) == {}
        assert persona.tts_kwargs(object()) == {}
        assert persona.tts_kwargs() == {}

    def test_only_gpt_4o_mini_tts_takes_voice_instructions(self):
        from deepeval.models import OpenAITTSModel

        def takes(model: str) -> bool:
            return OpenAITTSModel(
                model=model, api_key="test"
            ).supports_voice_instructions()

        assert takes("gpt-4o-mini-tts")
        assert takes("gpt-4o-mini-tts-2025-03-20")
        assert not takes("tts-1")
        assert not takes("tts-1-hd")

    def test_a_voice_per_provider_speaks_with_the_models_own(self):
        persona = Persona(
            characteristics="Anyone",
            voice={"openai": "ash", "elevenlabs": "el-voice"},
        )

        assert persona.tts_kwargs(_TTS(provider="elevenlabs")) == {
            "voice": "el-voice"
        }
        assert persona.tts_kwargs(_TTS(provider=ProviderSlug.OPENAI)) == {
            "voice": "ash"
        }

    def test_a_provider_without_a_voice_gets_the_models_default(self):
        persona = Persona(characteristics="Anyone", voice={"openai": "ash"})

        assert persona.tts_kwargs(_TTS(provider="cartesia")) == {}
        assert persona.tts_kwargs(_TTS(provider=None)) == {}
        assert persona.tts_kwargs() == {}

    def test_voice_providers_are_keyed_by_slug(self):
        persona = Persona(
            characteristics="Anyone",
            voice={"OpenAI": "ash", ProviderSlug.DEEPGRAM: "zeus"},
        )

        assert persona.voice == {"openai": "ash", "deepgram": "zeus"}

    def test_rejects_an_unknown_voice_provider(self):
        with pytest.raises(ValueError, match="Unknown voice provider"):
            Persona(characteristics="Anyone", voice={"open_ai": "ash"})

    def test_each_built_in_tts_model_names_its_provider(self):
        from deepeval.models import (
            CartesiaTTSModel,
            DeepgramTTSModel,
            ElevenLabsTTSModel,
            OpenAITTSModel,
        )

        assert OpenAITTSModel.provider == "openai"
        assert ElevenLabsTTSModel.provider == "elevenlabs"
        assert CartesiaTTSModel.provider == "cartesia"
        assert DeepgramTTSModel.provider == "deepgram"

    def test_tts_model_takes_a_name_or_a_model(self):
        model = _TTS()

        assert Persona(characteristics="Anyone").tts_model is None
        assert (
            Persona(characteristics="Anyone", tts_model="tts-1").tts_model
            == "tts-1"
        )
        assert (
            Persona(characteristics="Anyone", tts_model=model).tts_model
            is model
        )

    def test_rejects_a_tts_model_that_cannot_synthesize(self):
        with pytest.raises(ValueError, match="tts_model must be"):
            Persona(characteristics="Anyone", tts_model=5)

    def test_files_keep_a_tts_model_name_but_not_a_model_object(self):
        named = Persona(characteristics="Anyone", tts_model="tts-1")
        built = Persona(characteristics="Anyone", tts_model=_TTS())

        assert serialize_persona(named)["tts_model"] == "tts-1"
        assert serialize_persona(built) == {"characteristics": "Anyone"}

    def test_line_quality_defaults_to_wideband(self):
        assert Persona(characteristics="Anyone").line_quality == "wideband"

    def test_rejects_an_unknown_line_quality(self):
        with pytest.raises(ValueError):
            Persona(characteristics="Anyone", line_quality="satellite")

    def test_background_volume_is_bounded(self):
        with pytest.raises(ValueError):
            BackgroundNoiseSettings(audio="cafe.wav", volume=1.5)

    def test_hold_timeout_must_be_positive(self):
        with pytest.raises(ValueError):
            Persona(characteristics="Anyone", hold_timeout=0)


class TestInterruptionBehavior:
    def test_rejects_an_unknown_frequency(self):
        with pytest.raises(ValueError, match="interruption frequency"):
            InterruptionBehavior(frequency="always")

    def test_rejects_an_unknown_overlap(self):
        with pytest.raises(ValueError, match="overlap behavior"):
            InterruptionBehavior(overlap="shout")


class TestUserDescriptionDeprecation:
    def test_user_description_warns_and_builds_a_persona(self):
        with pytest.warns(DeprecationWarning, match="user_description"):
            golden = ConversationalGolden(
                scenario="Refund", user_description="An impatient caller."
            )

        assert golden.persona == Persona(characteristics="An impatient caller.")
        assert golden.user_description == "An impatient caller."

    def test_persona_backfills_user_description(self):
        golden = ConversationalGolden(
            scenario="Refund",
            persona=Persona(
                name="Dana", characteristics="An impatient caller."
            ),
        )

        assert golden.user_description == "An impatient caller."

    def test_rejects_conflicting_text(self):
        with pytest.raises(ValueError, match="conflicting text"):
            ConversationalGolden(
                scenario="Refund",
                persona=Persona(characteristics="An impatient caller."),
                user_description="A patient caller.",
            )

    def test_neither_field_stays_none(self):
        golden = ConversationalGolden(scenario="Refund")

        assert golden.persona is None
        assert golden.user_description is None


class TestTestCaseRoundTrip:
    def test_golden_flattens_its_persona_onto_the_test_case(self):
        golden = ConversationalGolden(
            scenario="Refund",
            turns=[Turn(role="user", content="Hi")],
            persona=Persona(
                characteristics="An impatient caller.", voice="coral"
            ),
        )

        test_case = convert_convo_goldens_to_convo_test_cases([golden])[0]

        assert test_case.user_description == "An impatient caller."

    def test_test_case_converts_back_without_a_deprecation_warning(self):
        golden = ConversationalGolden(
            scenario="Refund",
            turns=[Turn(role="user", content="Hi")],
            persona=Persona(characteristics="An impatient caller."),
        )
        test_case = convert_convo_goldens_to_convo_test_cases([golden])[0]

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            round_tripped = convert_convo_test_cases_to_convo_goldens(
                [test_case]
            )[0]

        assert round_tripped.persona == Persona(
            characteristics="An impatient caller."
        )

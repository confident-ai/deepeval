import warnings

import pytest

from deepeval.dataset import (
    BackgroundNoiseSettings,
    ConversationalGolden,
    InterruptionBehavior,
    Persona,
)
from deepeval.dataset.utils import (
    convert_convo_goldens_to_convo_test_cases,
    convert_convo_test_cases_to_convo_goldens,
)
from deepeval.test_case import Turn


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

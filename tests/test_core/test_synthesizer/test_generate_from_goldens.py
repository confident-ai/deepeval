import pytest
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import Golden, ConversationalGolden
from typing import List

original_goldens: List[Golden] = [
    Golden(
        input="What is the capital of France?",
        output="The capital of France is Paris.",
    ),
    Golden(
        input="What is the largest planet in our solar system?",
        output="The largest planet in our solar system is Jupiter.",
    ),
]

conversational_goldens = [
    ConversationalGolden(
        scenario="On a snowy evening before the school science fair, a parent and their child rehearse an experiment at the kitchen sink, debating how H2O molecules behave as the tap water in their beaker approaches 0°C and why the freezing point matters for their project display.",
        expected_outcome="The parent and child clarify that H2O molecules slow down and form a solid structure as water reaches 0°C, understanding why the freezing point is important for their science fair project.",
    ),
    ConversationalGolden(
        scenario="At a science museum, a child asks their parent why Earth is considered a planet and how its movement around the Sun differs from other celestial bodies, prompting a multi-step discussion about planetary classification, Earth's orbit, and the distinction between planets and other objects in the solar system.",
        expected_outcome="The child learns that Earth is considered a planet because it orbits the Sun, is spherical, and clears its orbit, and understands how this distinguishes planets from other celestial bodies like asteroids and comets.",
    ),
]


@pytest.fixture
def synthesizer():
    return Synthesizer()


def test_expand_dataset_from_inputs(synthesizer: Synthesizer):
    goldens = synthesizer.generate_goldens_from_goldens(original_goldens)
    assert goldens is not None, "Generated goldens should not be None"
    assert isinstance(
        goldens, list
    ), "Generated goldens should be a list of Golden objects"
    assert len(goldens) > 0, "Should generate at least one golden"
    assert all(
        isinstance(g, Golden) for g in goldens
    ), "All items should be Golden instances"


def test_expand_dataset_from_scenarios(synthesizer: Synthesizer):
    goldens = synthesizer.generate_conversational_goldens_from_goldens(
        conversational_goldens
    )
    assert goldens is not None, "Generated convo goldens should not be None"
    assert isinstance(
        goldens, list
    ), "Generated goldens should be a list of ConversationalGolden objects"
    assert len(goldens) > 0, "Should generate at least one convo golden"
    assert all(
        isinstance(g, ConversationalGolden) for g in goldens
    ), "All items should be ConversationalGolden instances"

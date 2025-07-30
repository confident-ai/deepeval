import pytest
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import Golden
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

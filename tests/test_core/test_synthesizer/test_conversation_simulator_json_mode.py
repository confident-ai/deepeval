import pytest
import deepeval.simulator.conversation_simulator as cs

from deepeval.dataset.golden import ConversationalGolden
from deepeval.test_case.conversational_test_case import Turn
from tests.test_core.stubs import AlwaysJsonModel


extractor = AlwaysJsonModel.balanced_json_after_anchor("Example JSON Output:")
model = AlwaysJsonModel(extractor)


def assistant_ok_callback(input: str, turns, thread_id=None) -> Turn:
    return Turn(role="assistant", content="ok")


@pytest.fixture
def patch_initialize_model(monkeypatch):
    monkeypatch.setattr(
        cs,
        "initialize_model",
        lambda _sim_model: (model, False),
        raising=True,
    )


def _golden():
    # Setting expected_outcome causes stop_conversation() to run first.
    # That path builds the prompt from the template's stop_simulation.
    return ConversationalGolden(
        scenario="Forgot password and needs reset",
        expected_outcome="User successfully resets password.",
        user_description="Samwise Gamgee",
        turns=[],
    )


@pytest.mark.parametrize("async_mode", [False, True])
def test_simulator_handles_example_json_from_prompt(
    patch_initialize_model, async_mode
):
    """
    RED: The template's Example JSON uses 'True' or 'False' resulting in trimAndLoadJson raises.
    GREEN after fix: Change 'False' -> 'false' in the template; parsing succeeds.
    """
    sim = cs.ConversationSimulator(
        simulator_model="whatever",
        model_callback=assistant_ok_callback,
        async_mode=async_mode,
    )

    # Expect this to run without exception after template fix.
    cases = sim.simulate([_golden()], max_user_simulations=1)
    assert isinstance(cases, list) and len(cases) == 1
    tc = cases[0]
    assert len(tc.turns) >= 2
    assert tc.turns[0].role == "user"
    assert tc.turns[1].role == "assistant"

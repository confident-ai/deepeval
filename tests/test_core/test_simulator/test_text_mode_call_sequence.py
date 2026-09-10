from deepeval.dataset import ConversationalGolden
from deepeval.simulator import ConversationSimulator
from tests.test_core.test_simulator.helpers import (
    StaticSimulatorModel,
    async_static_callback,
)


def test_text_mode_makes_the_same_calls_in_the_same_order():
    model = StaticSimulatorModel()
    simulator = ConversationSimulator(
        model_callback=async_static_callback, simulator_model=model
    )

    cases = simulator.simulate(
        [
            ConversationalGolden(
                scenario="Refund", expected_outcome="The refund is issued."
            )
        ],
        max_user_simulations=3,
    )

    assert model.schema_calls == [
        "SimulatedInput",
        "ConversationCompletion",
        "SimulatedInput",
        "ConversationCompletion",
        "SimulatedInput",
    ]
    assert [turn.role for turn in cases[0].turns] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [turn.content for turn in cases[0].turns] == [
        "simulated user input 1",
        "Assistant response to simulated user input 1",
        "simulated user input 2",
        "Assistant response to simulated user input 2",
        "simulated user input 3",
        "Assistant response to simulated user input 3",
    ]
    assert all(turn.audio is None for turn in cases[0].turns)
    assert simulator.model_callback is async_static_callback
    assert simulator.tts_cost == 0.0 and simulator.stt_cost == 0.0

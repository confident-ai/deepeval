from deepeval.dataset.golden import ConversationalGolden
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.controller import end, proceed
from tests.test_core.test_simulator.helpers import (
    StaticSimulatorModel,
    async_static_callback,
    static_callback,
)


def test_sync_controller_can_end_simulation():
    golden = ConversationalGolden(
        scenario="Purchase a concert ticket",
        expected_outcome=None,
        user_description="Test User",
    )

    controller_calls = []

    def controller(last_assistant_turn, simulated_user_turns):
        controller_calls.append(
            {
                "last_assistant_turn": last_assistant_turn,
                "simulated_user_turns": simulated_user_turns,
            }
        )
        if last_assistant_turn is not None:
            return end(reason="Assistant has responded")
        return proceed()

    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=StaticSimulatorModel(),
        async_mode=False,
        controller=controller,
    )

    cases = simulator.simulate([golden], max_user_simulations=5)

    assert len(cases[0].turns) == 2
    assert len(controller_calls) == 2
    assert controller_calls[-1]["last_assistant_turn"] is not None


def test_async_controller_can_run_in_sync_mode():
    golden = ConversationalGolden(
        scenario="Purchase a concert ticket",
        expected_outcome=None,
        user_description="Test User",
    )
    simulated_user_turn_counts = []

    async def controller(simulated_user_turns):
        simulated_user_turn_counts.append(simulated_user_turns)
        return proceed()

    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=StaticSimulatorModel(),
        async_mode=False,
        controller=controller,
    )

    cases = simulator.simulate([golden], max_user_simulations=1)

    assert len(cases[0].turns) == 2
    assert simulated_user_turn_counts == [0]


def test_sync_controller_can_run_in_async_mode():
    golden = ConversationalGolden(
        scenario="Purchase a concert ticket",
        expected_outcome=None,
        user_description="Test User",
    )
    simulated_user_turn_counts = []

    def controller(simulated_user_turns):
        simulated_user_turn_counts.append(simulated_user_turns)
        return proceed()

    simulator = ConversationSimulator(
        model_callback=async_static_callback,
        simulator_model=StaticSimulatorModel(),
        async_mode=True,
        controller=controller,
    )

    cases = simulator.simulate([golden], max_user_simulations=1)

    assert len(cases[0].turns) == 2
    assert simulated_user_turn_counts == [0]


def test_controller_replaces_expected_outcome_completion():
    golden = ConversationalGolden(
        scenario="Complete flow",
        expected_outcome="User successfully completes the task.",
        user_description="Stop User",
    )
    simulator_model = StaticSimulatorModel(expected_outcome_complete=True)

    def controller(turns):
        return proceed()

    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=simulator_model,
        async_mode=False,
        controller=controller,
    )

    cases = simulator.simulate([golden], max_user_simulations=1)

    assert len(cases[0].turns) == 2
    assert "ConversationCompletion" not in simulator_model.schema_calls


def test_max_user_simulations_is_checked_before_controller():
    golden = ConversationalGolden(
        scenario="Purchase a concert ticket",
        expected_outcome=None,
        user_description="Test User",
    )
    simulated_user_turn_counts = []

    def controller(simulated_user_turns, max_user_simulations):
        simulated_user_turn_counts.append(simulated_user_turns)
        if simulated_user_turns >= max_user_simulations:
            raise AssertionError("Controller should not run after max gate")

    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=StaticSimulatorModel(),
        async_mode=False,
        controller=controller,
    )

    cases = simulator.simulate([golden], max_user_simulations=1)

    assert len(cases[0].turns) == 2
    assert simulated_user_turn_counts == [0]


def test_async_controller_none_defaults_to_proceed():
    golden = ConversationalGolden(
        scenario="Purchase a concert ticket",
        expected_outcome=None,
        user_description="Test User",
    )
    simulated_user_turn_counts = []

    async def controller(simulated_user_turns):
        simulated_user_turn_counts.append(simulated_user_turns)

    simulator = ConversationSimulator(
        model_callback=async_static_callback,
        simulator_model=StaticSimulatorModel(),
        async_mode=True,
        controller=controller,
    )

    cases = simulator.simulate([golden], max_user_simulations=2)

    assert len(cases[0].turns) == 4
    assert simulated_user_turn_counts == [0, 1]

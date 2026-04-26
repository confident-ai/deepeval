from typing import List

import pytest

from deepeval.simulator import ConversationSimulator, ConversationSimulatorTemplate
from deepeval.test_case.conversational_test_case import (
    ConversationalTestCase,
    Turn,
)
from deepeval.dataset.golden import ConversationalGolden
from tests.test_core.test_simulator.helpers import (
    StaticSimulatorModel,
    async_callback_complete,
    static_callback,
    sync_callback,
)


def test_no_existing_turns():
    golden = ConversationalGolden(
        scenario="Purchase a concert ticket",
        expected_outcome=None,
        user_description="Test User",
        turns=None,
    )
    simulator = ConversationSimulator(
        model_callback=sync_callback,
        simulator_model="gpt-4.1-mini",
        # opening_message="Hi, I'm here to help you purchase a ticket.",
        async_mode=True,
        max_concurrent=2,
    )
    cases = simulator.simulate([golden], max_user_simulations=1)
    assert isinstance(cases, list) and len(cases) == 1
    tc = cases[0]
    assert len(tc.turns) == 2
    assert tc.turns[0].role == "user"
    assert isinstance(tc.turns[0].content, str)
    assert tc.turns[1].role == "assistant"
    assert isinstance(tc.turns[1].content, str)


def test_existing_turns():
    golden = ConversationalGolden(
        scenario="Ask about availability",
        expected_outcome=None,
        user_description="Another User",
        turns=[Turn(role="assistant", content="How can I help?")],
    )
    simulator = ConversationSimulator(
        model_callback=sync_callback,
        simulator_model="gpt-4.1-mini",
        async_mode=True,
    )
    cases = simulator.simulate([golden], max_user_simulations=1)
    tc = cases[0]
    assert len(tc.turns) == 3
    assert (
        tc.turns[0].role == "assistant"
        and tc.turns[0].content == "How can I help?"
    )
    assert tc.turns[1].role == "user" and isinstance(tc.turns[1].content, str)
    assert tc.turns[2].role == "assistant"
    assert isinstance(tc.turns[2].content, str)


def test_stop_early():
    golden = ConversationalGolden(
        scenario="Complete flow",
        expected_outcome="User successfully completes the task.",
        user_description="Stop User",
        turns=None,
    )
    simulator = ConversationSimulator(
        model_callback=async_callback_complete,
        simulator_model="gpt-4.1-mini",
        async_mode=True,
    )
    cases = simulator.simulate([golden], max_user_simulations=2)
    tc = cases[0]
    assert len(tc.turns) <= 4
    assert tc.turns[0].role == "user"
    assert isinstance(tc.turns[0].content, str)
    assert tc.turns[1].role == "assistant"
    assert isinstance(tc.turns[1].content, str)


def test_invalid_max_user_simulations():
    golden = ConversationalGolden(
        scenario="Any",
        expected_outcome=None,
        user_description="Any",
        turns=None,
    )

    simulator = ConversationSimulator(
        model_callback=sync_callback,
        simulator_model="gpt-4.1-mini",
        async_mode=True,
    )

    with pytest.raises(ValueError):
        simulator.simulate([golden], max_user_simulations=0)


def test_custom_simulation_template_is_used():
    class FormalTemplate(ConversationSimulatorTemplate):
        @staticmethod
        def simulate_first_user_turn(golden, language):
            return (
                "Generate a formal user message. "
                "Use the phrase FORMAL_STYLE. "
                'Return JSON: {"simulated_input": "hello"}'
            )

    golden = ConversationalGolden(
        scenario="Purchase a concert ticket",
        expected_outcome=None,
        user_description="Test User",
        turns=None,
    )
    simulator_model = StaticSimulatorModel()
    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=simulator_model,
        async_mode=False,
        simulation_template=FormalTemplate,
    )

    simulator.simulate([golden], max_user_simulations=1)

    assert any("FORMAL_STYLE" in prompt for prompt in simulator_model.prompts)


def test_custom_simulation_template_must_inherit_base_template():
    class InvalidTemplate:
        pass

    with pytest.raises(TypeError):
        ConversationSimulator(
            model_callback=static_callback,
            simulator_model=StaticSimulatorModel(),
            async_mode=False,
            simulation_template=InvalidTemplate,
        )


def test_custom_simulation_template_validates_first_turn_signature():
    class InvalidTemplate(ConversationSimulatorTemplate):
        @staticmethod
        def simulate_first_user_turn(scenario, language):
            return "bad"

    with pytest.raises(TypeError):
        ConversationSimulator(
            model_callback=static_callback,
            simulator_model=StaticSimulatorModel(),
            async_mode=False,
            simulation_template=InvalidTemplate,
        )


def test_custom_simulation_template_validates_next_turn_signature():
    class InvalidTemplate(ConversationSimulatorTemplate):
        @staticmethod
        def simulate_user_turn(golden, language):
            return "bad"

    with pytest.raises(TypeError):
        ConversationSimulator(
            model_callback=static_callback,
            simulator_model=StaticSimulatorModel(),
            async_mode=False,
            simulation_template=InvalidTemplate,
        )


def test_turn_alternation():
    golden = ConversationalGolden(
        scenario="Purchase a concert ticket",
        expected_outcome=None,
        user_description="Test User",
        turns=[
            Turn(role="assistant", content="How can I help?"),
            Turn(role="user", content="I need a ticket."),
        ],
    )
    simulator = ConversationSimulator(
        model_callback=sync_callback,
        simulator_model="gpt-4.1-mini",
        async_mode=True,
    )
    cases = simulator.simulate([golden], max_user_simulations=3)
    tc = cases[0]

    num_existing = len(golden.turns)
    for i in range(num_existing, len(tc.turns)):
        assert tc.turns[i].role != tc.turns[i - 1].role


def test_max_simulations_ignores_existing_turns():
    golden = ConversationalGolden(
        scenario="Book a flight",
        expected_outcome=None,
        user_description="Traveler",
        turns=[
            Turn(role="assistant", content="Welcome! How can I help?"),
            Turn(role="user", content="I want to book a flight."),
            Turn(role="assistant", content="Where would you like to go?"),
            Turn(role="user", content="To New York."),
            Turn(role="assistant", content="When would you like to travel?"),
            Turn(role="user", content="Next Monday."),
        ],
    )

    simulator = ConversationSimulator(
        model_callback=sync_callback,
        simulator_model="gpt-4.1-mini",
        async_mode=True,
    )

    max_sims = 3
    cases = simulator.simulate([golden], max_user_simulations=max_sims)
    tc = cases[0]

    num_existing_turns = len(golden.turns)
    new_turns = tc.turns[num_existing_turns:]
    new_user_turns = sum(1 for turn in new_turns if turn.role == "user")

    assert new_user_turns == max_sims


def test_on_simulation_complete_hook_single_conversation():
    golden = ConversationalGolden(
        scenario="Purchase a concert ticket",
        expected_outcome=None,
        user_description="Test User",
        turns=None,
    )

    hook_calls = []

    def on_complete(test_case, index):
        hook_calls.append({"test_case": test_case, "index": index})

    simulator = ConversationSimulator(
        model_callback=sync_callback,
        simulator_model="gpt-4.1-mini",
        async_mode=True,
    )

    cases = simulator.simulate(
        [golden], max_user_simulations=2, on_simulation_complete=on_complete
    )

    assert len(hook_calls) == 1
    assert hook_calls[0]["index"] == 0
    assert hook_calls[0]["test_case"] == cases[0]
    assert isinstance(hook_calls[0]["test_case"], ConversationalTestCase)
    assert hook_calls[0]["test_case"].scenario == golden.scenario


def test_on_simulation_complete_hook_multiple_conversations():
    goldens = [
        ConversationalGolden(
            scenario=f"Scenario {i}",
            expected_outcome=None,
            user_description=f"User {i}",
            turns=None,
        )
        for i in range(3)
    ]

    hook_calls = []

    def on_complete(test_case, index):
        hook_calls.append({"test_case": test_case, "index": index})

    simulator = ConversationSimulator(
        model_callback=sync_callback,
        simulator_model="gpt-4.1-mini",
        async_mode=True,
        max_concurrent=2,
    )

    cases = simulator.simulate(
        goldens, max_user_simulations=1, on_simulation_complete=on_complete
    )

    assert len(hook_calls) == 3
    indices = {call["index"] for call in hook_calls}
    assert indices == {0, 1, 2}

    for call in hook_calls:
        idx = call["index"]
        assert call["test_case"] == cases[idx]
        assert call["test_case"].scenario == goldens[idx].scenario

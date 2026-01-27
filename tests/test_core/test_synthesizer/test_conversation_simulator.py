from typing import List, Optional
from openai import AsyncOpenAI, OpenAI
import pytest

from deepeval.simulator import ConversationSimulator
from deepeval.test_case.conversational_test_case import (
    Turn,
    ConversationalTestCase,
)
from deepeval.dataset.golden import ConversationalGolden


def sync_callback(
    input: str, turns: List[Turn], thread_id: Optional[str] = None
) -> Turn:
    client = OpenAI()
    messages = [{"role": turn.role, "content": turn.content} for turn in turns]
    messages.append({"role": "user", "content": input})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    print(thread_id)
    return Turn(role="assistant", content=response.choices[0].message.content)


async def async_callback_complete(
    input: str, turns: List[Turn], thread_id: Optional[str] = None
) -> Turn:
    client = AsyncOpenAI()
    messages = [{"role": turn.role, "content": turn.content} for turn in turns]
    messages.append({"role": "user", "content": input})
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    print(thread_id)
    return Turn(role="assistant", content=response.choices[0].message.content)


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
        model_callback=async_callback_complete,  # async callback path
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

    # Start from existing turns and check alternation using modulo pattern
    num_existing = len(golden.turns)
    for i in range(num_existing, len(tc.turns)):
        # Check alternation: each turn should differ from previous turn
        assert tc.turns[i].role != tc.turns[i - 1].role


def test_max_simulations_ignores_existing_turns():
    """Test that max_user_simulations only counts new simulated user turns,
    not existing user turns in the golden."""
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

    # Count existing user turns
    num_existing_turns = len(golden.turns)

    # Count new user turns (after existing turns)
    new_turns = tc.turns[num_existing_turns:]
    new_user_turns = sum(1 for turn in new_turns if turn.role == "user")

    # Verify that new user turns equals max_user_simulations
    assert new_user_turns == max_sims


def test_on_simulation_complete_hook_single_conversation():
    """Test that on_simulation_complete hook is called with correct parameters
    for a single conversation."""
    golden = ConversationalGolden(
        scenario="Purchase a concert ticket",
        expected_outcome=None,
        user_description="Test User",
        turns=None,
    )

    # Track hook calls
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

    # Verify hook was called once
    assert len(hook_calls) == 1

    # Verify hook was called with correct index
    assert hook_calls[0]["index"] == 0

    # Verify hook was called with the correct test case
    assert hook_calls[0]["test_case"] == cases[0]
    assert isinstance(hook_calls[0]["test_case"], ConversationalTestCase)
    assert hook_calls[0]["test_case"].scenario == golden.scenario


def test_on_simulation_complete_hook_multiple_conversations():
    """Test that on_simulation_complete hook is called for each conversation
    with correct indices and test cases."""
    goldens = [
        ConversationalGolden(
            scenario=f"Scenario {i}",
            expected_outcome=None,
            user_description=f"User {i}",
            turns=None,
        )
        for i in range(3)
    ]

    # Track hook calls
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

    # Verify hook was called for each conversation
    assert len(hook_calls) == 3

    # Verify all indices are present (order may vary due to async)
    indices = {call["index"] for call in hook_calls}
    assert indices == {0, 1, 2}

    # Verify each test case matches the corresponding case
    for call in hook_calls:
        idx = call["index"]
        assert call["test_case"] == cases[idx]
        assert call["test_case"].scenario == goldens[idx].scenario

from typing import List, Optional
from openai import AsyncOpenAI, OpenAI
import pytest

from deepeval.simulator import ConversationSimulator
from deepeval.test_case.conversational_test_case import Turn
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


# def test_no_existing_turns():
#     golden = ConversationalGolden(
#         scenario="Purchase a concert ticket",
#         expected_outcome=None,
#         user_description="Test User",
#         turns=None,
#     )
#     simulator = ConversationSimulator(
#         model_callback=sync_callback,
#         simulator_model="gpt-4.1-mini",
#         opening_message="Hi, I'm here to help you purchase a ticket.",
#         async_mode=True,
#         max_concurrent=2,
#         run_remote=True,
#     )
#     cases = simulator.simulate([golden], max_user_simulations=1)
#     assert isinstance(cases, list) and len(cases) == 1
#     tc = cases[0]
#     assert len(tc.turns) == 3
#     assert tc.turns[0].role == "assistant"
#     assert tc.turns[0].content == "Hi, I'm here to help you purchase a ticket."
#     assert tc.turns[1].role == "user"
#     assert isinstance(tc.turns[1].content, str)
#     assert tc.turns[2].role == "assistant"
#     assert isinstance(tc.turns[2].content, str)


# def test_existing_turns():
#     golden = ConversationalGolden(
#         scenario="Ask about availability",
#         expected_outcome=None,
#         user_description="Another User",
#         turns=[Turn(role="assistant", content="How can I help?")],
#     )
#     simulator = ConversationSimulator(
#         model_callback=sync_callback,
#         simulator_model="gpt-4.1-mini",
#         opening_message="This should NOT be inserted because turns exist",
#         async_mode=True,
#         run_remote=True,
#     )
#     cases = simulator.simulate([golden], max_user_simulations=1)
#     tc = cases[0]
#     assert len(tc.turns) == 3
#     assert (
#         tc.turns[0].role == "assistant"
#         and tc.turns[0].content == "How can I help?"
#     )
#     assert tc.turns[1].role == "user" and isinstance(tc.turns[1].content, str)
#     assert tc.turns[2].role == "assistant"
#     assert isinstance(tc.turns[2].content, str)


# def test_stop_early():
#     golden = ConversationalGolden(
#         scenario="Complete flow",
#         expected_outcome="User successfully completes the task.",
#         user_description="Stop User",
#         turns=None,
#     )
#     simulator = ConversationSimulator(
#         model_callback=async_callback_complete,  # async callback path
#         simulator_model="gpt-4.1-mini",
#         opening_message="Let's start.",
#         async_mode=True,
#         run_remote=True,
#     )
#     cases = simulator.simulate([golden], max_user_simulations=2)
#     tc = cases[0]
#     assert len(tc.turns) <= 5
#     assert tc.turns[0].role == "assistant"
#     assert tc.turns[1].role == "user"
#     assert isinstance(tc.turns[1].content, str)
#     assert tc.turns[2].role == "assistant"
#     assert isinstance(tc.turns[2].content, str)


# def test_invalid_max_user_simulations():
#     golden = ConversationalGolden(
#         scenario="Any",
#         expected_outcome=None,
#         user_description="Any",
#         turns=None,
#     )

#     simulator = ConversationSimulator(
#         model_callback=sync_callback,
#         simulator_model="gpt-4.1-mini",
#         opening_message="Start",
#         async_mode=True,
#         run_remote=True,
#     )

#     with pytest.raises(ValueError):
#         simulator.simulate([golden], max_user_simulations=0)

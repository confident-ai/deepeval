from deepeval.simulator import ConversationSimulator
from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Turn
from openai import AsyncOpenAI
from typing import List

########################################
# Define Model Callback
########################################

client = AsyncOpenAI()


async def sync_model_callback(
    input: str,
    turns: List[Turn],
    thread_id: str,
):
    messages = [
        {"role": "system", "content": "You are a ticket purchasing assistant"},
        *[{"role": t.role, "content": t.content} for t in turns],
        {"role": "user", "content": input},
    ]
    response = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    return Turn(role="assistant", content=response.choices[0].message.content)


async def async_model_callback(
    input: str,
    turns: List[Turn],
    thread_id: str,
):
    messages = [
        {"role": "system", "content": "You are a ticket purchasing assistant"},
        *[{"role": t.role, "content": t.content} for t in turns],
        {"role": "user", "content": input},
    ]
    response = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    return Turn(role="assistant", content=response.choices[0].message.content)


########################################
# Initialize Conversation Simulator
########################################

simulator_async = ConversationSimulator(
    model_callback=async_model_callback,
    simulator_model="gpt-4o-mini",
    opening_message="You are a helpful assistant that answers questions concisely.",
    max_concurrent=5,
    async_mode=False,
)

simulator_sync = ConversationSimulator(
    model_callback=sync_model_callback,
    simulator_model="gpt-4o-mini",
    opening_message="You are a helpful assistant that answers questions concisely.",
    max_concurrent=5,
    async_mode=False,
)

########################################
# Run Simulation
########################################

conversation_golden = ConversationalGolden(
    scenario="Andy Byron wants to purchase a VIP ticket to a cold play concert.",
    expected_outcome="Successful purchase of ticket to a cold play concert.",
    user_description="Andy Byron is the former CEO of Astronomer.",
)

simulator_async.simulate(
    [conversation_golden, conversation_golden, conversation_golden],
    max_turns=3,
)

simulator_sync.simulate(
    [conversation_golden, conversation_golden, conversation_golden],
    max_turns=3,
)

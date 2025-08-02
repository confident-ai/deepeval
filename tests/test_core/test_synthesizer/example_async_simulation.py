from deepeval.conversation_simulator import ConversationSimulator
from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Turn
from openai import AsyncOpenAI
from typing import List

########################################
# Define Model Callback
########################################

client = AsyncOpenAI()


async def async_model_callback_1(input: str):
    messages = [
        {"role": "system", "content": "You are a ticket purchasing assistant"},
        {"role": "user", "content": input},
    ]
    response = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    return response.choices[0].message.content


async def async_model_callback_2(input: str, thread_id: str):
    messages = [
        {"role": "system", "content": "You are a ticket purchasing assistant"},
        {"role": "user", "content": input},
    ]
    response = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    return response.choices[0].message.content


async def async_model_callback_3(
    input: str,
    turns: List[Turn],
):
    messages = [
        {"role": "system", "content": "You are a ticket purchasing assistant"},
        *[{"role": t.role, "content": t.content} for t in turns],
        {"role": "user", "content": input},
    ]
    response = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    return response.choices[0].message.content


async def async_model_callback_4(
    input: str,
    turns: List[Turn],
    thread_id: str,
):
    print(thread_id)
    messages = [
        {"role": "system", "content": "You are a ticket purchasing assistant"},
        *[{"role": t.role, "content": t.content} for t in turns],
        {"role": "user", "content": input},
    ]
    response = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    return response.choices[0].message.content


########################################
# Initialize Conversation Simulator
########################################

simulator = ConversationSimulator(
    model_callback=async_model_callback_4,
    simulator_model="gpt-4o-mini",
    opening_message="You are a helpful assistant that answers questions concisely.",
    max_concurrent=5,
    async_mode=True,
)

########################################
# Run Simulation
########################################

conversation_golden = ConversationalGolden(
    scenario="Andy Byron wants to purchase a VIP ticket to a cold play concert.",
    expected_outcome="Successful purchase of ticket to a cold play concert.",
    user_description="Andy Byron is the former CEO of Astronomer.",
)

test_cases = simulator.simulate([conversation_golden, conversation_golden, conversation_golden], max_turns=3)

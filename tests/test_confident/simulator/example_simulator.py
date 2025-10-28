from deepeval.test_case import Turn
from deepeval.simulator import ConversationSimulator
from deepeval.dataset import ConversationalGolden
from openai import AsyncOpenAI, OpenAI
from typing import List

# Create ConversationalGolden
conversation_golden_1 = ConversationalGolden(
    scenario="Andy Byron wants to purchase a VIP ticket to a cold play concert.",
    expected_outcome="Successful purchase of a ticket.",
    user_description="Andy Byron is the CEO of Astronomer.",
    turns=[
        Turn(
            role="assistant",
            content="Hi, I'm here to help you purchase a ticket.",
        ),
        # Turn(role="user", content="I want to purchase a VIP ticket to a cold play concert."),
    ],
)

conversation_golden_2 = ConversationalGolden(
    scenario="Donald Trump wants to ask about ticket availability for a world cup final match.",
    expected_outcome="Donald Trump knows that the ticket is available or not available.",
    user_description="Donald Trump is the President of the United States.",
    turns=[
        Turn(
            role="assistant",
            content="Hi, I'm here to help you purchase a ticket.",
        ),
        # Turn(role="user", content="I want to ask about ticket availability for a world cup final match."),
    ],
)

conversation_golden_3 = ConversationalGolden(
    scenario="Barack Obama wants to book 2 tickets for jazz pub concert.",
    expected_outcome="Successful purchase of 2 tickets.",
    user_description="Barack Obama is the former President of the United States.",
)

goldens = [
    conversation_golden_1,
    # conversation_golden_2,
    # conversation_golden_3,
]

# Define chatbot callback
client = AsyncOpenAI()


async def chatbot_callback(input, turns: List[Turn]):
    messages = []
    for turn in turns:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": input})
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    return Turn(role="assistant", content=response.choices[0].message.content)

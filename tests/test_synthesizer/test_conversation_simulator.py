import asyncio
import random

from deepeval.conversation_simulator import ConversationSimulator
from deepeval.models import GPTModel

user_profile_requirements_1 = [
    "name (first and last)",
    "phone number",
    "availabilities (between monday and friday)",
]
user_intentions_1 = {
    "Ordering new products": 1,
    "Repair for existing products": 1,
    "Picking products up because somebody died or moved into intensive care": 1,
    "Questions regarding open invoices": 1,
    "Questions regarding the order status": 1,
}
user_profile_requirements_2 = "name (first and last), medical condition, availabilities (between monday and friday)"
user_intentions_2 = {
    # "Seeking medical advice for a chronic condition",
    "Booking an appointment with a specialist": 1,
    "Following up on test results": 1,
    # "Requesting prescription refills",
    # "Exploring treatment options for a new diagnosis",
}


async def test_user_profile(conversation_simulator: ConversationSimulator):
    tasks = [conversation_simulator._simulate_user_profile() for _ in range(5)]
    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results, start=1):
        print(f"Task {i}: {result}")
    return results


async def test_scenario(
    conversation_simulator: ConversationSimulator,
    user_profile: str,
    user_intention: str,
):
    for _ in range(5):
        scenario = await conversation_simulator._a_simulate_scenario(
            user_profile, user_intention
        )
        print(scenario)
        print(("================================"))


async def test_generate_conversations(
    conversation_simulator: ConversationSimulator,
):
    conversational_test_cases = await conversation_simulator._a_simulate()
    for tc in conversational_test_cases:
        conversation_str = conversation_simulator._format_conversational_turns(
            tc.turns
        )
        print(("================================"))


async def a_callback(prompt: str, conversation_history):
    model = GPTModel()
    res, _ = await model.a_generate(prompt)
    return res


def callback(prompt: str, conversation_history):
    model = GPTModel()
    res, _ = model.generate(prompt)
    return res


async def main():
    user_intentions = user_intentions_2
    conversational_synthesizer = ConversationSimulator(
        # user_profile_items=user_profile_requirements_2,
        user_profiles=[
            "Jeff Seid is available on Monday and Thursday afternoons, and his phone number is 0010281839."
        ],
        user_intentions=user_intentions,
        opening_message="Hi, I'm your personal medical chatbot.",
        async_mode=False,
    )
    conversational_synthesizer.simulate(
        min_turns=2,
        max_turns=4,
        model_callback=callback,
        stopping_criteria="The user has succesfully booked an appointment",
    )

    # user_profiles = await test_user_profile(conversational_synthesizer)
    # await test_scenario(
    #     conversational_synthesizer,
    #     random.choice(user_profiles),
    #     random.choice(user_intentions),
    # )
    # await test_generate_conversations(conversational_synthesizer)


# Run the main async function
if __name__ == "__main__":
    asyncio.run(main())

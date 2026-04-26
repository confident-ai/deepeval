import json

from deepeval.dataset import ConversationalGolden
from deepeval.simulator import (
    ConversationSimulator,
    ConversationSimulatorTemplate,
)
from deepeval.simulator.controller import end, proceed
from deepeval.test_case import Turn


conversation_goldens = [
    ConversationalGolden(
        scenario="A customer wants to return a damaged laptop purchased last week.",
        expected_outcome="The customer understands the return process and receives the next step.",
        user_description="A frustrated but cooperative customer.",
    ),
    ConversationalGolden(
        scenario="A user wants to upgrade from a free plan to a team plan.",
        expected_outcome="The user knows which plan to choose and how to complete the upgrade.",
        user_description="A startup founder comparing pricing options.",
    ),
    ConversationalGolden(
        scenario="A patient wants to reschedule an appointment because of a work conflict.",
        expected_outcome="The patient gets a suitable new appointment time.",
        user_description="A busy patient who prefers concise answers.",
    ),
    ConversationalGolden(
        scenario="A traveler needs help changing a flight after a weather delay.",
        expected_outcome="The traveler understands available rebooking options.",
        user_description="An anxious traveler stuck at the airport.",
    ),
    ConversationalGolden(
        scenario="A developer is debugging a failed API authentication request.",
        expected_outcome="The developer identifies the likely authentication issue and next debugging step.",
        user_description="A technical user who can understand API terminology.",
    ),
]


async def model_callback(input: str, turns: list[Turn], thread_id: str) -> Turn:
    return Turn(
        role="assistant",
        content=f"I can help with that. You said: {input}",
    )


def controller(simulated_user_turns: int):
    if simulated_user_turns >= 1:
        return end(reason="Stopped after two simulated user turns.")
    return proceed()


simulator = ConversationSimulator(
    model_callback=model_callback,
    controller=controller,
)
conversational_test_cases = simulator.simulate(
    conversational_goldens=conversation_goldens,
    max_user_simulations=5,
)


for test_case in conversational_test_cases:
    print(test_case.turns)

# from typing import List
# import asyncio
# import pytest

# from deepeval.simulator import ConversationSimulator
# from deepeval.test_case import ConversationalTestCase
# from deepeval.models import GPTModel

# user_intentions = {
#     "Booking an appointment with a specialist": 1,
#     "Following up on test results": 1,
# }
# user_profile = "Jeff Seid is available on Monday and Thursday afternoons, and his phone number is 0010281839."
# user_intention = "Booking an appointment with a specialist"


# @pytest.fixture
# def callback_fn():
#     def callback(prompt: str, conversation_history):
#         model = GPTModel()
#         res, _ = model.generate(prompt)
#         return res

#     return callback


# @pytest.fixture
# async def async_callback_fn():
#     async def a_callback(prompt: str, conversation_history):
#         model = GPTModel()
#         res, _ = await model.a_generate(prompt)
#         return res

#     return a_callback


# @pytest.fixture
# def simulator_sync():
#     conversation_simulator = ConversationSimulator(
#         user_profiles=[user_profile],
#         user_intentions=user_intentions,
#         opening_message="Hi, I'm your personal medical chatbot.",
#         async_mode=False,
#     )
#     conversation_simulator.simulation_cost = 0
#     return conversation_simulator


# @pytest.fixture
# def simulator_async():
#     conversation_simulator = ConversationSimulator(
#         user_profiles=[user_profile],
#         user_intentions=user_intentions,
#         opening_message="Hi, I'm your personal medical chatbot.",
#         async_mode=True,
#     )
#     conversation_simulator.simulation_cost = 0
#     return conversation_simulator


# def test_simulate_sync(simulator_sync: ConversationSimulator, callback_fn):
#     test_cases = simulator_sync.simulate(
#         min_turns=2,
#         max_turns=4,
#         model_callback=callback_fn,
#         stopping_criteria="The user has successfully booked an appointment",
#     )
#     print(test_cases)
#     assert test_cases is not None, "Should generate test cases"
#     assert len(test_cases) > 0, "Should have at least one test case"
#     for tc in test_cases:
#         assert hasattr(tc, "turns"), "Test case should have turns attribute"
#         assert (
#             len(tc.turns) >= 2 * 2 + 1
#         ), "Test case should have at least min_turns"
#         assert (
#             len(tc.turns) <= 2 * 4 + 1
#         ), "Test case should have at most max_turns"


# def test_simulate_async(
#     simulator_async: ConversationSimulator, async_callback_fn
# ):
#     test_cases: List[ConversationalTestCase] = simulator_async.simulate(
#         min_turns=2,
#         max_turns=4,
#         model_callback=async_callback_fn,
#         stopping_criteria="The user has successfully booked an appointment",
#     )
#     assert test_cases is not None, "Should generate test cases"
#     assert len(test_cases) > 0, "Should have at least one test case"
#     for tc in test_cases:
#         assert hasattr(tc, "turns"), "Test case should have turns attribute"
#         assert (
#             len(tc.turns) >= 2 * 2 + 1
#         ), "Test case should have at least min_turns"
#         assert (
#             len(tc.turns) <= 2 * 4 + 1
#         ), "Test case should have at most max_turns"

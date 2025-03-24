from typing import Optional, List, Literal, Union
from tqdm import tqdm
import asyncio
import random
import json

from deepeval.metrics.utils import initialize_model
from deepeval.test_case import ConversationalTestCase, LLMTestCase
from deepeval.conversation_simulator.template import (
    ConversationSimulatorTemplate,
)
from deepeval.models import DeepEvalBaseLLM
from .schema import (
    FirstInput,
    NextInput,
    Scenario,
    UserProfile,
)


class ConversationSimulator:
    def __init__(
        self,
        user_profile_requirements: List[str],
        user_intentions: List[str],
        min_turns: int,
        max_turns: int,
        num_conversations: int,
        model_callback: callable[str, str],
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        opening_message: Optional[str] = None,
        max_concurrent: int = 5,
        async_model: bool = True,
    ):
        self.user_profile_requirements = user_profile_requirements
        self.user_intentions = user_intentions
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.num_conversations = num_conversations
        self.opening_message = opening_message
        self.language = "English"
        self.model_callback = model_callback
        self.simulator_model, self.using_native_model = initialize_model(
            simulator_model
        )
        self.async_model = async_model

        # Config
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def simulate():
        pass

    async def _a_simulate(self) -> List[ConversationalTestCase]:
        async def simulate_conversation_with_semaphore():
            async with self.semaphore:
                return await self._simulate_single_conversation()

        tasks = [
            simulate_conversation_with_semaphore()
            for _ in range(self.num_conversations)
        ]
        conversations = await asyncio.gather(*tasks)
        return conversations

    async def _simulate_single_conversation(self) -> ConversationalTestCase:
        # Determine the number of turns
        num_turns = random.randint(self.min_turns, self.max_turns)

        # Generate scenario
        intent = random.choice(self.user_intentions)
        user_profile = await self._a_simulate_user_profile()
        scenario = await self._a_simulate_scenario(user_profile, intent)
        additional_metadata = {
            "User Profile": user_profile,
            "User Intent": intent,
        }

        turns = []
        if self.opening_message:
            turns.append(
                LLMTestCase(input="", actual_output=self.opening_message)
            )

        current_input = await self._a_simulate_first_user_input(
            scenario, user_profile
        )
        turns.append(
            LLMTestCase(
                input=current_input,
                actual_output=await self._a_generate_chatbot_response(
                    current_input
                ),
            )
        )

        for _ in tqdm(
            range(num_turns - 1), desc="Generating conversation turns"
        ):
            conversation_history = self._format_conversational_turns(turns)
            prompt = ConversationSimulatorTemplate.generate_next_user_input(
                scenario, user_profile, conversation_history, self.language
            )
            response: NextInput = await self.simulator_model.a_generate(
                prompt, NextInput
            )
            current_input = response.next_input

            turns.append(
                LLMTestCase(
                    input=current_input,
                    actual_output=await self._a_generate_chatbot_response(
                        current_input
                    ),
                    additional_metadata=additional_metadata,
                )
            )

        # Return the full test case
        return ConversationalTestCase(
            turns=turns, additional_metadata=additional_metadata
        )

    async def _a_simulate_user_profile(self) -> str:
        prompt = ConversationSimulatorTemplate.generate_user_profile(
            self.user_profile_requirements, self.language
        )
        response: UserProfile = await self.simulator_model.a_generate(
            prompt, UserProfile
        )
        return response.user_profile

    async def _a_simulate_scenario(self, user_profile: str, intent: str) -> str:
        prompt = ConversationSimulatorTemplate.generate_scenario(
            user_profile, intent, self.language
        )
        response: Scenario = await self.simulator_model.a_generate(
            prompt, Scenario
        )
        return response.scenario

    async def _a_simulate_first_user_input(
        self, scenario: str, user_profile: str
    ) -> str:
        prompt = ConversationSimulatorTemplate.generate_first_input(
            scenario, user_profile, self.language
        )
        response: FirstInput = await self.simulator_model.a_generate(
            prompt, FirstInput
        )
        return response.first_input

    ############################################
    ### Helper Methods #########################
    ############################################

    def _format_conversational_turns(self, turns: List[LLMTestCase]) -> str:
        formatted_turns = []
        for turn in turns:
            formatted_turns.append({"role": "user", "content": turn.input})
            formatted_turns.append(
                {"role": "assistant", "content": turn.actual_output}
            )
        return json.dumps(formatted_turns, indent=4)

    async def _a_generate_chatbot_response(self, input: str):
        res = await self.model_callback(input)
        return res

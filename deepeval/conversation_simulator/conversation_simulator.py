from typing import Optional, List, Union, Callable
from tqdm import tqdm
import asyncio
import random
import json

from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import initialize_model, trimAndLoadJson
from deepeval.test_case import ConversationalTestCase, LLMTestCase
from deepeval.conversation_simulator.template import (
    ConversationSimulatorTemplate,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.conversation_simulator.schema import (
    SimulatedInput,
    Scenario,
    UserProfile,
)


class ConversationSimulator:
    def __init__(
        self,
        user_profile_items: List[str],
        user_intentions: List[str],
        model_callback: Callable[[str], str],
        min_turns: int,
        max_turns: int,
        num_conversations: int,
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        opening_message: Optional[str] = None,
        max_concurrent: int = 5,
        async_mode: bool = True,
    ):
        if min_turns > max_turns:
            raise ValueError("`min_turns` cannot be greater than `max_turns`.")

        self.user_profile_items = user_profile_items
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
        self.async_mode = async_mode
        self.simulated_conversational_test_cases: List[
            ConversationalTestCase
        ] = []
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def simulate(self) -> List[ConversationalTestCase]:
        self.simulation_cost = 0 if self.using_native_model else None
        if self.async_mode:
            loop = get_or_create_event_loop()
            loop.run_until_complete(self._a_simulate())
        else:
            conversational_test_cases: List[ConversationalTestCase] = []
            for _ in range(self.num_conversations):
                conversational_test_case = self._simulate_single_conversation()
                conversational_test_cases.append(conversational_test_case)
            self.simulated_conversational_test_cases = conversational_test_cases

        return self.simulated_conversational_test_cases

    async def _a_simulate(self) -> List[ConversationalTestCase]:
        self.simulation_cost = 0 if self.using_native_model else None

        async def limited_simulation():
            async with self.semaphore:
                return await self._a_simulate_single_conversation()

        tasks = [limited_simulation() for _ in range(self.num_conversations)]
        self.simulated_conversational_test_cases = await asyncio.gather(*tasks)

    def _simulate_user_profile(self) -> str:
        prompt = ConversationSimulatorTemplate.generate_user_profile(
            self.user_profile_items, self.language
        )

        if self.using_native_model:
            res, cost = self.simulator_model.generate(
                prompt, schema=UserProfile
            )
            self.simulation_cost += cost
            return res.user_profile
        else:
            try:
                res: UserProfile = self.simulator_model.generate(
                    prompt, UserProfile
                )
                return res.user_profile
            except TypeError:
                res = self.simulator_model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["user_profile"]

    def _simulate_scenario(self, user_profile: str, intent: str) -> str:
        prompt = ConversationSimulatorTemplate.generate_scenario(
            user_profile, intent, self.language
        )

        if self.using_native_model:
            res, cost = self.simulator_model.a_generate(prompt, schema=Scenario)
            self.simulation_cost += cost
            return res.scenario
        else:
            try:
                res: Scenario = self.simulator_model.a_generate(
                    prompt, Scenario
                )
                return res.scenario
            except TypeError:
                res = self.simulator_model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["scenario"]

    def _simulate_first_user_input(
        self, scenario: str, user_profile: str
    ) -> str:
        prompt = ConversationSimulatorTemplate.generate_first_input(
            scenario, user_profile, self.language
        )

        if self.using_native_model:
            res, cost = self.simulator_model.a_generate(
                prompt, schema=SimulatedInput
            )
            self.simulation_cost += cost
            return res.simulated_input
        else:
            try:
                res: SimulatedInput = self.simulator_model.a_generate(
                    prompt, SimulatedInput
                )
                return res.simulated_input
            except TypeError:
                res = self.simulator_model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["simulated_input"]

    def _simulate_single_conversation(self) -> ConversationalTestCase:
        num_turns = random.randint(self.min_turns, self.max_turns)
        intent = random.choice(self.user_intentions)
        user_profile = self._a_simulate_user_profile()
        scenario = self._a_simulate_scenario(user_profile, intent)
        additional_metadata = {
            "User Profile": user_profile,
            "User Intent": intent,
        }

        turns = []
        user_input = None
        for turn_index in tqdm(
            range(num_turns), desc="Generating conversation turns"
        ):
            if turn_index == 0 and self.opening_message:
                # Add optional opening message from chatbot
                turns.append(
                    LLMTestCase(input="", actual_output=self.opening_message)
                )

            if turn_index == 0:
                # First user input
                user_input = self._simulate_first_user_input(
                    scenario, user_profile
                )
            else:
                # Generate next user input based on conversation so far
                conversation_history = self._format_conversational_turns(turns)
                prompt = ConversationSimulatorTemplate.generate_next_user_input(
                    scenario, user_profile, conversation_history, self.language
                )

                if self.using_native_model:
                    res, cost = self.simulator_model.generate(
                        prompt, schema=SimulatedInput
                    )
                    self.simulation_cost += cost
                    user_input = res.simulated_input
                else:
                    try:
                        res: SimulatedInput = self.simulator_model.generate(
                            prompt, SimulatedInput
                        )
                        user_input = res.simulated_input
                    except TypeError:
                        res = self.simulator_model.generate(prompt)
                        data = trimAndLoadJson(res, self)
                        user_input = data["simluated_input"]

            turns.append(
                LLMTestCase(
                    input=user_input,
                    actual_output=self._generate_chatbot_response(user_input),
                    additional_metadata=additional_metadata,
                )
            )

        return ConversationalTestCase(
            turns=turns, additional_metadata=additional_metadata
        )

    async def _a_simulate_user_profile(self) -> str:
        prompt = ConversationSimulatorTemplate.generate_user_profile(
            self.user_profile_items, self.language
        )

        if self.using_native_model:
            res, cost = await self.simulator_model.a_generate(
                prompt, schema=UserProfile
            )
            self.simulation_cost += cost
            return res.user_profile
        else:
            try:
                res: UserProfile = await self.simulator_model.a_generate(
                    prompt, UserProfile
                )
                return res.user_profile
            except TypeError:
                res = await self.simulator_model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["user_profile"]

    async def _a_simulate_scenario(self, user_profile: str, intent: str) -> str:
        prompt = ConversationSimulatorTemplate.generate_scenario(
            user_profile, intent, self.language
        )

        if self.using_native_model:
            res, cost = await self.simulator_model.a_generate(
                prompt, schema=Scenario
            )
            self.simulation_cost += cost
            return res.scenario
        else:
            try:
                res: Scenario = await self.simulator_model.a_generate(
                    prompt, Scenario
                )
                return res.scenario
            except TypeError:
                res = await self.simulator_model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["scenario"]

    async def _a_simulate_first_user_input(
        self, scenario: str, user_profile: str
    ) -> str:
        prompt = ConversationSimulatorTemplate.generate_first_input(
            scenario, user_profile, self.language
        )

        if self.using_native_model:
            res, cost = await self.simulator_model.a_generate(
                prompt, schema=SimulatedInput
            )
            self.simulation_cost += cost
            return res.simulated_input
        else:
            try:
                res: SimulatedInput = await self.simulator_model.a_generate(
                    prompt, SimulatedInput
                )
                return res.simulated_input
            except TypeError:
                res = await self.simulator_model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["simluated_input"]

    async def _a_simulate_single_conversation(self) -> ConversationalTestCase:
        num_turns = random.randint(self.min_turns, self.max_turns)
        intent = random.choice(self.user_intentions)
        user_profile = await self._a_simulate_user_profile()
        scenario = await self._a_simulate_scenario(user_profile, intent)
        additional_metadata = {
            "User Profile": user_profile,
            "User Intent": intent,
        }

        turns = []
        user_input = None
        for turn_index in tqdm(
            range(num_turns), desc="Generating conversation turns"
        ):
            if turn_index == 0 and self.opening_message:
                turns.append(
                    LLMTestCase(input="", actual_output=self.opening_message)
                )

            if turn_index == 0:
                user_input = await self._a_simulate_first_user_input(
                    scenario, user_profile
                )
            else:
                # Generate next user input based on conversation so far
                conversation_history = self._format_conversational_turns(turns)
                prompt = ConversationSimulatorTemplate.generate_next_user_input(
                    scenario, user_profile, conversation_history, self.language
                )

                if self.using_native_model:
                    res, cost = await self.simulator_model.a_generate(
                        prompt, schema=SimulatedInput
                    )
                    self.simulation_cost += cost
                    user_input = res.simulated_input
                else:
                    try:
                        res: SimulatedInput = (
                            await self.simulator_model.a_generate(
                                prompt, SimulatedInput
                            )
                        )
                        user_input = res.simulated_input
                    except TypeError:
                        res = await self.simulator_model.a_generate(prompt)
                        data = trimAndLoadJson(res, self)
                        user_input = data["simluated_input"]

                user_input = res.simulated_input

            turns.append(
                LLMTestCase(
                    input=user_input,
                    actual_output=await self._a_generate_chatbot_response(
                        user_input
                    ),
                    additional_metadata=additional_metadata,
                )
            )

        return ConversationalTestCase(
            turns=turns, additional_metadata=additional_metadata
        )

    ############################################
    ### Helper Methods #########################
    ############################################

    def _generate_chatbot_response(self, input: str):
        res = self.model_callback(input)
        return res

    async def _a_generate_chatbot_response(self, input: str):
        res = await self.model_callback(input)
        return res

    def _format_conversational_turns(self, turns: List[LLMTestCase]) -> str:
        formatted_turns = []

        for i in range(len(turns)):
            turn = turns[i]
            if not (i == 0 and turn.input == ""):
                formatted_turns.append({"role": "user", "content": turn.input})

            formatted_turns.append(
                {"role": "assistant", "content": turn.actual_output}
            )

        return json.dumps(formatted_turns, indent=4)

import inspect
from typing import Optional, List, Union, Callable
import asyncio
import random
import json
import tqdm

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
from deepeval.progress_context import conversation_simulator_progress_context


class ConversationSimulator:
    def __init__(
        self,
        user_profile_items: List[str],
        user_intentions: List[str],
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        opening_message: Optional[str] = None,
        max_concurrent: int = 5,
        async_mode: bool = True,
    ):
        self.user_profile_items = user_profile_items
        self.user_intentions = user_intentions
        self.opening_message = opening_message
        self.language = "English"
        self.simulator_model, self.using_native_model = initialize_model(
            simulator_model
        )
        self.async_mode = async_mode
        self.simulated_conversations: List[ConversationalTestCase] = []
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def simulate(
        self,
        model_callback: Callable[[str], str],
        min_turns: int = 5,
        max_turns: int = 20,
        num_conversations: int = 5,
    ) -> List[ConversationalTestCase]:
        if min_turns > max_turns:
            raise ValueError("`min_turns` cannot be greater than `max_turns`.")

        self.simulation_cost = 0 if self.using_native_model else None
        with conversation_simulator_progress_context(
            simulator_model=self.simulator_model.get_model_name(),
            num_conversations=num_conversations,
            async_mode=self.async_mode,
        ) as progress_bar:
            if self.async_mode:
                if not inspect.iscoroutinefunction(model_callback):
                    raise TypeError(
                        "`model_callback` must be a coroutine function when using 'async_mode' is 'True'."
                    )

                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self._a_simulate(
                        model_callback=model_callback,
                        min_turns=min_turns,
                        max_turns=max_turns,
                        num_conversations=num_conversations,
                        _progress_bar=progress_bar,
                    )
                )
            else:
                if inspect.iscoroutinefunction(model_callback):
                    raise TypeError(
                        "`model_callback` must be a synchronous function when using 'async_mode' is 'False'."
                    )

                conversational_test_cases: List[ConversationalTestCase] = []
                for _ in range(num_conversations):
                    conversational_test_case = (
                        self._simulate_single_conversation(
                            model_callback=model_callback,
                            min_turns=min_turns,
                            max_turns=max_turns,
                        )
                    )
                    conversational_test_cases.append(conversational_test_case)
                    progress_bar.update(1)

                self.simulated_conversations = conversational_test_cases

        return self.simulated_conversations

    async def _a_simulate(
        self,
        model_callback: Callable[[str], str],
        min_turns: int,
        max_turns: int,
        num_conversations: int,
        _progress_bar: Optional[tqdm.std.tqdm] = None,
    ) -> List[ConversationalTestCase]:
        self.simulation_cost = 0 if self.using_native_model else None

        async def limited_simulation():
            async with self.semaphore:
                return await self._a_simulate_single_conversation(
                    model_callback=model_callback,
                    min_turns=min_turns,
                    max_turns=max_turns,
                    _progress_bar=_progress_bar,
                )

        tasks = [limited_simulation() for _ in range(num_conversations)]
        self.simulated_conversations = await asyncio.gather(*tasks)

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
            res, cost = self.simulator_model.generate(prompt, schema=Scenario)
            self.simulation_cost += cost
            return res.scenario
        else:
            try:
                res: Scenario = self.simulator_model.generate(prompt, Scenario)
                return res.scenario
            except TypeError:
                res = self.simulator_model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["scenario"]

    def _simulate_first_user_input(
        self, scenario: str, user_profile: str
    ) -> str:
        prompt = ConversationSimulatorTemplate.generate_first_input(
            scenario, user_profile, self.language
        )

        if self.using_native_model:
            res, cost = self.simulator_model.generate(
                prompt, schema=SimulatedInput
            )
            self.simulation_cost += cost
            return res.simulated_input
        else:
            try:
                res: SimulatedInput = self.simulator_model.generate(
                    prompt, SimulatedInput
                )
                return res.simulated_input
            except TypeError:
                res = self.simulator_model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["simulated_input"]

    def _simulate_single_conversation(
        self, model_callback: Callable, min_turns: int, max_turns: int
    ) -> ConversationalTestCase:
        num_turns = random.randint(min_turns, max_turns)
        intent = random.choice(self.user_intentions)
        user_profile = self._a_simulate_user_profile()
        scenario = self._a_simulate_scenario(user_profile, intent)
        additional_metadata = {
            "User Profile": user_profile,
            "User Intent": intent,
        }

        turns = []
        user_input = None
        for turn_index in range(num_turns):
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
                    actual_output=self._generate_chatbot_response(
                        user_input, model_callback=model_callback
                    ),
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

    async def _a_simulate_single_conversation(
        self,
        model_callback: Callable,
        min_turns: int,
        max_turns: int,
        _progress_bar: Optional[tqdm.std.tqdm] = None,
    ) -> ConversationalTestCase:
        num_turns = random.randint(min_turns, max_turns)
        intent = random.choice(self.user_intentions)
        user_profile = await self._a_simulate_user_profile()
        scenario = await self._a_simulate_scenario(user_profile, intent)
        additional_metadata = {
            "User Profile": user_profile,
            "User Intent": intent,
        }

        turns = []
        user_input = None
        for turn_index in range(num_turns):
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
                        user_input, model_callback=model_callback
                    ),
                    additional_metadata=additional_metadata,
                )
            )

        if _progress_bar:
            _progress_bar.update(1)

        return ConversationalTestCase(
            turns=turns, additional_metadata=additional_metadata
        )

    ############################################
    ### Helper Methods #########################
    ############################################

    def _generate_chatbot_response(self, input: str, model_callback: Callable):
        res = model_callback(input)
        return res

    async def _a_generate_chatbot_response(
        self, input: str, model_callback: Callable
    ):
        res = await model_callback(input)
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

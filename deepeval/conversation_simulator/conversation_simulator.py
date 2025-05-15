from typing import Optional, List, Union, Callable, Dict, Any, Tuple
import inspect
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
    ConversationCompletion,
)
from deepeval.progress_context import conversation_simulator_progress_context


class ConversationSimulator:
    def __init__(
        self,
        user_intentions: Dict[str, int],
        user_profile_items: Optional[List[str]] = None,
        user_profiles: Optional[List[str]] = None,
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        opening_message: Optional[str] = None,
        max_concurrent: int = 5,
        async_mode: bool = True,
    ):
        if user_profile_items is None and user_profiles is None:
            raise ValueError(
                "You must supply either `user_profile_items` or `user_profiles`."
            )
        self.user_profiles = user_profiles
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
        stopping_criteria: Optional[str] = None,
    ) -> List[ConversationalTestCase]:
        if min_turns > max_turns:
            raise ValueError("`min_turns` cannot be greater than `max_turns`.")

        self.simulation_cost = 0 if self.using_native_model else None
        total_conversations = sum(
            [num for _, num in self.user_intentions.items()]
        )
        with conversation_simulator_progress_context(
            simulator_model=self.simulator_model.get_model_name(),
            num_conversations=total_conversations,
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
                        stopping_criteria=stopping_criteria,
                        _progress_bar=progress_bar,
                    )
                )
            else:
                if inspect.iscoroutinefunction(model_callback):
                    raise TypeError(
                        "`model_callback` must be a synchronous function when using 'async_mode' is 'False'."
                    )

                conversational_test_cases: List[ConversationalTestCase] = []
                for (
                    intent,
                    num_conversations_per_intent,
                ) in self.user_intentions.items():
                    for _ in range(num_conversations_per_intent):
                        conversational_test_case = (
                            self._simulate_single_conversation(
                                intent=intent,
                                model_callback=model_callback,
                                min_turns=min_turns,
                                max_turns=max_turns,
                                stopping_criteria=stopping_criteria,
                            )
                        )
                        conversational_test_cases.append(
                            conversational_test_case
                        )
                        progress_bar.update(1)

                self.simulated_conversations = conversational_test_cases

        return self.simulated_conversations

    async def _a_simulate(
        self,
        model_callback: Callable[[str], str],
        min_turns: int,
        max_turns: int,
        stopping_criteria: Optional[str],
        _progress_bar: Optional[tqdm.std.tqdm] = None,
    ) -> List[ConversationalTestCase]:
        self.simulation_cost = 0 if self.using_native_model else None

        async def limited_simulation(intent: str):
            async with self.semaphore:
                return await self._a_simulate_single_conversation(
                    intent=intent,
                    model_callback=model_callback,
                    min_turns=min_turns,
                    max_turns=max_turns,
                    stopping_criteria=stopping_criteria,
                    _progress_bar=_progress_bar,
                )

        tasks = [
            limited_simulation(intent)
            for intent, num_conversations_per_intent in self.user_intentions.items()
            for _ in range(num_conversations_per_intent)
        ]
        self.simulated_conversations = await asyncio.gather(*tasks)

    def _simulate_user_profile(self) -> str:
        if self.user_profiles:
            return random.choice(self.user_profiles)
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
        self,
        intent: str,
        model_callback: Callable,
        min_turns: int,
        max_turns: int,
        stopping_criteria: Optional[str],
    ) -> ConversationalTestCase:
        num_turns = random.randint(min_turns, max_turns)
        user_profile = self._simulate_user_profile()
        scenario = self._simulate_scenario(user_profile, intent)
        additional_metadata = {
            "User Profile": user_profile,
            "User Intent": intent,
        }
        model_callback_kwargs = {}

        turns = []
        user_input = None
        conversation_history = None
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

                # Check if conversation should stop early
                prompt = (
                    ConversationSimulatorTemplate.check_conversation_completed(
                        conversation_history, stopping_criteria
                    )
                )
                if stopping_criteria is not None:
                    if self.using_native_model:
                        res, cost = self.simulator_model.generate(
                            prompt, schema=ConversationCompletion
                        )
                        self.simulation_cost += cost
                        is_complete = res.is_complete
                    else:
                        try:
                            res: ConversationCompletion = (
                                self.simulator_model.generate(
                                    prompt, ConversationCompletion
                                )
                            )
                            is_complete = res.is_complete
                        except TypeError:
                            res = self.simulator_model.generate(prompt)
                            data = trimAndLoadJson(res, self)
                            is_complete = data["is_complete"]
                    if is_complete:
                        break

                # Generate next user input
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
                        user_input = data["simulated_input"]

            actual_output, new_model_callback_kwargs = (
                self._generate_chatbot_response(
                    user_input,
                    model_callback=model_callback,
                    conversation_history=conversation_history,
                    callback_kwargs=model_callback_kwargs,
                )
            )
            model_callback_kwargs = new_model_callback_kwargs
            turns.append(
                LLMTestCase(
                    input=user_input,
                    actual_output=actual_output,
                    additional_metadata=additional_metadata,
                )
            )

        return ConversationalTestCase(
            turns=turns, additional_metadata=additional_metadata
        )

    async def _a_simulate_user_profile(self) -> str:
        if self.user_profiles:
            return random.choice(self.user_profiles)
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
                return data["simulated_input"]

    async def _a_simulate_single_conversation(
        self,
        intent: str,
        model_callback: Callable,
        min_turns: int,
        max_turns: int,
        stopping_criteria: Optional[str],
        _progress_bar: Optional[tqdm.std.tqdm] = None,
    ) -> ConversationalTestCase:
        num_turns = random.randint(min_turns, max_turns)
        user_profile = await self._a_simulate_user_profile()
        scenario = await self._a_simulate_scenario(user_profile, intent)
        additional_metadata = {
            "User Profile": user_profile,
            "User Intent": intent,
        }
        model_callback_kwargs = {}

        turns = []
        user_input = None
        conversation_history = None
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

                # Check if conversation should stop early
                prompt = (
                    ConversationSimulatorTemplate.check_conversation_completed(
                        conversation_history, stopping_criteria
                    )
                )
                if stopping_criteria is not None:
                    if self.using_native_model:
                        res, cost = await self.simulator_model.a_generate(
                            prompt, schema=ConversationCompletion
                        )
                        self.simulation_cost += cost
                        is_complete = res.is_complete
                    else:
                        try:
                            res: ConversationCompletion = (
                                await self.simulator_model.a_generate(
                                    prompt, ConversationCompletion
                                )
                            )
                            is_complete = res.is_complete
                        except TypeError:
                            res = await self.simulator_model.a_generate(prompt)
                            data = trimAndLoadJson(res, self)
                            is_complete = data["is_complete"]
                    if is_complete:
                        break

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
                        user_input = data["simulated_input"]

                user_input = res.simulated_input

            actual_output, new_model_callback_kwargs = (
                await self._a_generate_chatbot_response(
                    user_input,
                    model_callback=model_callback,
                    conversation_history=conversation_history,
                    callback_kwargs=model_callback_kwargs,
                )
            )
            model_callback_kwargs = new_model_callback_kwargs
            turns.append(
                LLMTestCase(
                    input=user_input,
                    actual_output=actual_output,
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

    def _generate_chatbot_response(
        self,
        input: str,
        model_callback: Callable,
        conversation_history: List[Dict[str, str]],
        callback_kwargs: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        try:
            res = model_callback(
                input,
                conversation_history=conversation_history,
                **callback_kwargs
            )
        except TypeError:
            res = model_callback(
                input, conversation_history=conversation_history
            )
        if type(res) is str:
            return res, {}
        elif type(res) is tuple:
            return res[0], res[1]

    async def _a_generate_chatbot_response(
        self,
        input: str,
        model_callback: Callable,
        conversation_history: List[Dict[str, str]],
        callback_kwargs: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        try:
            res = await model_callback(
                input,
                conversation_history=conversation_history,
                **callback_kwargs
            )
        except TypeError:
            res = await model_callback(
                input, conversation_history=conversation_history
            )
        if type(res) is str:
            return res, {}
        elif type(res) is tuple:
            return res[0], res[1]

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

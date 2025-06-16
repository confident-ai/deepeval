from typing import Optional, List, Union, Callable, Dict, Any, Tuple
from rich.progress import Progress
import inspect
import asyncio
import random
import json

from deepeval.utils import (
    get_or_create_event_loop,
    update_pbar,
    add_pbar,
    remove_pbars,
)
from deepeval.metrics.utils import (
    convert_turn_to_dict,
    initialize_model,
    trimAndLoadJson,
)
from deepeval.test_case import ConversationalTestCase, Turn
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
        ) as (progress, pbar_id), progress:

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
                        progress=progress,
                        pbar_id=pbar_id,
                    )
                )
            else:
                if inspect.iscoroutinefunction(model_callback):
                    raise TypeError(
                        "`model_callback` must be a synchronous function when using 'async_mode' is 'False'."
                    )

                conversational_test_cases: List[ConversationalTestCase] = []
                intent_list = []
                for intent, num_conversations in self.user_intentions.items():
                    for _ in range(num_conversations):
                        intent_list.append(intent)

                for conversation_index, intent in enumerate(intent_list):
                    conversational_test_case = (
                        self._simulate_single_conversation(
                            intent=intent,
                            model_callback=model_callback,
                            min_turns=min_turns,
                            max_turns=max_turns,
                            stopping_criteria=stopping_criteria,
                            conversation_index=conversation_index,
                            progress=progress,
                            pbar_id=pbar_id,
                        )
                    )
                    conversational_test_cases.append(conversational_test_case)

                self.simulated_conversations = conversational_test_cases
            remove_pbars(progress, [pbar_id])

        return self.simulated_conversations

    async def _a_simulate(
        self,
        model_callback: Callable[[str], str],
        min_turns: int,
        max_turns: int,
        stopping_criteria: Optional[str],
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
    ) -> List[ConversationalTestCase]:
        self.simulation_cost = 0 if self.using_native_model else None

        intent_list: List[str] = []
        for intent, num_conversations in self.user_intentions.items():
            for _ in range(num_conversations):
                intent_list.append(intent)

        async def limited_simulation(
            intent: str,
            conversation_index: int,
        ):
            async with self.semaphore:
                return await self._a_simulate_single_conversation(
                    intent=intent,
                    model_callback=model_callback,
                    min_turns=min_turns,
                    max_turns=max_turns,
                    stopping_criteria=stopping_criteria,
                    conversation_index=conversation_index,
                    progress=progress,
                    pbar_id=pbar_id,
                )

        tasks = [
            limited_simulation(intent, i)
            for i, intent in enumerate(intent_list)
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
        conversation_index: int,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
    ) -> ConversationalTestCase:
        num_turns = random.randint(min_turns, max_turns)

        # determine pbar length
        pbar_conversation_length = 2
        pbar_turns_length = (num_turns - 1) * (
            3 if stopping_criteria is not None else 2
        ) + 2
        pbar_generating_scenario_length = 2

        # add pbar
        pbar_conversation_id = add_pbar(
            progress,
            f"\t⚡ Test case #{conversation_index}",
            total=pbar_conversation_length,
        )
        pbar_generating_scenario_id = add_pbar(
            progress,
            f"\t\t🖼️  Setting scenario",
            total=pbar_generating_scenario_length,
        )
        pbar_turns_id = add_pbar(
            progress,
            f"\t\t💬 Conversing",
            total=pbar_turns_length,
        )

        user_profile = self._simulate_user_profile()
        update_pbar(progress, pbar_generating_scenario_id, remove=False)
        scenario = self._simulate_scenario(user_profile, intent)
        update_pbar(progress, pbar_generating_scenario_id, remove=False)
        update_pbar(progress, pbar_conversation_id, remove=False)
        additional_metadata = {
            "User Profile": user_profile,
            "User Intent": intent,
        }
        model_callback_kwargs = {}

        turns: List[Turn] = []
        user_input = None
        conversation_history = None

        for turn_index in range(num_turns):
            if turn_index == 0 and self.opening_message:
                # Add optional opening message from chatbot
                turns.append(
                    Turn(role="assistant", content=self.opening_message)
                )

            if turn_index == 0:
                # First user input
                user_input = self._simulate_first_user_input(
                    scenario, user_profile
                )
                update_pbar(progress, pbar_turns_id, remove=False)
            else:
                # Generate next user input based on conversation so far
                conversation_history = json.dumps(
                    [convert_turn_to_dict(turn) for turn in turns], indent=4
                )

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
                        update_pbar(
                            progress,
                            pbar_turns_id,
                            advance_to_end=True,
                            remove=False,
                        )
                        break
                    else:
                        update_pbar(progress, pbar_turns_id, remove=False)

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
                update_pbar(progress, pbar_turns_id, remove=False)

            ai_output, new_model_callback_kwargs = (
                self._generate_chatbot_response(
                    user_input,
                    model_callback=model_callback,
                    conversation_history=conversation_history,
                    callback_kwargs=model_callback_kwargs,
                )
            )
            update_pbar(progress, pbar_turns_id, remove=False)
            model_callback_kwargs = new_model_callback_kwargs
            turns.append(
                Turn(
                    role="user",
                    content=user_input,
                )
            )
            turns.append(
                Turn(
                    role="assistant",
                    content=ai_output,
                    additional_metadata=additional_metadata,
                )
            )

        update_pbar(progress, pbar_conversation_id, remove=False)
        update_pbar(progress, pbar_id, remove=False)
        remove_pbars(
            progress,
            [pbar_conversation_id, pbar_generating_scenario_id, pbar_turns_id],
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
        conversation_index: Optional[int] = None,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
    ) -> ConversationalTestCase:
        num_turns = random.randint(min_turns, max_turns)

        # determine pbar length
        pbar_conversation_length = 2
        pbar_turns_length = (num_turns - 1) * (
            3 if stopping_criteria is not None else 2
        ) + 2
        pbar_generating_scenario_length = 2

        # add pbar
        pbar_conversation_id = add_pbar(
            progress,
            f"\t⚡ Test case #{conversation_index}",
            total=pbar_conversation_length,
        )
        pbar_generating_scenario_id = add_pbar(
            progress,
            f"\t\t🖼️  Setting scenario",
            total=pbar_generating_scenario_length,
        )
        pbar_turns_id = add_pbar(
            progress,
            f"\t\t💬 Conversing with AI",
            total=pbar_turns_length,
        )

        user_profile = await self._a_simulate_user_profile()
        update_pbar(progress, pbar_generating_scenario_id, remove=False)
        scenario = await self._a_simulate_scenario(user_profile, intent)
        update_pbar(progress, pbar_generating_scenario_id, remove=False)
        update_pbar(progress, pbar_conversation_id, remove=False)
        additional_metadata = {
            "User Profile": user_profile,
            "User Intent": intent,
        }
        model_callback_kwargs = {}

        turns: List[Turn] = []
        user_input = None
        conversation_history = None

        for turn_index in range(num_turns):
            if turn_index == 0 and self.opening_message:
                turns.append(
                    Turn(role="assistant", content=self.opening_message)
                )
            if turn_index == 0:
                user_input = await self._a_simulate_first_user_input(
                    scenario, user_profile
                )
                update_pbar(progress, pbar_turns_id)

            else:
                # Generate next user input based on conversation so far
                conversation_history = json.dumps(
                    [convert_turn_to_dict(turn) for turn in turns], indent=4
                )

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
                        update_pbar(
                            progress,
                            pbar_turns_id,
                            advance_to_end=True,
                            remove=False,
                        )
                        break
                    else:
                        update_pbar(progress, pbar_turns_id, remove=False)

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
                update_pbar(progress, pbar_turns_id, remove=False)

                user_input = res.simulated_input

            ai_output, new_model_callback_kwargs = (
                await self._a_generate_chatbot_response(
                    user_input,
                    model_callback=model_callback,
                    conversation_history=conversation_history,
                    callback_kwargs=model_callback_kwargs,
                )
            )
            update_pbar(progress, pbar_turns_id, remove=False)
            model_callback_kwargs = new_model_callback_kwargs
            turns.append(
                Turn(
                    role="user",
                    content=user_input,
                )
            )
            turns.append(
                Turn(
                    role="assistant",
                    content=ai_output,
                    additional_metadata=additional_metadata,
                )
            )

        update_pbar(progress, pbar_conversation_id, remove=False)
        update_pbar(progress, pbar_id, remove=False)
        remove_pbars(
            progress,
            [pbar_turns_id, pbar_conversation_id, pbar_generating_scenario_id],
        )

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
                **callback_kwargs,
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
                **callback_kwargs,
            )
        except TypeError:
            res = await model_callback(
                input, conversation_history=conversation_history
            )
        if type(res) is str:
            return res, {}
        elif type(res) is tuple:
            return res[0], res[1]

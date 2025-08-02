from typing import Optional, List, Union, Callable, Dict, Any, Tuple
from rich.progress import Progress
from pydantic import BaseModel
from dataclasses import asdict
import inspect
import asyncio
import uuid
import json

from deepeval.utils import (
    get_or_create_event_loop,
    update_pbar,
    add_pbar,
    remove_pbars,
)
from deepeval.metrics.utils import (
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
    ConversationCompletion,
)
from deepeval.progress_context import conversation_simulator_progress_context
from deepeval.dataset import ConversationalGolden


class ConversationSimulator:
    def __init__(
        self,
        model_callback: Callable[[str], str],
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        opening_message: Optional[str] = None,
        max_concurrent: int = 5,
        async_mode: bool = True,
    ):
        self.model_callback = model_callback
        self.simulator_model, self.using_native_model = initialize_model(
            simulator_model
        )
        self.opening_message = opening_message
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.async_mode = async_mode
        self.language = "English"
        self.simulated_conversations: List[ConversationalTestCase] = []
        self.template = ConversationSimulatorTemplate

    def simulate(
        self,
        conversational_goldens: List[ConversationalGolden],
        max_turns: int = 10,
    ) -> List[ConversationalTestCase]:
        self.simulation_cost = 0 if self.using_native_model else None

        with conversation_simulator_progress_context(
            simulator_model=self.simulator_model.get_model_name(),
            num_conversations=len(conversational_goldens),
            async_mode=self.async_mode,
        ) as (progress, pbar_id), progress:

            if self.async_mode:
                if not inspect.iscoroutinefunction(self.model_callback):
                    raise TypeError(
                        "`model_callback` must be a coroutine function when using 'async_mode' is 'True'."
                    )

                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self._a_simulate(
                        conversational_goldens=conversational_goldens,
                        max_turns=max_turns,
                        progress=progress,
                        pbar_id=pbar_id,
                    )
                )
            else:
                if inspect.iscoroutinefunction(self.model_callback):
                    raise TypeError(
                        "`model_callback` must be a synchronous function when using 'async_mode' is 'False'."
                    )
                conversational_test_cases: List[ConversationalTestCase] = []
                for conversation_index, golden in enumerate(
                    conversational_goldens
                ):
                    conversational_test_case = (
                        self._simulate_single_conversation(
                            golden=golden,
                            max_turns=max_turns,
                            index=conversation_index,
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
        conversational_goldens: List[ConversationalGolden],
        max_turns: int,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
    ) -> List[ConversationalTestCase]:
        self.simulation_cost = 0 if self.using_native_model else None

        async def simulate_conversations(
            golden: ConversationalGolden,
            conversation_index: int,
        ):
            async with self.semaphore:
                return await self._a_simulate_single_conversation(
                    golden=golden,
                    max_turns=max_turns,
                    index=conversation_index,
                    progress=progress,
                    pbar_id=pbar_id,
                )

        tasks = [
            simulate_conversations(golden, i)
            for i, golden in enumerate(conversational_goldens)
        ]
        self.simulated_conversations = await asyncio.gather(*tasks)

    ############################################
    ### Simulate Single Conversation ###########
    ############################################

    def _simulate_single_conversation(
        self,
        golden: ConversationalGolden,
        max_turns: int,
        index: int,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
    ) -> ConversationalTestCase:
        additional_metadata = {"User Description": golden.user_description}
        turns = (
            [Turn(role="assistant", content=self.opening_message)]
            if self.opening_message
            else []
        )
        user_input = None
        thread_id = str(uuid.uuid4())

        # Define pbar
        pbar_turns_id = add_pbar(
            progress,
            f"\t⚡ Test case #{index}",
            total=(max_turns - 1)
            * (3 if golden.expected_outcome is not None else 2)
            + 2,
        )

        # Generate first turn (from user)
        prompt = self.template.simulate_first_user_turn(golden, self.language)
        user_input = self.generate_schema(prompt, SimulatedInput)
        update_pbar(progress, pbar_turns_id, remove=False)

        for _ in range(max_turns - 1):
            # Stop conversation if needed
            stop_conversation = self.stop_conversation(
                turns, golden.expected_outcome, progress, pbar_turns_id
            )
            if stop_conversation:
                break

            # Generate turn from user
            prompt = self.template.simulate_user_turn(
                golden, turns, self.language
            )
            simulated_input: SimulatedInput = self.generate_schema(
                prompt, SimulatedInput
            )
            user_input = simulated_input.simulated_input
            turns.append(Turn(role="user", content=user_input))
            update_pbar(progress, pbar_turns_id, remove=False)

            # Generate turn from assistant
            chatbot_response = self.generate_chatbot_response(
                user_input,
                model_callback=self.model_callback,
                turns=turns,
                thread_id=thread_id,
            )
            turns.append(Turn(role="assistant", content=chatbot_response))
            update_pbar(progress, pbar_turns_id, remove=False)

        update_pbar(progress, pbar_id, remove=False)
        remove_pbars(progress, [pbar_turns_id])
        return ConversationalTestCase(
            turns=turns, additional_metadata=additional_metadata
        )

    async def _a_simulate_single_conversation(
        self,
        golden: ConversationalGolden,
        max_turns: int,
        index: Optional[int] = None,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
    ) -> ConversationalTestCase:
        additional_metadata = {"User Description": golden.user_description}
        turns = (
            [Turn(role="assistant", content=self.opening_message)]
            if self.opening_message
            else []
        )
        user_input = None
        thread_id = str(uuid.uuid4())

        # Define pbar
        pbar_turns_id = add_pbar(
            progress,
            f"\t⚡ Test case #{index}",
            total=(max_turns - 1)
            * (3 if golden.expected_outcome is not None else 2)
            + 2,
        )

        # Generate first user input
        prompt = self.template.simulate_first_user_turn(golden, self.language)
        user_input = await self.a_generate_schema(prompt, SimulatedInput)
        update_pbar(progress, pbar_turns_id, remove=False)

        for _ in range(max_turns - 1):
            # Stop conversation if needed
            stop_conversation = await self.a_stop_conversation(
                turns, golden.expected_outcome, progress, pbar_turns_id
            )
            if stop_conversation:
                break

            # Generate turn from user
            prompt = self.template.simulate_user_turn(
                golden, turns, self.language
            )
            simulated_input: SimulatedInput = await self.a_generate_schema(
                prompt, SimulatedInput
            )
            user_input = simulated_input.simulated_input
            turns.append(Turn(role="user", content=user_input))
            update_pbar(progress, pbar_turns_id, remove=False)

            # Generate turn from assistant
            chatbot_response = await self.a_generate_chatbot_response(
                user_input,
                model_callback=self.model_callback,
                turns=turns,
                thread_id=thread_id,
            )
            turns.append(Turn(role="assistant", content=chatbot_response))
            update_pbar(progress, pbar_turns_id, remove=False)

        update_pbar(progress, pbar_id, remove=False)
        remove_pbars(progress, [pbar_turns_id])
        return ConversationalTestCase(
            turns=turns, additional_metadata=additional_metadata
        )

    ############################################
    ### Stop Conversation ######################
    ############################################

    def stop_conversation(
        self,
        turns: List[Turn],
        expected_outcome: Optional[str],
        progress: Optional[Progress] = None,
        pbar_turns_id: Optional[int] = None,
    ):
        conversation_history = json.dumps([asdict(t) for t in turns], indent=4)
        prompt = self.template.stop_simulation(
            conversation_history, expected_outcome
        )
        if expected_outcome is not None:
            is_complete: ConversationCompletion = self.generate_schema(
                prompt, ConversationCompletion
            )
            update_pbar(
                progress,
                pbar_turns_id,
                advance_to_end=is_complete.is_complete,
                remove=False,
            )
            return is_complete.is_complete
        return False

    async def a_stop_conversation(
        self,
        turns: List[Turn],
        expected_outcome: Optional[str],
        progress: Optional[Progress] = None,
        pbar_turns_id: Optional[int] = None,
    ):
        conversation_history = json.dumps([asdict(t) for t in turns], indent=4)
        prompt = self.template.stop_simulation(
            conversation_history, expected_outcome
        )
        if expected_outcome is not None:
            is_complete: ConversationCompletion = await self.a_generate_schema(
                prompt, ConversationCompletion
            )
            update_pbar(
                progress,
                pbar_turns_id,
                advance_to_end=is_complete.is_complete,
                remove=False,
            )
            return is_complete.is_complete
        return False

    ############################################
    ### Generate Structured Response ###########
    ############################################

    def generate_schema(
        self,
        prompt: str,
        schema: BaseModel,
    ) -> BaseModel:
        _, using_native_model = initialize_model(model=self.simulator_model)
        if using_native_model:
            res, cost = self.simulator_model.generate(prompt, schema=schema)
            self.simulation_cost += cost
            return res
        else:
            try:
                res = self.simulator_model.generate(prompt, schema=schema)
                return res
            except TypeError:
                res = self.simulator_model.generate(prompt)
                data = trimAndLoadJson(res)
                return schema(**data)

    async def a_generate_schema(
        self,
        prompt: str,
        schema: BaseModel,
    ) -> BaseModel:
        _, using_native_model = initialize_model(model=self.simulator_model)
        if using_native_model:
            res, cost = await self.simulator_model.a_generate(
                prompt, schema=schema
            )
            self.simulation_cost += cost
            return res
        else:
            try:
                res = await self.simulator_model.a_generate(
                    prompt, schema=schema
                )
                return res
            except TypeError:
                res = await self.simulator_model.a_generate(prompt)
            data = trimAndLoadJson(res)
            return schema(**data)

    ############################################
    ### Invoke Model Callback ##################
    ############################################

    def generate_chatbot_response(
        self,
        input: str,
        turns: List[Turn],
        thread_id: str,
        model_callback: Callable,
    ) -> Tuple[str, Dict[str, Any]]:
        callback_kwargs = {
            "input": input,
            "turns": turns,
            "thread_id": thread_id,
        }
        supported_args = set(
            inspect.signature(model_callback).parameters.keys()
        )
        return model_callback(
            **{k: v for k, v in callback_kwargs.items() if k in supported_args}
        )

    async def a_generate_chatbot_response(
        self,
        input: str,
        model_callback: Callable,
        turns: List[Turn],
        thread_id: str,
    ) -> Tuple[str, Dict[str, Any]]:
        candidate_kwargs = {
            "input": input,
            "turns": turns,
            "thread_id": thread_id,
        }
        supported_args = set(
            inspect.signature(model_callback).parameters.keys()
        )
        return await model_callback(
            **{k: v for k, v in candidate_kwargs.items() if k in supported_args}
        )

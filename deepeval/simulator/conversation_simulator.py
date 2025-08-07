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
)
from deepeval.metrics.utils import (
    initialize_model,
    trimAndLoadJson,
)
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.simulator.template import (
    ConversationSimulatorTemplate,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.simulator.schema import (
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
        language: str = "English",
    ):
        self.model_callback = model_callback
        self.is_callback_async = inspect.iscoroutinefunction(
            self.model_callback
        )
        self.simulator_model, self.using_native_model = initialize_model(
            simulator_model
        )
        self.opening_message = opening_message
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.async_mode = async_mode
        self.language = language
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
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than 0")

        # Define pbar
        max_turns_including_opening = (
            max_turns + 1 if self.opening_message else max_turns
        )
        pbar_turns_id = add_pbar(
            progress,
            f"\t⚡ Test case #{index}",
            total=max_turns_including_opening,
        )

        additional_metadata = {"User Description": golden.user_description}
        user_input = None
        thread_id = str(uuid.uuid4())
        turns = []
        if self.opening_message:
            turns.append(Turn(role="assistant", content=self.opening_message))
            update_pbar(progress, pbar_turns_id)

        while True:
            # Stop conversation if needed
            stop_conversation = self.stop_conversation(
                turns, golden.expected_outcome, progress, pbar_turns_id
            )
            if stop_conversation:
                break

            # Generate turn from user
            if len(turns) >= max_turns_including_opening:
                break
            if len(turns) == 0 or (len(turns) == 1 and self.opening_message):
                # Generate first user input
                prompt = self.template.simulate_first_user_turn(
                    golden, self.language
                )
                simulated_input: SimulatedInput = self.generate_schema(
                    prompt, SimulatedInput
                )
            else:
                prompt = self.template.simulate_user_turn(
                    golden, turns, self.language
                )
                simulated_input: SimulatedInput = self.generate_schema(
                    prompt, SimulatedInput
                )
            user_input = simulated_input.simulated_input
            turns.append(Turn(role="user", content=user_input))
            update_pbar(progress, pbar_turns_id)

            # Generate turn from assistant
            if len(turns) >= max_turns_including_opening:
                break
            if self.is_callback_async:
                turn = asyncio.run(
                    self.a_generate_turn_from_callback(
                        user_input,
                        model_callback=self.model_callback,
                        turns=turns,
                        thread_id=thread_id,
                    )
                )
            else:
                turn = self.generate_turn_from_callback(
                    user_input,
                    model_callback=self.model_callback,
                    turns=turns,
                    thread_id=thread_id,
                )
            turns.append(turn)
            update_pbar(progress, pbar_turns_id)

        update_pbar(progress, pbar_id)
        return ConversationalTestCase(
            turns=turns,
            scenario=golden.scenario,
            expected_outcome=golden.expected_outcome,
            user_description=golden.user_description,
            context=golden.context,
            name=golden.name,
            additional_metadata={
                **(golden.additional_metadata or {}),
                **additional_metadata,
            },
            comments=golden.comments,
            _dataset_rank=golden._dataset_rank,
            _dataset_alias=golden._dataset_alias,
            _dataset_id=golden._dataset_id,
        )

    async def _a_simulate_single_conversation(
        self,
        golden: ConversationalGolden,
        max_turns: int,
        index: Optional[int] = None,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
    ) -> ConversationalTestCase:
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than 0")

        # Define pbar
        max_turns_including_opening = (
            max_turns + 1 if self.opening_message else max_turns
        )
        pbar_turns_id = add_pbar(
            progress,
            f"\t⚡ Test case #{index}",
            total=max_turns_including_opening,
        )

        additional_metadata = {"User Description": golden.user_description}
        user_input = None
        thread_id = str(uuid.uuid4())
        turns = []
        if self.opening_message:
            turns.append(Turn(role="assistant", content=self.opening_message))
            update_pbar(progress, pbar_turns_id)

        while True:
            # Stop conversation if needed
            stop_conversation = await self.a_stop_conversation(
                turns, golden.expected_outcome, progress, pbar_turns_id
            )
            if stop_conversation:
                break

            # Generate turn from user
            if len(turns) >= max_turns_including_opening:
                break
            if len(turns) == 0 or (len(turns) == 1 and self.opening_message):
                # Generate first user input
                prompt = self.template.simulate_first_user_turn(
                    golden, self.language
                )
                simulated_input: SimulatedInput = await self.a_generate_schema(
                    prompt, SimulatedInput
                )
            else:
                prompt = self.template.simulate_user_turn(
                    golden, turns, self.language
                )
                simulated_input: SimulatedInput = await self.a_generate_schema(
                    prompt, SimulatedInput
                )
            user_input = simulated_input.simulated_input
            turns.append(Turn(role="user", content=user_input))
            update_pbar(progress, pbar_turns_id)

            # Generate turn from assistant
            if len(turns) >= max_turns_including_opening:
                break
            if self.is_callback_async:
                turn = await self.a_generate_turn_from_callback(
                    user_input,
                    model_callback=self.model_callback,
                    turns=turns,
                    thread_id=thread_id,
                )
            else:
                turn = self.generate_turn_from_callback(
                    user_input,
                    model_callback=self.model_callback,
                    turns=turns,
                    thread_id=thread_id,
                )
            turns.append(turn)
            update_pbar(progress, pbar_turns_id)

        update_pbar(progress, pbar_id)
        return ConversationalTestCase(
            turns=turns,
            scenario=golden.scenario,
            expected_outcome=golden.expected_outcome,
            user_description=golden.user_description,
            context=golden.context,
            name=golden.name,
            additional_metadata={
                **(golden.additional_metadata or {}),
                **additional_metadata,
            },
            comments=golden.comments,
            _dataset_rank=golden._dataset_rank,
            _dataset_alias=golden._dataset_alias,
            _dataset_id=golden._dataset_id,
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
            if is_complete.is_complete:
                update_pbar(
                    progress,
                    pbar_turns_id,
                    advance_to_end=is_complete.is_complete,
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
            if is_complete.is_complete:
                update_pbar(
                    progress,
                    pbar_turns_id,
                    advance_to_end=is_complete.is_complete,
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

    def generate_turn_from_callback(
        self,
        input: str,
        turns: List[Turn],
        thread_id: str,
        model_callback: Callable,
    ) -> Turn:
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

    async def a_generate_turn_from_callback(
        self,
        input: str,
        model_callback: Callable,
        turns: List[Turn],
        thread_id: str,
    ) -> Turn:
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

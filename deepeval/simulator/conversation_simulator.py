from typing import Optional, List, Union, Callable
from rich.progress import Progress
from pydantic import BaseModel
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
from deepeval.confident.api import Api, Endpoints, HttpMethods, is_confident
from deepeval.simulator.schema import SimulateHttpResponse


class ConversationSimulator:
    def __init__(
        self,
        model_callback: Callable[[str], str],
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        max_concurrent: int = 5,
        async_mode: bool = True,
        language: str = "English",
        run_remote: bool = False,
    ):
        self.model_callback = model_callback
        self.is_callback_async = inspect.iscoroutinefunction(
            self.model_callback
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.async_mode = async_mode
        self.language = language
        self.simulated_conversations: List[ConversationalTestCase] = []
        self.template = ConversationSimulatorTemplate
        self.run_remote = run_remote
        if not run_remote:
            self.simulator_model, self.using_native_model = initialize_model(
                simulator_model
            )
        else:
            if not is_confident():
                raise ValueError(
                    "Confident API key not found. Run `deepeval login` to login."
                )
            self.simulator_model = None
            self.using_native_model = False

    def simulate(
        self,
        conversational_goldens: List[ConversationalGolden],
        max_user_simulations: int = 10,
        on_simulation_complete: Optional[
            Callable[[ConversationalTestCase, int], None]
        ] = None,
    ) -> List[ConversationalTestCase]:
        self.simulation_cost = 0 if self.using_native_model else None

        with conversation_simulator_progress_context(
            simulator_model=(
                self.simulator_model.get_model_name()
                if self.simulator_model
                else "Confident API"
            ),
            num_conversations=len(conversational_goldens),
            async_mode=self.async_mode,
        ) as (progress, pbar_id), progress:

            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self._a_simulate(
                        conversational_goldens=conversational_goldens,
                        max_user_simulations=max_user_simulations,
                        on_simulation_complete=on_simulation_complete,
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
                            max_user_simulations=max_user_simulations,
                            index=conversation_index,
                            progress=progress,
                            pbar_id=pbar_id,
                            on_simulation_complete=on_simulation_complete,
                        )
                    )
                    conversational_test_cases.append(conversational_test_case)

                self.simulated_conversations = conversational_test_cases

        return self.simulated_conversations

    async def _a_simulate(
        self,
        conversational_goldens: List[ConversationalGolden],
        max_user_simulations: int,
        on_simulation_complete: Optional[
            Callable[[ConversationalTestCase, int], None]
        ] = None,
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
                    max_user_simulations=max_user_simulations,
                    index=conversation_index,
                    progress=progress,
                    pbar_id=pbar_id,
                    on_simulation_complete=on_simulation_complete,
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
        max_user_simulations: int,
        index: int,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
        on_simulation_complete: Optional[
            Callable[[ConversationalTestCase, int], None]
        ] = None,
    ) -> ConversationalTestCase:
        simulation_counter = 0
        if max_user_simulations <= 0:
            raise ValueError("max_user_simulations must be greater than 0")

        # Define pbar
        pbar_max_user_simluations_id = add_pbar(
            progress,
            f"\t⚡ Test case #{index}",
            total=max_user_simulations + 1,
        )

        additional_metadata = {"User Description": golden.user_description}
        user_input = None
        thread_id = str(uuid.uuid4())
        turns: List[Turn] = []

        if golden.turns is not None:
            turns.extend(golden.turns)

        while True:
            # Stop conversation if needed
            stop_conversation = self.stop_conversation(
                turns,
                golden,
                progress,
                pbar_max_user_simluations_id,
            )
            if stop_conversation:
                break

            # Generate turn from user
            if simulation_counter >= max_user_simulations:
                update_pbar(progress, pbar_max_user_simluations_id)
                break
            if len(turns) == 0:
                # Generate first user input
                user_input = self.generate_first_user_input(golden)
                turns.append(Turn(role="user", content=user_input))
                update_pbar(progress, pbar_max_user_simluations_id)
                simulation_counter += 1
            elif turns[-1].role != "user":
                user_input = self.generate_next_user_input(golden, turns)
                turns.append(Turn(role="user", content=user_input))
                update_pbar(progress, pbar_max_user_simluations_id)
                simulation_counter += 1
            else:
                user_input = turns[-1].content

            # Generate turn from assistant
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

        update_pbar(progress, pbar_id)
        conversational_test_case = ConversationalTestCase(
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
        if on_simulation_complete:
            on_simulation_complete(conversational_test_case, index)
        return conversational_test_case

    async def _a_simulate_single_conversation(
        self,
        golden: ConversationalGolden,
        max_user_simulations: int,
        index: Optional[int] = None,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
        on_simulation_complete: Optional[
            Callable[[ConversationalTestCase, int], None]
        ] = None,
    ) -> ConversationalTestCase:
        simulation_counter = 0
        if max_user_simulations <= 0:
            raise ValueError("max_user_simulations must be greater than 0")

        # Define pbar
        pbar_max_user_simluations_id = add_pbar(
            progress,
            f"\t⚡ Test case #{index}",
            total=max_user_simulations + 1,
        )

        additional_metadata = {"User Description": golden.user_description}
        user_input = None
        thread_id = str(uuid.uuid4())
        turns: List[Turn] = []

        if golden.turns is not None:
            turns.extend(golden.turns)

        while True:
            # Stop conversation if needed
            stop_conversation = await self.a_stop_conversation(
                turns,
                golden,
                progress,
                pbar_max_user_simluations_id,
            )
            if stop_conversation:
                break

            # Generate turn from user
            if simulation_counter >= max_user_simulations:
                update_pbar(progress, pbar_max_user_simluations_id)
                break
            if len(turns) == 0:
                # Generate first user input
                user_input = await self.a_generate_first_user_input(golden)
                turns.append(Turn(role="user", content=user_input))
                update_pbar(progress, pbar_max_user_simluations_id)
                simulation_counter += 1
            elif turns[-1].role != "user":
                user_input = await self.a_generate_next_user_input(
                    golden, turns
                )
                turns.append(Turn(role="user", content=user_input))
                update_pbar(progress, pbar_max_user_simluations_id)
                simulation_counter += 1
            else:
                user_input = turns[-1].content

            # Generate turn from assistant
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

        update_pbar(progress, pbar_id)
        conversational_test_case = ConversationalTestCase(
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
        if on_simulation_complete:
            on_simulation_complete(conversational_test_case, index)
        return conversational_test_case

    ############################################
    ### Generate User Inputs ###################
    ############################################

    def generate_first_user_input(self, golden: ConversationalGolden):
        if not self.run_remote:
            prompt = self.template.simulate_first_user_turn(
                golden, self.language
            )
            simulated_input: SimulatedInput = self.generate_schema(
                prompt, SimulatedInput
            )
            user_input = simulated_input.simulated_input
        else:
            api = Api()
            data, _ = api.send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.SIMULATE_ENDPOINT,
                body=self.dump_conversational_golden(golden),
            )
            res = SimulateHttpResponse(
                user_input=data["userResponse"], complete=data["completed"]
            )
            user_input = res.user_input
        return user_input

    async def a_generate_first_user_input(self, golden: ConversationalGolden):
        if not self.run_remote:
            prompt = self.template.simulate_first_user_turn(
                golden, self.language
            )
            simulated_input: SimulatedInput = await self.a_generate_schema(
                prompt, SimulatedInput
            )
            user_input = simulated_input.simulated_input
        else:
            api = Api()
            data, _ = await api.a_send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.SIMULATE_ENDPOINT,
                body=self.dump_conversational_golden(golden),
            )
            res = SimulateHttpResponse(
                user_input=data["userResponse"], complete=data["completed"]
            )
            user_input = res.user_input
        return user_input

    def generate_next_user_input(
        self, golden: ConversationalGolden, turns: List[Turn]
    ):
        if not self.run_remote:
            prompt = self.template.simulate_user_turn(
                golden, turns, self.language
            )
            simulated_input: SimulatedInput = self.generate_schema(
                prompt, SimulatedInput
            )
            user_input = simulated_input.simulated_input
        else:
            api = Api()
            temp_golden = ConversationalGolden(
                scenario=golden.scenario,
                expected_outcome=golden.expected_outcome,
                user_description=golden.user_description,
                context=golden.context,
                turns=turns,
            )
            data, _ = api.send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.SIMULATE_ENDPOINT,
                body=self.dump_conversational_golden(temp_golden),
            )
            res = SimulateHttpResponse(
                user_input=data["userResponse"], complete=data["completed"]
            )
            user_input = res.user_input
        return user_input

    async def a_generate_next_user_input(
        self, golden: ConversationalGolden, turns: List[Turn]
    ):
        if not self.run_remote:
            prompt = self.template.simulate_user_turn(
                golden, turns, self.language
            )
            simulated_input: SimulatedInput = await self.a_generate_schema(
                prompt, SimulatedInput
            )
            user_input = simulated_input.simulated_input
        else:
            api = Api()
            temp_golden = ConversationalGolden(
                scenario=golden.scenario,
                expected_outcome=golden.expected_outcome,
                user_description=golden.user_description,
                context=golden.context,
                turns=turns,
            )
            data, _ = await api.a_send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.SIMULATE_ENDPOINT,
                body=self.dump_conversational_golden(temp_golden),
            )
            res = SimulateHttpResponse(
                user_input=data["userResponse"], complete=data["completed"]
            )
            user_input = res.user_input
        return user_input

    ############################################
    ### Stop Conversation ######################
    ############################################

    def stop_conversation(
        self,
        turns: List[Turn],
        golden: ConversationalGolden,
        progress: Optional[Progress] = None,
        pbar_turns_id: Optional[int] = None,
    ):
        if not self.run_remote:
            conversation_history = json.dumps(
                [t.model_dump() for t in turns], indent=4
            )
            prompt = self.template.stop_simulation(
                conversation_history, golden.expected_outcome
            )
            if golden.expected_outcome is not None:
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
        else:
            api = Api()
            temp_golden = ConversationalGolden(
                scenario=golden.scenario,
                expected_outcome=golden.expected_outcome,
                user_description=golden.user_description,
                context=golden.context,
                turns=turns,
            )
            data, _ = api.send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.SIMULATE_ENDPOINT,
                body=self.dump_conversational_golden(temp_golden),
            )
            res = SimulateHttpResponse(
                user_input=data["userResponse"], complete=data["completed"]
            )
            return res.complete

    async def a_stop_conversation(
        self,
        turns: List[Turn],
        golden: ConversationalGolden,
        progress: Optional[Progress] = None,
        pbar_turns_id: Optional[int] = None,
    ):
        if not self.run_remote:
            conversation_history = json.dumps(
                [t.model_dump() for t in turns], indent=4
            )
            prompt = self.template.stop_simulation(
                conversation_history, golden.expected_outcome
            )
            if golden.expected_outcome is not None:
                is_complete: ConversationCompletion = (
                    await self.a_generate_schema(prompt, ConversationCompletion)
                )
                if is_complete.is_complete:
                    update_pbar(
                        progress,
                        pbar_turns_id,
                        advance_to_end=is_complete.is_complete,
                    )
                return is_complete.is_complete
            return False
        else:
            api = Api()
            temp_golden = ConversationalGolden(
                scenario=golden.scenario,
                expected_outcome=golden.expected_outcome,
                user_description=golden.user_description,
                context=golden.context,
                turns=turns,
            )
            data, _ = api.send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.SIMULATE_ENDPOINT,
                body=self.dump_conversational_golden(temp_golden),
            )
            res = SimulateHttpResponse(
                user_input=data["userResponse"], complete=data["completed"]
            )
            return res.complete

    ############################################
    ### Generate Structured Response ###########
    ############################################

    def generate_schema(
        self,
        prompt: str,
        schema: BaseModel,
    ) -> BaseModel:
        if self.using_native_model:
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
        if self.using_native_model:
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

    ############################################
    ### Invoke Model Callback ##################
    ############################################

    def dump_conversational_golden(self, golden: ConversationalGolden):
        new_golden = ConversationalGolden(
            scenario=golden.scenario,
            expected_outcome=golden.expected_outcome,
            user_description=golden.user_description,
            context=golden.context,
            turns=(
                [
                    Turn(
                        role=turn.role,
                        content=turn.content,
                        user_id=turn.user_id,
                        retrieval_context=turn.retrieval_context,
                        tools_called=turn.tools_called,
                    )
                    for turn in golden.turns
                ]
                if golden.turns is not None
                else None
            ),
        )
        try:
            body = new_golden.model_dump(
                by_alias=True,
                exclude_none=True,
                exclude={"turns": {"__all__": {"_mcp_interaction"}}},
            )
        except AttributeError:
            body = new_golden.dict(
                by_alias=True,
                exclude_none=True,
                exclude={"turns": {"__all__": {"_mcp_interaction"}}},
            )
        return {"conversationalGolden": body}

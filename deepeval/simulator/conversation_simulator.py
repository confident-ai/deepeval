from typing import Any, Optional, List, Type, Union, Callable
from rich.progress import Progress
from pydantic import BaseModel
import inspect
import asyncio
import uuid
import warnings

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
    SimulationTemplate,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import MULTIMODAL_SUPPORTED_MODELS
from deepeval.simulator.schema import (
    SimulatedInput,
)
from deepeval.simulator.controller.controller import (
    SimulationController,
    expected_outcome_controller,
)
from deepeval.simulator.simulation_graph import (
    SimulationNode,
    default_simulation_node,
)
from deepeval.simulator.simulation_graph.runner import (
    _SimulationGraphRunner,
    _GraphConversationState,
)
from deepeval.progress_context import conversation_simulator_progress_context
from deepeval.dataset import ConversationalGolden

_MISSING = object()


class ConversationSimulator:
    def __init__(
        self,
        model_callback: Callable[[str], str],
        simulation_graph: Optional[SimulationNode] = None,
        stopping_controller: Callable = expected_outcome_controller,
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        max_concurrent: int = 5,
        async_mode: bool = True,
        language: str = "English",
        controller: Any = _MISSING,
    ):
        if controller is not _MISSING:
            if stopping_controller is not expected_outcome_controller:
                raise TypeError(
                    "Pass either `stopping_controller` or the deprecated "
                    "`controller`, not both."
                )
            warnings.warn(
                "`controller` is deprecated; use `stopping_controller` "
                "instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            stopping_controller = controller

        self.model_callback = model_callback
        self.is_callback_async = inspect.iscoroutinefunction(
            self.model_callback
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.async_mode = async_mode
        self.language = language
        self.simulated_conversations: List[ConversationalTestCase] = []
        self.simulator_model, self.using_native_model = initialize_model(
            simulator_model
        )
        # `None` is rewritten to the default node so the runtime path is
        # uniform: `_SimulationGraphRunner` always drives user-turn generation.
        # To customize the prompt template, pass
        # `simulation_graph=default_simulation_node(template=MyTemplate)`.
        self.simulation_graph = (
            simulation_graph
            if simulation_graph is not None
            else default_simulation_node()
        )
        self._graph_runner = _SimulationGraphRunner(root=self.simulation_graph)
        self.stopping_controller = SimulationController(
            controller=stopping_controller,
            generate_schema=self.generate_schema,
            a_generate_schema=self.a_generate_schema,
        )

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
            simulator_model=self.simulator_model.get_model_name(),
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
                multimodal = any(
                    [golden.multimodal for golden in conversational_goldens]
                )
                if multimodal:
                    if (
                        not self.simulator_model
                        or not self.simulator_model.supports_multimodal()
                    ):
                        if (
                            self.simulator_model
                            and type(self.simulator_model)
                            in MULTIMODAL_SUPPORTED_MODELS
                        ):
                            raise ValueError(
                                f"The evaluation model {self.simulator_model.name} does not support multimodal evaluations at the moment. Available multi-modal models for the {self.simulator_model.__class__.__name__} provider includes {', '.join(self.simulator_model.__class__.valid_multimodal_models)}."
                            )
                        else:
                            raise ValueError(
                                f"The evaluation model {self.simulator_model.name} does not support multimodal inputs, please use one of the following evaluation models: {', '.join([cls.__name__ for cls in MULTIMODAL_SUPPORTED_MODELS])}"
                            )
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

        multimodal = any(
            [golden.multimodal for golden in conversational_goldens]
        )
        if multimodal:
            if (
                not self.simulator_model
                or not self.simulator_model.supports_multimodal()
            ):
                if (
                    self.simulator_model
                    and type(self.simulator_model)
                    in MULTIMODAL_SUPPORTED_MODELS
                ):
                    raise ValueError(
                        f"The evaluation model {self.simulator_model.name} does not support multimodal evaluations at the moment. Available multi-modal models for the {self.simulator_model.__class__.__name__} provider includes {', '.join(self.simulator_model.__class__.valid_multimodal_models)}."
                    )
                else:
                    raise ValueError(
                        f"The evaluation model {self.simulator_model.name} does not support multimodal inputs, please use one of the following evaluation models: {', '.join([cls.__name__ for cls in MULTIMODAL_SUPPORTED_MODELS])}"
                    )

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
        graph_state: _GraphConversationState = (
            self._graph_runner.new_conversation_state()
        )

        if golden.turns is not None:
            turns.extend(golden.turns)

        while True:
            if simulation_counter >= max_user_simulations:
                update_pbar(progress, pbar_max_user_simluations_id)
                break

            # Stop conversation if needed
            should_stop_simulation = self.stopping_controller.run(
                turns=turns,
                golden=golden,
                index=index,
                thread_id=thread_id,
                simulation_counter=simulation_counter,
                max_user_simulations=max_user_simulations,
                progress=progress,
                pbar_turns_id=pbar_max_user_simluations_id,
            )
            if should_stop_simulation:
                break

            # Generate turn from user (via simulation graph)
            emission_end = False
            if len(turns) > 0 and turns[-1].role == "user":
                user_input = turns[-1].content
            else:
                emission = self._graph_runner.run(
                    self,
                    graph_state,
                    turns,
                    golden,
                    thread_id,
                    self.language,
                )
                emission_end = emission.end
                if emission.turn is None:
                    # max_visits exhausted on entry; end without another turn.
                    update_pbar(progress, pbar_max_user_simluations_id)
                    break
                turns.append(emission.turn)
                user_input = emission.turn.content
                update_pbar(progress, pbar_max_user_simluations_id)
                simulation_counter += 1

            # Generate turn from assistant
            if self.is_callback_async:
                assistant_turn = asyncio.run(
                    self.a_generate_turn_from_callback(
                        user_input,
                        model_callback=self.model_callback,
                        turns=turns,
                        thread_id=thread_id,
                    )
                )
            else:
                assistant_turn = self.generate_turn_from_callback(
                    user_input,
                    model_callback=self.model_callback,
                    turns=turns,
                    thread_id=thread_id,
                )
            turns.append(assistant_turn)

            # Route to the next graph node based on the assistant reply.
            self._graph_runner.advance(
                self, graph_state, assistant_turn.content
            )

            if emission_end:
                break

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
        graph_state: _GraphConversationState = (
            self._graph_runner.new_conversation_state()
        )

        if golden.turns is not None:
            turns.extend(golden.turns)

        while True:
            if simulation_counter >= max_user_simulations:
                update_pbar(progress, pbar_max_user_simluations_id)
                break

            # Stop conversation if needed
            should_stop_simulation = await self.stopping_controller.a_run(
                turns=turns,
                golden=golden,
                index=index if index is not None else 0,
                thread_id=thread_id,
                simulation_counter=simulation_counter,
                max_user_simulations=max_user_simulations,
                progress=progress,
                pbar_turns_id=pbar_max_user_simluations_id,
            )
            if should_stop_simulation:
                break

            # Generate turn from user (via simulation graph)
            emission_end = False
            if len(turns) > 0 and turns[-1].role == "user":
                user_input = turns[-1].content
            else:
                emission = await self._graph_runner.a_run(
                    self,
                    graph_state,
                    turns,
                    golden,
                    thread_id,
                    self.language,
                )
                emission_end = emission.end
                if emission.turn is None:
                    update_pbar(progress, pbar_max_user_simluations_id)
                    break
                turns.append(emission.turn)
                user_input = emission.turn.content
                update_pbar(progress, pbar_max_user_simluations_id)
                simulation_counter += 1

            # Generate turn from assistant
            if self.is_callback_async:
                assistant_turn = await self.a_generate_turn_from_callback(
                    user_input,
                    model_callback=self.model_callback,
                    turns=turns,
                    thread_id=thread_id,
                )
            else:
                assistant_turn = self.generate_turn_from_callback(
                    user_input,
                    model_callback=self.model_callback,
                    turns=turns,
                    thread_id=thread_id,
                )
            turns.append(assistant_turn)

            # Route to the next graph node based on the assistant reply.
            await self._graph_runner.a_advance(
                self, graph_state, assistant_turn.content
            )

            if emission_end:
                break

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

    def generate_first_user_input(
        self,
        golden: ConversationalGolden,
        template: Optional[Type[SimulationTemplate]] = None,
    ):
        tmpl = template or SimulationTemplate
        prompt = tmpl.simulate_first_user_turn(golden, self.language)
        simulated_input: SimulatedInput = self.generate_schema(
            prompt, SimulatedInput
        )
        return simulated_input.simulated_input

    async def a_generate_first_user_input(
        self,
        golden: ConversationalGolden,
        template: Optional[Type[SimulationTemplate]] = None,
    ):
        tmpl = template or SimulationTemplate
        prompt = tmpl.simulate_first_user_turn(golden, self.language)
        simulated_input: SimulatedInput = await self.a_generate_schema(
            prompt, SimulatedInput
        )
        return simulated_input.simulated_input

    def generate_next_user_input(
        self,
        golden: ConversationalGolden,
        turns: List[Turn],
        template: Optional[Type[SimulationTemplate]] = None,
    ):
        tmpl = template or SimulationTemplate
        prompt = tmpl.simulate_user_turn(golden, turns, self.language)
        simulated_input: SimulatedInput = self.generate_schema(
            prompt, SimulatedInput
        )
        return simulated_input.simulated_input

    async def a_generate_next_user_input(
        self,
        golden: ConversationalGolden,
        turns: List[Turn],
        template: Optional[Type[SimulationTemplate]] = None,
    ):
        tmpl = template or SimulationTemplate
        prompt = tmpl.simulate_user_turn(golden, turns, self.language)
        simulated_input: SimulatedInput = await self.a_generate_schema(
            prompt, SimulatedInput
        )
        return simulated_input.simulated_input

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
            if cost is not None:
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
            if cost is not None:
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

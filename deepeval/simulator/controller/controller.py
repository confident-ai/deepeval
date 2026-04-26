import asyncio
import inspect
import json
from typing import Awaitable, Callable, List, Optional

from pydantic import BaseModel
from rich.progress import Progress

from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.dataset import ConversationalGolden
from deepeval.simulator.controller.template import SimulatorControllerTemplate
from deepeval.simulator.controller.types import Context, Decision
from deepeval.simulator.schema import ConversationCompletion
from deepeval.simulator.schema import SimulateHttpResponse
from deepeval.simulator.utils import dump_conversational_golden
from deepeval.test_case import Turn
from deepeval.utils import update_pbar


def proceed() -> Decision:
    return Decision(should_end=False)


def end(reason: Optional[str] = None) -> Decision:
    return Decision(should_end=True, reason=reason)


class SimulationController:
    def __init__(
        self,
        generate_schema: Callable[[str, BaseModel], BaseModel],
        a_generate_schema: Callable[
            [str, BaseModel], Awaitable[BaseModel]
        ],
        controller: Callable,
        run_remote: bool = False,
    ):
        self.controller = controller
        self.template = SimulatorControllerTemplate
        self.run_remote = run_remote
        self.generate_schema = generate_schema
        self.a_generate_schema = a_generate_schema

    def run(
        self,
        turns: List[Turn],
        golden: ConversationalGolden,
        index: int,
        thread_id: str,
        simulation_counter: int,
        max_user_simulations: int,
        progress: Optional[Progress] = None,
        pbar_turns_id: Optional[int] = None,
    ) -> bool:
        if self.controller is expected_outcome_controller:
            return self.controller.run(
                self, turns, golden, progress, pbar_turns_id
            )

        ctx = self._build_context(
            turns=turns,
            golden=golden,
            index=index,
            thread_id=thread_id,
            simulation_counter=simulation_counter,
            max_user_simulations=max_user_simulations,
        )
        decision = self._invoke_controller(ctx)
        if inspect.isawaitable(decision):
            decision = asyncio.run(decision)

        return self._should_end(decision, progress, pbar_turns_id)

    async def a_run(
        self,
        turns: List[Turn],
        golden: ConversationalGolden,
        index: int,
        thread_id: str,
        simulation_counter: int,
        max_user_simulations: int,
        progress: Optional[Progress] = None,
        pbar_turns_id: Optional[int] = None,
    ) -> bool:
        if self.controller is expected_outcome_controller:
            return await self.controller.a_run(
                self, turns, golden, progress, pbar_turns_id
            )

        ctx = self._build_context(
            turns=turns,
            golden=golden,
            index=index,
            thread_id=thread_id,
            simulation_counter=simulation_counter,
            max_user_simulations=max_user_simulations,
        )
        decision = self._invoke_controller(ctx)
        if inspect.isawaitable(decision):
            decision = await decision

        return self._should_end(decision, progress, pbar_turns_id)

    def check_expected_outcome(
        self,
        turns: List[Turn],
        golden: ConversationalGolden,
        progress: Optional[Progress] = None,
        pbar_turns_id: Optional[int] = None,
    ) -> bool:
        if not self.run_remote:
            if golden.expected_outcome is None:
                return False

            conversation_history = json.dumps(
                [t.model_dump() for t in turns],
                indent=4,
                ensure_ascii=False,
            )
            prompt = self.template.check_expected_outcome(
                conversation_history, golden.expected_outcome
            )
            is_complete: ConversationCompletion = self._generate_schema(
                prompt, ConversationCompletion
            )
            if is_complete.is_complete:
                update_pbar(
                    progress,
                    pbar_turns_id,
                    advance_to_end=is_complete.is_complete,
                )
            return is_complete.is_complete

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
            body=dump_conversational_golden(temp_golden),
        )
        res = SimulateHttpResponse(
            user_input=data["userResponse"], complete=data["completed"]
        )
        return res.complete

    async def a_check_expected_outcome(
        self,
        turns: List[Turn],
        golden: ConversationalGolden,
        progress: Optional[Progress] = None,
        pbar_turns_id: Optional[int] = None,
    ) -> bool:
        if not self.run_remote:
            if golden.expected_outcome is None:
                return False

            conversation_history = json.dumps(
                [t.model_dump() for t in turns],
                indent=4,
                ensure_ascii=False,
            )
            prompt = self.template.check_expected_outcome(
                conversation_history, golden.expected_outcome
            )
            is_complete: ConversationCompletion = await self._a_generate_schema(
                prompt, ConversationCompletion
            )
            if is_complete.is_complete:
                update_pbar(
                    progress,
                    pbar_turns_id,
                    advance_to_end=is_complete.is_complete,
                )
            return is_complete.is_complete

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
            body=dump_conversational_golden(temp_golden),
        )
        res = SimulateHttpResponse(
            user_input=data["userResponse"], complete=data["completed"]
        )
        return res.complete

    def _build_context(
        self,
        turns: List[Turn],
        golden: ConversationalGolden,
        index: int,
        thread_id: str,
        simulation_counter: int,
        max_user_simulations: int,
    ) -> Context:
        last_user_turn = next(
            (turn for turn in reversed(turns) if turn.role == "user"), None
        )
        last_assistant_turn = next(
            (turn for turn in reversed(turns) if turn.role == "assistant"),
            None,
        )

        return Context(
            turns=list(turns),
            golden=golden,
            index=index,
            thread_id=thread_id,
            simulated_user_turns=simulation_counter,
            max_user_simulations=max_user_simulations,
            last_user_turn=last_user_turn,
            last_assistant_turn=last_assistant_turn,
        )

    def _invoke_controller(self, ctx: Context):
        controller_kwargs = {
            "turns": ctx.turns,
            "golden": ctx.golden,
            "index": ctx.index,
            "thread_id": ctx.thread_id,
            "simulated_user_turns": ctx.simulated_user_turns,
            "max_user_simulations": ctx.max_user_simulations,
            "last_user_turn": ctx.last_user_turn,
            "last_assistant_turn": ctx.last_assistant_turn,
        }
        supported_args = set(
            inspect.signature(self.controller).parameters.keys()
        )
        return self.controller(
            **{
                key: value
                for key, value in controller_kwargs.items()
                if key in supported_args
            }
        )

    def _normalize_decision(self, decision: Optional[Decision]) -> Decision:
        if not isinstance(decision, Decision):
            return Decision(should_end=False)
        return decision

    def _should_end(
        self,
        decision: Optional[Decision],
        progress: Optional[Progress],
        pbar_turns_id: Optional[int],
    ) -> bool:
        should_end = self._normalize_decision(decision).should_end
        if should_end:
            update_pbar(progress, pbar_turns_id, advance_to_end=True)
        return should_end

    def _generate_schema(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate_schema(prompt, schema)

    async def _a_generate_schema(
        self, prompt: str, schema: BaseModel
    ) -> BaseModel:
        return await self.a_generate_schema(prompt, schema)


class _ExpectedOutcomeController:
    def run(
        self,
        simulation_controller: SimulationController,
        turns: List[Turn],
        golden: ConversationalGolden,
        progress: Optional[Progress] = None,
        pbar_turns_id: Optional[int] = None,
    ) -> bool:
        return simulation_controller.check_expected_outcome(
            turns, golden, progress, pbar_turns_id
        )

    async def a_run(
        self,
        simulation_controller: SimulationController,
        turns: List[Turn],
        golden: ConversationalGolden,
        progress: Optional[Progress] = None,
        pbar_turns_id: Optional[int] = None,
    ) -> bool:
        return await simulation_controller.a_check_expected_outcome(
            turns, golden, progress, pbar_turns_id
        )


expected_outcome_controller = _ExpectedOutcomeController()


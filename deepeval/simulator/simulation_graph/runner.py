from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from deepeval.dataset import ConversationalGolden
from deepeval.simulator.simulation_graph.node import SimulationNode
from deepeval.simulator.simulation_graph.template import SimulationGraphTemplate
from deepeval.simulator.schema import EdgeChoice
from deepeval.test_case import Turn

if TYPE_CHECKING:
    from deepeval.simulator.conversation_simulator import (
        ConversationSimulator,
    )


@dataclass
class TurnEmission:
    """Per-step result from the decision-graph runner.

    - `turn=None, end=True`: `max_visits` was already at its cap on entry,
      so no user turn is emitted and the simulation ends immediately.
    - `turn=<Turn>, end=True`: the current node is `terminal=True`; emit one
      last user turn, the assistant replies, then the simulation ends.
    - `turn=<Turn>, end=False`: normal step; continue.
    """

    turn: Optional[Turn]
    end: bool


@dataclass
class _GraphConversationState:
    """Per-conversation runtime state for the graph runner."""

    current: SimulationNode
    visits: Dict[int, int] = field(default_factory=dict)


class _SimulationGraphRunner:
    """Drives a `SimulationNode` graph during a single
    `ConversationSimulator.simulate(...)` call.

    A fresh `_GraphConversationState` is created per conversation so visit
    counts and the current node don't leak across goldens.
    """

    def __init__(self, root: SimulationNode):
        if not isinstance(root, SimulationNode):
            raise TypeError(
                "simulation_graph must be a SimulationNode (the "
                "root of the graph)."
            )
        self.root = root

    def new_conversation_state(self) -> _GraphConversationState:
        return _GraphConversationState(current=self.root, visits={})

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def run(
        self,
        simulator: "ConversationSimulator",
        state: _GraphConversationState,
        turns: List[Turn],
        golden: ConversationalGolden,
        thread_id: str,
        language: str,
    ) -> TurnEmission:
        node = state.current
        visits = state.visits.get(id(node), 0)
        if node.max_visits is not None and visits >= node.max_visits:
            return TurnEmission(turn=None, end=True)

        result = self._invoke_action(
            node,
            simulator=simulator,
            turns=turns,
            golden=golden,
            thread_id=thread_id,
            language=language,
            async_mode=False,
        )
        if inspect.isawaitable(result):
            # Action is async but we're in sync mode: run it via asyncio,
            # matching the existing `is_callback_async` shim used by
            # `model_callback`.
            result = asyncio.run(result)

        turn = _normalize_user_turn(result, node)
        state.visits[id(node)] = visits + 1

        return TurnEmission(turn=turn, end=bool(node.terminal))

    def advance(
        self,
        simulator: "ConversationSimulator",
        state: _GraphConversationState,
        assistant_reply: str,
    ) -> None:
        node = state.current
        if not node.edges:
            return  # No edges -> stay on current node (no LLM call).
        choices = [when for _, when in node.edges]
        prompt = SimulationGraphTemplate.classify_edge(assistant_reply, choices)
        choice: EdgeChoice = simulator.generate_schema(prompt, EdgeChoice)
        next_node = _resolve_choice(node, choice)
        if next_node is not None:
            state.current = next_node

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def a_run(
        self,
        simulator: "ConversationSimulator",
        state: _GraphConversationState,
        turns: List[Turn],
        golden: ConversationalGolden,
        thread_id: str,
        language: str,
    ) -> TurnEmission:
        node = state.current
        visits = state.visits.get(id(node), 0)
        if node.max_visits is not None and visits >= node.max_visits:
            return TurnEmission(turn=None, end=True)

        result = self._invoke_action(
            node,
            simulator=simulator,
            turns=turns,
            golden=golden,
            thread_id=thread_id,
            language=language,
            async_mode=True,
        )
        if inspect.isawaitable(result):
            result = await result

        turn = _normalize_user_turn(result, node)
        state.visits[id(node)] = visits + 1

        return TurnEmission(turn=turn, end=bool(node.terminal))

    async def a_advance(
        self,
        simulator: "ConversationSimulator",
        state: _GraphConversationState,
        assistant_reply: str,
    ) -> None:
        node = state.current
        if not node.edges:
            return
        choices = [when for _, when in node.edges]
        prompt = SimulationGraphTemplate.classify_edge(assistant_reply, choices)
        choice: EdgeChoice = await simulator.a_generate_schema(
            prompt, EdgeChoice
        )
        next_node = _resolve_choice(node, choice)
        if next_node is not None:
            state.current = next_node

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _invoke_action(
        node: SimulationNode,
        *,
        simulator: "ConversationSimulator",
        turns: List[Turn],
        golden: ConversationalGolden,
        thread_id: str,
        language: str,
        async_mode: bool,
    ) -> Any:
        last_user_turn = next(
            (t for t in reversed(turns) if t.role == "user"), None
        )
        last_assistant_turn = next(
            (t for t in reversed(turns) if t.role == "assistant"), None
        )
        candidate_kwargs = {
            "simulator": simulator,
            "turns": turns,
            "golden": golden,
            "last_assistant_turn": last_assistant_turn,
            "last_user_turn": last_user_turn,
            "thread_id": thread_id,
            "language": language,
        }
        try:
            sig = inspect.signature(node.action)
        except (TypeError, ValueError):
            # Builtin or C-implemented callable; pass nothing.
            return node.action()
        accepts_var_keyword = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if accepts_var_keyword:
            return node.action(**candidate_kwargs)
        supported = set(sig.parameters.keys())
        kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}
        return node.action(**kwargs)


def _normalize_user_turn(
    result: Union[str, Turn], node: SimulationNode
) -> Turn:
    if isinstance(result, str):
        return Turn(role="user", content=result)
    if isinstance(result, Turn):
        if result.role != "user":
            raise TypeError(
                f"SimulationNode {node.name!r} returned a Turn with "
                f"role={result.role!r}; must be 'user'."
            )
        return result
    raise TypeError(
        f"SimulationNode {node.name!r} action must return str or "
        f"Turn(role='user', ...); got {type(result).__name__}."
    )


def _resolve_choice(
    node: SimulationNode, choice: EdgeChoice
) -> Optional[SimulationNode]:
    index = getattr(choice, "index", None)
    if index is None:
        return None
    if not (1 <= index <= len(node.edges)):
        return None
    return node.edges[index - 1][0]

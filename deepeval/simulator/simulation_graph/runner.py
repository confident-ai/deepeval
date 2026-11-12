from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

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
    events: List[Any] = field(default_factory=list)


@dataclass
class _GraphConversationState:
    """Per-conversation runtime state for the graph runner."""

    current: SimulationNode
    visits: Dict[int, int] = field(default_factory=dict)
    pending: Optional[SimulationNode] = None
    path: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    ending: Optional[str] = None
    off_script: bool = False


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
        self.shortest_path = _shortest_path(root)

    def summary(self, state: _GraphConversationState) -> Dict[str, Any]:
        """What a conversation did on the graph, for the test case metadata."""
        if not (self.root.edges or self.root.terminal):
            return {}
        failed = bool(state.failed) or state.off_script
        return {
            "Simulation path": list(state.path),
            "Simulation ending": state.ending,
            "Simulation failed": failed,
            "Failed nodes": list(state.failed),
            "Shortest path": self.shortest_path,
        }

    def _enter(
        self, state: _GraphConversationState
    ) -> Optional[SimulationNode]:
        if state.off_script:
            return None
        node = state.pending or state.current
        state.pending = None
        visits = state.visits.get(id(node), 0)
        if node.max_visits is not None and visits >= node.max_visits:
            return None
        return node

    @staticmethod
    def _emitted(state: _GraphConversationState, node: SimulationNode) -> bool:
        state.visits[id(node)] = state.visits.get(id(node), 0) + 1
        state.path.append(node.name)
        if node.fails:
            state.failed.append(node.name)
        if node.terminal and node is state.current:
            state.ending = node.ending or node.name
            return True
        return False

    def _choices(
        self, state: _GraphConversationState
    ) -> List[
        Tuple[str, str, Optional[SimulationNode], Optional[SimulationNode]]
    ]:
        """Every way out of the current node: its edges, then the reactions."""
        said = lambda node: bool(state.visits.get(id(node)))
        choices = []
        for child, when in state.current.edges:
            said_once = child.group is not None and said(child)
            # A line already said in another place, out of turn, is not
            # said again.
            twin = getattr(child, "same_line_as", None)
            if not said_once and not (twin is not None and said(twin)):
                choices.append(("edge", when, child, None))
        for reaction, when, then in self.root.reactions:
            twin = getattr(reaction, "same_line_as", None)
            if twin is not None and (said(twin) or said(reaction)):
                continue
            choices.append(("reaction", when, reaction, then))
        return choices

    def _follow(
        self,
        state: _GraphConversationState,
        choices: List[
            Tuple[str, str, Optional[SimulationNode], Optional[SimulationNode]]
        ],
        choice: EdgeChoice,
    ) -> None:
        index = getattr(choice, "index", None)
        if index is None or not (1 <= index <= len(choices)):
            fallback = self.root.fallback
            if fallback is None:
                return
            used = state.visits.get(id(fallback), 0)
            if fallback.max_visits is not None and used >= fallback.max_visits:
                state.off_script = True
                state.ending = "off_script"
            else:
                state.pending = fallback
            return
        kind, _, node, then = choices[index - 1]
        if kind == "edge":
            state.current = node
            return
        used = state.visits.get(id(node), 0)
        if node.max_visits is None or used < node.max_visits:
            state.pending = node
        elif then is not None:
            state.current = then

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
        node = self._enter(state)
        if node is None:
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
        if node.events:
            turn.metadata = {
                **(turn.metadata or {}),
                "events": [event.record() for event in node.events],
            }
        return TurnEmission(
            turn=turn, end=self._emitted(state, node), events=node.events
        )

    def advance(
        self,
        simulator: "ConversationSimulator",
        state: _GraphConversationState,
        assistant_reply: str,
        turns: Optional[List[Turn]] = None,
    ) -> None:
        choices = self._choices(state)
        if not choices:
            return  # No edges -> stay on current node (no LLM call).
        prompt = SimulationGraphTemplate.classify_edge(
            assistant_reply,
            [when for _, when, _, _ in choices],
            _recent(turns),
        )
        choice: EdgeChoice = simulator.generate_schema(prompt, EdgeChoice)
        self._follow(state, choices, choice)

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
        node = self._enter(state)
        if node is None:
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
        if node.events:
            turn.metadata = {
                **(turn.metadata or {}),
                "events": [event.record() for event in node.events],
            }
        return TurnEmission(
            turn=turn, end=self._emitted(state, node), events=node.events
        )

    async def a_advance(
        self,
        simulator: "ConversationSimulator",
        state: _GraphConversationState,
        assistant_reply: str,
        turns: Optional[List[Turn]] = None,
    ) -> None:
        choices = self._choices(state)
        if not choices:
            return
        prompt = SimulationGraphTemplate.classify_edge(
            assistant_reply,
            [when for _, when, _, _ in choices],
            _recent(turns),
        )
        choice: EdgeChoice = await simulator.a_generate_schema(
            prompt, EdgeChoice
        )
        self._follow(state, choices, choice)

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


def _recent(turns: Optional[List[Turn]], keep: int = 8) -> str:
    """The last few turns before the assistant's latest reply."""
    earlier = list(turns or [])
    if earlier and earlier[-1].role == "assistant":
        earlier = earlier[:-1]
    earlier = earlier[-keep:]
    return "\n".join(
        f'{"User" if turn.role == "user" else "Assistant"}: {turn.content}'
        for turn in earlier
    )


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


def _shortest_path(root: SimulationNode) -> Optional[int]:
    """Fewest user turns from the root to an ending that is not a failure.

    Nodes in a group are all said, so entering a group costs its size.
    """
    best: Dict[int, int] = {id(root): 1}
    queue: List[Tuple[int, SimulationNode]] = [(1, root)]
    shortest: Optional[int] = None
    while queue:
        queue.sort(key=lambda item: item[0])
        cost, node = queue.pop(0)
        if cost > best.get(id(node), cost):
            continue
        if node.terminal:
            if node.ending is not None and not node.fails:
                shortest = cost if shortest is None else min(shortest, cost)
            continue
        for child, _ in node.edges:
            if child.fails:
                continue
            if child.group is not None:
                if child.group is node.group:
                    continue
                step = len(child.group.members)
            else:
                step = 1
            if cost + step < best.get(id(child), cost + step + 1):
                best[id(child)] = cost + step
                queue.append((cost + step, child))
    return shortest

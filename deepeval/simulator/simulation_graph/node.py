import inspect
from typing import Callable, List, Optional, Tuple, Union

from deepeval.test_case import Turn


class SimulationNode:
    """A single node in a `simulation_graph`.

    Each node represents one state of the simulated user. When the runner
    visits a node it invokes the node's `action(...)` to produce the next
    user turn, then (if the node has outgoing edges) asks the simulator's
    LLM to classify which edge to follow based on the assistant's reply.

    Args:
        action: Callable returning either a `str` (wrapped into
            `Turn(role="user", content=str)`) or a `Turn` with role=="user".
            The action receives any subset of the kwargs
            `{simulator, turns, golden, last_assistant_turn, last_user_turn,
            thread_id, language}`; only declared parameters are passed.
            May be sync or async.
        terminal: If True, the simulation ends immediately after this node
            emits a user turn and the assistant replies.
        max_visits: Optional cap. The node will be emitted at most this many
            times. On the (max_visits+1)-th entry attempt the runner emits
            nothing and ends the simulation. `None` means no cap.
        name: Optional debug name; defaults to `action.__name__`.
    """

    def __init__(
        self,
        action: Callable[..., Union[str, Turn]],
        *,
        terminal: bool = False,
        max_visits: Optional[int] = None,
        name: Optional[str] = None,
    ):
        if not callable(action):
            raise TypeError(
                "`action` must be a callable returning str or Turn."
            )
        if max_visits is not None and max_visits <= 0:
            raise ValueError("`max_visits` must be a positive integer.")

        self.action = action
        self.is_action_async = inspect.iscoroutinefunction(action)
        self.terminal = bool(terminal)
        self.max_visits = max_visits
        self.name = name or getattr(action, "__name__", "simulation_node")
        self.edges: List[Tuple["SimulationNode", str]] = []

    def add_node(self, child: "SimulationNode", when: str) -> "SimulationNode":
        """Append an outgoing edge to `child`, taken when the LLM router
        classifies the assistant's reply as matching `when` (a natural-language
        description).

        Returns the child so calls can be chained.
        """
        if not isinstance(child, SimulationNode):
            raise TypeError("`child` must be a SimulationNode instance.")
        if not isinstance(when, str) or not when.strip():
            raise ValueError("`when=` must be a non-empty string description.")
        self.edges.append((child, when))
        return child

    def __repr__(self) -> str:
        return (
            f"SimulationNode(name={self.name!r}, terminal={self.terminal}, "
            f"max_visits={self.max_visits}, edges={len(self.edges)})"
        )

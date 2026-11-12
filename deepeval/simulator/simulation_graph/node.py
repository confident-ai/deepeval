import inspect
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

from deepeval.test_case import Turn

if TYPE_CHECKING:
    from deepeval.simulator.events import VoiceEvent


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
        ending: Names the outcome this node ends the conversation on, such as
            `"refunded"`. A node with an ending is terminal.
        fails: If True, reaching this node marks the conversation as failed.
            On a node with an `ending` it makes that a failure ending.
        events: Voice events from `deepeval.simulator.events` that happen on
            this node's turn, such as a pause in the middle of the line.
    """

    def __init__(
        self,
        action: Callable[..., Union[str, Turn]],
        *,
        terminal: bool = False,
        max_visits: Optional[int] = None,
        name: Optional[str] = None,
        ending: Optional[str] = None,
        fails: bool = False,
        events: Optional[List["VoiceEvent"]] = None,
    ):
        if not callable(action):
            raise TypeError(
                "`action` must be a callable returning str or Turn."
            )
        if max_visits is not None and max_visits <= 0:
            raise ValueError("`max_visits` must be a positive integer.")

        self.action = action
        self.is_action_async = inspect.iscoroutinefunction(action)
        self.terminal = bool(terminal) or ending is not None
        self.max_visits = max_visits
        self.name = name or getattr(action, "__name__", "simulation_node")
        self.ending = ending
        self.fails = bool(fails)
        self.events: List["VoiceEvent"] = list(events or [])
        self.edges: List[Tuple["SimulationNode", str]] = []
        self.group: Optional["SimulationGroup"] = None
        self.reactions: List[
            Tuple["SimulationNode", str, Optional["SimulationNode"]]
        ] = []
        self.fallback: Optional["SimulationNode"] = None

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

    def add_group(self, group: "SimulationGroup") -> "SimulationGroup":
        """Append an edge to every node in `group`, each taken when the
        assistant's reply matches that node's own `when`.

        Returns the group so calls can be chained.
        """
        if not isinstance(group, SimulationGroup):
            raise TypeError("`group` must be a SimulationGroup instance.")
        for member, when in group.members:
            self.add_node(member, when)
        return group

    def add_reaction(
        self,
        reaction: "SimulationNode",
        when: str,
        then: Optional["SimulationNode"] = None,
    ) -> "SimulationNode":
        """Register a line the simulated user says from any node in the graph
        when the assistant's reply matches `when`, after which the
        conversation carries on from the node it interrupted.

        Call this on the root node. `reaction.max_visits` caps how many times
        it is said; once used up, the conversation moves to `then` instead.

        Returns the reaction so calls can be chained.
        """
        if not isinstance(reaction, SimulationNode):
            raise TypeError("`reaction` must be a SimulationNode instance.")
        if not isinstance(when, str) or not when.strip():
            raise ValueError("`when=` must be a non-empty string description.")
        if then is not None and not isinstance(then, SimulationNode):
            raise TypeError("`then` must be a SimulationNode instance.")
        self.reactions.append((reaction, when, then))
        return reaction

    def off_script(self, fallback: "SimulationNode") -> "SimulationNode":
        """Set the line the simulated user says when the assistant's reply
        matches no edge and no reaction, such as "Sorry, can you say that
        again?".

        Call this on the root node. Once `fallback.max_visits` is used up the
        conversation ends with the ending `"off_script"`.

        Returns the fallback so calls can be chained.
        """
        if not isinstance(fallback, SimulationNode):
            raise TypeError("`fallback` must be a SimulationNode instance.")
        self.fallback = fallback
        return fallback

    def __repr__(self) -> str:
        return (
            f"SimulationNode(name={self.name!r}, terminal={self.terminal}, "
            f"max_visits={self.max_visits}, edges={len(self.edges)})"
        )


class SimulationGroup:
    """Nodes the assistant may ask for in any order, such as a name, a date
    of birth and a reference number.

    Args:
        members: `(node, when)` pairs. Each node is said once, when the
            assistant's reply matches its `when`.
    """

    def __init__(self, members: List[Tuple[SimulationNode, str]]):
        if len(members) < 2:
            raise ValueError("A SimulationGroup needs at least two nodes.")
        for member, when in members:
            if not isinstance(member, SimulationNode):
                raise TypeError(
                    "Group members must be SimulationNode instances."
                )
            if not isinstance(when, str) or not when.strip():
                raise ValueError("Each group member needs a non-empty `when`.")
        self.members = list(members)
        for member, _ in self.members:
            member.group = self
            for other, when in self.members:
                if other is not member:
                    member.add_node(other, when)

    def add_node(self, child: SimulationNode, when: str) -> SimulationNode:
        """Append an edge from every node in the group to `child`."""
        for member, _ in self.members:
            member.add_node(child, when)
        return child

    def add_group(self, group: "SimulationGroup") -> "SimulationGroup":
        """Append an edge from every node in this group to every node in
        `group`."""
        for member, _ in self.members:
            member.add_group(group)
        return group

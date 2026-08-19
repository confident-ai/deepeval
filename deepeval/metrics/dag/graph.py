from typing import Dict, List, Tuple, Union

from deepeval.metrics.dag import (
    BaseNode,
    NonBinaryJudgementNode,
    BinaryJudgementNode,
    VerdictNode,
)
from deepeval.metrics.conversational_dag import (
    ConversationalBaseNode,
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
    ConversationalVerdictNode,
)
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.metrics import BaseMetric, BaseConversationalMetric

Node = Union[BaseNode, ConversationalBaseNode]

_VERDICT_TYPES = (VerdictNode, ConversationalVerdictNode)
_JUDGEMENT_TYPES = (
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
)


def validate_root_nodes(root_nodes: List[Node]):
    if not all(isinstance(node, type(root_nodes[0])) for node in root_nodes):
        raise ValueError("You cannot mix multi and single turn nodes")
    return True


def _edges_of(node: Node) -> List[Node]:
    edges: List[Node] = []
    children = getattr(node, "children", None)
    if children:
        edges.extend(children)
    child = getattr(node, "child", None)
    if child is not None and isinstance(
        child, (BaseNode, ConversationalBaseNode)
    ):
        edges.append(child)
    return edges


def _collapse_exclusive_verdict_edges(
    indegree: Dict[Node, int],
    parents: Dict[Node, List[Node]],
    root_nodes: List[Node],
) -> None:
    """Count mutually exclusive verdict edges as a single incoming edge.

    Exactly one verdict child of a judgement node ever fires. When several of
    them are direct parents of the same node, counting one edge each leaves
    that node permanently short of indegree zero, because the verdicts that
    lost can never arrive. Collectively they deliver at most one arrival, so
    they contribute one edge.

    Only a verdict owned by exactly one judgement and not itself a declared
    root qualifies. A verdict shared between two judgements can fire through
    either of them, and a root verdict is visited unconditionally, so in both
    cases the edge stays counted on its own -- collapsing it could drop the
    count below the number of arrivals and let the node run more than once.
    """
    for node, node_parents in parents.items():
        exclusive: Dict[Node, int] = {}
        for parent in node_parents:
            if not isinstance(parent, _VERDICT_TYPES):
                continue
            if any(parent is root for root in root_nodes):
                # A declared root is visited unconditionally, so its edge
                # always arrives and must keep its own count.
                continue
            owners = parents.get(parent) or ()
            if len(owners) != 1 or not isinstance(owners[0], _JUDGEMENT_TYPES):
                continue
            exclusive[owners[0]] = exclusive.get(owners[0], 0) + 1
        for count in exclusive.values():
            if count > 1:
                indegree[node] -= count - 1


class DeepAcyclicGraph:
    multiturn: bool

    def __init__(self, root_nodes: List[Node]):
        validate_root_nodes(root_nodes)
        self.multiturn = isinstance(root_nodes[0], ConversationalBaseNode)

        judgement_types = (
            (
                ConversationalBinaryJudgementNode,
                ConversationalNonBinaryJudgementNode,
            )
            if self.multiturn
            else (BinaryJudgementNode, NonBinaryJudgementNode)
        )
        if len(root_nodes) > 1 and any(
            isinstance(n, judgement_types) for n in root_nodes
        ):
            raise ValueError(
                "You cannot provide more than one root node when a "
                "BinaryJudgementNode or NonBinaryJudgementNode is a root."
            )

        self.root_nodes = root_nodes
        self.indegree, self.parents = self._build_graph()

        for root in self.root_nodes:
            if self.indegree[root] > 0:
                raise ValueError(
                    "You cannot declare a node as a root node when it is "
                    "also reachable from another root node."
                )

    def _build_graph(self) -> Tuple[Dict[Node, int], Dict[Node, List[Node]]]:
        indegree: Dict[Node, int] = {}
        parents: Dict[Node, List[Node]] = {}
        visited: set = set()
        stack: set = set()

        def visit(node: Node):
            if node in stack:
                raise ValueError("Cycle detected in DAG graph.")
            if node in visited:
                return
            visited.add(node)
            node._validate()
            stack.add(node)
            indegree.setdefault(node, 0)
            for child in _edges_of(node):
                indegree[child] = indegree.get(child, 0) + 1
                parents.setdefault(child, []).append(node)
                visit(child)
            stack.discard(node)

        for root in self.root_nodes:
            visit(root)
        _collapse_exclusive_verdict_edges(indegree, parents, self.root_nodes)
        return indegree, parents

    def _execute(
        self,
        metric: Union[BaseMetric, BaseConversationalMetric],
        test_case: Union[LLMTestCase, ConversationalTestCase],
    ) -> None:
        from deepeval.metrics.dag.runner import _DeepAcyclicGraphRunner

        _DeepAcyclicGraphRunner(self).run(metric, test_case)

    async def _a_execute(
        self,
        metric: Union[BaseMetric, BaseConversationalMetric],
        test_case: Union[LLMTestCase, ConversationalTestCase],
    ) -> None:
        from deepeval.metrics.dag.runner import _DeepAcyclicGraphRunner

        await _DeepAcyclicGraphRunner(self).a_run(metric, test_case)

    def to_dict(self) -> dict:
        """Serialize this DAG to a JSON-friendly dict (structure only)."""
        from deepeval.metrics.dag.serialization import dag_to_dict

        return dag_to_dict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize this DAG to a JSON string (structure only)."""
        from deepeval.metrics.dag.serialization import dag_to_json

        return dag_to_json(self, indent=indent)

    @classmethod
    def from_dict(
        cls, data: dict, multiturn: bool = False
    ) -> "DeepAcyclicGraph":
        """Re-create a DAG from a dict produced by ``to_dict``.

        ``multiturn`` selects between single-turn and conversational node
        classes; the JSON document itself is mode-agnostic.
        """
        from deepeval.metrics.dag.serialization import dag_from_dict

        return dag_from_dict(data, multiturn=multiturn)

    @classmethod
    def from_json(cls, s: str, multiturn: bool = False) -> "DeepAcyclicGraph":
        """Re-create a DAG from a JSON string produced by ``to_json``."""
        from deepeval.metrics.dag.serialization import dag_from_json

        return dag_from_json(s, multiturn=multiturn)

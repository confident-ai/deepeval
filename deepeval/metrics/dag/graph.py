from typing import Dict, List, Tuple, Union

from deepeval.metrics.dag import (
    BaseNode,
    NonBinaryJudgementNode,
    BinaryJudgementNode,
)
from deepeval.metrics.conversational_dag import (
    ConversationalBaseNode,
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
)
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.metrics import BaseMetric, BaseConversationalMetric

Node = Union[BaseNode, ConversationalBaseNode]


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

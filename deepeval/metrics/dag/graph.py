import asyncio
from typing import List, Union

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


def validate_root_nodes(
    root_nodes: Union[List[BaseNode], List[ConversationalBaseNode]],
):
    # see if all root nodes are of the same type, more verbose error message, actualy we should say we cannot mix multi and single turn nodes
    if not all(isinstance(node, type(root_nodes[0])) for node in root_nodes):
        raise ValueError("You cannot mix multi and single turn nodes")
    return True


class DeepAcyclicGraph:
    multiturn: bool

    def __init__(
        self,
        root_nodes: Union[List[BaseNode], List[ConversationalBaseNode]],
    ):
        validate_root_nodes(root_nodes)
        self.multiturn = isinstance(root_nodes[0], ConversationalBaseNode)

        if not self.multiturn:
            for root_node in root_nodes:
                if isinstance(root_node, NonBinaryJudgementNode) or isinstance(
                    root_node, BinaryJudgementNode
                ):
                    if len(root_nodes) > 1:
                        raise ValueError(
                            "You cannot provide more than one root node when using 'BinaryJudgementNode' or 'NonBinaryJudgementNode' in root_nodes."
                        )
        else:
            for root_node in root_nodes:
                if isinstance(
                    root_node, ConversationalNonBinaryJudgementNode
                ) or isinstance(root_node, ConversationalBinaryJudgementNode):
                    if len(root_nodes) > 1:
                        raise ValueError(
                            "You cannot provide more than one root node when using 'ConversationalBinaryJudgementNode' or 'ConversationalNonBinaryJudgementNode' in root_nodes."
                        )
        self.root_nodes = root_nodes

    def _execute(
        self,
        metric: Union[BaseMetric, BaseConversationalMetric],
        test_case: Union[LLMTestCase, ConversationalTestCase],
    ) -> None:
        for root_node in self.root_nodes:
            root_node._execute(metric=metric, test_case=test_case, depth=0)

    async def _a_execute(
        self,
        metric: Union[BaseMetric, BaseConversationalMetric],
        test_case: Union[LLMTestCase, ConversationalTestCase],
    ) -> None:
        await asyncio.gather(
            *(
                root_node._a_execute(
                    metric=metric, test_case=test_case, depth=0
                )
                for root_node in self.root_nodes
            )
        )

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

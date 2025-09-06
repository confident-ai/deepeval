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

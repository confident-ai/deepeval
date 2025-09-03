import asyncio
from typing import List

from deepeval.metrics.conversational_dag import (
    ConversationalBaseNode,
    ConversationalNonBinaryJudgementNode,
    ConversationalBinaryJudgementNode,
)
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics import BaseConversationalMetric


class DeepAcyclicGraph:
    def __init__(
        self,
        root_nodes: List[ConversationalBaseNode],
    ):
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
        metric: BaseConversationalMetric,
        test_case: ConversationalTestCase,
    ) -> None:
        for root_node in self.root_nodes:
            root_node._execute(metric=metric, test_case=test_case, depth=0)

    async def _a_execute(
        self,
        metric: BaseConversationalMetric,
        test_case: ConversationalTestCase,
    ) -> None:
        await asyncio.gather(
            *(
                root_node._a_execute(
                    metric=metric, test_case=test_case, depth=0
                )
                for root_node in self.root_nodes
            )
        )

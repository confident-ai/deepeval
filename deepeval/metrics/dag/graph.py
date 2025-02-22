import asyncio
from typing import List

from deepeval.metrics.dag import (
    BaseNode,
    NonBinaryJudgementNode,
    BinaryJudgementNode,
)
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric


class DeepAcyclicGraph:
    def __init__(
        self,
        root_nodes: List[BaseNode],
    ):
        for root_node in root_nodes:
            if isinstance(root_node, NonBinaryJudgementNode) or isinstance(
                root_node, BinaryJudgementNode
            ):
                if len(root_nodes) > 1:
                    raise ValueError(
                        "You cannot provide more than one root node when using 'BinaryJudgementNode' or 'NonBinaryJudgementNode' in root_nodes."
                    )

        self.root_nodes = root_nodes

    def _execute(self, metric: BaseMetric, test_case: LLMTestCase) -> None:
        for root_node in self.root_nodes:
            root_node._execute(metric=metric, test_case=test_case, depth=0)

    async def _a_execute(
        self,
        metric: BaseMetric,
        test_case: LLMTestCase,
    ) -> None:
        await asyncio.gather(
            *(
                root_node._a_execute(
                    metric=metric, test_case=test_case, depth=0
                )
                for root_node in self.root_nodes
            )
        )

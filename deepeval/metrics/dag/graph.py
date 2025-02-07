import asyncio
from typing import List

from deepeval.metrics.dag import BaseNode
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric


class DeepAcyclicGraph:
    def __init__(
        self,
        root_nodes: List[BaseNode],
    ):
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

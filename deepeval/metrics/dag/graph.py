from typing import Set, List

from deepeval.metrics.dag import BaseNode
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric


class DeepAcyclicGraph:
    def __init__(
        self,
        root_node: BaseNode,
        metric: BaseMetric,
        test_case: LLMTestCase,
    ):
        self.root_node = root_node
        self.metric = metric
        self.test_case = test_case

    def execute(self) -> None:
        self.root_node._execute(metric=self.metric, test_case=self.test_case)

    async def a_execute(self) -> None:
        self.root_node._a_execute(metric=self.metric, test_case=self.test_case)

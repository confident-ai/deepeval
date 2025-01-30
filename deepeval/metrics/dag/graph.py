from deepeval.metrics.dag import BaseNode
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric


class DeepAcyclicGraph:
    def __init__(
        self,
        root_node: BaseNode,
    ):
        self.root_node = root_node

    def _execute(self, metric: BaseMetric, test_case: LLMTestCase) -> None:
        self.root_node._execute(metric=metric, test_case=test_case)

    async def _a_execute(
        self,
        metric: BaseMetric,
        test_case: LLMTestCase,
    ) -> None:
        self.root_node._a_execute(metric=metric, test_case=test_case)

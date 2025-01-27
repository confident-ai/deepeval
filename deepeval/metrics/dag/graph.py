from typing import Set, List

from deepeval.metrics.dag import BaseNode, VerdictNode
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

    def _topological_sort(
        self, node: BaseNode, visited: Set[BaseNode], stack: List[BaseNode]
    ) -> None:
        if node in visited:
            return
        visited.add(node)

        if isinstance(node, VerdictNode):
            if node.child is not None:
                self._topological_sort(node.child, visited, stack)
        elif hasattr(node, "children"):
            for child in node.children:
                self._topological_sort(child, visited, stack)

        stack.append(node)

    def evaluate(self) -> None:
        visited: Set[BaseNode] = set()
        stack: List[BaseNode] = []

        self._topological_sort(self.root_node, visited, stack)

        while stack:
            node = stack.pop()
            if isinstance(node, BaseNode):
                node._execute(metric=self.metric, test_case=self.test_case)

    async def a_evaluate(self) -> None:
        visited: Set[BaseNode] = set()
        stack: List[BaseNode] = []

        self._topological_sort(self.root_node, visited, stack)

        while stack:
            node = stack.pop()
            if isinstance(node, BaseNode):
                await node._a_execute(
                    metric=self.metric, test_case=self.test_case
                )

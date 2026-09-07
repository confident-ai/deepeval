from __future__ import annotations

import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from deepeval.metrics.dag.nodes import (
    BaseNode,
    VerdictNode,
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    construct_node_verbose_log as _st_verbose_log,
)
from deepeval.metrics.conversational_dag.nodes import (
    ConversationalBaseNode,
    ConversationalVerdictNode,
    ConversationalTaskNode,
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
    construct_node_verbose_log as _mt_verbose_log,
)
from deepeval.metrics.base_metric import BaseMetric, BaseConversationalMetric
from deepeval.test_case import LLMTestCase, ConversationalTestCase

if TYPE_CHECKING:
    from deepeval.metrics.dag.graph import DeepAcyclicGraph

Node = Union[BaseNode, ConversationalBaseNode]
Metric = Union[BaseMetric, BaseConversationalMetric]
TestCase = Union[LLMTestCase, ConversationalTestCase]

_NODE = (BaseNode, ConversationalBaseNode)
_VERDICT = (VerdictNode, ConversationalVerdictNode)
_TASK = (TaskNode, ConversationalTaskNode)
_JUDGEMENT = (
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
)


class _DeepAcyclicGraphRunner:
    def __init__(self, graph: DeepAcyclicGraph):
        self.graph = graph
        self.remaining: Dict[Node, int] = dict(graph.indegree)
        self.outputs: Dict[Node, Any] = {}
        self.verdicts: Dict[Node, Any] = {}
        self.depth: Dict[Node, int] = {}
        self._verbose_log = (
            _mt_verbose_log if graph.multiturn else _st_verbose_log
        )
        self._order = self._build_order()
        self._score_events: List[Tuple[int, float, Optional[str]]] = []

    def _build_order(self) -> Dict[Node, int]:
        """Depth-first pre-order index for every node, roots in declaration order.

        Used to pick a winner when more than one verdict assigns a score. This
        is the order the synchronous traversal already visits nodes in, so
        resolving on the highest index reproduces today's sync result exactly
        while bringing the async result into line with it.
        """
        order: Dict[Node, int] = {}
        seen: set = set()

        def walk(node: Node) -> None:
            if node in seen:
                return
            seen.add(node)
            order[node] = len(order)
            children = list(getattr(node, "children", None) or ())
            child = getattr(node, "child", None)
            if child is not None and isinstance(child, _NODE):
                children.append(child)
            for c in children:
                walk(c)

        for root in self.graph.root_nodes:
            walk(root)
        return order

    def _record_score(
        self, node: Node, score: float, reason: Optional[str]
    ) -> None:
        """Record a scoring verdict so a winner can be chosen deterministically.

        The metric is still written on arrival, because
        `VerdictNode._generate_reason` builds its prompt from `metric.score`.
        What changes is that the last arrival no longer decides the result:
        `_settle` re-applies the highest-ordered event once the traversal
        finishes. Previously the surviving value was whichever branch completed
        last, and `asyncio.gather` interleaves completion, so the same graph
        with the same judgements scored differently under `async_mode=True`.
        """
        self._score_events.append(
            (self._order.get(node, len(self._order)), score, reason)
        )

    def _settle(self, metric: Metric) -> None:
        if not self._score_events:
            return
        _, score, reason = max(self._score_events, key=lambda e: e[0])
        metric.score = score
        if metric.include_reason and reason is not None:
            metric.reason = reason

    def _check_remaining(self, node: Node) -> bool:
        self.remaining[node] -= 1
        return self.remaining[node] <= 0

    def _verdict_matches(self, node: Node) -> bool:
        parents = self.graph.parents.get(node)
        parent = parents[0] if parents else None
        if isinstance(parent, _JUDGEMENT):
            return self.verdicts[parent].verdict == node.verdict
        return True

    def _store_result(self, node: Node, result: Any) -> None:
        if isinstance(node, _TASK):
            self.outputs[node] = result
        else:
            self.verdicts[node] = result

    def _apply_child_metric(
        self, node: Node, child_metric: Metric, metric: Metric, depth: int
    ) -> None:
        metric._verbose_steps.append(
            self._verbose_log(node, depth, child_metric)
        )
        metric.score = child_metric.score
        if metric.include_reason:
            metric.reason = child_metric.reason
        self._record_score(
            node,
            child_metric.score,
            child_metric.reason if metric.include_reason else None,
        )
        metric._accrue_cost(child_metric.evaluation_cost)
        metric._accrue_tokens(
            child_metric.input_tokens, child_metric.output_tokens
        )

    # ------------------------------------------------------------------ sync
    def run(self, metric: Metric, test_case: TestCase) -> None:
        for root in self.graph.root_nodes:
            self._visit(root, metric, test_case, 0)
        self._settle(metric)

    def _visit(
        self, node: Node, metric: Metric, test_case: TestCase, depth: int
    ) -> None:
        if isinstance(node, _VERDICT):
            self._visit_verdict(node, metric, test_case, depth)
            return
        self.depth[node] = max(0, self.depth.get(node, 0), depth)
        if not self._check_remaining(node):
            return
        result = node._execute(
            metric, test_case, self.graph.parents.get(node), self.outputs
        )
        self._store_result(node, result)
        node_depth = self.depth[node]
        metric._verbose_steps.append(
            self._verbose_log(
                node,
                node_depth,
                output=self.outputs.get(node),
                verdict=self.verdicts.get(node),
            )
        )
        for child in node.children:
            self._visit(child, metric, test_case, node_depth + 1)

    def _visit_verdict(
        self, node: Node, metric: Metric, test_case: TestCase, depth: int
    ) -> None:
        if not self._check_remaining(node):
            return
        if not self._verdict_matches(node):
            return
        child = node.child
        if child is None:
            metric._verbose_steps.append(self._verbose_log(node, depth))
            metric.score = node.score / 10
            reason = (
                node._generate_reason(metric=metric)
                if metric.include_reason
                else None
            )
            if reason is not None:
                metric.reason = reason
            self._record_score(node, node.score / 10, reason)
        elif isinstance(child, _NODE):
            self._visit(child, metric, test_case, depth)
        else:
            copied = node._run_child_metric(metric, test_case)
            self._apply_child_metric(node, copied, metric, depth)

    async def a_run(self, metric: Metric, test_case: TestCase) -> None:
        await asyncio.gather(
            *(
                self._a_visit(root, metric, test_case, 0)
                for root in self.graph.root_nodes
            )
        )
        self._settle(metric)

    async def _a_visit(
        self, node: Node, metric: Metric, test_case: TestCase, depth: int
    ) -> None:
        if isinstance(node, _VERDICT):
            await self._a_visit_verdict(node, metric, test_case, depth)
            return
        self.depth[node] = max(0, self.depth.get(node, 0), depth)
        if not self._check_remaining(node):
            return
        result = await node._a_execute(
            metric, test_case, self.graph.parents.get(node), self.outputs
        )
        self._store_result(node, result)
        node_depth = self.depth[node]
        metric._verbose_steps.append(
            self._verbose_log(
                node,
                node_depth,
                output=self.outputs.get(node),
                verdict=self.verdicts.get(node),
            )
        )
        await asyncio.gather(
            *(
                self._a_visit(child, metric, test_case, node_depth + 1)
                for child in node.children
            )
        )

    async def _a_visit_verdict(
        self, node: Node, metric: Metric, test_case: TestCase, depth: int
    ) -> None:
        if not self._check_remaining(node):
            return
        if not self._verdict_matches(node):
            return
        child = node.child
        if child is None:
            metric._verbose_steps.append(self._verbose_log(node, depth))
            metric.score = node.score / 10
            reason = (
                await node._a_generate_reason(metric=metric)
                if metric.include_reason
                else None
            )
            if reason is not None:
                metric.reason = reason
            self._record_score(node, node.score / 10, reason)
        elif isinstance(child, _NODE):
            await self._a_visit(child, metric, test_case, depth)
        else:
            copied = await node._a_run_child_metric(metric, test_case)
            self._apply_child_metric(node, copied, metric, depth)

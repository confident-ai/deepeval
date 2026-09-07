from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

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


@dataclass
class _ScoreCandidate:
    """A scoring verdict that fired during a run.

    ``order`` is the node's position in a pre-order walk of the graph, so it
    reflects how the DAG was declared and never how the run happened to
    interleave.
    """

    order: int
    score: float
    node: Node
    child_metric: Optional[Metric] = None


class _DeepAcyclicGraphRunner:
    def __init__(self, graph: DeepAcyclicGraph):
        self.graph = graph
        self.remaining: Dict[Node, int] = dict(graph.indegree)
        self.outputs: Dict[Node, Any] = {}
        self.verdicts: Dict[Node, Any] = {}
        self.depth: Dict[Node, int] = {}
        self.score_candidates: List[_ScoreCandidate] = []
        self._order = self._build_order(graph)
        self._verbose_log = (
            _mt_verbose_log if graph.multiturn else _st_verbose_log
        )

    @staticmethod
    def _build_order(graph: DeepAcyclicGraph) -> Dict[Node, int]:
        """Number every node by a pre-order walk of the declared graph."""
        order: Dict[Node, int] = {}

        def visit(node: Node) -> None:
            if node in order:
                return
            order[node] = len(order)
            for child in getattr(node, "children", None) or ():
                visit(child)
            child = getattr(node, "child", None)
            if isinstance(child, _NODE):
                visit(child)

        for root in graph.root_nodes:
            visit(root)
        return order

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
        # Cost and tokens were really spent, so accrue them for every branch
        # that ran. Only the score and reason are deferred to `_finalize`.
        metric._accrue_cost(child_metric.evaluation_cost)
        metric._accrue_tokens(
            child_metric.input_tokens, child_metric.output_tokens
        )
        self._record_score(node, child_metric.score, child_metric)

    def _record_score(
        self,
        node: Node,
        score: float,
        child_metric: Optional[Metric] = None,
    ) -> None:
        self.score_candidates.append(
            _ScoreCandidate(
                order=self._order.get(node, len(self._order)),
                score=score,
                node=node,
                child_metric=child_metric,
            )
        )

    def _winning_candidate(self) -> Optional[_ScoreCandidate]:
        """Pick the metric's score from every scoring verdict that fired.

        A DAG can have more than one scoring verdict reachable at once. Writing
        each one straight to ``metric.score`` made the result depend on which
        branch happened to finish last, which differs between the depth-first
        sync walk and the interleaved async one. Choosing here instead makes
        both agree.

        The lowest score wins: a metric that measures several things should
        report its weakest, in the same spirit as a failing check. ``order``
        breaks ties by declaration order so the choice is fully determined by
        how the DAG was written.
        """
        if not self.score_candidates:
            return None
        return min(self.score_candidates, key=lambda c: (c.score, c.order))

    def _finalize(self, metric: Metric) -> Optional[_ScoreCandidate]:
        """Commit the winning score. Returns the winner so the caller can
        produce its reason, which needs `metric.score` to already be set."""
        winner = self._winning_candidate()
        if winner is not None:
            metric.score = winner.score
        return winner

    # ------------------------------------------------------------------ sync
    def run(self, metric: Metric, test_case: TestCase) -> None:
        for root in self.graph.root_nodes:
            self._visit(root, metric, test_case, 0)
        winner = self._finalize(metric)
        if winner is not None and metric.include_reason:
            if winner.child_metric is not None:
                metric.reason = winner.child_metric.reason
            else:
                metric.reason = winner.node._generate_reason(metric=metric)

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
            self._record_score(node, node.score / 10)
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
        winner = self._finalize(metric)
        if winner is not None and metric.include_reason:
            if winner.child_metric is not None:
                metric.reason = winner.child_metric.reason
            else:
                metric.reason = await winner.node._a_generate_reason(
                    metric=metric
                )

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
            self._record_score(node, node.score / 10)
        elif isinstance(child, _NODE):
            await self._a_visit(child, metric, test_case, depth)
        else:
            copied = await node._a_run_child_metric(metric, test_case)
            self._apply_child_metric(node, copied, metric, depth)

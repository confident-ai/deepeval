from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

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
        # Every fired scoring verdict/child metric contributes a (score, reason)
        # pair; the final metric score is resolved deterministically at the end
        # instead of "last write wins" (which depended on node order and on the
        # sync vs async traversal).
        self.scores: List[Tuple[float, Optional[str]]] = []
        self._verbose_log = (
            _mt_verbose_log if graph.multiturn else _st_verbose_log
        )

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
        reason = child_metric.reason if metric.include_reason else None
        self.scores.append((child_metric.score, reason))
        metric._accrue_cost(child_metric.evaluation_cost)
        metric._accrue_tokens(
            child_metric.input_tokens, child_metric.output_tokens
        )

    def _finalize_score(self, metric: Metric) -> None:
        """Resolve the final metric score deterministically.

        Every fired scoring verdict/child metric contributes a (score, reason)
        pair. The score is picked with a conservative ``min`` aggregation so
        the result no longer depends on node declaration order or on whether
        the graph ran in sync or async mode. A DAG with a single scoring
        path is unaffected.
        """
        scored = [(s, r) for s, r in self.scores if s is not None]
        if not scored:
            return
        score, reason = min(scored, key=lambda item: item[0])
        metric.score = score
        if metric.include_reason and reason is not None:
            metric.reason = reason

    # ------------------------------------------------------------------ sync
    def run(self, metric: Metric, test_case: TestCase) -> None:
        for root in self.graph.root_nodes:
            self._visit(root, metric, test_case, 0)
        self._finalize_score(metric)

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
            self.scores.append((metric.score, reason))
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
        self._finalize_score(metric)

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
            self.scores.append((metric.score, reason))
        elif isinstance(child, _NODE):
            await self._a_visit(child, metric, test_case, depth)
        else:
            copied = await node._a_run_child_metric(metric, test_case)
            self._apply_child_metric(node, copied, metric, depth)

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass, field
from pydantic import create_model

from deepeval.metrics.dag.schema import (
    MetricScoreReason,
    BinaryJudgementVerdict,
    TaskNodeOutput,
)
from deepeval.metrics.base_metric import BaseMetric, PromptMixin
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.metrics.g_eval.utils import G_EVAL_PARAMS
from deepeval.metrics.utils import (
    copy_metrics,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall
from deepeval.utils import prettify_list


class BaseNode(PromptMixin):
    def _validate(self) -> None:
        pass

    def _resolve_text(
        self,
        test_case: LLMTestCase,
        parents: Optional[List[BaseNode]],
        outputs: Dict[BaseNode, Any],
        sep: str = "\n\n",
    ) -> str:
        text = ""
        if parents is not None:
            for parent in parents:
                if isinstance(parent, TaskNode):
                    text += f"{parent.output_label}:\n{outputs[parent]}{sep}"
        if self.evaluation_params is not None:
            for param in self.evaluation_params:
                value = getattr(test_case, param.value)
                if isinstance(value, ToolCall):
                    value = repr(value)
                text += f"{G_EVAL_PARAMS[param]}:\n{value}\n"
        return text


@dataclass
class VerdictNode(BaseNode):
    verdict: Union[str, bool]
    score: Optional[int] = None
    child: Optional[Union[BaseNode, GEval, BaseMetric]] = None

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        if self.score is not None and self.child is not None:
            raise ValueError(
                "A VerdictNode can have either a 'score' or a 'child', but not both."
            )
        if self.score is None and self.child is None:
            raise ValueError(
                "A VerdictNode must have either a 'score' or a 'child'."
            )
        if self.score is not None and not (0 <= self.score <= 10):
            raise ValueError("The score must be between 0 and 10, inclusive.")

    def _generate_reason(self, metric: BaseMetric) -> str:
        prompt = self._get_prompt(
            "generate_reason",
            template_class="VerdictNode",
            verbose_steps=metric._verbose_steps,
            score=metric.score,
            name=metric.__name__,
        )
        return generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=MetricScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_generate_reason(self, metric: BaseMetric) -> str:
        prompt = self._get_prompt(
            "generate_reason",
            template_class="VerdictNode",
            verbose_steps=metric._verbose_steps,
            score=metric.score,
            name=metric.__name__,
        )
        return await a_generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=MetricScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    def _build_child_metric(self, metric: BaseMetric):
        if isinstance(self.child, GEval):
            args = {
                "name": self.child.name,
                "evaluation_params": self.child.evaluation_params,
                "model": metric.model,
                "verbose_mode": False,
            }
            if self.child.criteria:
                args["criteria"] = self.child.criteria
            else:
                args["evaluation_steps"] = self.child.evaluation_steps
            return GEval(**args)
        copied = copy_metrics([self.child])[0]
        copied.verbose_mode = False
        return copied

    def _run_child_metric(self, metric: BaseMetric, test_case: LLMTestCase):
        copied = self._build_child_metric(metric)
        copied.measure(
            test_case=test_case,
            _show_indicator=False,
            _log_metric_to_confident=False,
        )
        return copied

    async def _a_run_child_metric(
        self, metric: BaseMetric, test_case: LLMTestCase
    ):
        copied = self._build_child_metric(metric)
        await copied.a_measure(
            test_case=test_case,
            _show_indicator=False,
            _log_metric_to_confident=False,
        )
        return copied


@dataclass
class TaskNode(BaseNode):
    instructions: str
    output_label: str
    children: List[BaseNode] = field(default_factory=list)
    evaluation_params: Optional[List[SingleTurnParams]] = None
    label: Optional[str] = None

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        if self.children:
            self._validate()

    def add_node(self, child: BaseNode) -> BaseNode:
        self.children.append(child)
        return child

    def _validate(self) -> None:
        for child in self.children:
            if not isinstance(child, BaseNode):
                raise TypeError(
                    "A TaskNode's children must be BaseNode instances."
                )
            if isinstance(child, VerdictNode):
                raise ValueError(
                    "A TaskNode must not have a VerdictNode as one of their 'children'."
                )

    def _execute(
        self,
        metric: BaseMetric,
        test_case: LLMTestCase,
        parents: Optional[List[BaseNode]],
        outputs: Dict[BaseNode, Any],
    ) -> Any:
        if self.evaluation_params is None and parents is None:
            raise ValueError(
                "A TaskNode must have either a 'evaluation_params' or parent node(s)."
            )
        prompt = self._get_prompt(
            "generate_task_output",
            template_class="TaskNode",
            instructions=self.instructions,
            text=self._resolve_text(test_case, parents, outputs),
        )
        return generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=TaskNodeOutput,
            extract_schema=lambda s: s.output,
            extract_json=lambda data: data["output"],
        )

    async def _a_execute(
        self,
        metric: BaseMetric,
        test_case: LLMTestCase,
        parents: Optional[List[BaseNode]],
        outputs: Dict[BaseNode, Any],
    ) -> Any:
        if self.evaluation_params is None and parents is None:
            raise ValueError(
                "A TaskNode must have either a 'evaluation_params' or parent node(s)."
            )
        prompt = self._get_prompt(
            "generate_task_output",
            template_class="TaskNode",
            instructions=self.instructions,
            text=self._resolve_text(test_case, parents, outputs),
        )
        return await a_generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=TaskNodeOutput,
            extract_schema=lambda s: s.output,
            extract_json=lambda data: data["output"],
        )


@dataclass
class BinaryJudgementNode(BaseNode):
    criteria: str
    children: List[VerdictNode] = field(default_factory=list)
    evaluation_params: Optional[List[SingleTurnParams]] = None
    label: Optional[str] = None

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        if self.children:
            self._validate()

    def add_verdict(
        self,
        verdict: bool,
        *,
        score: Optional[int] = None,
        then: Optional[Union[BaseNode, GEval, BaseMetric]] = None,
    ) -> VerdictNode:
        node = VerdictNode(verdict=verdict, score=score, child=then)
        self.children.append(node)
        return node

    def _validate(self) -> None:
        if len(self.children) != 2:
            raise ValueError(
                "BinaryJudgementNode must have exactly 2 children."
            )
        for child in self.children:
            if not isinstance(child, VerdictNode):
                raise TypeError("All children must be of type VerdictNode.")
            if not isinstance(child.verdict, bool):
                raise ValueError(
                    "All children BinaryJudgementNode must have a boolean verdict."
                )
        verdicts = [child.verdict for child in self.children]
        if verdicts.count(True) != 1 or verdicts.count(False) != 1:
            raise ValueError(
                "BinaryJudgementNode must have one True and one False VerdictNode child."
            )

    def _execute(
        self,
        metric: BaseMetric,
        test_case: LLMTestCase,
        parents: Optional[List[BaseNode]],
        outputs: Dict[BaseNode, Any],
    ) -> BinaryJudgementVerdict:
        prompt = self._get_prompt(
            "generate_binary_verdict",
            template_class="BinaryJudgement",
            criteria=self.criteria,
            text=self._resolve_text(test_case, parents, outputs),
        )
        return generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=BinaryJudgementVerdict,
            extract_schema=lambda s: s,
            extract_json=lambda data: BinaryJudgementVerdict(**data),
        )

    async def _a_execute(
        self,
        metric: BaseMetric,
        test_case: LLMTestCase,
        parents: Optional[List[BaseNode]],
        outputs: Dict[BaseNode, Any],
    ) -> BinaryJudgementVerdict:
        prompt = self._get_prompt(
            "generate_binary_verdict",
            template_class="BinaryJudgement",
            criteria=self.criteria,
            text=self._resolve_text(test_case, parents, outputs),
        )
        return await a_generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=BinaryJudgementVerdict,
            extract_schema=lambda s: s,
            extract_json=lambda data: BinaryJudgementVerdict(**data),
        )


@dataclass
class NonBinaryJudgementNode(BaseNode):
    criteria: str
    children: List[VerdictNode] = field(default_factory=list)
    evaluation_params: Optional[List[SingleTurnParams]] = None
    label: Optional[str] = None

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        if self.children:
            self._validate()

    def add_verdict(
        self,
        verdict: str,
        *,
        score: Optional[int] = None,
        then: Optional[Union[BaseNode, GEval, BaseMetric]] = None,
    ) -> VerdictNode:
        node = VerdictNode(verdict=verdict, score=score, child=then)
        self.children.append(node)
        return node

    def _validate(self) -> None:
        if not self.children:
            raise ValueError(
                "NonBinaryJudgementNode must have at least one child."
            )
        verdicts_set = set()
        for child in self.children:
            if not isinstance(child, VerdictNode):
                raise TypeError("All children must be of type VerdictNode.")
            if not isinstance(child.verdict, str):
                raise ValueError(
                    "The verdict attribute of all NonBinaryJudgementNode children must be a string."
                )
            if child.verdict in verdicts_set:
                raise ValueError(
                    f"Duplicate verdict found: {child.verdict} in children of NonBinaryJudgementNode."
                )
            verdicts_set.add(child.verdict)

        self._verdict_options = list(verdicts_set)
        self._verdict_schema = create_model(
            "NonBinaryJudgementVerdict",
            verdict=(Literal[tuple(self._verdict_options)], ...),
            reason=(str, ...),
        )

    def _execute(
        self,
        metric: BaseMetric,
        test_case: LLMTestCase,
        parents: Optional[List[BaseNode]],
        outputs: Dict[BaseNode, Any],
    ) -> Any:
        prompt = self._get_prompt(
            "generate_non_binary_verdict",
            template_class="NonBinaryJudgement",
            criteria=self.criteria,
            text=self._resolve_text(test_case, parents, outputs, sep="\n"),
            options=self._verdict_options,
        )
        return generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=self._verdict_schema,
            extract_schema=lambda s: s,
            extract_json=lambda data: self._verdict_schema(**data),
        )

    async def _a_execute(
        self,
        metric: BaseMetric,
        test_case: LLMTestCase,
        parents: Optional[List[BaseNode]],
        outputs: Dict[BaseNode, Any],
    ) -> Any:
        prompt = self._get_prompt(
            "generate_non_binary_verdict",
            template_class="NonBinaryJudgement",
            criteria=self.criteria,
            text=self._resolve_text(test_case, parents, outputs, sep="\n"),
            options=self._verdict_options,
        )
        return await a_generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=self._verdict_schema,
            extract_schema=lambda s: s,
            extract_json=lambda data: self._verdict_schema(**data),
        )


def construct_node_verbose_log(
    node: BaseNode,
    depth: int,
    node_metric: Optional[Union[GEval, BaseMetric]] = None,
    *,
    output: Optional[Any] = None,
    verdict: Optional[Any] = None,
) -> str:
    if isinstance(
        node, (BinaryJudgementNode, NonBinaryJudgementNode, TaskNode)
    ):
        label = node.label if node.label else "None"

    if isinstance(node, (BinaryJudgementNode, NonBinaryJudgementNode)):
        is_binary_node = isinstance(node, BinaryJudgementNode)
        node_type = (
            "BinaryJudgementNode"
            if is_binary_node
            else "NonBinaryJudgementNode"
        )
        underscore_multiple = 34 if is_binary_node else 37
        star_multiple = 48 if is_binary_node else 53
        return (
            f"{'_' * underscore_multiple}\n"
            f"| {node_type} | Level == {depth} |\n"
            f"{'*' * star_multiple}\n"
            f"Label: {label}\n\n"
            "Criteria:\n"
            f"{node.criteria}\n\n"
            f"Verdict: {verdict.verdict}\n"
            f"Reason: {verdict.reason}\n"
        )
    elif isinstance(node, TaskNode):
        return (
            "______________________\n"
            f"| TaskNode | Level == {depth} |\n"
            "*******************************\n"
            f"Label: {label}\n\n"
            "Instructions:\n"
            f"{node.instructions}\n\n"
            f"{node.output_label}:\n{output}\n"
        )
    elif isinstance(node, VerdictNode):
        type = None
        if node_metric:
            if isinstance(node_metric, (GEval, BaseMetric)):
                type = f"{node_metric.__name__} Metric"
        else:
            type = "Deterministic"

        verbose_log = (
            "________________________\n"
            f"| VerdictNode | Level == {depth} |\n"
            "**********************************\n"
            f"Verdict: {node.verdict}\n"
            f"Type: {type}"
        )
        if isinstance(node_metric, GEval):
            verbose_log += f"\n\nCriteria:\n{node_metric.criteria}\n"
            verbose_log += f"Evaluation Steps:\n{prettify_list(node_metric.evaluation_steps)}"
        elif isinstance(node_metric, BaseMetric):
            verbose_log += f"\n\n{node_metric.verbose_logs}"

        return verbose_log

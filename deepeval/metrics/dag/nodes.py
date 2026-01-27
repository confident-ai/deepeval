from typing import Optional, List, Union, Literal
from dataclasses import dataclass
from pydantic import create_model
import asyncio
import ast
import inspect

from deepeval.metrics.dag.schema import (
    MetricScoreReason,
    BinaryJudgementVerdict,
    NonBinaryJudgementVerdict,
    TaskNodeOutput,
)
from deepeval.metrics.dag.templates import (
    VerdictNodeTemplate,
    TaskNodeTemplate,
    BinaryJudgementTemplate,
    NonBinaryJudgementTemplate,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.metrics.g_eval.utils import G_EVAL_PARAMS
from deepeval.metrics.utils import (
    copy_metrics,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall
from deepeval.utils import prettify_list


class BaseNode:
    _indegree: int = 0
    _depth: int = 0

    def set_parent(self, parent: "BaseNode"):
        if hasattr(self, "_parent"):
            self._parent = parent
        elif hasattr(self, "_parents"):
            if self._parents is None:
                self._parents = []
            self._parents.append(parent)

    def _execute(self, metric: BaseMetric, test_case: LLMTestCase, depth: int):
        raise NotImplementedError(
            "This node type must implement the _execute method."
        )

    async def _a_execute(
        self, metric: BaseMetric, test_case: LLMTestCase, depth: int
    ):
        raise NotImplementedError(
            "This node type must implement the _a_execute method."
        )


def increment_indegree(node: BaseNode):
    node._indegree += 1


def decrement_indegree(node: BaseNode):
    node._indegree -= 1


@dataclass
class VerdictNode(BaseNode):
    verdict: Union[str, bool]
    score: Optional[int] = None
    child: Optional[Union[BaseNode, GEval, BaseMetric]] = None
    _parent: Optional[BaseNode] = None

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        # Ensure either `score` or `g_eval` is set, but not both
        if self.score is not None and self.child is not None:
            raise ValueError(
                "A VerdictNode can have either a 'score' or a 'child', but not both."
            )
        if self.score is None and self.child is None:
            raise ValueError(
                "A VerdictNode must have either a 'score' or a 'child'."
            )

        if self.score is not None:
            if not (0 <= self.score <= 10):
                raise ValueError(
                    "The score must be between 0 and 10, inclusive."
                )

    def _execute(self, metric: BaseMetric, test_case: LLMTestCase, depth: int):
        decrement_indegree(self)
        if self._indegree > 0:
            return

        if isinstance(self._parent, NonBinaryJudgementNode) or isinstance(
            self._parent, BinaryJudgementNode
        ):
            if self._parent._verdict.verdict != self.verdict:
                return

        if self.child is not None:
            if isinstance(self.child, GEval):
                g_eval_args = {
                    "name": self.child.name,
                    "evaluation_params": self.child.evaluation_params,
                    "model": metric.model,
                    "verbose_mode": False,
                }
                if self.child.criteria:
                    g_eval_args["criteria"] = self.child.criteria
                else:
                    g_eval_args["evaluation_steps"] = (
                        self.child.evaluation_steps
                    )
                copied_g_eval = GEval(**g_eval_args)

                copied_g_eval.measure(
                    test_case=test_case,
                    _show_indicator=False,
                    _log_metric_to_confident=False,
                )
                metric._verbose_steps.append(
                    construct_node_verbose_log(self, depth, copied_g_eval)
                )
                metric.score = copied_g_eval.score
                if metric.include_reason:
                    metric.reason = copied_g_eval.reason
            elif isinstance(self.child, BaseMetric):
                copied_metric: BaseMetric = copy_metrics([self.child])[0]
                copied_metric.verbose_mode = False

                copied_metric.measure(
                    test_case=test_case,
                    _show_indicator=False,
                    _log_metric_to_confident=False,
                )
                metric._verbose_steps.append(
                    construct_node_verbose_log(self, depth, copied_metric)
                )
                metric.score = copied_metric.score
                if metric.include_reason:
                    metric.reason = copied_metric.reason
            else:
                self.child._execute(
                    metric=metric, test_case=test_case, depth=depth
                )
        else:
            metric._verbose_steps.append(
                construct_node_verbose_log(self, depth)
            )
            metric.score = self.score / 10
            if metric.include_reason:
                metric.reason = self._generate_reason(metric=metric)

    async def _a_execute(
        self, metric: BaseMetric, test_case: LLMTestCase, depth: int
    ):
        decrement_indegree(self)
        if self._indegree > 0:
            return

        if isinstance(self._parent, NonBinaryJudgementNode) or isinstance(
            self._parent, BinaryJudgementNode
        ):
            if self._parent._verdict.verdict != self.verdict:
                return

        if self.child is not None:
            if isinstance(self.child, GEval):
                g_eval_args = {
                    "name": self.child.name,
                    "evaluation_params": self.child.evaluation_params,
                    "model": metric.model,
                    "verbose_mode": False,
                }
                if self.child.criteria:
                    g_eval_args["criteria"] = self.child.criteria
                else:
                    g_eval_args["evaluation_steps"] = (
                        self.child.evaluation_steps
                    )
                copied_g_eval = GEval(**g_eval_args)

                await copied_g_eval.a_measure(
                    test_case=test_case,
                    _show_indicator=False,
                    _log_metric_to_confident=False,
                )
                metric._verbose_steps.append(
                    construct_node_verbose_log(self, depth, copied_g_eval)
                )
                metric.score = copied_g_eval.score
                if metric.include_reason:
                    metric.reason = copied_g_eval.reason

            elif isinstance(self.child, BaseMetric):
                copied_metric: BaseMetric = copy_metrics([self.child])[0]
                copied_metric.verbose_mode = False

                await copied_metric.a_measure(
                    test_case=test_case,
                    _show_indicator=False,
                    _log_metric_to_confident=False,
                )
                metric._verbose_steps.append(
                    construct_node_verbose_log(self, depth, copied_metric)
                )
                metric.score = copied_metric.score
                if metric.include_reason:
                    metric.reason = copied_metric.reason
            else:
                await self.child._a_execute(
                    metric=metric, test_case=test_case, depth=depth
                )
        else:
            metric._verbose_steps.append(
                construct_node_verbose_log(self, depth)
            )
            metric.score = self.score / 10
            if metric.include_reason:
                metric.reason = await self._a_generate_reason(metric=metric)

    def _generate_reason(self, metric: BaseMetric):
        prompt = VerdictNodeTemplate.generate_reason(
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

    async def _a_generate_reason(self, metric: BaseMetric):
        prompt = VerdictNodeTemplate.generate_reason(
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


@dataclass
class TaskNode(BaseNode):
    instructions: str
    output_label: str
    children: List[BaseNode]
    evaluation_params: List[LLMTestCaseParams] = None
    label: Optional[str] = None
    _verbose_logs: Optional[str] = None
    _output: Optional[str] = None
    _parents: Optional[List[BaseNode]] = None

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        for child in self.children:
            if isinstance(child, VerdictNode):
                raise ValueError(
                    "A TaskNode must not have a VerdictNode as one of their 'children'."
                )
            if isinstance(child, LoopNode) and not len(self.children) == 1:
                raise ValueError(
                    "A TaskNode can only have one child if it cointains a 'LoopNode'."
                )

        # print("-------")
        for child in self.children:
            child.set_parent(self)
            increment_indegree(child)
        #     print("task node", child.__class__.__name__, id(child), child._indegree)
        # print("-------")

    def _execute(self, metric: BaseMetric, test_case: LLMTestCase, depth: int):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            return

        if self.evaluation_params is None and self._parents is None:
            raise ValueError(
                "A TaskNode must have either a 'evaluation_params' or parent node(s)."
            )

        text = """"""
        if self._parents is not None:
            for parent in self._parents:
                if isinstance(parent, TaskNode) or isinstance(parent, LoopNode):
                    text += f"{parent.output_label}:\n{parent._output}\n\n"

        if self.evaluation_params is not None:
            for param in self.evaluation_params:
                value = getattr(test_case, param.value)
                if isinstance(value, ToolCall):
                    value = repr(value)
                text += f"{G_EVAL_PARAMS[param]}:\n{value}\n"

        prompt = TaskNodeTemplate.generate_task_output(
            instructions=self.instructions,
            text=text,
        )
        self._output = generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=TaskNodeOutput,
            extract_schema=lambda s: s.output,
            extract_json=lambda data: data["output"],
        )

        metric._verbose_steps.append(
            construct_node_verbose_log(self, self._depth)
        )
        for children in self.children:
            children._execute(
                metric=metric, test_case=test_case, depth=self._depth + 1
            )

    async def _a_execute(
        self, metric: BaseMetric, test_case: LLMTestCase, depth: int
    ):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            return

        if self.evaluation_params is None and self._parents is None:
            raise ValueError(
                "A TaskNode must have either a 'evaluation_params' or parent node(s)."
            )

        text = """"""
        if self._parents is not None:
            for parent in self._parents:
                if isinstance(parent, TaskNode) or isinstance(parent, LoopNode):
                    text += f"{parent.output_label}:\n{parent._output}\n\n"

        if self.evaluation_params is not None:
            for param in self.evaluation_params:
                value = getattr(test_case, param.value)
                if isinstance(value, ToolCall):
                    value = repr(value)
                text += f"{G_EVAL_PARAMS[param]}:\n{value}\n"

        prompt = TaskNodeTemplate.generate_task_output(
            instructions=self.instructions,
            text=text,
        )

        self._output = await a_generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=TaskNodeOutput,
            extract_schema=lambda s: s.output,
            extract_json=lambda data: data["output"],
        )

        metric._verbose_steps.append(
            construct_node_verbose_log(self, self._depth)
        )
        await asyncio.gather(
            *(
                child._a_execute(
                    metric=metric, test_case=test_case, depth=self._depth + 1
                )
                for child in self.children
            )
        )


@dataclass
class LoopNode(BaseNode):
    output_label: str
    children: List[TaskNode]
    instructions: Optional[str] = None
    loop_function: Optional[callable] = None
    _parent: Optional[BaseNode] = None

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        for child in self.children:
            if not isinstance(child, TaskNode) and len(self.children) > 1:
                raise ValueError(
                    "A LoopNode must only have a single TaskNode as its 'child'."
                )

        if self.loop_function and self.instructions:
            raise ValueError(
                "A 'LoopNode' can have either 'loop_function' or 'instructions' not both."
            )

        if self.loop_function and not callable(self.loop_function):
            raise ValueError(
                "The 'loop_function' must be a callable function with a single argument taking your TaskNode's expected python output"
            )

        self.child = self.children[0]
        self.child.set_parent(self)
        increment_indegree(self.child)

    def _execute(self, metric: BaseMetric, test_case: LLMTestCase, depth: int):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            raise ValueError("Too many indegrees for a 'LoopNode'")

        if self._parent is None:
            raise ValueError("A LoopNode must have a single parent TaskNode.")

        task_node_output = self._parent._output
        try:
            task_node_output = ast.literal_eval(task_node_output)
        except:
            raise ValueError(
                f"The 'TaskNode' has produced an output that is not a valid python expression: {task_node_output}"
            )

        if self.loop_function:
            self._output = str(self.loop_function(task_node_output))
        else:
            if not isinstance(task_node_output, list):
                raise ValueError(
                    "The 'TaskNode' did not generate a python list. Task Node Output: ",
                    task_node_output,
                )

            final_list = []

            for item in task_node_output:
                prompt = TaskNodeTemplate.generate_task_output(
                    instructions=self.instructions,
                    text=str(item),
                )

                if metric.using_native_model:
                    res, cost = metric.model.generate(
                        prompt, schema=TaskNodeOutput
                    )
                    metric.evaluation_cost += cost
                    final_list.append(res.output)
                else:
                    try:
                        res: TaskNodeOutput = metric.model.generate(
                            prompt, schema=TaskNodeOutput
                        )
                        final_list.append(res.output)
                    except TypeError:
                        res = metric.model.generate(prompt)
                        data = trimAndLoadJson(res, self)
                        final_list.append(TaskNodeOutput(**data).output)

            self._output = str(final_list)

        metric._verbose_steps.append(
            construct_node_verbose_log(self, self._depth)
        )

        self.child._execute(
            metric=metric, test_case=test_case, depth=self._depth + 1
        )

    async def _a_execute(
        self, metric: BaseMetric, test_case: LLMTestCase, depth: int
    ):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            raise ValueError("Too many indegrees for a 'LoopNode'")

        if self._parent is None:
            raise ValueError("A LoopNode must have a single parent TaskNode.")

        task_node_output = self._parent._output
        try:
            task_node_output = ast.literal_eval(task_node_output)
        except:
            raise ValueError(
                f"The 'TaskNode' has produced an output that is not a valid python expression: {task_node_output}"
            )

        if self.loop_function:
            self._output = str(self.loop_function(task_node_output))
        else:
            if not isinstance(task_node_output, list):
                raise ValueError(
                    "The 'TaskNode' did not generate a python list. Task Node Output: ",
                    task_node_output,
                )

            async def generate_result(item):
                prompt = TaskNodeTemplate.generate_task_output(
                    instructions=self.instructions,
                    text=str(item),
                )

                if metric.using_native_model:
                    res, cost = await metric.model.a_generate(
                        prompt, schema=TaskNodeOutput
                    )
                    metric.evaluation_cost += cost
                    return res.output
                else:
                    try:
                        res: TaskNodeOutput = await metric.model.a_generate(
                            prompt, schema=TaskNodeOutput
                        )
                        return res.output
                    except TypeError:
                        res = await metric.model.a_generate(prompt)
                        data = trimAndLoadJson(res, self)
                        return TaskNodeOutput(**data).output

            tasks = [generate_result(item) for item in task_node_output]

            final_list = await asyncio.gather(*tasks)

            self._output = str(final_list)

        metric._verbose_steps.append(
            construct_node_verbose_log(self, self._depth)
        )

        await self.child._a_execute(
            metric=metric, test_case=test_case, depth=self._depth + 1
        )


@dataclass
class BinaryJudgementNode(BaseNode):
    criteria: str
    children: List[VerdictNode]
    evaluation_params: Optional[List[LLMTestCaseParams]] = None
    label: Optional[str] = None
    _verbose_logs: Optional[str] = None
    _verdict: Optional[BinaryJudgementVerdict] = None
    _parents: Optional[List[BaseNode]] = None

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        if len(self.children) != 2:
            raise ValueError(
                "BinaryJudgementNode must have exactly 2 children."
            )

        # Check if all children are ClassificationResultNode and their classifications are boolean
        for child in self.children:
            if not isinstance(child, VerdictNode):
                raise TypeError("All children must be of type VerdictNode.")

            if not isinstance(child.verdict, bool):
                raise ValueError(
                    "All children BinaryJudgementNode must have a boolean verdict."
                )

        # Check if there is one True and one False classification
        verdicts = [child.verdict for child in self.children]
        if verdicts.count(True) != 1 or verdicts.count(False) != 1:
            raise ValueError(
                "BinaryJudgementNode must have one True and one False VerdictNode child."
            )

        # print("-------")
        for child in self.children:
            child.set_parent(self)
            increment_indegree(child)
            if child.child is not None and isinstance(child.child, BaseNode):
                increment_indegree(child.child)
        #         print("binary node nested", child.child.__class__.__name__, id(child.child), child.child._indegree)
        #     print("binary node", child.__class__.__name__, id(child), child._indegree)
        # print("-------")

    def _execute(self, metric: BaseMetric, test_case: LLMTestCase, depth: int):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            return

        text = """"""
        if self._parents is not None:
            for parent in self._parents:
                if isinstance(parent, TaskNode):
                    text += f"{parent.output_label}:\n{parent._output}\n\n"

        if self.evaluation_params is not None:
            for param in self.evaluation_params:
                value = getattr(test_case, param.value)
                if isinstance(value, ToolCall):
                    value = repr(value)
                text += f"{G_EVAL_PARAMS[param]}:\n{value}\n"

        prompt = BinaryJudgementTemplate.generate_binary_verdict(
            criteria=self.criteria,
            text=text,
        )
        self._verdict = generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=BinaryJudgementVerdict,
            extract_schema=lambda s: s,
            extract_json=lambda data: BinaryJudgementVerdict(**data),
        )
        metric._verbose_steps.append(
            construct_node_verbose_log(self, self._depth)
        )
        for children in self.children:
            children._execute(
                metric=metric, test_case=test_case, depth=self._depth + 1
            )

    async def _a_execute(
        self, metric: BaseMetric, test_case: LLMTestCase, depth: int
    ):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            return

        text = """"""
        if self._parents is not None:
            for parent in self._parents:
                if isinstance(parent, TaskNode):
                    text += f"{parent.output_label}:\n{parent._output}\n\n"

        if self.evaluation_params is not None:
            for param in self.evaluation_params:
                value = getattr(test_case, param.value)
                if isinstance(value, ToolCall):
                    value = repr(value)
                text += f"{G_EVAL_PARAMS[param]}:\n{value}\n"

        prompt = BinaryJudgementTemplate.generate_binary_verdict(
            criteria=self.criteria,
            text=text,
        )
        self._verdict = await a_generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=BinaryJudgementVerdict,
            extract_schema=lambda s: s,
            extract_json=lambda data: BinaryJudgementVerdict(**data),
        )

        metric._verbose_steps.append(
            construct_node_verbose_log(self, self._depth)
        )
        await asyncio.gather(
            *(
                child._a_execute(
                    metric=metric, test_case=test_case, depth=self._depth + 1
                )
                for child in self.children
            )
        )


@dataclass
class NonBinaryJudgementNode(BaseNode):
    criteria: str
    children: List[VerdictNode]
    evaluation_params: Optional[List[LLMTestCaseParams]] = None
    label: Optional[str] = None
    _verbose_logs: Optional[str] = None
    _verdict: Optional[NonBinaryJudgementVerdict] = None
    _parents: Optional[List[BaseNode]] = None

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        # Check if children is not empty
        if not self.children:
            raise ValueError(
                "NonBinaryJudgementNode must have at least one child."
            )

        verdicts_set = set()
        for child in self.children:
            if not isinstance(child, VerdictNode):
                raise TypeError("All children must be of type VerdictNode.")

            # Check if the verdict attribute of each child is a string
            if not isinstance(child.verdict, str):
                raise ValueError(
                    "The verdict attribute of all children must be a string."
                )

            # Check for duplicate verdicts
            if child.verdict in verdicts_set:
                raise ValueError(
                    f"Duplicate verdict found: {child.verdict} in children of NonBinaryJudgementNode."
                )
            verdicts_set.add(child.verdict)

        self._verdict_options = list(verdicts_set)

        # Dynamically create NonBinaryJudgementVerdict class
        self._verdict_schema = create_model(
            "NonBinaryJudgementVerdict",
            verdict=(Literal[tuple(self._verdict_options)], ...),
            reason=(str, ...),
        )

        # print("-------")
        for child in self.children:
            child.set_parent(self)
            increment_indegree(child)
            if child.child is not None and isinstance(child.child, BaseNode):
                increment_indegree(child.child)
        #         print("non binary node nested", child.child.__class__.__name__, id(child.child), child.child._indegree)
        #     print("non binary node", child.__class__.__name__, id(child), child._indegree)
        # print("-------")

    def _execute(self, metric: BaseMetric, test_case: LLMTestCase, depth: int):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            return

        text = """"""
        if self._parents is not None:
            for parent in self._parents:
                if isinstance(parent, TaskNode):
                    text += f"{parent.output_label}:\n{parent._output}\n"

        if self.evaluation_params is not None:
            for param in self.evaluation_params:
                value = getattr(test_case, param.value)
                if isinstance(value, ToolCall):
                    value = repr(value)
                text += f"{G_EVAL_PARAMS[param]}:\n{value}\n"

        prompt = NonBinaryJudgementTemplate.generate_non_binary_verdict(
            criteria=self.criteria, text=text, options=self._verdict_options
        )

        self._verdict = generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=self._verdict_schema,
            extract_schema=lambda s: s,
            extract_json=lambda data: self._verdict_schema(**data),
        )

        metric._verbose_steps.append(
            construct_node_verbose_log(self, self._depth)
        )
        for children in self.children:
            children._execute(
                metric=metric, test_case=test_case, depth=self._depth + 1
            )

    async def _a_execute(
        self, metric: BaseMetric, test_case: LLMTestCase, depth: int
    ):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            return

        text = """"""
        if self._parents is not None:
            for parent in self._parents:
                if isinstance(parent, TaskNode):
                    text += f"{parent.output_label}:\n{parent._output}\n"

        if self.evaluation_params is not None:
            for param in self.evaluation_params:
                value = getattr(test_case, param.value)
                if isinstance(value, ToolCall):
                    value = repr(value)
                text += f"{G_EVAL_PARAMS[param]}:\n{value}\n"

        prompt = NonBinaryJudgementTemplate.generate_non_binary_verdict(
            criteria=self.criteria, text=text, options=self._verdict_options
        )

        self._verdict = await a_generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=self._verdict_schema,
            extract_schema=lambda s: s,
            extract_json=lambda data: self._verdict_schema(**data),
        )

        metric._verbose_steps.append(
            construct_node_verbose_log(self, self._depth)
        )
        await asyncio.gather(
            *(
                child._a_execute(
                    metric=metric, test_case=test_case, depth=self._depth + 1
                )
                for child in self.children
            )
        )


def construct_node_verbose_log(
    node: BaseNode,
    depth: int,
    node_metric: Optional[Union[GEval, BaseMetric]] = None,
) -> str:
    if (
        isinstance(node, BinaryJudgementNode)
        or isinstance(node, NonBinaryJudgementNode)
        or isinstance(node, TaskNode)
    ):
        label = node.label if node.label else "None"

    if isinstance(node, BinaryJudgementNode) or isinstance(
        node, NonBinaryJudgementNode
    ):
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
            f"Verdict: {node._verdict.verdict}\n"
            f"Reason: {node._verdict.reason}\n"
        )
    elif isinstance(node, TaskNode):
        return (
            "______________________\n"
            f"| TaskNode | Level == {depth} |\n"
            "*******************************\n"
            f"Label: {label}\n\n"
            "Instructions:\n"
            f"{node.instructions}\n\n"
            f"{node.output_label}:\n{node._output}\n"
        )
    elif isinstance(node, LoopNode):
        return (
            "______________________\n"
            f"| LoopNode | Level == {depth} |\n"
            "*******************************\n"
            f"{node.output_label}:\n{node._output}\n"
        )
    elif isinstance(node, VerdictNode):
        type = None
        if node_metric:
            if isinstance(node_metric, GEval) or isinstance(
                node_metric, BaseMetric
            ):
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

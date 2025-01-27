from typing import Optional, List, Union, Literal
from dataclasses import dataclass
from pydantic import create_model

from deepeval.metrics.dag.types import (
    BinaryJudgementVerdict,
    NonBinaryJudgementVerdict,
)
from deepeval.metrics.dag.templates import (
    TaskNodeTemplate,
    BinaryJudgementTemplate,
    NonBinaryJudgementTemplate,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.metrics.g_eval.g_eval import G_EVAL_PARAMS
from deepeval.metrics.utils import trimAndLoadJson
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall


class BaseNode:
    def set_parent(self, parent: "BaseNode"):
        if hasattr(self, "_parent"):
            self._parent = parent
        elif hasattr(self, "_parents"):
            if self._parents is None:
                self._parents = []
            self._parents.append(parent)

    def _execute(self, metric: BaseMetric, test_case: LLMTestCase):
        raise NotImplementedError(
            "This node type must implement the _execute method."
        )

    async def _a_execute(self, metric: BaseMetric, test_case: LLMTestCase):
        raise NotImplementedError(
            "This node type must implement the _a_execute method."
        )


@dataclass
class VerdictNode(BaseNode):
    verdict: Union[str, bool]
    score: Optional[int]
    child: Optional[BaseNode]
    _parent: BaseNode

    def __post_init__(self):
        # Ensure either `score` or `child` is set, but not both
        if self.score is not None and self.child is not None:
            raise ValueError(
                "A VerdictNode can have either a 'score' or a 'child', but not both."
            )
        if self.score is None and self.child is None:
            raise ValueError(
                "A VerdictNode must have either a 'score' or a 'child'."
            )

        if isinstance(self.child, "VerdictNode"):
            raise ValueError(
                "A VerdictNode must not have another VerdictNode as a 'child'."
            )

        if self.score is not None:
            if not (0 <= self.score <= 10):
                raise ValueError(
                    "The score must be between 0 and 10, inclusive."
                )

        if self.child is not None:
            self.child.set_parent(self)

    def _execute(self, metric: BaseMetric, test_case: LLMTestCase):
        if (
            not hasattr(self._parent, "verdict")
            or self.verdict != self._parent.verdict.verdict
        ):
            return

        if self.score is not None:
            # generate reason
            return

    async def _a_execute(self, metric: BaseMetric, test_case: LLMTestCase):
        if (
            not hasattr(self._parent, "verdict")
            or self.verdict != self._parent.verdict.verdict
        ):
            return

        if self.score is not None:
            # generate reason
            return


@dataclass
class TaskNode(BaseNode):
    instructions: str
    evaluation_params: Optional[List[LLMTestCaseParams]]
    output_label: str
    output: Optional[str]
    children: List[BaseNode]
    _parents: Optional[List[BaseNode]]

    def __post_init__(self):
        for child in self.children:
            if isinstance(child, VerdictNode):
                raise ValueError(
                    "A TaskNode must not have a VerdictNode as one of their 'children'."
                )

        for child in self.children:
            child.set_parent(self)

    def _execute(self, metric: BaseMetric, test_case: LLMTestCase):
        text = """"""

        for child in self.children:
            if isinstance(child, TaskNode):
                text += f"{child.output_label}:\n{child.output}\n\n"

        for param in self.evaluation_params:
            value = getattr(test_case, param.value)
            if isinstance(value, ToolCall):
                value = repr(value)
            text += f"{G_EVAL_PARAMS[param]}:\n{value}\n\n"

        prompt = TaskNodeTemplate.generate_task_output(
            instructions=self.instructions,
            text=text,
        )
        if metric.using_native_model:
            res, cost = metric.model.a_generate(prompt)
            metric.evaluation_cost += cost
            self.output = res
        else:
            res = self.model.a_generate(prompt=prompt)
            self.output = res

    async def _a_execute(self, metric: BaseMetric, test_case: LLMTestCase):
        pass


@dataclass
class BinaryJudgementNode(BaseNode):
    criteria: str
    evaluation_params: Optional[List[LLMTestCaseParams]]
    verdict: Optional[BinaryJudgementVerdict]
    children: List[VerdictNode]
    _parents: Optional[List[BaseNode]]

    def __post_init__(self):
        if len(self.children) != 2:
            raise ValueError(
                "BinaryJudgementNode must have exactly 2 children."
            )

        # Check if all children are ClassificationResultNode and their classifications are boolean
        for child in self.children:
            if not isinstance(child, VerdictNode):
                raise TypeError(
                    "All children must be of type BinaryJudgementNode."
                )
            if not isinstance(child.verdict, bool):
                raise ValueError(
                    "All children BinaryJudgementNode must have a boolean vedict."
                )

        # Check if there is one True and one False classification
        verdicts = [child.verdict for child in self.children]
        if verdicts.count(True) != 1 or verdicts.count(False) != 1:
            raise ValueError(
                "BinaryJudgementNode must have one True and one False child."
            )

        for child in self.children:
            child.set_parent(self)

    def _execute(self, metric: BaseMetric, test_case: LLMTestCase):
        text = """"""

        for child in self.children:
            if isinstance(child, TaskNode):
                text += f"{child.output_label}:\n{child.output}\n\n"

        for param in self.evaluation_params:
            value = getattr(test_case, param.value)
            if isinstance(value, ToolCall):
                value = repr(value)
            text += f"{G_EVAL_PARAMS[param]}:\n{value}\n\n"

        prompt = BinaryJudgementTemplate.generate_binary_verdict(
            criteria=self.criteria,
            text=text,
        )
        if metric.using_native_model:
            res, cost = metric.model.generate(
                prompt, schema=BinaryJudgementVerdict
            )
            metric.evaluation_cost += cost
            self.verdict = res
        else:
            try:
                res: BinaryJudgementVerdict = metric.model.generate(
                    prompt, schema=BinaryJudgementVerdict
                )
                self.verdict = res
            except TypeError:
                res = metric.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                metric.verdict = BinaryJudgementVerdict(**data)

    async def _a_execute(self, metric: BaseMetric, test_case: LLMTestCase):
        pass


@dataclass
class NonBinaryJudgementNode(BaseNode):
    criteria: str
    evaluation_params: Optional[List[LLMTestCaseParams]]
    verdict: Optional[NonBinaryJudgementVerdict]
    children: List[VerdictNode]
    _parents: Optional[List[BaseNode]]
    _verdicts_options: Optional[List[str]]

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
        self.verdict_type = create_model(
            "NonBinaryJudgementVerdict",
            verdict=(Literal[tuple(self._verdict_options)], ...),
            reason=(str, ...),
        )

        for child in self.children:
            child.set_parent(self)

    def _execute(self, metric: BaseMetric, test_case: LLMTestCase):
        text = """"""

        for child in self.children:
            if isinstance(child, TaskNode):
                text += f"{child.output_label}:\n{child.output}\n\n"

        for param in self.evaluation_params:
            value = getattr(test_case, param.value)
            if isinstance(value, ToolCall):
                value = repr(value)
            text += f"{G_EVAL_PARAMS[param]}:\n{value}\n\n"

        prompt = NonBinaryJudgementTemplate.generate_non_binary_verdict(
            criteria=self.criteria, text=text, options=self._verdict_options
        )
        if metric.using_native_model:
            res, cost = metric.model.generate(prompt, schema=self.verdict_type)
            metric.evaluation_cost += cost
            self.verdict = res
        else:
            try:
                res: self.verdict_type = metric.model.generate(
                    prompt, schema=self.verdict_type
                )
                self.verdict = res
            except TypeError:
                res = metric.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                self.verdict = self.verdict_type(**data)

    async def _a_execute(self, metric: BaseMetric, test_case: LLMTestCase):
        pass

from typing import Optional, List, Union, Literal, Tuple
from dataclasses import dataclass
from pydantic import create_model
import asyncio

from deepeval.metrics.base_metric import BaseConversationalMetric
from deepeval.metrics.conversational_g_eval.conversational_g_eval import (
    ConversationalGEval,
)
from deepeval.metrics.g_eval.utils import CONVERSATIONAL_G_EVAL_PARAMS
from deepeval.metrics.utils import copy_metrics, trimAndLoadJson
from deepeval.test_case import (
    ConversationalTestCase,
    TurnParams,
    ToolCall,
    Turn,
)
from deepeval.utils import prettify_list

from .templates import (
    ConversationalBinaryJudgementTemplate,
    ConversationalNonBinaryJudgementTemplate,
    ConversationalTaskNodeTemplate,
    ConversationalVerdictNodeTemplate,
)
from deepeval.metrics.dag.schema import (
    BinaryJudgementVerdict,
    MetricScoreReason,
    NonBinaryJudgementVerdict,
    TaskNodeOutput,
)


class ConversationalBaseNode:
    _indegree: int = 0
    _depth: int = 0

    def set_parent(self, parent: "ConversationalBaseNode"):
        if hasattr(self, "_parent"):
            self._parent = parent
        elif hasattr(self, "_parents"):
            if self._parents is None:
                self._parents = []
            self._parents.append(parent)

    def _execute(
        self,
        metric: BaseConversationalMetric,
        test_case: ConversationalTestCase,
        depth: int,
    ):
        raise NotImplementedError(
            "This node type must implement the _execute method."
        )

    async def _a_execute(
        self,
        metric: BaseConversationalMetric,
        test_case: ConversationalTestCase,
        depth: int,
    ):
        raise NotImplementedError(
            "This node type must implement the _a_execute method."
        )


def increment_indegree(node: ConversationalBaseNode):
    node._indegree += 1


def decrement_indegree(node: ConversationalBaseNode):
    node._indegree -= 1


@dataclass
class ConversationalVerdictNode(ConversationalBaseNode):
    verdict: Union[str, bool]
    score: Optional[int] = None
    child: Optional[
        Union[
            ConversationalBaseNode,
            ConversationalGEval,
            BaseConversationalMetric,
        ]
    ] = None
    _parent: Optional[ConversationalBaseNode] = None

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        # Ensure either `score` or `child` is set, but not both
        if self.score is not None and self.child is not None:
            raise ValueError(
                "A ConversationalVerdictNode can have either a 'score' or a 'child', but not both."
            )
        if self.score is None and self.child is None:
            raise ValueError(
                "A ConversationalVerdictNode must have either a 'score' or a 'child'."
            )

        if self.score is not None:
            if not (0 <= self.score <= 10):
                raise ValueError(
                    "The score must be between 0 and 10, inclusive."
                )

    def _execute(
        self,
        metric: BaseConversationalMetric,
        test_case: ConversationalTestCase,
        depth: int,
    ):
        decrement_indegree(self)
        if self._indegree > 0:
            return

        if isinstance(
            self._parent, ConversationalNonBinaryJudgementNode
        ) or isinstance(self._parent, ConversationalBinaryJudgementNode):
            if self._parent._verdict.verdict != self.verdict:
                return

        if self.child is not None:
            if isinstance(self.child, ConversationalGEval):
                convo_g_eval_args = {
                    "name": self.child.name,
                    "model": metric.model,
                    "verbose_mode": False,
                }
                if self.child.criteria:
                    convo_g_eval_args["criteria"] = self.child.criteria
                else:
                    convo_g_eval_args["evaluation_steps"] = (
                        self.child.evaluation_steps
                    )
                if self.child.evaluation_params:
                    convo_g_eval_args["evaluation_params"] = (
                        self.child.evaluation_params
                    )
                copied_convo_g_eval = ConversationalGEval(**convo_g_eval_args)

                copied_convo_g_eval.measure(
                    test_case=test_case,
                    _show_indicator=False,
                    _log_metric_to_confident=False,
                )
                metric._verbose_steps.append(
                    construct_node_verbose_log(self, depth, copied_convo_g_eval)
                )
                metric.score = copied_convo_g_eval.score
                if metric.include_reason:
                    metric.reason = copied_convo_g_eval.reason

            elif isinstance(self.child, BaseConversationalMetric):
                copied_metric: BaseConversationalMetric = copy_metrics(
                    [self.child]
                )[0]
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
        self,
        metric: BaseConversationalMetric,
        test_case: ConversationalTestCase,
        depth: int,
    ):
        decrement_indegree(self)
        if self._indegree > 0:
            return

        if isinstance(
            self._parent, ConversationalNonBinaryJudgementNode
        ) or isinstance(self._parent, ConversationalBinaryJudgementNode):
            if self._parent._verdict.verdict != self.verdict:
                return

        if self.child is not None:
            if isinstance(self.child, ConversationalGEval):
                convo_g_eval_args = {
                    "name": self.child.name,
                    "model": metric.model,
                    "verbose_mode": False,
                }
                if self.child.criteria:
                    convo_g_eval_args["criteria"] = self.child.criteria
                else:
                    convo_g_eval_args["evaluation_steps"] = (
                        self.child.evaluation_steps
                    )
                if self.child.evaluation_params:
                    convo_g_eval_args["evaluation_params"] = (
                        self.child.evaluation_params
                    )
                copied_convo_g_eval = ConversationalGEval(**convo_g_eval_args)

                await copied_convo_g_eval.a_measure(
                    test_case=test_case,
                    _show_indicator=False,
                    _log_metric_to_confident=False,
                )
                metric._verbose_steps.append(
                    construct_node_verbose_log(self, depth, copied_convo_g_eval)
                )
                metric.score = copied_convo_g_eval.score
                if metric.include_reason:
                    metric.reason = copied_convo_g_eval.reason

            elif isinstance(self.child, BaseConversationalMetric):
                copied_metric: BaseConversationalMetric = copy_metrics(
                    [self.child]
                )[0]
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

    def _generate_reason(self, metric: BaseConversationalMetric):
        prompt = ConversationalVerdictNodeTemplate.generate_reason(
            verbose_steps=metric._verbose_steps,
            score=metric.score,
            name=metric.__name__,
        )
        if metric.using_native_model:
            res, cost = metric.model.generate(prompt, schema=MetricScoreReason)
            metric.evaluation_cost += cost
        else:
            try:
                res: MetricScoreReason = metric.model.generate(
                    prompt, schema=MetricScoreReason
                )
            except TypeError:
                res = metric.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                res = MetricScoreReason(**data)

        return res.reason

    async def _a_generate_reason(self, metric: BaseConversationalMetric):
        prompt = ConversationalVerdictNodeTemplate.generate_reason(
            verbose_steps=metric._verbose_steps,
            score=metric.score,
            name=metric.__name__,
        )
        if metric.using_native_model:
            res, cost = await metric.model.a_generate(
                prompt, schema=MetricScoreReason
            )
            metric.evaluation_cost += cost
        else:
            try:
                res: MetricScoreReason = await metric.model.a_generate(
                    prompt, schema=MetricScoreReason
                )
            except TypeError:
                res = await metric.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                res = MetricScoreReason(**data)

        return res.reason


@dataclass
class ConversationalTaskNode(ConversationalBaseNode):
    instructions: str
    output_label: str
    children: List[ConversationalBaseNode]
    evaluation_params: List[TurnParams] = None
    turn_window: Tuple[int, int] = None
    label: Optional[str] = None
    _verbose_logs: Optional[str] = None
    _output: Optional[str] = None
    _parents: Optional[List[ConversationalBaseNode]] = None

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        for child in self.children:
            if isinstance(child, ConversationalVerdictNode):
                raise ValueError(
                    "A ConversationalTaskNode must not have a ConversationalVerdictNode as one of their 'children'."
                )

        for child in self.children:
            child.set_parent(self)
            increment_indegree(child)

    def _execute(
        self,
        metric: BaseConversationalMetric,
        test_case: ConversationalTestCase,
        depth: int,
    ):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            return

        if self.evaluation_params is None and self._parents is None:
            raise ValueError(
                "A ConversationalTaskNode must have either a 'evaluation_params' or parent node(s)."
            )

        if self.turn_window is not None:
            is_valid_turn_window(self.turn_window, test_case.turns)

        if not self.turn_window:
            self.turn_window = 0, len(test_case.turns) - 1

        text = """"""
        start, end = self.turn_window
        if self._parents is not None:
            for parent in self._parents:
                if isinstance(parent, ConversationalTaskNode):
                    text += f"{parent.output_label}:\n{parent._output}\n\n"

        if self.evaluation_params is not None:
            text += "Full Conversation: \n"
            for index in range(start, end + 1):
                turn = test_case.turns[index]
                for param in self.evaluation_params:
                    value = getattr(turn, param.value)
                    if isinstance(value, ToolCall):
                        value = repr(value)
                    text += f"{CONVERSATIONAL_G_EVAL_PARAMS[param]}:\n{value}\n"
                    text += "\n"

        prompt = ConversationalTaskNodeTemplate.generate_task_output(
            instructions=self.instructions,
            text=text,
        )
        if metric.using_native_model:
            res, cost = metric.model.generate(prompt, schema=TaskNodeOutput)
            metric.evaluation_cost += cost
            self._output = res.output
        else:
            try:
                res: TaskNodeOutput = metric.model.generate(
                    prompt, schema=TaskNodeOutput
                )
                self._output = res.output
            except TypeError:
                res = metric.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                self._output = TaskNodeOutput(**data).output

        metric._verbose_steps.append(
            construct_node_verbose_log(self, self._depth)
        )
        for children in self.children:
            children._execute(
                metric=metric, test_case=test_case, depth=self._depth + 1
            )

    async def _a_execute(
        self,
        metric: BaseConversationalMetric,
        test_case: ConversationalTestCase,
        depth: int,
    ):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            return

        if self.evaluation_params is None and self._parents is None:
            raise ValueError(
                "A ConversationalTaskNode must have either a 'evaluation_params' or parent node(s)."
            )

        if self.turn_window is not None:
            is_valid_turn_window(self.turn_window, test_case.turns)

        if not self.turn_window:
            self.turn_window = 0, len(test_case.turns) - 1

        text = """"""
        start, end = self.turn_window
        if self._parents is not None:
            for parent in self._parents:
                if isinstance(parent, ConversationalTaskNode):
                    text += f"{parent.output_label}:\n{parent._output}\n\n"

        if self.evaluation_params is not None:
            text += "Full Conversation: \n"
            for index in range(start, end + 1):
                turn = test_case.turns[index]
                for param in self.evaluation_params:
                    value = getattr(turn, param.value)
                    if isinstance(value, ToolCall):
                        value = repr(value)
                    text += f"{CONVERSATIONAL_G_EVAL_PARAMS[param]}:\n{value}\n"
                    text += "\n"

        prompt = ConversationalTaskNodeTemplate.generate_task_output(
            instructions=self.instructions,
            text=text,
        )
        if metric.using_native_model:
            res, cost = await metric.model.a_generate(
                prompt, schema=TaskNodeOutput
            )
            metric.evaluation_cost += cost
            self._output = res.output
        else:
            try:
                res: TaskNodeOutput = await metric.model.a_generate(
                    prompt, schema=TaskNodeOutput
                )
                self._output = res.output
            except TypeError:
                res = await metric.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                self._output = TaskNodeOutput(**data).output

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
class ConversationalBinaryJudgementNode(ConversationalBaseNode):
    criteria: str
    children: List[ConversationalVerdictNode]
    evaluation_params: Optional[List[TurnParams]] = None
    turn_window: Tuple[int, int] = None
    label: Optional[str] = None
    _verbose_logs: Optional[str] = None
    _verdict: Optional[BinaryJudgementVerdict] = None
    _parents: Optional[List[ConversationalBaseNode]] = None

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        if len(self.children) != 2:
            raise ValueError(
                "ConversationalBinaryJudgementNode must have exactly 2 children."
            )

        # Check if all children are ClassificationResultNode and their classifications are boolean
        for child in self.children:
            if not isinstance(child, ConversationalVerdictNode):
                raise TypeError(
                    "All children of ConversationalBinaryJudgementNode must be of type ConversationalVerdictNode."
                )

            if not isinstance(child.verdict, bool):
                raise ValueError(
                    "All children of ConversationalBinaryJudgementNode must have a boolean verdict."
                )

        # Check if there is one True and one False classification
        verdicts = [child.verdict for child in self.children]
        if verdicts.count(True) != 1 or verdicts.count(False) != 1:
            raise ValueError(
                "ConversationalBinaryJudgementNode must have one True and one False ConversationalVerdictNode child."
            )

        # print("-------")
        for child in self.children:
            child.set_parent(self)
            increment_indegree(child)
            if child.child is not None and isinstance(
                child.child, ConversationalBaseNode
            ):
                increment_indegree(child.child)
        #         print("binary node nested", child.child.__class__.__name__, id(child.child), child.child._indegree)
        #     print("binary node", child.__class__.__name__, id(child), child._indegree)
        # print("-------")

    def _execute(
        self,
        metric: BaseConversationalMetric,
        test_case: ConversationalTestCase,
        depth: int,
    ):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            return

        if self.turn_window is not None:
            is_valid_turn_window(self.turn_window, test_case.turns)

        if not self.turn_window:
            self.turn_window = 0, len(test_case.turns) - 1

        text = """"""
        start, end = self.turn_window
        if self._parents is not None:
            for parent in self._parents:
                if isinstance(parent, ConversationalTaskNode):
                    text += f"{parent.output_label}:\n{parent._output}\n\n"

        if self.evaluation_params is not None:
            text += "Full Conversation: \n"
            for index in range(start, end + 1):
                turn = test_case.turns[index]
                for param in self.evaluation_params:
                    value = getattr(turn, param.value)
                    if isinstance(value, ToolCall):
                        value = repr(value)
                    text += f"{CONVERSATIONAL_G_EVAL_PARAMS[param]}:\n{value}\n"
                    text += "\n"

        prompt = ConversationalBinaryJudgementTemplate.generate_binary_verdict(
            criteria=self.criteria,
            text=text,
        )
        if metric.using_native_model:
            res, cost = metric.model.generate(
                prompt, schema=BinaryJudgementVerdict
            )
            metric.evaluation_cost += cost
            self._verdict = res
        else:
            try:
                res: BinaryJudgementVerdict = metric.model.generate(
                    prompt, schema=BinaryJudgementVerdict
                )
                self._verdict = res
            except TypeError:
                res = metric.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                self._verdict = BinaryJudgementVerdict(**data)

        metric._verbose_steps.append(
            construct_node_verbose_log(self, self._depth)
        )
        for children in self.children:
            children._execute(
                metric=metric, test_case=test_case, depth=self._depth + 1
            )

    async def _a_execute(
        self,
        metric: BaseConversationalMetric,
        test_case: ConversationalTestCase,
        depth: int,
    ):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            return

        if self.turn_window is not None:
            is_valid_turn_window(self.turn_window, test_case.turns)

        if not self.turn_window:
            self.turn_window = 0, len(test_case.turns) - 1

        text = """"""
        start, end = self.turn_window
        if self._parents is not None:
            for parent in self._parents:
                if isinstance(parent, ConversationalTaskNode):
                    text += f"{parent.output_label}:\n{parent._output}\n\n"

        if self.evaluation_params is not None:
            text += "Full Conversation: \n"
            for index in range(start, end + 1):
                turn = test_case.turns[index]
                for param in self.evaluation_params:
                    value = getattr(turn, param.value)
                    if isinstance(value, ToolCall):
                        value = repr(value)
                    text += f"{CONVERSATIONAL_G_EVAL_PARAMS[param]}:\n{value}\n"
                    text += "\n"

        prompt = ConversationalBinaryJudgementTemplate.generate_binary_verdict(
            criteria=self.criteria,
            text=text,
        )
        if metric.using_native_model:
            res, cost = await metric.model.a_generate(
                prompt, schema=BinaryJudgementVerdict
            )
            metric.evaluation_cost += cost
            self._verdict = res
        else:
            try:
                res: BinaryJudgementVerdict = await metric.model.a_generate(
                    prompt, schema=BinaryJudgementVerdict
                )
                self._verdict = res
            except TypeError:
                res = await metric.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                self._verdict = BinaryJudgementVerdict(**data)

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
class ConversationalNonBinaryJudgementNode(ConversationalBaseNode):
    criteria: str
    children: List[ConversationalVerdictNode]
    evaluation_params: Optional[List[TurnParams]] = None
    turn_window: Tuple[int, int] = None
    label: Optional[str] = None
    _verbose_logs: Optional[str] = None
    _verdict: Optional[NonBinaryJudgementVerdict] = None
    _parents: Optional[List[ConversationalBaseNode]] = None

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        # Check if children is not empty
        if not self.children:
            raise ValueError(
                "ConversationalNonBinaryJudgementNode must have at least one child."
            )

        verdicts_set = set()
        for child in self.children:
            if not isinstance(child, ConversationalVerdictNode):
                raise TypeError(
                    "All children must be of type ConversationalVerdictNode."
                )

            # Check if the verdict attribute of each child is a string
            if not isinstance(child.verdict, str):
                raise ValueError(
                    "The verdict attribute of all children must be a string."
                )

            # Check for duplicate verdicts
            if child.verdict in verdicts_set:
                raise ValueError(
                    f"Duplicate verdict found: {child.verdict} in children of ConversationalNonBinaryJudgementNode."
                )
            verdicts_set.add(child.verdict)

        self._verdict_options = list(verdicts_set)

        # Dynamically create ConversationalNonBinaryJudgementNode class
        self._verdict_schema = create_model(
            "ConversationalNonBinaryJudgementNode",
            verdict=(Literal[tuple(self._verdict_options)], ...),
            reason=(str, ...),
        )

        # print("-------")
        for child in self.children:
            child.set_parent(self)
            increment_indegree(child)
            if child.child is not None and isinstance(
                child.child, ConversationalBaseNode
            ):
                increment_indegree(child.child)
        #         print("non binary node nested", child.child.__class__.__name__, id(child.child), child.child._indegree)
        #     print("non binary node", child.__class__.__name__, id(child), child._indegree)
        # print("-------")

    def _execute(
        self,
        metric: BaseConversationalMetric,
        test_case: ConversationalTestCase,
        depth: int,
    ):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            return

        if self.turn_window is not None:
            is_valid_turn_window(self.turn_window, test_case.turns)

        if not self.turn_window:
            self.turn_window = 0, len(test_case.turns) - 1

        text = """"""
        start, end = self.turn_window
        if self._parents is not None:
            for parent in self._parents:
                if isinstance(parent, ConversationalTaskNode):
                    text += f"{parent.output_label}:\n{parent._output}\n\n"

        if self.evaluation_params is not None:
            text += "Full Conversation: \n"
            for index in range(start, end + 1):
                turn = test_case.turns[index]
                for param in self.evaluation_params:
                    value = getattr(turn, param.value)
                    if isinstance(value, ToolCall):
                        value = repr(value)
                    text += f"{CONVERSATIONAL_G_EVAL_PARAMS[param]}:\n{value}\n"
                    text += "\n"

        prompt = ConversationalNonBinaryJudgementTemplate.generate_non_binary_verdict(
            criteria=self.criteria, text=text, options=self._verdict_options
        )
        if metric.using_native_model:
            res, cost = metric.model.generate(
                prompt, schema=self._verdict_schema
            )
            metric.evaluation_cost += cost
            self._verdict = res
        else:
            try:
                res: self._verdict_schema = metric.model.generate(
                    prompt, schema=self._verdict_schema
                )
                self._verdict = res
            except TypeError:
                res = metric.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                self._verdict = self._verdict_schema(**data)

        metric._verbose_steps.append(
            construct_node_verbose_log(self, self._depth)
        )
        for children in self.children:
            children._execute(
                metric=metric, test_case=test_case, depth=self._depth + 1
            )

    async def _a_execute(
        self,
        metric: BaseConversationalMetric,
        test_case: ConversationalTestCase,
        depth: int,
    ):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            return

        if self.turn_window is not None:
            is_valid_turn_window(self.turn_window, test_case.turns)

        if not self.turn_window:
            self.turn_window = 0, len(test_case.turns) - 1

        text = """"""
        start, end = self.turn_window
        if self._parents is not None:
            for parent in self._parents:
                if isinstance(parent, ConversationalTaskNode):
                    text += f"{parent.output_label}:\n{parent._output}\n\n"

        if self.evaluation_params is not None:
            text += "Full Conversation: \n"
            for index in range(start, end + 1):
                turn = test_case.turns[index]
                for param in self.evaluation_params:
                    value = getattr(turn, param.value)
                    if isinstance(value, ToolCall):
                        value = repr(value)
                    text += f"{CONVERSATIONAL_G_EVAL_PARAMS[param]}:\n{value}\n"
                    text += "\n"

        prompt = ConversationalNonBinaryJudgementTemplate.generate_non_binary_verdict(
            criteria=self.criteria, text=text, options=self._verdict_options
        )
        if metric.using_native_model:
            res, cost = await metric.model.a_generate(
                prompt, schema=self._verdict_schema
            )
            metric.evaluation_cost += cost
            self._verdict = res
        else:
            try:
                res: self._verdict_schema = await metric.model.a_generate(
                    prompt, schema=self._verdict_schema
                )
                self._verdict = res
            except TypeError:
                res = await metric.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                self._verdict = self._verdict_schema(**data)

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
    node: ConversationalBaseNode,
    depth: int,
    node_metric: Optional[
        Union[ConversationalGEval, BaseConversationalMetric]
    ] = None,
) -> str:
    if (
        isinstance(node, ConversationalBinaryJudgementNode)
        or isinstance(node, ConversationalNonBinaryJudgementNode)
        or isinstance(node, ConversationalTaskNode)
    ):
        label = node.label if node.label else "None"

    if isinstance(node, ConversationalBinaryJudgementNode) or isinstance(
        node, ConversationalNonBinaryJudgementNode
    ):
        is_binary_node = isinstance(node, ConversationalBinaryJudgementNode)
        node_type = (
            "ConversationalBinaryJudgementNode"
            if is_binary_node
            else "ConversationalNonBinaryJudgementNode"
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
    elif isinstance(node, ConversationalTaskNode):
        return (
            "______________________________________________\n"
            f"| ConversationalTaskNode | Level == {depth} |\n"
            "**********************************************\n"
            f"Label: {label}\n\n"
            "Instructions:\n"
            f"{node.instructions}\n\n"
            f"{node.output_label}:\n{node._output}\n"
        )
    elif isinstance(node, ConversationalVerdictNode):
        type = None
        if node_metric:
            if isinstance(node_metric, ConversationalGEval) or isinstance(
                node_metric, BaseConversationalMetric
            ):
                type = f"{node_metric.__name__} Metric"
        else:
            type = "Deterministic"

        verbose_log = (
            "_________________________________________________\n"
            f"| ConversationalVerdictNode | Level == {depth} |\n"
            "*************************************************\n"
            f"Verdict: {node.verdict}\n"
            f"Type: {type}"
        )
        if isinstance(node_metric, ConversationalGEval):
            verbose_log += f"\n\nCriteria:\n{node_metric.criteria}\n"
            verbose_log += f"Evaluation Steps:\n{prettify_list(node_metric.evaluation_steps)}"
        elif isinstance(node_metric, BaseConversationalMetric):
            verbose_log += f"\n\n{node_metric.verbose_logs}"

        return verbose_log


def is_valid_turn_window(
    turn_window: Tuple[int, int], turns: List[Turn]
) -> bool:
    if len(turn_window) != 2:
        raise ValueError(
            "A 'turn_window' must have only 2 indices representing start and end"
        )
    start, end = turn_window
    if (
        start > end
        or start == end
        or (end - start) >= len(turns)
        or start < 0
        or end < 0
        or end == len(turns)
    ):
        raise ValueError(
            "The 'turn_window' passed is invalid. Please recheck your 'turn_window' values."
        )
    return True

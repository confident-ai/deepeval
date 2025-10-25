import asyncio
from typing import Optional, Union, List

from deepeval.metrics import BaseConversationalMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
    get_unit_interactions,
    trimAndLoadJson,
    initialize_model,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.test_case import ConversationalTestCase, TurnParams
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.mcp.schema import Task, ArgsScore, ToolScore
from deepeval.metrics.mcp.template import MCPTaskCompletionTemplate
from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.api import metric_data_manager


class MultiTurnMCPUseMetric(BaseConversationalMetric):
    _required_test_case_params = [
        TurnParams.ROLE,
        TurnParams.CONTENT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ):
        check_conversational_test_case_params(
            test_case, self._required_test_case_params, self
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                        _log_metric_to_confident=_log_metric_to_confident,
                    )
                )
            else:
                if not test_case.mcp_servers:
                    error_str = "'mcp_servers' in a conversational test case cannot be empty for the 'MultiTurnMCPUseMetric' metric."
                    self.error = error_str
                    raise MissingTestCaseParamsError(error_str)
                self.unit_interactions = get_unit_interactions(test_case.turns)
                self.tasks = self._get_tasks(self.unit_interactions)
                primitives_accuracy_scores = [
                    self._get_tool_accuracy_score(task, test_case)
                    for task in self.tasks
                ]
                args_accuracy_scores = [
                    self._get_args_score(task, test_case) for task in self.tasks
                ]
                self.score = self._calculate_score(
                    primitives_accuracy_scores, args_accuracy_scores
                )
                self.reason = self._generate_reason(
                    primitives_accuracy_scores, args_accuracy_scores
                )
                self.tools_scores_reasons_list = [
                    (tool_score.score, tool_score.reason)
                    for tool_score in primitives_accuracy_scores
                ]
                self.args_scores_reasons_list = [
                    (args_score.score, args_score.reason)
                    for args_score in args_accuracy_scores
                ]
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Tasks:\n{prettify_list(self.tasks)}",
                        f"Individual Scores & Reasons for Primitives:\n{prettify_list(self.tools_scores_reasons_list)}",
                        f"Individual Scores & Reasons for Arguments:\n{prettify_list(self.args_scores_reasons_list)}",
                        f"Score: {self.score}",
                    ],
                )
                if _log_metric_to_confident:
                    metric_data_manager.post_metric_if_enabled(
                        self, test_case=test_case
                    )

            return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ):
        check_conversational_test_case_params(
            test_case, self._required_test_case_params, self
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if not test_case.mcp_servers:
                error_str = "'mcp_servers' in a conversational test case cannot be empty for the 'MultiTurnMCPUseMetric' metric."
                self.error = error_str
                raise MissingTestCaseParamsError(error_str)

            self.unit_interactions = get_unit_interactions(test_case.turns)
            self.tasks = self._get_tasks(self.unit_interactions)
            primitives_accuracy_scores = await asyncio.gather(
                *[
                    self._a_get_tool_accuracy_score(task, test_case)
                    for task in self.tasks
                ]
            )
            args_accuracy_scores = await asyncio.gather(
                *[
                    self._a_get_args_score(task, test_case)
                    for task in self.tasks
                ]
            )
            self.score = self._calculate_score(
                primitives_accuracy_scores, args_accuracy_scores
            )
            self.reason = self._generate_reason(
                primitives_accuracy_scores, args_accuracy_scores
            )
            self.tools_scores_reasons_list = [
                (tool_score.score, tool_score.reason)
                for tool_score in primitives_accuracy_scores
            ]
            self.args_scores_reasons_list = [
                (args_score.score, args_score.reason)
                for args_score in args_accuracy_scores
            ]
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Tasks:\n{prettify_list(self.tasks)}",
                    f"Individual Scores & Reasons for Primitives:\n{prettify_list(self.tools_scores_reasons_list)}",
                    f"Individual Scores & Reasons for Arguments:\n{prettify_list(self.args_scores_reasons_list)}",
                    f"Score: {self.score}",
                ],
            )
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )
        return self.score

    def _get_tool_accuracy_score(
        self, task: Task, test_case: ConversationalTestCase
    ) -> ToolScore:
        prompt = MCPTaskCompletionTemplate.get_tool_correctness_score(
            task, test_case.mcp_servers
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=ToolScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: ToolScore = self.model.generate(prompt, schema=ToolScore)
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return ToolScore(**data)

    async def _a_get_tool_accuracy_score(
        self, task: Task, test_case: ConversationalTestCase
    ) -> ToolScore:
        prompt = MCPTaskCompletionTemplate.get_tool_correctness_score(
            task, test_case.mcp_servers
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=ToolScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: ToolScore = await self.model.a_generate(
                    prompt, schema=ToolScore
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return ToolScore(**data)

    def _get_args_score(
        self, task: Task, test_case: ConversationalTestCase
    ) -> ArgsScore:
        prompt = MCPTaskCompletionTemplate.get_args_correctness_score(
            task, test_case.mcp_servers
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=ArgsScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: ArgsScore = self.model.generate(prompt, schema=ArgsScore)
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return ArgsScore(**data)

    async def _a_get_args_score(
        self, task: Task, test_case: ConversationalTestCase
    ) -> ArgsScore:
        prompt = MCPTaskCompletionTemplate.get_args_correctness_score(
            task, test_case.mcp_servers
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=ArgsScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: ArgsScore = await self.model.a_generate(
                    prompt, schema=ArgsScore
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return ArgsScore(**data)

    def _get_tasks(self, unit_interactions: List) -> List[Task]:
        tasks = []
        for unit_interaction in unit_interactions:
            if len(unit_interaction) <= 2:
                continue
            user_messages = ""
            for turn in unit_interaction:
                if turn.role == "user":
                    user_messages += turn.content + "\n"
                else:
                    break
            new_task = Task(task=user_messages, steps_taken=[])
            for turn in unit_interaction[1:]:
                if turn._mcp_interaction:
                    mcp_interaction = "Tools called by agent: \n"
                    if turn.mcp_tools_called is not None:
                        for tool in turn.mcp_tools_called:
                            mcp_interaction += (
                                f"\n<Tool Called>\n"
                                f"\n**This does not appear to user**\n"
                                f"Name: {tool.name}\n"
                                f"Args: {tool.args}\n"
                                f"Result: \n{tool.result.structuredContent['result']}\n"
                                f"</Tool Called>\n"
                            )
                    if turn.mcp_resources_called is not None:
                        for resource in turn.mcp_resources_called:
                            mcp_interaction += (
                                f"\n<Resource Called>\n"
                                f"\n**This does not appear to user**\n"
                                f"URI: {resource.uri}\n"
                                f"Result: {str(resource.result)}\n"
                                f"</Resource Called>\n"
                            )
                    if turn.mcp_prompts_called is not None:
                        for prompt in turn.mcp_prompts_called:
                            mcp_interaction += (
                                f"\n<Prompt Called>\n"
                                f"\n**This does not appear to user**\n"
                                f"Name: {prompt.name}\n"
                                f"Result: {str(prompt.result)}\n"
                                f"</Prompt Called>\n"
                            )
                    new_task.steps_taken.append(mcp_interaction)
                else:
                    new_task.steps_taken.append(
                        "Agent's response to user: \n" + turn.content
                    )
            tasks.append(new_task)
        return tasks

    def _calculate_score(
        self,
        tool_accuracy_score: List[ToolScore],
        args_accuracy_score: List[ArgsScore],
    ) -> float:
        tool_divisor = (
            len(tool_accuracy_score) if len(tool_accuracy_score) > 0 else 1
        )
        args_divisor = (
            len(args_accuracy_score) if len(args_accuracy_score) > 0 else 1
        )
        tool_score = (
            sum(score.score for score in tool_accuracy_score) / tool_divisor
        )
        args_score = (
            sum(score.score for score in args_accuracy_score) / args_divisor
        )
        score = min(tool_score, args_score)
        return 0 if self.strict_mode and score < self.threshold else score

    def _generate_reason(
        self,
        tool_accuracy_score: List[ToolScore],
        args_accuracy_score: List[ArgsScore],
    ) -> str:
        reason = "["
        for task_score in tool_accuracy_score:
            if task_score.score < self.threshold:
                reason += "\nPrimitives Used\n"
                reason += (
                    f"Score: {task_score.score}\n"
                    f"Reason: {task_score.reason}\n"
                )
        for task_score in args_accuracy_score:
            if task_score.score < self.threshold:
                reason += "\nArguments Generated\n"
                reason += (
                    f"Score: {task_score.score}\n"
                    f"Reason: {task_score.reason}\n"
                )
        reason += "]"
        return reason

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Multi-Turn MCP Use"

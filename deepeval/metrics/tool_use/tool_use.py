from typing import Optional, List, Union
import asyncio
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    get_unit_interactions,
    print_tools_called,
    check_conversational_test_case_params,
    initialize_model,
)
from deepeval.test_case import ConversationalTestCase, TurnParams, ToolCall, Turn
from deepeval.metrics import BaseConversationalMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.tool_use.template import ToolUseTemplate
from deepeval.metrics.tool_use.schema import ToolUseScore, UserInputAndTools
from deepeval.metrics.api import metric_data_manager


class ToolUseMetric(BaseConversationalMetric):

    _required_test_case_params = [
        TurnParams.ROLE,
        TurnParams.CONTENT,
    ]

    def __init__(
        self,
        available_tools: List[ToolCall],
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.available_tools = available_tools
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
                unit_interactions = get_unit_interactions(test_case.turns)
                user_input_and_tools = self._get_user_input_and_turns(unit_interactions)
                tool_use_scores = [
                    self._get_tool_use_score(user_and_tools) for user_and_tools in user_input_and_tools
                ]
                self.score = self._calculate_score(tool_use_scores)
                self.reason = self._generate_reason(tool_use_scores)

                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Tool Selection Scores: {prettify_list(tool_use_scores)} \n\n"
                        f"Final Score: {self.score}",
                        f"Final Reason: {self.reason}",
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
            unit_interactions = get_unit_interactions(test_case.turns)
            user_input_and_tools = self._get_user_input_and_turns(unit_interactions)
            tool_use_scores = await asyncio.gather(
                *[
                    self._a_get_tool_use_score(user_and_tools) for user_and_tools in user_input_and_tools
                ]
            )
            self.score = self._calculate_score(tool_use_scores)
            self.reason = await self._a_generate_reason(tool_use_scores)

            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Tool Selection Scores: {prettify_list(tool_use_scores)} \n\n"
                    f"Final Score: {self.score}",
                    f"Final Reason: {self.reason}",
                ],
            )

            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )

            return self.score
    
    def _get_tool_use_score(
        self,
        user_and_tools: UserInputAndTools,
    ):
        prompt = ToolUseTemplate.get_tool_selection_score(
            user_and_tools.user_messages,
            user_and_tools.assistant_messages,
            user_and_tools.tools_called,
            user_and_tools.available_tools
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=ToolUseScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: ToolUseScore = self.model.generate(prompt, schema=ToolUseScore)
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return ToolUseScore(**data)
    
    async def _a_get_tool_use_score(
        self,
        user_and_tools: UserInputAndTools,
    ):
        prompt = ToolUseTemplate.get_tool_selection_score(
            user_and_tools.user_messages,
            user_and_tools.assistant_messages,
            user_and_tools.tools_called,
            user_and_tools.available_tools
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=ToolUseScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: ToolUseScore = await self.model.a_generate(prompt, schema=ToolUseScore)
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return ToolUseScore(**data)

    def _get_user_input_and_turns(
        self,
        unit_interactions: List[List[Turn]],
    ) -> List[UserInputAndTools]:
        user_inputs_and_tools = []
        available_tools = ",".join([repr(tool) for tool in self.available_tools])
        for unit_interaction in unit_interactions:
            if len(unit_interaction) < 2:
                continue
            user_messages = ""
            assistant_messages = ""
            tools_called = []
            for turn in unit_interaction:
                if turn.role == "user":
                    user_messages += f"{turn.content} \n"
                else:
                    break
            for turn in unit_interaction[1:]:
                if turn.role == "assistant":
                    assistant_messages += f"{turn.content} \n"
                    if turn.tools_called:
                        tools_called.extend(turn.tools_called)
            tools_called = ",".join([repr(tool) for tool in tools_called])
            new_user_input_tools = UserInputAndTools(
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                tools_called=tools_called,
                available_tools=available_tools
            )
            user_inputs_and_tools.append(new_user_input_tools)
        return user_inputs_and_tools
    
    def _calculate_score(self, tool_use_scores: List[ToolUseScore]):
        scores_sum = sum([tool_use_score.score for tool_use_score in tool_use_scores])
        scores_divisor = len(tool_use_scores) if len(tool_use_scores) > 0 else 1
        return scores_sum / scores_divisor
    
    def _generate_reason(self, tool_use_scores: List[ToolUseScore]):
        scores_and_reasons = ""
        for tool_use in tool_use_scores:
            scores_and_reasons += f"\nScore: {tool_use.score} \nReason: {tool_use.reason} \n"
        prompt = ToolUseTemplate.get_tool_selection_final_reason(
            scores_and_reasons, self.score, self.threshold
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            return res
        else:
            res = self.model.generate(prompt)
            return res
    
    async def _a_generate_reason(self, tool_use_scores: List[ToolUseScore]):
        scores_and_reasons = ""
        for tool_use in tool_use_scores:
            scores_and_reasons += f"\nScore: {tool_use.score} \nReason: {tool_use.reason} \n"
        prompt = ToolUseTemplate.get_tool_selection_final_reason(
            scores_and_reasons, self.score, self.threshold
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            return res
        else:
            res = await self.model.a_generate(prompt)
            return res
        
    def is_successful(self) -> bool:
        try:
            self.success = self.score >= self.threshold
        except (AttributeError, TypeError):
            self.success = False
        return self.success

    @property
    def __name__(self):
        return "Tool Use"



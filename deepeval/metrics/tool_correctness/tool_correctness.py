from typing import List, Dict, Optional, Union

from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    trimAndLoadJson,
    initialize_model,
    print_tools_called,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ToolCallParams,
    ToolCall,
)
from deepeval.metrics import BaseMetric
from deepeval.metrics.api import metric_data_manager
from deepeval.metrics.tool_correctness.template import ToolCorrectnessTemplate
from deepeval.metrics.tool_correctness.schema import ToolSelectionScore


class ToolCorrectnessMetric(BaseMetric):

    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.TOOLS_CALLED,
        LLMTestCaseParams.EXPECTED_TOOLS,
    ]

    def __init__(
        self,
        available_tools: List[ToolCall] = None,
        threshold: float = 0.5,
        evaluation_params: List[ToolCallParams] = [],
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        should_exact_match: bool = False,
        should_consider_ordering: bool = False,
    ):
        self.available_tools = available_tools
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.async_mode = async_mode
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_params: List[ToolCallParams] = evaluation_params
        self.should_exact_match = should_exact_match
        self.should_consider_ordering = should_consider_ordering

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)
        self.test_case = test_case
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
                self.tools_called: List[ToolCall] = test_case.tools_called
                self.expected_tools: List[ToolCall] = test_case.expected_tools
                tool_calling_score = self._calculate_score()
                if self.available_tools:
                    tool_selection_score = self._get_tool_selection_score(
                        test_case.input,
                        test_case.tools_called,
                        self.available_tools,
                    )
                else:
                    tool_selection_score = tool_selection_score = (
                        ToolSelectionScore(
                            score=1,
                            reason="No available tools were provided to assess tool selection criteria",
                        )
                    )
                score = min(tool_calling_score, tool_selection_score.score)
                self.score = (
                    0 if self.strict_mode and score < self.threshold else score
                )
                tool_calling_reason = self._generate_reason()
                self.reason = self._construct_final_reason(
                    tool_calling_reason, tool_selection_score.reason
                )
                self.success = self.score >= self.threshold

                expected_tools_formatted = (
                    "Expected Tools:\n[\n"
                    + ",\n".join(
                        self.indent_multiline_string(
                            repr(tool_call), indent_level=4
                        )
                        for tool_call in self.expected_tools
                    )
                    + "\n]"
                )
                tools_called_formatted = (
                    "Tools Called:\n[\n"
                    + ",\n".join(
                        self.indent_multiline_string(
                            repr(tool_call), indent_level=4
                        )
                        for tool_call in self.tools_called
                    )
                    + "\n]"
                )
                available_tools_formatted = (
                    (
                        "Available Tools:\n[\n"
                        + ",\n".join(
                            self.indent_multiline_string(
                                repr(tool_call), indent_level=4
                            )
                            for tool_call in self.available_tools
                        )
                        + "\n]"
                    )
                    if self.available_tools
                    else "Available Tools: []"
                )
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"{expected_tools_formatted}",
                        f"{tools_called_formatted}",
                        f"{available_tools_formatted}",
                        f"Tool Selection Score: {tool_selection_score.score}",
                        f"Tool Selection Reason: {tool_selection_score.reason}",
                        f"Final Score: {self.score}\nFinal Reason: {self.reason}",
                    ],
                )

                if _log_metric_to_confident:
                    metric_data_manager.post_metric_if_enabled(
                        self, test_case=test_case
                    )
                return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.tools_called: List[ToolCall] = test_case.tools_called
            self.expected_tools: List[ToolCall] = test_case.expected_tools
            tool_calling_score = self._calculate_score()
            if self.available_tools:
                tool_selection_score = await self._a_get_tool_selection_score(
                    test_case.input,
                    test_case.tools_called,
                    self.available_tools,
                )
            else:
                tool_selection_score = ToolSelectionScore(
                    score=1,
                    reason="No available tools were provided to assess tool selection criteria",
                )
            score = min(tool_calling_score, tool_selection_score.score)
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            tool_calling_reason = self._generate_reason()
            self.reason = self._construct_final_reason(
                tool_calling_reason, tool_selection_score.reason
            )
            self.success = self.score >= self.threshold

            expected_tools_formatted = (
                "Expected Tools:\n[\n"
                + ",\n".join(
                    self.indent_multiline_string(
                        repr(tool_call), indent_level=4
                    )
                    for tool_call in self.expected_tools
                )
                + "\n]"
            )
            tools_called_formatted = (
                "Tools Called:\n[\n"
                + ",\n".join(
                    self.indent_multiline_string(
                        repr(tool_call), indent_level=4
                    )
                    for tool_call in self.tools_called
                )
                + "\n]"
            )
            available_tools_formatted = (
                (
                    "Available Tools:\n[\n"
                    + ",\n".join(
                        self.indent_multiline_string(
                            repr(tool_call), indent_level=4
                        )
                        for tool_call in self.available_tools
                    )
                    + "\n]"
                )
                if self.available_tools
                else "Available Tools: []"
            )
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"{expected_tools_formatted}",
                    f"{tools_called_formatted}",
                    f"{available_tools_formatted}",
                    f"Tool Selection Score: {tool_selection_score.score}",
                    f"Tool Selection Reason: {tool_selection_score.reason}",
                    f"Final Score: {self.score}\nFinal Reason: {self.reason}",
                ],
            )

            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )
            return self.score

    ##################################################
    ### Tool Correctness (Tool) ######################
    ##################################################

    def _generate_reason(self):
        tools_called_names = [
            tool_called.name for tool_called in self.tools_called
        ]
        expected_tools_names = [
            expected_tool.name for expected_tool in self.expected_tools
        ]

        if self.should_exact_match:
            return f"{'Exact match' if self._calculate_exact_match_score() else 'Not an exact match'}: expected {expected_tools_names}, called {tools_called_names}. See details above."

        elif self.should_consider_ordering:
            lcs, weighted_length = self._compute_weighted_lcs()
            if (
                len(self.tools_called) == len(self.expected_tools)
                and len(self.expected_tools) == 0
            ):
                score = 1.0
            elif len(self.expected_tools) == 0:
                score = 0.0
            else:
                score = weighted_length / len(self.expected_tools)
            missing = set(expected_tools_names) - set(tools_called_names)
            out_of_order = set(expected_tools_names) - set(
                [tool.name for tool in lcs]
            )
            if score == 1:
                return f"Correct ordering: all expected tools {expected_tools_names} were called in the correct order."
            else:
                issues = []
                if missing:
                    issues.append(f"missing tools {list(missing)}")
                if out_of_order:
                    issues.append(f"out-of-order tools {list(out_of_order)}")
                return f"Incorrect tool usage: {' and '.join(issues)}; expected {expected_tools_names}, called {tools_called_names}. See more details above."
        else:
            used_expected = set(self.tools_called).intersection(
                set(self.expected_tools)
            )
            missing = set(self.expected_tools) - used_expected
            if self._calculate_non_exact_match_score() == 1:
                return f"All expected tools {expected_tools_names} were called (order not considered)."
            else:
                return f"Incomplete tool usage: missing tools {list(missing)}; expected {expected_tools_names}, called {tools_called_names}. See more details above."

    def _construct_final_reason(
        self,
        tool_calling_reason,
        tool_selection_reason,
    ):
        final_reason = "[\n"
        final_reason += "\t Tool Calling Reason: " + tool_calling_reason + "\n"
        final_reason += (
            "\t Tool Selection Reason: " + tool_selection_reason + "\n"
        )
        final_reason += "]\n"
        return final_reason

    ##################################################
    ### Score Helper Functions #######################
    ##################################################

    def _get_tool_selection_score(
        self, user_input, tools_called, available_tools
    ):
        tools_called_formatted = print_tools_called(tools_called)
        available_tools_formatted = print_tools_called(available_tools)
        prompt = ToolCorrectnessTemplate.get_tool_selection_score(
            user_input, tools_called_formatted, available_tools_formatted
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=ToolSelectionScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res = self.model.generate(prompt, schema=ToolSelectionScore)
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return ToolSelectionScore(**data)

    async def _a_get_tool_selection_score(
        self, user_input, tools_called, available_tools
    ):
        tools_called_formatted = print_tools_called(tools_called)
        available_tools_formatted = print_tools_called(available_tools)
        prompt = ToolCorrectnessTemplate.get_tool_selection_score(
            user_input, tools_called_formatted, available_tools_formatted
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=ToolSelectionScore
            )
            self.evaluation_cost += cost
            return res
        else:
            try:
                res = await self.model.a_generate(
                    prompt, schema=ToolSelectionScore
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return ToolSelectionScore(**data)

    # Calculate score
    def _calculate_score(self):
        if self.should_exact_match:
            score = self._calculate_exact_match_score()
        elif self.should_consider_ordering:
            _, weighted_length = self._compute_weighted_lcs()
            if (
                len(self.tools_called) == len(self.expected_tools)
                and len(self.expected_tools) == 0
            ):
                score = 1.0
            elif len(self.expected_tools) == 0:
                score = 0.0
            else:
                score = weighted_length / len(self.expected_tools)
        else:
            score = self._calculate_non_exact_match_score()
        return 0 if self.strict_mode and score < self.threshold else score

    # Exact matching score
    def _calculate_exact_match_score(self):
        if len(self.tools_called) != len(self.expected_tools):
            return 0.0
        if (
            len(self.tools_called) == len(self.expected_tools)
            and len(self.expected_tools) == 0
        ):
            return 1.0
        for i in range(len(self.tools_called)):
            if self.tools_called[i].name != self.expected_tools[i].name:
                return 0.0
            if ToolCallParams.INPUT_PARAMETERS in self.evaluation_params:
                if (
                    self.tools_called[i].input_parameters
                    != self.expected_tools[i].input_parameters
                ):
                    return 0.0
            if ToolCallParams.OUTPUT in self.evaluation_params:
                if self.tools_called[i].output != self.expected_tools[i].output:
                    return 0.0
        return 1.0

    # Non exact matching score
    def _calculate_non_exact_match_score(self):
        total_score = 0.0
        matched_called_tools = set()
        for expected_tool in self.expected_tools:
            best_score = 0.0
            for called_tool in self.tools_called:
                if called_tool in matched_called_tools:
                    continue
                if expected_tool.name == called_tool.name:
                    match_score = 1.0
                    if (
                        ToolCallParams.INPUT_PARAMETERS
                        in self.evaluation_params
                    ):
                        match_score *= self._compare_dicts(
                            expected_tool.input_parameters,
                            called_tool.input_parameters,
                        )
                    if (
                        ToolCallParams.OUTPUT in self.evaluation_params
                        and expected_tool.output != called_tool.output
                    ):
                        match_score = 0.0
                    if match_score > best_score:
                        best_score = match_score
                        best_called_tool = called_tool
            if best_score > 0:
                total_score += best_score
                matched_called_tools.add(best_called_tool)
        return (
            1.0
            if not self.expected_tools and not self.tools_called
            else (
                0.0
                if not self.expected_tools
                else total_score / len(self.expected_tools)
            )
        )

    # Consider ordering score
    def _compute_weighted_lcs(self):
        m, n = len(self.expected_tools), len(self.tools_called)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                expected_tool, called_tool = (
                    self.expected_tools[i - 1],
                    self.tools_called[j - 1],
                )
                if expected_tool.name != called_tool.name:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    continue
                score = 1.0
                if ToolCallParams.INPUT_PARAMETERS in self.evaluation_params:
                    score *= self._compare_dicts(
                        expected_tool.input_parameters,
                        called_tool.input_parameters,
                    )
                if (
                    ToolCallParams.OUTPUT in self.evaluation_params
                    and expected_tool.output != called_tool.output
                ):
                    score = 0.0
                dp[i][j] = max(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1] + score if score > 0 else 0,
                )
        i, j, total_score = m, n, 0.0
        lcs = []
        while i > 0 and j > 0:
            if dp[i][j] == dp[i - 1][j]:
                i -= 1
            elif dp[i][j] == dp[i][j - 1]:
                j -= 1
            else:
                lcs.append(self.expected_tools[i - 1])
                total_score += dp[i][j] - dp[i - 1][j - 1]
                i, j = i - 1, j - 1
        return lcs[::-1], total_score

    # For matching input parameters
    def _compare_dicts(self, dict1: Dict, dict2: Dict):
        if dict1 == dict2:
            return 1.0
        if self.should_exact_match:
            return 1.0 if dict1 == dict2 else 0.0
        match_score = 0
        matched_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        total_keys = set(dict1.keys()).union(set(dict2.keys()))
        for key in matched_keys:
            if dict1[key] == dict2[key]:
                match_score += 1 / len(total_keys)
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                match_score += self._compare_dicts(
                    dict1[key], dict2[key]
                ) / len(total_keys)
        return match_score

    ##################################################
    ### Others #######################################
    ##################################################

    def is_successful(self) -> bool:
        try:
            self.success = self.score >= self.threshold
        except (AttributeError, TypeError):
            self.success = False
        return self.success

    @property
    def __name__(self):
        return "Tool Correctness"

    def indent_multiline_string(self, s, indent_level=4):
        indent = " " * indent_level
        return "\n".join(f"{indent}{line}" for line in s.splitlines())

from typing import List, Dict, Optional, Union, Tuple, Type

from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    prepare_measure,
    construct_verbose_logs,
    initialize_model,
    initialize_system_one_model,
    print_tools_called,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
    SystemOneEvalSpec,
    SystemOneScoreSpec,
    format_decision_reason,
    system_one_score,
    a_system_one_score,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
    ToolCallParams,
    ToolCall,
)
from deepeval.metrics import BaseMetric
from deepeval.metrics.tool_correctness.schema import ToolSelectionScore
from deepeval.templates import make_template_class


ToolCorrectnessTemplate = make_template_class("ToolCorrectnessMetric")

TOOL_SELECTION_LEVELS = [
    "Misaligned",
    "Poor selection",
    "Mixed selection",
    "Mostly appropriate",
    "Fully appropriate",
]


class ToolCorrectnessMetric(BaseMetric):

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.TOOLS_CALLED,
        SingleTurnParams.EXPECTED_TOOLS,
    ]

    def __init__(
        self,
        available_tools: List[ToolCall] = None,
        threshold: Optional[float] = 0.5,
        evaluation_params: List[ToolCallParams] = [],
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        eval_mode: Optional[EvalModeName] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        should_exact_match: bool = False,
        should_consider_ordering: bool = False,
        flaky: bool = False,
        evaluation_template: Type[
            ToolCorrectnessTemplate
        ] = ToolCorrectnessTemplate,
    ):
        self.available_tools = available_tools
        self.threshold = 1 if strict_mode else threshold
        self.eval_mode = resolve_eval_mode(eval_mode)
        self.model, self.using_native_model = initialize_model(
            model, self.eval_mode
        )
        self.system_one_model = initialize_system_one_model(
            system_one_model, self.eval_mode
        )
        self.evaluation_model = (
            self.model or self.system_one_model
        ).get_model_name()
        self.async_mode = async_mode
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.evaluation_params: List[ToolCallParams] = evaluation_params
        self.should_exact_match = should_exact_match
        self.should_consider_ordering = should_consider_ordering
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:

        prepare_measure(self, test_case)
        self.test_case = test_case

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
                    )
                )
            else:
                self.tools_called: List[ToolCall] = test_case.tools_called
                self.expected_tools: List[ToolCall] = test_case.expected_tools
                tool_calling_score = self._calculate_score()
                if self.available_tools and not test_case.multimodal:
                    tool_selection_score = self._get_tool_selection_score(
                        test_case.input,
                        test_case.tools_called,
                        self.available_tools,
                        multimodal=test_case.multimodal,
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
                self.reason = (
                    self._construct_final_reason(
                        self._generate_reason(), tool_selection_score.reason
                    )
                    if self.include_reason
                    else None
                )
                self.success = self.is_successful()

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

                return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case)
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.tools_called: List[ToolCall] = test_case.tools_called
            self.expected_tools: List[ToolCall] = test_case.expected_tools
            tool_calling_score = self._calculate_score()
            if self.available_tools and not test_case.multimodal:
                tool_selection_score = await self._a_get_tool_selection_score(
                    test_case.input,
                    test_case.tools_called,
                    self.available_tools,
                    multimodal=test_case.multimodal,
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
            self.reason = (
                self._construct_final_reason(
                    self._generate_reason(), tool_selection_score.reason
                )
                if self.include_reason
                else None
            )
            self.success = self.is_successful()

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

            return self.score

    ##################################################
    ### Tool Correctness (Tool) ######################
    ##################################################

    def _generate_reason(self):
        expected_names = [tool.name for tool in self.expected_tools]
        called_names = [tool.name for tool in self.tools_called]
        summary = f"expected {expected_names}, called {called_names}"

        if self.should_exact_match:
            type_mismatches = self._get_type_mismatches()
            return (
                f"{'Exact match' if self._calculate_exact_match_score() else 'Not an exact match'}: {summary}."
                + (
                    f" Tool type mismatches: {type_mismatches}."
                    if type_mismatches
                    else ""
                )
                + " See details above."
            )
        if self._calculate_score() == 1:
            if self.should_consider_ordering:
                return f"Correct ordering: all expected tools {expected_names} were called in the correct order."
            return f"All expected tools {expected_names} were called (order not considered)."

        issues = self._issues() or ["tools did not fully match"]
        label = (
            "Incorrect tool usage"
            if self.should_consider_ordering
            else "Incomplete tool usage"
        )
        return (
            f"{label}: {'; '.join(issues)}; {summary}. See more details above."
        )

    def _issues(self) -> List[str]:
        missing, wrong_inputs, wrong_outputs, matched = [], [], [], []
        unpaired = list(self.tools_called)
        for expected_tool in self.expected_tools:
            same_name = [t for t in unpaired if t.name == expected_tool.name]
            if not same_name:
                missing.append(expected_tool.name)
                continue
            call = max(
                same_name, key=lambda t: self._match_score(expected_tool, t)
            )
            unpaired.remove(call)
            if self._input_parameters_score(expected_tool, call) < 1:
                wrong_inputs.append(expected_tool.name)
            if not self._output_matches(expected_tool, call):
                wrong_outputs.append(expected_tool.name)
            if self._match_score(expected_tool, call) == 1:
                matched.append(expected_tool.name)

        out_of_order = []
        if self.should_consider_ordering:
            in_order = [tool.name for tool in self._compute_weighted_lcs()[0]]
            for name in matched:
                if name in in_order:
                    in_order.remove(name)
                else:
                    out_of_order.append(name)

        issues = []
        for label, names in [
            ("missing tools", missing),
            ("out-of-order tools", out_of_order),
            ("different input parameters for", wrong_inputs),
            ("different output for", wrong_outputs),
            ("tool type mismatches", self._get_type_mismatches()),
        ]:
            if names:
                issues.append(f"{label} {names}")
        if not self.expected_tools and self.tools_called:
            called_names = [tool.name for tool in self.tools_called]
            issues.append(
                f"no tools were expected, but {called_names} were called"
            )
        return issues

    def _match_score(
        self, expected_tool: ToolCall, called_tool: ToolCall
    ) -> float:
        if (
            expected_tool.name != called_tool.name
            or expected_tool.type != called_tool.type
            or not self._output_matches(expected_tool, called_tool)
        ):
            return 0.0
        return self._input_parameters_score(expected_tool, called_tool)

    def _input_parameters_score(
        self, expected_tool: ToolCall, called_tool: ToolCall
    ) -> float:
        if (
            ToolCallParams.INPUT_PARAMETERS not in self.evaluation_params
            or expected_tool.input_parameters is None
        ):
            return 1.0
        return self._compare_dicts(
            expected_tool.input_parameters, called_tool.input_parameters or {}
        )

    def _output_matches(
        self, expected_tool: ToolCall, called_tool: ToolCall
    ) -> bool:
        return (
            ToolCallParams.OUTPUT not in self.evaluation_params
            or expected_tool.output is None
            or expected_tool.output == called_tool.output
        )

    def _get_type_mismatches(self) -> List[str]:
        mismatches = []
        for expected_tool in self.expected_tools:
            for called_tool in self.tools_called:
                if (
                    expected_tool.name == called_tool.name
                    and expected_tool.type != called_tool.type
                ):
                    mismatches.append(
                        f"{expected_tool.name} (expected {expected_tool.type.value}, called {called_tool.type.value})"
                    )
                    break
        return mismatches

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
        self, user_input, tools_called, available_tools, *, multimodal: bool
    ):
        value = system_one_score(
            self,
            self._system_one_tool_selection_spec(
                user_input, tools_called, available_tools, multimodal
            ),
        )
        if value is not None:
            return self._system_one_tool_selection_score(value)
        tools_called_formatted = print_tools_called(tools_called)
        available_tools_formatted = print_tools_called(available_tools)
        prompt = self._get_prompt(
            "get_tool_selection_score",
            user_input=user_input,
            tools_called=tools_called_formatted,
            available_tools=available_tools_formatted,
            multimodal=multimodal,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ToolSelectionScore,
            extract_schema=lambda s: s,
            extract_json=lambda data: ToolSelectionScore(**data),
        )

    async def _a_get_tool_selection_score(
        self, user_input, tools_called, available_tools, *, multimodal: bool
    ):
        value = await a_system_one_score(
            self,
            self._system_one_tool_selection_spec(
                user_input, tools_called, available_tools, multimodal
            ),
        )
        if value is not None:
            return self._system_one_tool_selection_score(value)
        tools_called_formatted = print_tools_called(tools_called)
        available_tools_formatted = print_tools_called(available_tools)
        prompt = self._get_prompt(
            "get_tool_selection_score",
            user_input=user_input,
            tools_called=tools_called_formatted,
            available_tools=available_tools_formatted,
            multimodal=multimodal,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ToolSelectionScore,
            extract_schema=lambda s: s,
            extract_json=lambda data: ToolSelectionScore(**data),
        )

    def _system_one_tool_selection_spec(
        self,
        user_input,
        tools_called: List[ToolCall],
        available_tools: List[ToolCall],
        multimodal: bool,
    ) -> Optional[SystemOneScoreSpec]:
        if multimodal:
            return None
        return SystemOneScoreSpec(
            instructions=self._get_prompt(
                "_experimental_system_one_tool_selection_score"
            ),
            levels=TOOL_SELECTION_LEVELS,
            state={
                "input": user_input,
                "tools_called": [
                    t.model_dump(mode="json", exclude_none=True)
                    for t in tools_called or []
                ],
                "available_tools": [
                    t.model_dump(mode="json", exclude_none=True)
                    for t in available_tools or []
                ],
            },
        )

    def _system_one_tool_selection_score(
        self, value: float
    ) -> ToolSelectionScore:
        return ToolSelectionScore(
            score=value,
            reason=format_decision_reason(self, "tool selection", value),
        )

    def _system_one_eval_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the score is decided in code by matching
        `tools_called` against `expected_tools`, so there is no whole-metric
        Jev request; the tool-selection check (when `available_tools` is
        given) is the metric's only Jev decision. See EXPERIMENTAL.md."""
        return None

    # Calculate score
    def _calculate_score(self) -> float:
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
    def _calculate_exact_match_score(self) -> float:
        if len(self.tools_called) != len(self.expected_tools):
            return 0.0
        if (
            len(self.tools_called) == len(self.expected_tools)
            and len(self.expected_tools) == 0
        ):
            return 1.0
        return float(
            all(
                self._match_score(expected_tool, called_tool) == 1
                for expected_tool, called_tool in zip(
                    self.expected_tools, self.tools_called
                )
            )
        )

    # Non exact matching score
    def _calculate_non_exact_match_score(self) -> float:
        total_score = 0.0
        used = set()
        for expected_tool in self.expected_tools:
            scores = [
                (self._match_score(expected_tool, called_tool), index)
                for index, called_tool in enumerate(self.tools_called)
                if index not in used
            ]
            best_score, best_index = max(scores, default=(0.0, None))
            if best_score > 0:
                total_score += best_score
                used.add(best_index)
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
    def _compute_weighted_lcs(self) -> Tuple[List[ToolCall], float]:
        m, n = len(self.expected_tools), len(self.tools_called)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                expected_tool, called_tool = (
                    self.expected_tools[i - 1],
                    self.tools_called[j - 1],
                )
                score = self._match_score(expected_tool, called_tool)
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

    @property
    def __name__(self):
        return "Tool Correctness"

    def indent_multiline_string(self, s, indent_level=4):
        indent = " " * indent_level
        return "\n".join(f"{indent}{line}" for line in s.splitlines())

from typing import List, Union, Dict

from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.utils import prettify_list, get_lcs
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
)
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
    ToolCallParams,
)
from deepeval.metrics import BaseMetric
import json

required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.TOOLS_CALLED,
    LLMTestCaseParams.EXPECTED_TOOLS,
]


class ToolCorrectnessMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        evaluation_param: ToolCallParams = ToolCallParams.TOOL,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        should_exact_match: bool = False,
        should_consider_ordering: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_param: ToolCallParams = evaluation_param
        self.should_exact_match = should_exact_match
        self.should_consider_ordering = should_consider_ordering

    def measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = test_case.turns[0]
        check_llm_test_case_params(test_case, required_params, self)
        self.test_case = test_case
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            expected_tools_formatted = (
                "Expected Tools:\n[\n"
                + ",\n".join(
                    self.indent_multiline_string(
                        repr(tool_call), indent_level=4
                    )
                    for tool_call in test_case.expected_tools
                )
                + "\n]"
            )
            tools_called_formatted = (
                "Tools Called:\n[\n"
                + ",\n".join(
                    self.indent_multiline_string(
                        repr(tool_call), indent_level=4
                    )
                    for tool_call in test_case.tools_called
                )
                + "\n]"
            )
            self.steps = [
                f"{expected_tools_formatted}",
                f"{tools_called_formatted}",
            ]
            if self.evaluation_param == ToolCallParams.TOOL:
                self.tools_called: List[str] = [
                    tool_call.name for tool_call in test_case.tools_called
                ]
                self.expected_tools: List[str] = [
                    tool_call.name for tool_call in test_case.expected_tools
                ]
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
            if self.evaluation_param == ToolCallParams.INPUT_PARAMETERS:
                self.input_parameters_called_list: List[str] = [
                    tool_call.input_parameters
                    for tool_call in test_case.tools_called
                ]
                self.expected_input_parameters_list: List[str] = [
                    tool_call.input_parameters
                    for tool_call in test_case.expected_tools
                ]
                self.score = self._calculate_input_parameter_correctness_score()
                self.reason = (
                    self._generate_input_parameter_correctness_reason()
                )
            if self.evaluation_param == ToolCallParams.OUTPUT:
                self.outputs: List[str] = [
                    tool_call.output for tool_call in test_case.tools_called
                ]
                self.expected_outputs: List[str] = [
                    tool_call.output for tool_call in test_case.expected_tools
                ]
                self.score = self._calculate_output_correctness_score()
                self.reason = self._generate_output_correctness_reason()

            self.success = self.score >= self.threshold
            self.steps.append(f"Score: {self.score}\nReason: {self.reason}")
            self.verbose_logs = construct_verbose_logs(self, steps=self.steps)
            return self.score

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ) -> float:
        return self.measure(test_case, _show_indicator=_show_indicator)

    ##################################################
    ### Tool Correctness (Tool) ######################
    ##################################################

    def _generate_reason(self):
        if self.should_exact_match:
            return f"{'Exact match' if self.tools_called == self.expected_tools else 'Not an exact match'}: expected {self.expected_tools}, called {self.tools_called}."

        elif self.should_consider_ordering:
            lcs = get_lcs(self.expected_tools, self.tools_called)
            missing = set(self.expected_tools) - set(self.tools_called)
            out_of_order = set(self.expected_tools) - set(lcs)

            if len(lcs) == len(self.expected_tools):
                return f"Correct ordering: all expected tools {self.expected_tools} were called in the correct order."
            else:
                issues = []
                if missing:
                    issues.append(f"missing tools {list(missing)}")
                if out_of_order:
                    issues.append(f"out-of-order tools {list(out_of_order)}")
                return f"Incorrect tool usage: {' and '.join(issues)}; expected {self.expected_tools}, called {self.tools_called}."

        else:
            used_expected = set(self.tools_called).intersection(
                set(self.expected_tools)
            )
            missing = set(self.expected_tools) - used_expected

            if len(used_expected) == len(self.expected_tools):
                return f"All expected tools {self.expected_tools} were called (order not considered)."
            else:
                return f"Incomplete tool usage: missing tools {list(missing)}; expected {self.expected_tools}, called {self.tools_called}."

    def _calculate_score(self):
        if self.should_exact_match:
            return 1.0 if self.tools_called == self.expected_tools else 0.0

        elif self.should_consider_ordering:
            longest_common_subsequence = get_lcs(
                self.expected_tools, self.tools_called
            )
            score = len(longest_common_subsequence) / len(self.expected_tools)

        else:
            used_expected_tools = set(self.tools_called).intersection(
                set(self.expected_tools)
            )
            score = len(used_expected_tools) / len(self.expected_tools)

        return 0 if self.strict_mode and score < self.threshold else score

    ##################################################
    ### Tool Correctness (Input Params) ##############
    ##################################################

    def _calculate_input_parameter_correctness_score(self):
        total_score = 0
        used_expected_indices = set()
        for input_parameters_called in self.input_parameters_called_list:
            best_match_score = 0
            best_match_index = -1
            for idx, expected_input_parameters in enumerate(
                self.expected_input_parameters_list
            ):
                if idx in used_expected_indices:
                    continue
                match_score = self._compare_dicts(
                    input_parameters_called, expected_input_parameters
                )
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_index = idx
            total_score += best_match_score
            if best_match_index != -1:
                used_expected_indices.add(best_match_index)
        score = total_score / len(self.expected_input_parameters_list)
        return 0 if self.strict_mode and score < self.threshold else score

    def _compare_dicts(self, dict1: Dict, dict2: Dict):
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

    def _generate_input_parameter_correctness_reason(self):
        if self.should_exact_match:
            return f"{'Exact match' if self.input_parameters_called_list == self.expected_input_parameters_list else 'Not an exact match'}: expected {self.expected_input_parameters_list}, called {self.input_parameters_called_list}."
        else:
            if self.score == 1:
                return f"All expected tools were called with the correct parameters: {self.expected_input_parameters_list}."
            matched_parameters = []
            unmatched_parameters = []
            used_expected_indices = set()
            for called_tool in self.input_parameters_called_list:
                best_match = None
                best_score = 0
                for idx, expected_tool in enumerate(
                    self.expected_input_parameters_list
                ):
                    if idx in used_expected_indices:
                        continue
                    score = self._compare_dicts(called_tool, expected_tool)
                    if score > best_score:
                        best_score = score
                        best_match = expected_tool
                if best_match is not None:
                    used_expected_indices.add(
                        self.expected_input_parameters_list.index(best_match)
                    )
                    matched_parameters.append(
                        {
                            "Input Parameters Called": called_tool,
                            "Expected Input Parameters": best_match,
                            "Score": round(best_score, 2),
                        }
                    )
                else:
                    unmatched_parameters.append(
                        {
                            "Input Parameters Called": called_tool,
                            "Expected Input Parameters": None,
                            "Score": 0,
                        }
                    )
            self.steps.append(
                f"Matched Input Parameters:\n{json.dumps(matched_parameters, indent=4)}"
            )
            self.steps.append(
                f"Unmatched Input Parameters:\n{json.dumps(unmatched_parameters, indent=4)}"
            )
            if unmatched_parameters:
                return f"Some input parameters did not match: {unmatched_parameters}"
            return f"All input parameters matched the expected inputs."

    ##################################################
    ### Tool Correctness (Output) ####################
    ##################################################

    def _calculate_output_correctness_score(self):
        total_score = 0
        used_expected_indices = set()
        for output in self.outputs:
            best_match_score = 0
            best_match_index = -1
            for idx, expected_output in enumerate(self.expected_outputs):
                if idx in used_expected_indices:
                    continue
                if output == expected_output:
                    best_match_score = 1.0
                    best_match_index = idx
                    break
            total_score += best_match_score
            if best_match_index != -1:
                used_expected_indices.add(best_match_index)
        score = total_score / len(self.expected_outputs)
        return 0 if self.strict_mode and score < self.threshold else score

    def _generate_output_correctness_reason(self):
        matched_outputs = []
        unmatched_outputs = []
        used_expected_indices = set()
        for tools_called_idx, output in enumerate(self.outputs):
            best_match = None
            for idx, expected_output in enumerate(self.expected_outputs):
                if idx in used_expected_indices:
                    continue
                if output == expected_output:
                    best_match = expected_output
                    used_expected_indices.add(idx)
                    break
            if best_match:
                matched_outputs.append(
                    {
                        "Tool Name": self.test_case.tools_called[
                            tools_called_idx
                        ].name,
                        "Called Output": output,
                        "Expected Output": expected_output,
                    }
                )
            else:
                unmatched_outputs.append(
                    {
                        "Tool Name": self.test_case.tools_called[
                            tools_called_idx
                        ].name,
                        "Called Output": output,
                        "Expected Output": None,
                    }
                )
        self.steps.append(
            f"Matched Outputs:\n{json.dumps(matched_outputs, indent=4)}"
        )
        self.steps.append(
            f"Unmatched Outputs:\n{json.dumps(unmatched_outputs, indent=4)}"
        )
        if unmatched_outputs:
            return f"Some outputs did not match: {unmatched_outputs}"
        return f"All outputs matched the expected outputs."

    ##################################################
    ### Others #######################################
    ##################################################

    def is_successful(self) -> bool:
        try:
            self.success = self.score >= self.threshold
        except:
            self.success = False
        return self.success

    @property
    def __name__(self):
        if self.evaluation_param == ToolCallParams.INPUT_PARAMETERS:
            return "Tool Correctness (Input Parameter)"
        elif self.evaluation_param == ToolCallParams.OUTPUT:
            return "Tool Correctness (Output)"
        else:
            return "Tool Correctness"

    def indent_multiline_string(self, s, indent_level=4):
        indent = " " * indent_level
        return "\n".join(f"{indent}{line}" for line in s.splitlines())

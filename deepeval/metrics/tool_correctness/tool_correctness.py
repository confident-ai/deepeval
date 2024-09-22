from typing import List, Union

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
)
from deepeval.metrics import BaseMetric

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

        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            self.tools_called: List[str] = test_case.tools_called
            self.expected_tools: List[str] = test_case.expected_tools
            self.score = self._calculate_score()
            self.reason = self._generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Expected Tools:\n{prettify_list(self.expected_tools)}",
                    f"Tools Called:\n{prettify_list(self.tools_called)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ) -> float:
        return self.measure(test_case, _show_indicator=_show_indicator)

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

    def is_successful(self) -> bool:
        try:
            self.success = self.score >= self.threshold
        except:
            self.success = False
        return self.success

    @property
    def __name__(self):
        return "Tool Correctness"

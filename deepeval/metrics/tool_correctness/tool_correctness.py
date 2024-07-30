from typing import Set, List, Union

from deepeval.utils import prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    validate_conversational_test_case,
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
    LLMTestCaseParams.TOOLS_USED,
    LLMTestCaseParams.EXPECTED_TOOLS,
]


class ToolCorrectnessMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def measure(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = validate_conversational_test_case(test_case, self)
        check_llm_test_case_params(test_case, required_params, self)

        self.tools_used: Set[str] = set(test_case.tools_used)
        self.expected_tools: Set[str] = set(test_case.expected_tools)
        self.expected_tools_used = self.tools_used.intersection(
            self.expected_tools
        )
        self.score = self._calculate_score()
        self.reason = self._generate_reason()
        self.success = self.score >= self.threshold
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Expected Tools:\n{prettify_list(list(self.expected_tools))}",
                f"Expected Tools Used:\n{prettify_list(list(self.expected_tools_used))}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )
        return self.score

    async def a_measure(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> float:
        return self.measure(test_case)

    def _generate_reason(self):
        reason = f"The score is {self.score} because {len(self.expected_tools_used)} out of {len(self.expected_tools)} expected tools were used. "
        tools_unused = list(self.expected_tools - self.expected_tools_used)
        if len(tools_unused) > 0:
            reason += f""
            reason += (
                f"Tool {tools_unused} was "
                if len(tools_unused) == 1
                else f"Tools {tools_unused} were "
            )
            reason += "expected but not used."

        return reason

    def _calculate_score(self):
        number_of_expected_tools_used = len(self.expected_tools_used)
        number_of_expected_tools = len(self.expected_tools)
        score = number_of_expected_tools_used / number_of_expected_tools
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

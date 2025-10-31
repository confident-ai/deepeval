import re
from typing import List
from collections import Counter

from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.utils import prettify_list
from deepeval.test_case import LLMTestCase
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
    print_tools_called,
)
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
)
from deepeval.metrics.api import metric_data_manager
from deepeval.metrics import BaseMetric

class ExactMatchMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]

    def __init__(
        self,
        compute_f1: bool = True,
        threshold: float = 0.5,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.compute_f1 = compute_f1

    def measure(
        self, 
        test_case: LLMTestCase, 
        _show_indicator: bool = True, 
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_llm_test_case_params(test_case, self._required_params, self)

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            expected = test_case.expected_output.strip()
            actual = test_case.actual_output.strip()

            if expected == actual:
                self.score = 1.0
                self.reason = "The actual output and expected output are exact matches"
                self.precision, self.recall, self.f1 = 1.0, 1.0, 1.0
            elif self.compute_f1:
                self.precision, self.recall, self.f1 = self._compute_precision_recall_f1(expected, actual)
                self.score = self.f1
                self.reason = (
                    f"The actual output and expected output are not exact matches, the other scores are \nPrecision={self.precision:.2f}, \nRecall={self.recall:.2f}, \nF1={self.f1:.2f}"
                )
            else:
                self.score = 0
                self.precision, self.recall, self.f1 = 0.0, 0.0, 0.0
                self.reason = "The actual output and expected output are not exact matches, the other score were opted out of calculation."

            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Precision Score:\n{self.precision}",
                    f"Recall Score:\n{self.recall}",
                    f"F1 Score:\n{self.f1}",
                    f"Final Score:\n{self.score}",
                    f"Reason: {self.reason}",
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
        _in_component: bool = False
    ) -> float:
        return self.measure(
            test_case, 
            _show_indicator=_show_indicator,
            _in_component=_in_component
        )
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _compute_precision_recall_f1(self, expected: str, actual: str):
        expected_tokens = self._tokenize(expected)
        actual_tokens = self._tokenize(actual)

        expected_counts = Counter(expected_tokens)
        actual_counts = Counter(actual_tokens)

        overlap = sum((expected_counts & actual_counts).values())

        precision = overlap / len(actual_tokens) if actual_tokens else 0
        recall = overlap / len(expected_tokens) if expected_tokens else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return precision, recall, f1

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Exact Match"

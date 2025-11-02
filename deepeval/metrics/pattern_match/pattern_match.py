import re
from typing import List

from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.metrics.api import metric_data_manager
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


class PatternMatchMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        pattern: str,
        ignore_case: bool = False,
        threshold: float = 1.0,
        verbose_mode: bool = False,
    ):
        self.pattern = pattern.strip()
        self.ignore_case = ignore_case
        self.verbose_mode = verbose_mode
        self.threshold = threshold

        flags = re.IGNORECASE if ignore_case else 0
        try:
            self._compiled_pattern = re.compile(self.pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {pattern} — {e}")

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
            actual = test_case.actual_output.strip()
            full_match = self._compiled_pattern.fullmatch(actual)

            self.score = 1.0 if full_match else 0.0
            self.reason = (
                f"The actual output fully matches the pattern."
                if full_match
                else f"The actual output does not match the pattern."
            )
            self.success = self.score >= self.threshold

            if self.verbose_mode:
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Pattern: {self.pattern}",
                        f"Actual: {actual}",
                        f"Score: {self.score:.2f}",
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
        _in_component: bool = False,
    ) -> float:
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

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
        return "Pattern Match"

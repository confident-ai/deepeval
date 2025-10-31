import re
from typing import List, Tuple
from collections import Counter

from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.metrics.api import metric_data_manager
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

_TOKENIZER = re.compile(r"\w+")

class F1Metric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        verbose_mode: bool = False,
    ):
        self.threshold = threshold
        self.verbose_mode = verbose_mode

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

            self.score, expected_tokens, actual_tokens, matched_tokens = self._compute_f1(expected, actual)
            self.reason = (
                f"Expected Output tokens: {expected_tokens} "
                f"Actual Output tokens: {actual_tokens} "
                f"Matched tokens: {matched_tokens} "
                f"Precision = num of matched tokens / num of actual tokens "
                f"Recall = num of matched tokens / num of expected tokens "
                f"F1 = 2 * recall * precision / (recall + precision) "
            )

            self.success = self.score >= self.threshold

            if self.verbose_mode:
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"F1 Score: {self.score:.2f}",
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

    def _tokenize(self, text: str) -> List[str]:
        return _TOKENIZER.findall(text.lower())

    def _compute_f1(
        self, expected: str, actual: str
    ):
        expected_tokens = self._tokenize(expected)
        actual_tokens = self._tokenize(actual)

        if not expected_tokens and not actual_tokens:
            return 0.0, [], [], []

        expected_counts = Counter(expected_tokens)
        actual_counts = Counter(actual_tokens)
        overlap = sum((expected_counts & actual_counts).values())
        overlap_tokens = list((expected_counts & actual_counts).elements())

        len_expected, len_actual = len(expected_tokens), len(actual_tokens)
        if len_expected == 0 or len_actual == 0:
            return 0.0, [], [], []

        precision = overlap / len_actual
        recall = overlap / len_expected
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )

        return f1, expected_tokens, actual_tokens, overlap_tokens

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
        return "F1"

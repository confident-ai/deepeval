import re
from typing import List, Optional

from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams


class PatternMatchMetric(BaseMetric):
    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        pattern: str,
        ignore_case: bool = False,
        regex: bool = True,
        threshold: Optional[float] = 1.0,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if not isinstance(pattern, str):
            raise TypeError(
                f"`pattern` must be a string for the 'Pattern Match' metric, "
                f"got {type(pattern).__name__}."
            )
        self.pattern = pattern.strip()
        self.ignore_case = ignore_case
        self.regex = regex
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.threshold = threshold

        flags = re.IGNORECASE if ignore_case else 0
        # In literal mode (`regex=False`) every regex metacharacter is inert,
        # so `re.escape` turns e.g. "C++" into "C\+\+" and the pattern matches
        # plain text exactly. `ignore_case` still applies.
        compiled_pattern = self.pattern if regex else re.escape(self.pattern)
        try:
            self._compiled_pattern = re.compile(compiled_pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {pattern} — {e}")

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            None,
            test_case.multimodal,
        )

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            actual = test_case.actual_output.strip()
            full_match = self._compiled_pattern.fullmatch(actual)

            self.score = 1.0 if full_match else 0.0
            self.reason = (
                "The actual output fully matches the pattern."
                if full_match
                else "The actual output does not match the pattern."
            )
            self.success = self.is_successful()

            if self.verbose_mode:
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Pattern: {self.pattern}",
                        f"Mode: {'regex' if self.regex else 'literal (regex disabled)'}",
                        f"Actual: {actual}",
                        f"Score: {self.score:.2f}",
                        f"Reason: {self.reason}",
                    ],
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

    @property
    def __name__(self):
        return "Pattern Match"

import re
from typing import List, Literal, Optional

from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams

PatternMatchMode = Literal["search", "match", "fullmatch"]


class PatternMatchMetric(BaseMetric):
    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        pattern: str,
        ignore_case: bool = False,
        match_mode: PatternMatchMode = "fullmatch",
        threshold: Optional[float] = 1.0,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.pattern = pattern.strip()
        self.ignore_case = ignore_case
        self.match_mode = self._validate_match_mode(match_mode)
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.threshold = threshold

        if len(self.pattern) == 0:
            # An empty pattern compiles fine but `fullmatch("")` / `search("")`
            # matches *everything*, so it silently turns every test case into a
            # pass. Fail loudly instead of making the metric useless.
            raise ValueError(
                "`pattern` cannot be empty for the 'Pattern Match' metric."
            )

        flags = re.IGNORECASE if ignore_case else 0
        try:
            self._compiled_pattern = re.compile(self.pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {pattern} — {e}") from e

    @staticmethod
    def _validate_match_mode(match_mode: str) -> PatternMatchMode:
        if match_mode not in ("search", "match", "fullmatch"):
            raise ValueError(
                f"`match_mode` must be one of 'search', 'match', or "
                f"'fullmatch', got {match_mode!r}."
            )
        return match_mode  # type: ignore[return-value]

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

            if self.match_mode == "fullmatch":
                matched = self._compiled_pattern.fullmatch(actual)
            elif self.match_mode == "match":
                # `match` anchors at the start of the string only (like
                # ``re.match``); the tail may differ.
                matched = self._compiled_pattern.match(actual)
            else:  # "search"
                # `search` scans the whole string for the first location
                # where the pattern produces a match.
                matched = self._compiled_pattern.search(actual)

            self.score = 1.0 if matched else 0.0
            self.reason = f"The actual output {'fully matches' if matched else 'does not match'} the pattern."
            self.success = self.is_successful()

            if self.verbose_mode:
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Pattern: {self.pattern}",
                        f"Match mode: {self.match_mode}",
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

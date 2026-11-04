from typing import List, Optional

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

_UNITS = ("words", "chars")


class OutputLengthBudgetMetric(BaseMetric):
    """Is the response within an allowed length range?

    Counts the words (whitespace-separated) or characters of
    ``actual_output`` and passes when the count sits between
    ``min_length`` and ``max_length``. Below the minimum the score is
    ``count / min``; above the maximum it is ``max / count``; inside the
    range it is ``1.0``. Fully **deterministic**: no LLM, no API key,
    zero token cost. Use it to keep verbose answers from driving up
    cost, or one-word answers from slipping through.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        unit: str = "words",
        threshold: Optional[float] = 1.0,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if unit not in _UNITS:
            raise ValueError(
                f"Unknown unit '{unit}'. Supported: {list(_UNITS)}."
            )
        if min_length is None and max_length is None:
            raise ValueError(
                "OutputLengthBudgetMetric requires at least one of "
                "`min_length` or `max_length`."
            )
        if min_length is not None and min_length < 0:
            raise ValueError("`min_length` cannot be negative.")
        if max_length is not None and max_length <= 0:
            raise ValueError("`max_length` must be positive.")
        if (
            min_length is not None
            and max_length is not None
            and min_length > max_length
        ):
            raise ValueError("`min_length` cannot exceed `max_length`.")
        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit
        self.threshold = 1.0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        # Deterministic metric: no evaluation model is used.
        self.model = None
        self.using_native_model = False
        self.async_mode = False

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
        self.test_case = test_case
        with metric_progress_indicator(
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            length = self._length(test_case.actual_output)
            score = self._score(length)
            self.score_breakdown = {
                "length": float(length),
                "min_length": (
                    float(self.min_length)
                    if self.min_length is not None
                    else 0.0
                ),
                "max_length": (
                    float(self.max_length)
                    if self.max_length is not None
                    else 0.0
                ),
            }
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(length)
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Unit: {self.unit}",
                    f"Length: {length}",
                    f"Allowed: [{self.min_length}, {self.max_length}]",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        # Deterministic metric — no async work to do; reuse the sync path.
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    def _length(self, text: str) -> int:
        if self.unit == "chars":
            return len(text)
        return len(text.split())

    def _score(self, length: int) -> float:
        if self.min_length is not None and length < self.min_length:
            return length / self.min_length if self.min_length else 1.0
        if self.max_length is not None and length > self.max_length:
            return self.max_length / length if length else 0.0
        return 1.0

    def _generate_reason(self, length: int) -> Optional[str]:
        if not self.include_reason:
            return None
        if self._score(length) == 1.0:
            return (
                f"Output length {length} {self.unit} is within "
                f"[{self.min_length}, {self.max_length}]."
            )
        return (
            f"Output length {length} {self.unit} is outside "
            f"[{self.min_length}, {self.max_length}]."
        )

    @property
    def __name__(self):
        return "Output Length Budget"

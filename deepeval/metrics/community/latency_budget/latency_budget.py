from typing import List, Optional

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams


class LatencyBudgetMetric(BaseMetric):
    """Did the system under test finish within a latency SLO?

    Scores ``LLMTestCase.completion_time`` (seconds) against a
    ``max_completion_time`` budget. This metric is fully **deterministic**
    and requires no LLM, so it is cheap and reliable to run as a CI gate
    for production latency SLOs.

    Score:

    - ``1.0`` when ``completion_time <= max_completion_time``
    - otherwise ``max(0, max_completion_time / completion_time)``
      (e.g. twice the budget yields ``0.5``)
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.COMPLETION_TIME,
    ]

    def __init__(
        self,
        max_completion_time: float,
        threshold: Optional[float] = 1.0,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if max_completion_time <= 0:
            raise ValueError(
                "LatencyBudgetMetric requires `max_completion_time` "
                "to be greater than 0 (seconds)."
            )
        self.max_completion_time = max_completion_time
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
            test_case, self._required_params, None, None, self
        )
        self.test_case = test_case
        with metric_progress_indicator(
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            completion_time = test_case.completion_time
            if completion_time < 0:
                raise ValueError(
                    "`completion_time` cannot be negative for the "
                    f"'{self.__name__}' metric"
                )

            score = self._score(completion_time)
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(completion_time)
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Max completion time: {self.max_completion_time}s",
                    f"Observed completion time: {completion_time}s",
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

    def _score(self, completion_time: float) -> float:
        if completion_time <= self.max_completion_time:
            return 1.0
        return max(0.0, self.max_completion_time / completion_time)

    def _generate_reason(self, completion_time: float) -> Optional[str]:
        if not self.include_reason:
            return None
        if completion_time <= self.max_completion_time:
            return (
                f"Completion time {completion_time}s is within the "
                f"{self.max_completion_time}s latency budget."
            )
        return (
            f"Completion time {completion_time}s exceeds the "
            f"{self.max_completion_time}s latency budget "
            f"({completion_time / self.max_completion_time:.2f}x over)."
        )

    @property
    def __name__(self):
        return "Latency Budget"

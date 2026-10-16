from typing import List, Optional

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams


class CostBudgetMetric(BaseMetric):
    """Did the system under test stay within a per-request cost budget?

    Scores ``LLMTestCase.token_cost`` (the cost of the application's own LLM
    calls, not the evaluation judge) against a ``max_token_cost`` budget.
    This metric is fully **deterministic** and requires no LLM, so it is
    cheap and reliable to run as a CI gate for production cost budgets.

    Score:

    - ``1.0`` when ``token_cost <= max_token_cost``
    - otherwise ``max(0, max_token_cost / token_cost)``
      (e.g. twice the budget yields ``0.5``)
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.TOKEN_COST,
    ]

    def __init__(
        self,
        max_token_cost: float,
        threshold: Optional[float] = 1.0,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if max_token_cost <= 0:
            raise ValueError(
                "CostBudgetMetric requires `max_token_cost` to be greater "
                "than 0."
            )
        self.max_token_cost = max_token_cost
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
            token_cost = test_case.token_cost
            if token_cost < 0:
                raise ValueError(
                    "`token_cost` cannot be negative for the "
                    f"'{self.__name__}' metric"
                )

            score = self._score(token_cost)
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(token_cost)
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Max token cost: {self.max_token_cost}",
                    f"Observed token cost: {token_cost}",
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

    def _score(self, token_cost: float) -> float:
        if token_cost <= self.max_token_cost:
            return 1.0
        return max(0.0, self.max_token_cost / token_cost)

    def _generate_reason(self, token_cost: float) -> Optional[str]:
        if not self.include_reason:
            return None
        if token_cost <= self.max_token_cost:
            return (
                f"Token cost {token_cost} is within the "
                f"{self.max_token_cost} cost budget."
            )
        return (
            f"Token cost {token_cost} exceeds the "
            f"{self.max_token_cost} cost budget "
            f"({token_cost / self.max_token_cost:.2f}x over)."
        )

    @property
    def __name__(self):
        return "Cost Budget"

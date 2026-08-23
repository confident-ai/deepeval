from abc import abstractmethod
from typing import Dict, Optional, Tuple

from deepeval.metrics.base_metric import BaseConversationalMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import ConversationalTestCase


VoiceMetricResult = Tuple[Optional[float], str, Dict]


class BaseVoiceMetric(BaseConversationalMetric):
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.threshold = 1.0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.async_mode = False
        self.evaluation_model = None

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> Optional[float]:
        check_conversational_test_case_params(test_case, [], self)
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            return self._run(test_case)

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> Optional[float]:
        check_conversational_test_case_params(test_case, [], self)
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            return self._run(test_case)

    def _run(self, test_case: ConversationalTestCase) -> Optional[float]:
        self.error = None
        score, reason, breakdown = self._evaluate(test_case)
        if (
            score is not None
            and self.strict_mode
            and score < (self.threshold or 1.0)
        ):
            score = 0.0
        self.score = score
        self.score_breakdown = breakdown
        self.skipped = score is None
        self.reason = reason if self.include_reason else None
        self.success = self.is_successful()
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Breakdown: {breakdown}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )
        return self.score

    @abstractmethod
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        raise NotImplementedError

from typing import List, Optional

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.voice.utils import summarize_turn_scores, TurnRecord
from deepeval.test_case import ConversationalTestCase, MultiTurnParams


class InterruptionMetric(BaseConversationalMetric):

    _required_test_case_params = [
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
    ]

    def __init__(
        self,
        threshold: float = 1.0,
        include_reason: bool = True,
        verbose_mode: bool = False,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.verbose_mode = verbose_mode

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_conversational_test_case_params(
            test_case, self._required_test_case_params, self
        )
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            self.error = None
            records: List[TurnRecord] = []
            assistant_index = 0
            for turn in test_case.turns:
                if turn.role != "assistant":
                    continue
                assistant_index += 1
                if turn.interrupted is None:
                    continue
                interrupted = bool(turn.interrupted)
                records.append(
                    (
                        assistant_index,
                        1.0 if interrupted else 0.0,
                        not interrupted,
                    )
                )

            self.score, self.score_breakdown, reason = summarize_turn_scores(
                records,
                label="interruption",
                unit="",
                limit_description="not interrupted",
            )
            self.skipped = self.score is None
            self.reason = reason if self.include_reason else None
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Eligible assistant turns: "
                    f"{self.score_breakdown.get('eligible', 0)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    def is_successful(self) -> Optional[bool]:
        if self.error is not None:
            self.success = False
        elif self.score is None:
            self.success = None
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Interruption"

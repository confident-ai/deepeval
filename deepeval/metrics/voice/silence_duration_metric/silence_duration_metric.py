from typing import List, Optional

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.voice.utils import (
    summarize_turn_scores,
    compute_longest_silence_ms,
    TurnRecord,
)
from deepeval.test_case import ConversationalTestCase, MultiTurnParams


class SilenceDurationMetric(BaseConversationalMetric):

    _required_test_case_params = [
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
    ]

    def __init__(
        self,
        max_silence_ms: float = 1500.0,
        threshold: float = 1.0,
        include_reason: bool = True,
        verbose_mode: bool = False,
    ):
        self.max_silence_ms = max_silence_ms
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
                if turn.audio is None:
                    continue
                value = compute_longest_silence_ms(turn.audio)
                if value is None:
                    continue
                records.append(
                    (assistant_index, value, value <= self.max_silence_ms)
                )

            self.score, self.score_breakdown, reason = summarize_turn_scores(
                records,
                label="silence",
                unit=" ms",
                limit_description=f"<= {self.max_silence_ms:g} ms",
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
        return "Silence Duration"

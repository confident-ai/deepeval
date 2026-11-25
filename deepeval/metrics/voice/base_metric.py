import json
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from deepeval.metrics.base_metric import BaseConversationalMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import ConversationalTestCase, MultiTurnParams


VoiceMetricResult = Tuple[Optional[float], str, Dict]


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    return value


def _format_breakdown(breakdown: Dict) -> str:
    try:
        return json.dumps(_rounded(breakdown), indent=2, default=str)
    except (TypeError, ValueError):
        return str(breakdown)


def _place_reply_events(breakdown: Dict, test_case: ConversationalTestCase):
    """An event about a whole reply, such as its loudness or pitch, spans
    that reply's audio; give it the reply's place when the metric gave none."""
    # Imported here: the voice utils import this module.
    from deepeval.metrics.voice.utils.analysis import decode_audio
    from deepeval.metrics.voice.utils.events import placed

    events = breakdown.get("events") if isinstance(breakdown, dict) else None
    for event in events or []:
        turn = event.get("turn") if isinstance(event, dict) else None
        if turn is None or "offset_seconds" in event or "start_time" in event:
            continue
        if not 0 <= turn < len(test_case.turns):
            continue
        audio = test_case.turns[turn].audio
        if audio is None:
            continue
        try:
            samples, sample_rate = decode_audio(audio)
        except (TypeError, ValueError):
            continue
        event.update(
            {
                key: value
                for key, value in placed(
                    audio, 0.0, len(samples) / sample_rate
                ).items()
                if value is not None
            }
        )


class BaseVoiceMetric(BaseConversationalMetric):
    _required_test_case_params: List[MultiTurnParams] = []

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
        _place_reply_events(breakdown, test_case)
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
                f"Breakdown:\n{_format_breakdown(breakdown)}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )
        return self.score

    @abstractmethod
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        raise NotImplementedError

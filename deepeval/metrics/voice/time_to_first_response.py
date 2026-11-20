from typing import Optional

from deepeval.metrics.voice._defects import (
    defect_share_result,
    untimed_call_reason,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class TimeToFirstResponseMetric(BaseVoiceMetric):
    # The 2 s latency limit plus 1 s for the line to connect, since the clock
    # starts when the call connects.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        max_wait_seconds: int = 3,
        flaky: bool = False,
    ):
        if max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be greater than 0.")
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.max_wait_seconds = max_wait_seconds

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        defect_label = (
            f"a first response over {self.max_wait_seconds} seconds after "
            "the call connected"
        )
        untimed_reason = untimed_call_reason(test_case)
        audio_turns = (
            []
            if untimed_reason
            else sorted(
                (turn.audio.start_time, index)
                for index, turn in enumerate(test_case.turns)
                if turn.audio is not None
            )
        )
        if not audio_turns:
            return defect_share_result(
                [],
                [],
                [],
                defect_label,
                no_eligible_reason=untimed_reason
                or "The call had no timed audio.",
            )
        greeting_at, greeting_turn = audio_turns[0]
        if test_case.turns[greeting_turn].role != "assistant":
            return defect_share_result(
                [],
                [],
                [],
                defect_label,
                no_eligible_reason="The caller spoke first, so the agent was "
                "not expected to greet.",
            )
        defects = []
        if greeting_at > self.max_wait_seconds:
            defects.append(
                {
                    "type": "slow_first_response",
                    "turn": greeting_turn,
                    "first_response_seconds": round(greeting_at, 2),
                }
            )
        return defect_share_result([greeting_turn], defects, [], defect_label)

    @property
    def __name__(self):
        return "Time to First Response"

from deepeval.metrics.voice.audio_defects.echo.utils import (
    ECHO_MIN_STRENGTH,
    echo_delay_and_strength,
)
from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent


class EchoMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        measurements, skipped = agent_reply_measurements(test_case)
        defects = []
        for turn, reply in measurements.items():
            # A replayed buffer also repeats at a fixed delay; Loop owns it.
            if reply.loop_events:
                continue
            delay, strength = echo_delay_and_strength(
                test_case.turns[turn].audio
            )
            if strength >= ECHO_MIN_STRENGTH:
                defects.append(
                    VoiceEvent(
                        type="echo",
                        turn=turn,
                        delay_seconds=round(delay, 3),
                        strength=round(strength, 4),
                    )
                )
        return defect_share_result(
            list(measurements), defects, skipped, "an echo"
        )

    @property
    def __name__(self):
        return "Echo"

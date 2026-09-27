from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.metrics.voice._detectors import (
    detect_audio_integrity,
    detect_responsiveness,
)
from deepeval.test_case import ConversationalTestCase


class VoiceReliabilityMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        responsiveness = detect_responsiveness(test_case)
        integrity = detect_audio_integrity(test_case)
        critical = responsiveness.critical or integrity.critical
        score = (
            0.0
            if critical
            else 0.5 * responsiveness.score + 0.5 * integrity.score
        )
        if critical:
            reason = (
                "Voice reliability was 0 because at least one critical "
                "responsiveness or audio-integrity failure occurred."
            )
        else:
            reason = (
                f"Voice reliability was {score:.2f} "
                f"(responsiveness={responsiveness.score:.2f}, "
                f"audio integrity={integrity.score:.2f})."
            )
        return (
            score,
            reason,
            {
                "critical_failure": critical,
                "responsiveness": {
                    "score": responsiveness.score,
                    "critical_failure": responsiveness.critical,
                    "events": responsiveness.events,
                },
                "audio_integrity": {
                    "score": integrity.score,
                    "critical_failure": integrity.critical,
                    "events": integrity.events,
                },
            },
        )

    @property
    def __name__(self):
        return "Voice Reliability"

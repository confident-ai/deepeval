from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.metrics.voice._detectors import detect_audio_integrity
from deepeval.test_case import ConversationalTestCase


class AudioIntegrityMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        report = detect_audio_integrity(test_case)
        if not report.events:
            reason = "No audio-integrity failures were detected."
        elif report.critical:
            reason = (
                "A critical audio-integrity failure occurred: "
                + ", ".join(event["type"] for event in report.events)
                + "."
            )
        else:
            reason = (
                f"Audio integrity was {report.score:.2f}; "
                f"{len(report.events)} defect event(s) were detected."
            )
        return (
            report.score,
            reason,
            {
                "critical_failure": report.critical,
                "events": report.events,
            },
        )

    @property
    def __name__(self):
        return "Audio Integrity"

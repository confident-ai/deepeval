from deepeval.metrics.voice.utils.events import events_to_dicts
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.metrics.voice.utils.detectors import detect_responsiveness
from deepeval.test_case import ConversationalTestCase


class AgentResponsivenessMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        report = detect_responsiveness(test_case)
        if not report.events:
            reason = "The agent responded without requiring a reprompt."
        elif report.critical:
            reason = (
                "A critical responsiveness failure occurred: "
                + ", ".join(event.type for event in report.events)
                + "."
            )
        else:
            reason = (
                f"Responsiveness was {report.score:.2f}; "
                f"{len(report.events)} reprompt event(s) were detected."
            )
        return (
            report.score,
            reason,
            {
                "critical_failure": report.critical,
                "events": events_to_dicts(report.events),
            },
        )

    @property
    def __name__(self):
        return "Agent Responsiveness"

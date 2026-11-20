from deepeval.metrics.voice._defects import (
    AGENT_OVERLAP_ALLOWANCE_SECONDS,
    agent_reply_gaps,
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class AgentInterruptionMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        gaps, skipped, untimed_reason = agent_reply_gaps(test_case)
        defects = [
            {
                "type": "agent_overlap",
                "turn": turn,
                "overlap_seconds": round(-gap, 2),
            }
            for turn, gap in gaps.items()
            if gap < -AGENT_OVERLAP_ALLOWANCE_SECONDS
        ]
        return defect_share_result(
            list(gaps),
            defects,
            skipped,
            "the agent talking over the caller",
            no_eligible_reason=untimed_reason
            or "No assistant reply followed a timed caller turn.",
        )

    @property
    def __name__(self):
        return "Agent Interruption"

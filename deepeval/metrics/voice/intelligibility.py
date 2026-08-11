from statistics import mean
from typing import Dict, List

from deepeval.metrics.voice._analysis import (
    analyze_audio,
    intelligibility_score,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class SpeechIntelligibilityMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        scores: List[float] = []
        turns: List[Dict] = []
        assistant_index = 0
        for turn in test_case.turns:
            if turn.role != "assistant":
                continue
            assistant_index += 1
            if turn.audio is None:
                continue
            try:
                measurements = analyze_audio(turn.audio)
            except (TypeError, ValueError) as error:
                turns.append({"turn": assistant_index, "error": str(error)})
                continue
            score = intelligibility_score(measurements)
            scores.append(score)
            turns.append(
                {
                    "turn": assistant_index,
                    "score": score,
                    "estimated_snr_db": measurements.estimated_snr_db,
                    "rms_dbfs": measurements.rms_dbfs,
                    "clipping_fraction": measurements.clipping_fraction,
                    "dropout_events": measurements.dropout_events,
                }
            )
        if not scores:
            return (
                None,
                "No decodable assistant audio was available to evaluate.",
                {"turns": turns},
            )
        score = mean(scores)
        return (
            score,
            f"Assistant speech intelligibility was {score:.2f} across "
            f"{len(scores)} eligible turn(s).",
            {"eligible_turns": len(scores), "turns": turns},
        )

    @property
    def __name__(self):
        return "Speech Intelligibility"

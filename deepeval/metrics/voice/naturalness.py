from statistics import mean
from typing import Dict, List

from deepeval.metrics.voice._analysis import (
    analyze_audio,
    naturalness_score,
    speaking_rate_wpm,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class VoiceNaturalnessMetric(BaseVoiceMetric):
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
                turns.append(
                    {
                        "turn": assistant_index,
                        "error": str(error),
                    }
                )
                continue
            wpm = speaking_rate_wpm(turn.content, measurements)
            score = naturalness_score(measurements, words_per_minute=wpm)
            scores.append(score)
            turns.append(
                {
                    "turn": assistant_index,
                    "score": score,
                    "speaking_rate_wpm": wpm,
                    "silence_fraction": measurements.silence_fraction,
                    "pitch_variation_hz": measurements.pitch_variation_hz,
                    "clipping_fraction": measurements.clipping_fraction,
                    "dropout_events": measurements.dropout_events,
                    "loop_events": measurements.loop_events,
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
            f"Assistant speech naturalness was {score:.2f} across "
            f"{len(scores)} eligible turn(s).",
            {"eligible_turns": len(scores), "turns": turns},
        )

    @property
    def __name__(self):
        return "Voice Naturalness"

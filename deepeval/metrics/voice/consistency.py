from statistics import mean, pstdev
from typing import Dict, List

from deepeval.metrics.voice._analysis import analyze_audio, clamp_score
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class VoiceConsistencyMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        records: List[Dict] = []
        for index, turn in enumerate(test_case.turns):
            if turn.role != "assistant" or turn.audio is None:
                continue
            try:
                measurements = analyze_audio(turn.audio)
            except (TypeError, ValueError):
                continue
            records.append(
                {
                    "turn": index,
                    "rms_dbfs": measurements.rms_dbfs,
                    "pitch_mean_hz": measurements.pitch_mean_hz,
                    "zero_crossing_rate": measurements.zero_crossing_rate,
                }
            )
        if len(records) < 2:
            return (
                None,
                "At least two decodable assistant audio turns are required "
                "to evaluate voice consistency.",
                {"eligible_turns": len(records), "turns": records},
            )

        loudness_values = [record["rms_dbfs"] for record in records]
        pitch_values = [
            record["pitch_mean_hz"]
            for record in records
            if record["pitch_mean_hz"] is not None
        ]
        zcr_values = [record["zero_crossing_rate"] for record in records]
        loudness_score = clamp_score(1.0 - pstdev(loudness_values) / 12.0)
        zcr_mean = max(mean(zcr_values), 1e-6)
        timbre_score = clamp_score(1.0 - pstdev(zcr_values) / zcr_mean)
        if len(pitch_values) > 1:
            pitch_mean = max(mean(pitch_values), 1e-6)
            pitch_score = clamp_score(1.0 - pstdev(pitch_values) / pitch_mean)
            score = (
                0.4 * pitch_score + 0.35 * loudness_score + 0.25 * timbre_score
            )
        else:
            pitch_score = None
            score = 0.6 * loudness_score + 0.4 * timbre_score
        breakdown = {
            "eligible_turns": len(records),
            "pitch_consistency": pitch_score,
            "loudness_consistency": loudness_score,
            "spectral_consistency": timbre_score,
            "turns": records,
        }
        return (
            score,
            f"Assistant voice consistency was {score:.2f} across "
            f"{len(records)} turns.",
            breakdown,
        )

    @property
    def __name__(self):
        return "Voice Consistency"

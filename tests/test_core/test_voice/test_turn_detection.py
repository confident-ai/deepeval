import pytest

from deepeval.voice.connectors.transports.callback import (
    CallbackVoiceConnector,
)
from deepeval.voice.turn_detection import turn_detection_timing


async def _agent(_audio):
    raise AssertionError("not called")


class TestTurnDetectionPresets:
    def test_balanced_is_the_default(self):
        assert CallbackVoiceConnector(_agent).turn_detection == "balanced"

    def test_patience_grows_with_the_level(self):
        silence = [
            turn_detection_timing(level).end_of_turn_silence_ms
            for level in ("eager", "balanced", "patient")
        ]
        assert silence == sorted(silence)

    def test_timeout_grows_with_the_level(self):
        timeouts = [
            turn_detection_timing(level).max_turn_timeout_s
            for level in ("eager", "balanced", "patient")
        ]
        assert timeouts == sorted(timeouts)

    def test_a_connector_resolves_the_preset_to_timings(self):
        connector = CallbackVoiceConnector(_agent, turn_detection="patient")
        patient = turn_detection_timing("patient")
        assert (
            connector.end_of_turn_silence_ms == patient.end_of_turn_silence_ms
        )
        assert connector.max_turn_timeout_s == patient.max_turn_timeout_s

    def test_an_unknown_level_names_the_ones_that_exist(self):
        with pytest.raises(ValueError, match="eager.*balanced.*patient"):
            CallbackVoiceConnector(_agent, turn_detection="snappy")

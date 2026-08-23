import math
from array import array

import pytest

from deepeval.metrics.voice import (
    AgentResponsivenessMetric,
    AudioIntegrityMetric,
    SpeechIntelligibilityMetric,
    TurnTakingNaturalnessMetric,
    VoiceConsistencyMetric,
    VoiceNaturalnessMetric,
    VoiceReliabilityMetric,
)
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from deepeval.voice.connectors import audio_utils
from deepeval.voice.timeline import render_timeline_wav


def make_tone(
    *,
    duration: float = 0.3,
    frequency: float = 180.0,
    amplitude: int = 6000,
    start_time: float | None = None,
    sample_rate: int = 8000,
) -> Audio:
    count = int(duration * sample_rate)
    fade = max(1, int(0.02 * sample_rate))
    samples = array("h")
    for index in range(count):
        envelope = min(1.0, index / fade, (count - index - 1) / fade)
        samples.append(
            int(
                amplitude
                * max(0.0, envelope)
                * math.sin(2 * math.pi * frequency * index / sample_rate)
            )
        )
    wav = audio_utils.pcm16_to_wav_bytes(
        samples.tobytes(), sample_rate, num_channels=1
    )
    return Audio.from_bytes(
        wav,
        "audio/wav",
        sampleRate=sample_rate,
        encoding="wav",
        duration=duration,
        start_time=start_time,
    )


def voice_case() -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="Can you help me?",
                audio=make_tone(start_time=0.0, frequency=160),
            ),
            Turn(
                role="assistant",
                content="Absolutely, what do you need?",
                audio=make_tone(start_time=0.5, frequency=190),
            ),
            Turn(
                role="user",
                content="Please check my order.",
                audio=make_tone(start_time=1.0, frequency=165),
            ),
            Turn(
                role="assistant",
                content="Your order is on the way.",
                audio=make_tone(start_time=1.5, frequency=192),
            ),
        ]
    )


def test_audio_start_time_is_preserved_on_turn_audio():
    audio = make_tone(start_time=1.25)
    turn = Turn(role="assistant", content="Hello", audio=audio)

    assert turn.audio.start_time == 1.25


def test_audio_rejects_negative_timeline_values():
    with pytest.raises(ValueError):
        make_tone(start_time=-0.1)


def test_quality_metrics_score_assistant_audio():
    test_case = voice_case()

    naturalness = VoiceNaturalnessMetric().measure(
        test_case, _show_indicator=False
    )
    intelligibility = SpeechIntelligibilityMetric().measure(
        test_case, _show_indicator=False
    )
    consistency = VoiceConsistencyMetric().measure(
        test_case, _show_indicator=False
    )

    assert naturalness is not None and 0 <= naturalness <= 1
    assert intelligibility is not None and 0 <= intelligibility <= 1
    assert consistency is not None and 0 <= consistency <= 1


@pytest.mark.asyncio
async def test_voice_metric_sync_async_parity():
    metric = SpeechIntelligibilityMetric()
    test_case = voice_case()

    sync_score = metric.measure(test_case, _show_indicator=False)
    async_score = await metric.a_measure(test_case, _show_indicator=False)

    assert async_score == sync_score


def test_turn_taking_uses_timestamped_audio_and_skips_without_it():
    test_case = voice_case()
    metric = TurnTakingNaturalnessMetric()

    score = metric.measure(test_case, _show_indicator=False)
    assert score is not None and 0 <= score <= 1
    assert metric.score_breakdown["transitions"]

    test_case.turns[0].audio.start_time = None
    assert metric.measure(test_case, _show_indicator=False) is None
    assert metric.skipped is True


def test_timeline_rendering_preserves_gap_and_overlap():
    turns = [
        Turn(role="user", content="One", audio=make_tone(start_time=0.0)),
        Turn(
            role="assistant",
            content="Two",
            audio=make_tone(start_time=0.2),
        ),
        Turn(role="user", content="Three", audio=make_tone(start_time=0.8)),
    ]

    combined = render_timeline_wav(turns)
    assert combined is not None
    pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(combined)
    assert channels == 1
    assert len(pcm) / 2 / rate == pytest.approx(1.1, abs=0.002)


def test_responsiveness_and_reliability_critical_failure_override():
    test_case = ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="Where is my order?",
                audio=make_tone(start_time=0.0),
            )
        ]
    )

    responsiveness = AgentResponsivenessMetric().measure(
        test_case, _show_indicator=False
    )
    reliability = VoiceReliabilityMetric().measure(
        test_case, _show_indicator=False
    )

    assert responsiveness == 0
    assert reliability == 0
    assert VoiceReliabilityMetric().threshold == 0.5


def test_audio_integrity_fails_undecodable_assistant_audio():
    test_case = ConversationalTestCase(
        turns=[
            Turn(
                role="assistant",
                content="Broken",
                audio=Audio.from_bytes(b"not-a-wav", "audio/wav"),
            )
        ]
    )
    metric = AudioIntegrityMetric()

    assert metric.measure(test_case, _show_indicator=False) == 0
    assert metric.score_breakdown["critical_failure"] is True

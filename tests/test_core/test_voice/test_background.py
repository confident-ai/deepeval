from array import array

import pytest

from deepeval.dataset import BackgroundNoiseSettings
from deepeval.test_case import Audio
from deepeval.voice.background import _load_pcm, mix_background
from deepeval.voice.connectors import audio_utils


def _wav(path, samples, sample_rate=16000, num_channels=1):
    pcm = array("h", samples).tobytes()
    wav = audio_utils.pcm16_to_wav_bytes(
        pcm, sample_rate=sample_rate, num_channels=num_channels
    )
    path.write_bytes(wav)
    return wav


@pytest.fixture(autouse=True)
def clear_pcm_cache():
    _load_pcm.cache_clear()
    yield
    _load_pcm.cache_clear()


def test_mix_background_loops_bed_and_scales_by_volume(tmp_path):
    bed_path = tmp_path / "cafe.wav"
    _wav(bed_path, [1000, -1000])
    speech = Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(
            array("h", [100, 100, 100, 100]).tobytes(), sample_rate=16000
        ),
        "audio/wav",
    )

    mixed = mix_background(
        speech, BackgroundNoiseSettings(audio=str(bed_path), volume=0.5)
    )

    pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(mixed.get_bytes())
    samples = array("h")
    samples.frombytes(pcm)
    assert (rate, channels) == (16000, 1)
    assert list(samples) == [600, -400, 600, -400]


def test_mix_background_resamples_bed_to_the_speech_rate(tmp_path):
    bed_path = tmp_path / "rain.wav"
    _wav(bed_path, [500] * 8, sample_rate=8000)
    speech = Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(
            array("h", [0] * 16).tobytes(), sample_rate=16000
        ),
        "audio/wav",
    )

    mixed = mix_background(
        speech, BackgroundNoiseSettings(audio=str(bed_path), volume=1.0)
    )

    pcm, rate, _ = audio_utils.wav_bytes_to_pcm16(mixed.get_bytes())
    samples = array("h")
    samples.frombytes(pcm)
    assert rate == 16000
    assert len(samples) == 16
    assert all(sample == 500 for sample in samples)


def test_mix_background_clips_instead_of_wrapping(tmp_path):
    bed_path = tmp_path / "loud.wav"
    _wav(bed_path, [32000, -32000])
    speech = Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(
            array("h", [32000, -32000]).tobytes(), sample_rate=16000
        ),
        "audio/wav",
    )

    mixed = mix_background(
        speech, BackgroundNoiseSettings(audio=str(bed_path), volume=1.0)
    )

    samples = array("h")
    samples.frombytes(audio_utils.wav_bytes_to_pcm16(mixed.get_bytes())[0])
    assert list(samples) == [32767, -32768]


def test_mix_background_is_a_no_op_without_settings_or_volume(tmp_path):
    bed_path = tmp_path / "cafe.wav"
    _wav(bed_path, [1000])
    speech = Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(
            array("h", [7]).tobytes(), sample_rate=16000
        ),
        "audio/wav",
    )

    assert mix_background(speech, None) is speech
    assert (
        mix_background(
            speech, BackgroundNoiseSettings(audio=str(bed_path), volume=0.0)
        )
        is speech
    )


def test_mix_background_keeps_the_speech_when_the_file_is_missing(tmp_path):
    speech = Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(
            array("h", [7]).tobytes(), sample_rate=16000
        ),
        "audio/wav",
    )

    mixed = mix_background(
        speech, BackgroundNoiseSettings(audio=str(tmp_path / "nope.wav"))
    )

    assert mixed is speech

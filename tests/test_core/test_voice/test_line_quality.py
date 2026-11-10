import math
from array import array

from deepeval.test_case import Audio, AudioChunk
from deepeval.voice.connectors import audio_utils
from deepeval.voice.line_quality import LineQualityFilter, apply_line_quality

RATE = 24000


def _tone(hz: float, seconds: float = 0.5, amplitude: int = 8000) -> bytes:
    return array(
        "h",
        (
            int(amplitude * math.sin(2 * math.pi * hz * i / RATE))
            for i in range(int(RATE * seconds))
        ),
    ).tobytes()


def _wav(pcm: bytes, channels: int = 1) -> Audio:
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(pcm, RATE, channels), "audio/wav"
    )


def _settled_rms(pcm: bytes) -> float:
    """RMS of the second half, after the filters' start-up transient."""
    samples = array("h")
    samples.frombytes(pcm)
    tail = samples[len(samples) // 2 :]
    return math.sqrt(sum(sample * sample for sample in tail) / len(tail))


def _phone(pcm: bytes) -> bytes:
    return LineQualityFilter("phone").filter_pcm(pcm, RATE)


def test_a_wideband_line_leaves_audio_untouched():
    audio = _wav(_tone(6000))

    assert apply_line_quality(audio, "wideband") is audio


def test_a_phone_line_keeps_the_speech_band():
    tone = _tone(1000)

    assert _settled_rms(_phone(tone)) > 0.9 * _settled_rms(tone)


def test_a_phone_line_cuts_below_and_above_the_speech_band():
    for hz in (100, 6000):
        tone = _tone(hz)

        assert _settled_rms(_phone(tone)) < 0.2 * _settled_rms(tone)


def test_filtering_frame_by_frame_matches_filtering_whole():
    pcm = _tone(1000) + _tone(250)
    line = LineQualityFilter("phone")
    frame_bytes = 2 * 480

    framed = b"".join(
        line.filter_chunk(
            AudioChunk.from_bytes(
                pcm[offset : offset + frame_bytes],
                "audio/pcm",
                sampleRate=RATE,
            )
        ).get_bytes()
        for offset in range(0, len(pcm), frame_bytes)
    )

    assert framed == _phone(pcm)


def test_silence_stays_silent():
    silence = b"\x00\x00" * 480

    assert _phone(silence) == silence


def test_a_stereo_clip_comes_back_mono_at_its_own_rate():
    stereo = array("h")
    for sample in array("h", _tone(1000)):
        stereo.extend((sample, sample))

    filtered = apply_line_quality(_wav(stereo.tobytes(), channels=2), "phone")

    _, rate, channels = audio_utils.wav_bytes_to_pcm16(filtered.get_bytes())
    assert (rate, channels) == (RATE, 1)


def test_audio_that_is_not_wav_is_left_alone():
    audio = Audio.from_bytes(b"ID3 not a wav", "audio/mpeg")

    assert apply_line_quality(audio, "phone") is audio

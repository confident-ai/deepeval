from deepeval.test_case import Audio, Turn
from deepeval.voice.connectors import audio_utils
from deepeval.voice.output import _concat_wav_turns


def test_concat_wav_turns_normalizes_formats_and_skips_missing_audio():
    mono_8khz = audio_utils.pcm16_to_wav_bytes(
        b"\x01\x00" * 80, sample_rate=8000, num_channels=1
    )
    stereo_16khz = audio_utils.pcm16_to_wav_bytes(
        b"\x01\x00" * 320, sample_rate=16000, num_channels=2
    )
    turns = [
        Turn(
            role="user",
            content="Hello",
            audio=Audio.from_bytes(mono_8khz, "audio/wav"),
        ),
        Turn(role="assistant", content="Missing audio is allowed"),
        Turn(
            role="user",
            content="Again",
            audio=Audio.from_bytes(stereo_16khz, "audio/wav"),
        ),
    ]

    combined = _concat_wav_turns(turns)

    assert combined is not None
    pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(combined)
    assert rate == 8000
    assert channels == 1
    assert len(pcm) == 320


def test_render_starts_at_the_first_clip_but_keeps_gaps_between_clips():
    """Setup time before the first word is trimmed; real gaps are kept."""
    clip = audio_utils.pcm16_to_wav_bytes(
        b"\x01\x00" * 8000, sample_rate=8000, num_channels=1
    )

    def timed_audio(start_time: float) -> Audio:
        return Audio.from_bytes(
            clip, "audio/wav", sampleRate=8000, start_time=start_time
        )

    turns = [
        Turn(role="user", content="Hi", audio=timed_audio(11.5)),
        Turn(role="assistant", content="Hello", audio=timed_audio(14.5)),
    ]

    combined = _concat_wav_turns(turns)

    assert combined is not None
    pcm, rate, _ = audio_utils.wav_bytes_to_pcm16(combined)
    # 11.5s of lead-in is gone: first clip (1s) + 2s gap + second clip (1s).
    assert len(pcm) / 2 / rate == 4.0

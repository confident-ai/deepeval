import io
import wave
from array import array
from math import sqrt
from typing import Iterator, Tuple

DEFAULT_FRAME_MS = 20
DEFAULT_SILENCE_RMS = 300.0


def wav_bytes_to_pcm16(wav_bytes: bytes) -> Tuple[bytes, int, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_width = wf.getsampwidth()
        if sample_width != 2:
            raise ValueError(
                f"Expected 16-bit PCM WAV (sample width 2), got {sample_width}."
            )
        num_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, sample_rate, num_channels


def pcm16_to_wav_bytes(
    pcm: bytes, sample_rate: int, num_channels: int = 1
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buffer.getvalue()


def downmix_to_mono(pcm: bytes, num_channels: int) -> bytes:
    if num_channels <= 1:
        return pcm
    samples = array("h")
    samples.frombytes(pcm)
    frame_count = len(samples) // num_channels
    mono = array("h", [0]) * frame_count
    for frame_index in range(frame_count):
        base = frame_index * num_channels
        channel_sum = sum(
            samples[base + channel] for channel in range(num_channels)
        )
        mono[frame_index] = int(channel_sum / num_channels)
    return mono.tobytes()


def iter_pcm16_frames(
    pcm: bytes,
    sample_rate: int,
    frame_ms: int = DEFAULT_FRAME_MS,
    num_channels: int = 1,
) -> Iterator[bytes]:
    samples_per_channel = int(sample_rate * frame_ms / 1000)
    bytes_per_frame = samples_per_channel * 2 * num_channels
    if bytes_per_frame <= 0:
        raise ValueError(
            "frame_ms and sample_rate must produce a positive frame size"
        )

    for offset in range(0, len(pcm), bytes_per_frame):
        frame = pcm[offset : offset + bytes_per_frame]
        if len(frame) < bytes_per_frame:
            frame += b"\x00" * (bytes_per_frame - len(frame))
        yield frame


def resample_pcm16(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    if from_rate == to_rate or not pcm:
        return pcm

    samples = array("h")
    samples.frombytes(pcm)
    n = len(samples)
    if n == 0:
        return pcm

    out_n = max(1, round(n * to_rate / from_rate))
    out = array("h", bytes(2 * out_n))
    step = (n - 1) / out_n if out_n > 1 else 0.0
    for i in range(out_n):
        pos = i * step
        left = int(pos)
        frac = pos - left
        right = left + 1 if left + 1 < n else left
        out[i] = int(samples[left] * (1.0 - frac) + samples[right] * frac)
    return out.tobytes()


def rms_pcm16(pcm: bytes) -> float:
    samples = array("h")
    samples.frombytes(pcm)
    if len(samples) == 0:
        return 0.0
    return sqrt(sum(sample * sample for sample in samples) / len(samples))


def is_silent(pcm: bytes, threshold_rms: float = DEFAULT_SILENCE_RMS) -> bool:
    return rms_pcm16(pcm) < threshold_rms


_ULAW_BIAS = 0x84
_ULAW_CLIP = 32635


def _build_ulaw_exponent_lut():
    lut = []
    for exponent, count in enumerate([2, 2, 4, 8, 16, 32, 64, 128]):
        lut.extend([exponent] * count)
    return lut


_ULAW_EXP_LUT = _build_ulaw_exponent_lut()


def pcm16_to_ulaw(pcm: bytes) -> bytes:
    samples = array("h")
    samples.frombytes(pcm)
    out = bytearray(len(samples))
    for i, sample in enumerate(samples):
        sign = 0x80 if sample < 0 else 0x00
        if sample < 0:
            sample = -sample
        if sample > _ULAW_CLIP:
            sample = _ULAW_CLIP
        sample += _ULAW_BIAS
        exponent = _ULAW_EXP_LUT[(sample >> 7) & 0xFF]
        mantissa = (sample >> (exponent + 3)) & 0x0F
        out[i] = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return bytes(out)


def ulaw_to_pcm16(data: bytes) -> bytes:
    out = array("h", bytes(2 * len(data)))
    for i, byte in enumerate(data):
        byte = ~byte & 0xFF
        t = ((byte & 0x0F) << 3) + _ULAW_BIAS
        t <<= (byte & 0x70) >> 4
        out[i] = (_ULAW_BIAS - t) if (byte & 0x80) else (t - _ULAW_BIAS)
    return out.tobytes()

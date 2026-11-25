import math

from deepeval.test_case import Audio
from deepeval.voice.connectors import audio_utils

DTMF_FREQUENCIES = {
    "1": (697, 1209),
    "2": (697, 1336),
    "3": (697, 1477),
    "4": (770, 1209),
    "5": (770, 1336),
    "6": (770, 1477),
    "7": (852, 1209),
    "8": (852, 1336),
    "9": (852, 1477),
    "*": (941, 1209),
    "0": (941, 1336),
    "#": (941, 1477),
}
DTMF_CODES = {**{str(digit): digit for digit in range(10)}, "*": 10, "#": 11}
TONE_S = 0.1
GAP_S = 0.08
TONE_AMPLITUDE = 0.25


def keypad_turn_text(digits: str) -> str:
    return f"Pressed {digits} on the keypad"


def dtmf_tones(digits: str, sample_rate: int) -> bytes:
    pcm = bytearray()
    for digit in digits:
        low, high = DTMF_FREQUENCIES[digit]
        for index in range(int(sample_rate * TONE_S)):
            t = index / sample_rate
            value = (
                math.sin(2 * math.pi * low * t)
                + math.sin(2 * math.pi * high * t)
            ) / 2
            pcm.extend(
                int(value * TONE_AMPLITUDE * 32767).to_bytes(
                    2, "little", signed=True
                )
            )
        pcm.extend(b"\x00\x00" * int(sample_rate * GAP_S))
    return bytes(pcm)


def keypad_audio(digits: str, sample_rate: int) -> Audio:
    pcm = dtmf_tones(digits, sample_rate)
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(pcm, sample_rate, 1),
        "audio/wav",
        sampleRate=sample_rate,
        encoding="wav",
        duration=len(pcm) / 2 / sample_rate,
    )

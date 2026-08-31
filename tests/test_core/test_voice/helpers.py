from deepeval.test_case import Audio
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.types import ConnectorTurn

RATE = 24000


def wav_audio(seconds: float = 0.02) -> Audio:
    pcm = b"\xe8\x03" * int(RATE * seconds)
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(pcm, sample_rate=RATE),
        "audio/wav",
        sampleRate=RATE,
        encoding="wav",
        duration=seconds,
    )


class EchoAgent:
    async def __call__(self, audio: Audio) -> ConnectorTurn:
        return ConnectorTurn(audio=wav_audio(), transcript="Agent reply")


class StubTTS:
    sample_rate = RATE

    async def a_synthesize(self, text: str, **kwargs):
        return wav_audio(), None

    def supports_streaming(self) -> bool:
        return False


class StubSTT:
    truncated_audio_pad_seconds = 0.0

    async def a_transcribe(self, audio, **kwargs):
        return "Agent reply", None

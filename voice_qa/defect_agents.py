import asyncio

from deepeval.models.tts import OpenAITTSModel
from deepeval.test_case import Audio
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.types import ConnectorTurn

AGENT_VOICE = "onyx"
RATE = 24000

CONTROL_LINES = [
    "Thanks for calling Confident support, how can I help you today?",
    "I understand. Let me pull that up for you right away.",
    "That is done on my end. Is there anything else I can help with?",
]

RAMBLE_TEXT = (
    "Well now, let me tell you, back when I started in this business we did "
    "everything on paper, and I mean everything, carbon copies and all, and "
    "the reason I bring that up is that your account here reminds me of a "
    "filing cabinet we had in the Cleveland office, drawer after drawer of "
    "orders just like yours, and if you give me a moment I will walk you "
    "through every single step of what happened to your order, starting from "
    "the very beginning, because context matters, it really does, and people "
    "these days are always in such a hurry that they never hear the full "
    "story, which in your case begins on a Tuesday at the sorting facility."
)

THINKER_PART_A = "Let me check that for you, one moment while I look it up."
THINKER_PART_B = "Yes, I found it. Your order will arrive on Friday."


def _pcm(audio: Audio) -> bytes:
    pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(audio.get_bytes())
    pcm = audio_utils.downmix_to_mono(pcm, channels)
    if rate != RATE:
        pcm = audio_utils.resample_pcm16(pcm, rate, RATE)
    return pcm


def _wav(pcm: bytes) -> Audio:
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(pcm, RATE),
        "audio/wav",
        sampleRate=RATE,
        encoding="wav",
        duration=len(pcm) / 2 / RATE,
    )


def _silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(RATE * seconds)


class CannedAgent:
    def __init__(self, turns):
        self.turns = turns
        self.index = 0

    async def __call__(self, audio: Audio) -> ConnectorTurn:
        clip, transcript = self.turns[min(self.index, len(self.turns) - 1)]
        self.index += 1
        copy = Audio.from_bytes(
            clip.get_bytes(),
            clip.mimeType or "audio/wav",
            sampleRate=clip.sampleRate,
            encoding=clip.encoding,
            duration=clip.duration,
        )
        return ConnectorTurn(audio=copy, transcript=transcript)


async def build_clip_library() -> dict:
    tts = OpenAITTSModel(voice=AGENT_VOICE)

    async def say(text: str) -> Audio:
        audio, _ = await tts.a_synthesize(text)
        return audio

    control_clips = [await say(line) for line in CONTROL_LINES]
    ramble_clip = await say(RAMBLE_TEXT)
    part_a = await say(THINKER_PART_A)
    part_b = await say(THINKER_PART_B)

    thinker_clip = _wav(_pcm(part_a) + _silence(3.5) + _pcm(part_b))
    silence_clip = _wav(_silence(25.0))

    return {
        "control": [
            (clip, line) for clip, line in zip(control_clips, CONTROL_LINES)
        ],
        "rambler": [(ramble_clip, RAMBLE_TEXT)],
        "thinker": [(thinker_clip, f"{THINKER_PART_A} {THINKER_PART_B}")],
        "corpse": [
            (control_clips[0], CONTROL_LINES[0]),
            (silence_clip, ""),
        ],
    }


def make_agent(library: dict, kind: str) -> CannedAgent:
    return CannedAgent(library[kind])


if __name__ == "__main__":
    library = asyncio.run(build_clip_library())
    for kind, turns in library.items():
        durations = [round(clip.duration or 0, 1) for clip, _ in turns]
        print(kind, durations)

"""Synthetic audio and shared helpers for the voice metric tests."""

import math
import random
from array import array

from deepeval.test_case import Audio, ConversationalTestCase, Turn
from deepeval.voice.connectors import audio_utils

RATE = 8000
# speech_like's syllables leave the last quarter second of a clip silent.
TAIL = 0.25


def speech_like(
    seconds, *, base_hz=200.0, seed=0, noise=20, glide_hz=30.0
) -> array:
    """A voice-shaped signal: pitch gliding ±`glide_hz`, harmonics, syllables
    of sound and near-silence, and faint room noise so pauses are never
    zero."""
    rng = random.Random(seed)
    out = array("h")
    phase = 0.0
    for index in range(int(seconds * RATE)):
        t = index / RATE
        hz = base_hz + glide_hz * math.sin(2 * math.pi * t / 1.0)
        phase += 2 * math.pi * hz / RATE
        envelope = max(0.0, math.sin(math.pi * 4.0 * t)) ** 0.7
        voice = (
            math.sin(phase)
            + 0.5 * math.sin(2 * phase)
            + 0.3 * math.sin(3 * phase)
        ) / 1.8
        out.append(int(8000 * envelope * voice + rng.gauss(0, noise)))
    return out


def vowel(seconds, *, start_hz=200.0, end_hz=185.0) -> array:
    """A held vowel: harmonics, steady level, and the slow pitch glide every
    voice has."""
    out = array("h")
    phase = 0.0
    count = int(seconds * RATE)
    for index in range(count):
        hz = start_hz + (end_hz - start_hz) * index / count
        phase += 2 * math.pi * hz / RATE
        out.append(
            int(
                5000
                * (
                    math.sin(phase)
                    + 0.5 * math.sin(2 * phase)
                    + 0.3 * math.sin(3 * phase)
                )
                / 1.8
            )
        )
    return out


def cracked_vowel(
    seconds, *, hz=200.0, crack_at=1.0, crack_seconds=0.2, semitones=8.0
) -> array:
    """A held vowel whose pitch leaps by `semitones` for `crack_seconds`."""
    out = array("h")
    phase = 0.0
    for index in range(int(seconds * RATE)):
        t = index / RATE
        cracked = crack_at <= t < crack_at + crack_seconds
        frame_hz = hz * 2 ** (semitones / 12) if cracked else hz
        phase += 2 * math.pi * frame_hz / RATE
        out.append(
            int(
                5000
                * (
                    math.sin(phase)
                    + 0.5 * math.sin(2 * phase)
                    + 0.3 * math.sin(3 * phase)
                )
                / 1.8
            )
        )
    return out


def tone(seconds, hz, *, amplitude=6000) -> array:
    out = array("h")
    count = int(seconds * RATE)
    fade = int(0.02 * RATE)
    for index in range(count):
        envelope = max(0.0, min(1.0, index / fade, (count - index - 1) / fade))
        out.append(
            int(
                amplitude * envelope * math.sin(2 * math.pi * hz * index / RATE)
            )
        )
    return out


def spoken_words(count=8, *, pause=0.1, noise=20) -> array:
    rng = random.Random(count)
    out = array("h")
    for index in range(count):
        out.extend(speech_like(0.25, seed=index, noise=noise))
        out.extend(int(rng.gauss(0, noise)) for _ in range(int(pause * RATE)))
    return out


def synthetic_syllable(seconds, *, hz=230.0, wobble=0.005, seed=0) -> array:
    """No glide, but the pitch moves a little every 40 ms frame."""
    rng = random.Random(seed)
    out = array("h")
    phase = 0.0
    step = int(0.04 * RATE)
    for index in range(int(seconds * RATE)):
        if index % step == 0:
            frame_hz = hz * (1 + rng.gauss(0, wobble))
        phase += 2 * math.pi * frame_hz / RATE
        out.append(
            int(
                5000
                * (
                    math.sin(phase)
                    + 0.5 * math.sin(2 * phase)
                    + 0.3 * math.sin(3 * phase)
                )
                / 1.8
            )
        )
    return out


def zeros(seconds) -> array:
    return array("h", [0] * int(seconds * RATE))


def clipped(seconds) -> array:
    return array(
        "h",
        [
            max(-32768, min(32767, sample * 8))
            for sample in speech_like(seconds)
        ],
    )


def cut_short(samples: array, seconds) -> array:
    return samples[: int(seconds * RATE)]


def with_pause(samples: array, at, seconds, *, noise=20) -> array:
    """Faint room noise for `seconds`, inserted `at` seconds in."""
    rng = random.Random(1)
    split = int(at * RATE)
    gap = array(
        "h", [int(rng.gauss(0, noise)) for _ in range(int(seconds * RATE))]
    )
    return samples[:split] + gap + samples[split:]


def _saturate(value) -> int:
    return max(-32768, min(32767, int(value)))


def with_noise(samples: array, dbfs, *, seed=0) -> array:
    rng = random.Random(seed)
    level = 32768 * 10 ** (dbfs / 20)
    return array("h", [_saturate(v + rng.gauss(0, level)) for v in samples])


def with_gain(samples: array, gain) -> array:
    return array("h", [_saturate(sample * gain) for sample in samples])


def faded(samples: array, db) -> array:
    """Level moves by `db` across the clip, evenly in dB: a fade when
    negative, a swell up to the original level when positive."""
    count = len(samples)
    return array(
        "h",
        [
            _saturate(
                sample
                * 10
                ** (
                    (db * index / count if db < 0 else db * (index / count - 1))
                    / 20
                )
            )
            for index, sample in enumerate(samples)
        ],
    )


def to_audio(samples: array, start_time=None) -> Audio:
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(samples.tobytes(), RATE, num_channels=1),
        "audio/wav",
        sampleRate=RATE,
        encoding="wav",
        duration=len(samples) / RATE,
        start_time=start_time,
    )


def clean_turns(count=6):
    turns = []
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        turns.append(
            Turn(
                role=role,
                content=f"Turn {index} spoken by the {role}.",
                audio=to_audio(
                    speech_like(
                        3.0,
                        base_hz=200.0 if role == "user" else 230.0,
                        seed=index,
                    ),
                    start_time=index * 3.5,
                ),
            )
        )
    return turns


def timed_call(gaps, *, greeting_at=None, line_seconds=2.0, reply_seconds=2.0):
    """Caller lines each answered by the agent ``gaps[i]`` seconds after the
    line ends (negative when the agent starts over it). With ``greeting_at``
    the agent greets first, that many seconds after the call connects."""
    turns = []
    cursor = 0.0
    if greeting_at is not None:
        turns.append(
            Turn(
                role="assistant",
                content="Hi, thanks for calling.",
                audio=to_audio(
                    speech_like(reply_seconds, base_hz=230.0),
                    start_time=greeting_at,
                ),
            )
        )
        cursor = greeting_at + reply_seconds + 0.5
    for index, gap in enumerate(gaps):
        line_end = cursor + line_seconds
        reply_start = line_end + gap
        turns.append(
            Turn(
                role="user",
                content=f"Caller line {index}.",
                audio=to_audio(
                    speech_like(line_seconds, base_hz=200.0, seed=index),
                    start_time=cursor,
                ),
            )
        )
        turns.append(
            Turn(
                role="assistant",
                content=f"Agent reply {index}.",
                audio=to_audio(
                    speech_like(reply_seconds, base_hz=230.0, seed=index),
                    start_time=reply_start,
                ),
            )
        )
        cursor = reply_start + reply_seconds + 0.5
    return turns


def stereo_recording(path, *, caller=(), agent=(), seconds=None) -> str:
    """A whole-call recording, caller left and agent right; `caller` and
    `agent` are (start seconds, clip) pairs."""
    end = max(
        (start + len(clip) / RATE for start, clip in (*caller, *agent)),
        default=0.0,
    )
    length = int((seconds if seconds is not None else end + 0.5) * RATE)
    channels = []
    for placed in (caller, agent):
        channel = array("h", [0]) * length
        for start, clip in placed:
            offset = int(start * RATE)
            channel[offset : offset + len(clip)] = clip[: length - offset]
        channels.append(channel)
    stereo = array("h", [0]) * (2 * length)
    stereo[0::2], stereo[1::2] = channels
    with open(path, "wb") as file:
        file.write(
            audio_utils.pcm16_to_wav_bytes(
                stereo.tobytes(), RATE, num_channels=2
            )
        )
    return str(path)


CLEAN_GAPS = [0.5, 0.5, 0.5]
STAMPS_NOTE = (
    " Judged from each turn's own audio and timestamps, as no stereo call "
    "recording was available."
)
AGENT_HUNG_UP = {"Stop reason": "The agent ended the call"}


def measure(metric, turns, metadata=None):
    metric.measure(
        ConversationalTestCase(turns=turns, metadata=metadata),
        _show_indicator=False,
    )
    return metric


def agent_speech(seconds=3.0, **kwargs):
    return speech_like(seconds, base_hz=230.0, seed=1, **kwargs)


def loop_clip():
    syllable = speech_like(0.25, seed=1)
    return syllable + syllable + syllable + syllable


def with_defect(index, build_clip):
    turns = clean_turns()
    turns[index].audio = to_audio(build_clip(), start_time=index * 3.5)
    return turns


def with_reply_audio(index, samples, turns=None):
    turns = turns or clean_turns()
    turns[index].audio = to_audio(samples, start_time=index * 3.5)
    return turns


def set_audio(samples):
    def apply(turn):
        turn.audio = to_audio(samples, start_time=turn.audio.start_time)

    return apply


def set_content(content):
    def apply(turn):
        turn.content = content

    return apply


def edit_reply(index, apply, turns=None):
    turns = turns or clean_turns()
    apply(turns[index])
    return turns


def caller_turn(content):
    return Turn(role="user", content=content, audio=to_audio(speech_like(1.0)))


def agent_turn(content):
    return Turn(
        role="assistant", content=content, audio=to_audio(speech_like(1.0))
    )


def timed_turn(role, seconds, start_time, seed=0):
    return Turn(
        role=role,
        content=f"{role} line.",
        audio=to_audio(speech_like(seconds, seed=seed), start_time=start_time),
    )


PLACE_KEYS = (
    "offset_seconds",
    "duration_seconds",
    "start_time",
    "mark",
    "timing_source",
    "description",
)


def without_place(event):
    """The event minus where it sits and how it reads on the call."""
    return {key: value for key, value in event.items() if key not in PLACE_KEYS}


def undecodable(turn):
    turn.audio = Audio.from_bytes(
        b"not-a-wav", "audio/wav", start_time=turn.audio.start_time
    )


def recording(path) -> Audio:
    """A stereo call recording on disk as the test case's `audio`."""
    return Audio(url=str(path), mimeType="audio/wav")


def recorded_call(tmp_path, *, caller, agent):
    path = stereo_recording(tmp_path / "call.wav", caller=caller, agent=agent)
    return timed_call(CLEAN_GAPS), path


def measure_recording(metric, tmp_path, *, caller, agent):
    turns, path = recorded_call(tmp_path, caller=caller, agent=agent)
    metric.measure(
        ConversationalTestCase(turns=turns, audio=recording(path)),
        _show_indicator=False,
    )
    return metric


def agent_backs_off_call(caller_pause):
    """The caller speaks, pauses, and starts again 0.4 s after the agent
    starts; the agent stops after 1 s."""
    agent_at = 5.0 + caller_pause - 0.4
    return [
        timed_turn("assistant", 2.0, 0.0),
        timed_turn("user", 2.0, 3.0),
        timed_turn("assistant", 1.0, agent_at, seed=1),
        timed_turn("user", 2.0, agent_at + 0.4, seed=1),
        timed_turn("assistant", 2.0, agent_at + 3.0, seed=2),
    ]


def clip_shape(clip):
    """The channel count and length in seconds of a WAV clip."""
    import io
    import wave

    with wave.open(io.BytesIO(clip)) as wav:
        return wav.getnchannels(), wav.getnframes() / wav.getframerate()

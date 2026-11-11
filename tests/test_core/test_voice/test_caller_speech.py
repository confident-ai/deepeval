"""The audio a persona's speech patterns, line and background speakers make.

Everything here runs on a stub TTS whose speech is a constant tone, so what a
setting does to the caller's audio shows up as plain arithmetic on samples.
"""

import asyncio
import random
from array import array
from typing import List, Optional, Tuple

import pytest

from deepeval.dataset.golden import (
    BackgroundNoiseSettings,
    BackgroundSpeaker,
    LineSettings,
    Persona,
    SpeechPatternSettings,
    VoiceSettings,
)
from deepeval.models.base_model import DeepEvalBaseTTS
from deepeval.models.tts._frames import frame_pcm_stream
from deepeval.test_case import Audio
from deepeval.voice.caller_speech import (
    ASIDE_GAIN,
    PATTERN_CHANCE,
    CallerSpeechTTS,
    LineFaults,
    split_for_pause,
)
from deepeval.voice.connectors import audio_utils
from deepeval.voice.phone_line import PhoneLineTTS

_RATE = 8000
_LEVEL = 10000
_SECONDS_PER_WORD = 0.1
_SENTENCE = "I was hoping, if possible, to move my appointment to Thursday"


class _ToneTTS(DeepEvalBaseTTS):
    """Speaks a constant tone, a tenth of a second for every word."""

    sample_rate = _RATE

    def __init__(self, streaming: bool = False):
        self.spoken: List[str] = []
        self.voices: List[Optional[str]] = []
        self.streaming = streaming
        super().__init__(model="tone-tts")

    def load_model(self):
        return self

    def _pcm(self, text: str) -> bytes:
        samples = int(len(text.split()) * _SECONDS_PER_WORD * _RATE)
        return array("h", [_LEVEL] * samples).tobytes()

    def synthesize(
        self, text: str, *args, voice: Optional[str] = None, **kwargs
    ) -> Tuple[Audio, Optional[float]]:
        self.spoken.append(text)
        self.voices.append(voice)
        return (
            Audio.from_bytes(
                audio_utils.pcm16_to_wav_bytes(self._pcm(text), _RATE, 1),
                "audio/wav",
                sampleRate=_RATE,
                encoding="wav",
            ),
            1.0,
        )

    async def a_synthesize(self, text, *args, voice=None, **kwargs):
        return self.synthesize(text, voice=voice, **kwargs)

    async def a_synthesize_stream(self, text, *args, voice=None, **kwargs):
        self.spoken.append(text)
        pcm = self._pcm(text)

        async def _reads():
            for start in range(0, len(pcm), 700):
                yield pcm[start : start + 700]

        async for chunk in frame_pcm_stream(_reads(), sample_rate=_RATE):
            yield chunk

    def synthesis_cost(self, text: str) -> Optional[float]:
        return 1.0

    def supports_streaming(self) -> bool:
        return self.streaming

    def get_model_name(self) -> str:
        return "tone-tts"


def _persona(**settings) -> Persona:
    return Persona(name="Sam", characteristics="A caller.", **settings)


def _samples(audio: Audio) -> array:
    pcm, _, _ = audio_utils.wav_bytes_to_pcm16(audio.get_bytes())
    return array("h", pcm)


def _spoken(persona: Persona, text: str = _SENTENCE):
    tts = _ToneTTS()
    audio, cost = CallerSpeechTTS(tts, persona).synthesize(text)
    return tts, _samples(audio), cost


def _always(monkeypatch) -> None:
    for frequency in PATTERN_CHANCE:
        monkeypatch.setitem(PATTERN_CHANCE, frequency, 1.0)


def _silent_run(samples: array) -> int:
    """The longest run of silent samples."""
    longest = run = 0
    for sample in samples:
        run = run + 1 if sample == 0 else 0
        longest = max(longest, run)
    return longest


def test_a_pause_falls_at_the_comma_nearest_the_middle():
    assert split_for_pause(_SENTENCE) == (
        "I was hoping, if possible,",
        "to move my appointment to Thursday",
    )


def test_a_short_utterance_has_nowhere_to_pause():
    assert split_for_pause("Yes, that works") is None


def test_a_persona_with_plain_speech_needs_no_wrapper():
    plain = _persona(
        speech_patterns=SpeechPatternSettings(fillers="frequent"),
        line=LineSettings(quality="phone"),
        background_noise=BackgroundNoiseSettings(audio="cafe.wav"),
    )

    tts_model = plain.caller_tts_model(_ToneTTS())

    assert not CallerSpeechTTS.applies_to(plain)
    assert isinstance(tts_model, PhoneLineTTS)


def test_line_faults_go_on_top_of_the_phone_line():
    persona = _persona(line=LineSettings(quality="phone", dropouts="rare"))

    tts_model = persona.caller_tts_model(_ToneTTS())

    assert isinstance(tts_model, CallerSpeechTTS)
    assert isinstance(tts_model.inner, PhoneLineTTS)


def test_a_mid_sentence_pause_puts_silence_between_two_halves(monkeypatch):
    _always(monkeypatch)
    persona = _persona(
        speech_patterns=SpeechPatternSettings(mid_sentence_pauses="frequent")
    )

    tts, samples, cost = _spoken(persona)

    assert tts.spoken == [
        "I was hoping, if possible,",
        "to move my appointment to Thursday",
    ]
    assert samples[0] == _LEVEL and samples[-1] == _LEVEL
    assert 1.5 * _RATE <= _silent_run(samples) <= 2.5 * _RATE
    assert cost == pytest.approx(2.0)


def test_a_slow_answer_starts_with_silence(monkeypatch):
    _always(monkeypatch)
    persona = _persona(
        speech_patterns=SpeechPatternSettings(slow_to_answer="frequent")
    )

    tts, samples, _ = _spoken(persona)

    assert tts.spoken == [_SENTENCE]
    assert all(sample == 0 for sample in samples[: 2 * _RATE])
    assert samples[-1] == _LEVEL


def test_side_speech_is_said_first_and_quieter(monkeypatch):
    _always(monkeypatch)
    persona = _persona(
        speech_patterns=SpeechPatternSettings(side_speech="frequent")
    )
    tts = _ToneTTS()
    caller = CallerSpeechTTS(tts, persona)
    caller.use_lines(asides=["Put that down."])

    audio, _ = caller.synthesize(_SENTENCE)
    samples = _samples(audio)

    assert tts.spoken == ["Put that down.", _SENTENCE]
    assert samples[0] == int(_LEVEL * ASIDE_GAIN)
    assert samples[-1] == _LEVEL


def test_network_delay_holds_back_every_utterance():
    tts, samples, _ = _spoken(_persona(line=LineSettings(delay_ms=500)))

    assert all(sample == 0 for sample in samples[: _RATE // 2])
    assert samples[_RATE // 2] == _LEVEL


def test_static_is_heard_over_silence_and_speech():
    _, samples, _ = _spoken(_persona(line=LineSettings(static=1.0)))

    assert len(set(samples)) > 100
    assert abs(sum(samples) / len(samples) - _LEVEL) < 200


def test_dropouts_cut_holes_in_the_speech():
    line = LineSettings(dropouts="frequent")
    pcm = array("h", [_LEVEL] * (_RATE * 30)).tobytes()

    damaged = array("h", LineFaults(_RATE, line, random.Random(1)).process(pcm))

    silent = sum(1 for sample in damaged if sample == 0)
    assert 0.02 < silent / len(damaged) < 0.4
    assert 0.1 * _RATE <= _silent_run(damaged) <= 1.0 * _RATE


def test_garbled_spans_change_the_speech_without_silencing_it():
    line = LineSettings(garbled="frequent")
    ramp = array(
        "h", [(index * 37) % 20000 - 10000 for index in range(_RATE * 30)]
    )

    damaged = array(
        "h", LineFaults(_RATE, line, random.Random(1)).process(ramp.tobytes())
    )

    changed = sum(1 for before, after in zip(ramp, damaged) if before != after)
    assert 0.02 < changed / len(ramp) < 0.5


def test_faults_fall_in_the_same_places_fed_whole_or_in_frames():
    line = LineSettings(dropouts="frequent", garbled="frequent")
    pcm = array("h", [_LEVEL] * (_RATE * 20)).tobytes()

    whole = LineFaults(_RATE, line, random.Random(7)).process(pcm)
    framed_faults = LineFaults(_RATE, line, random.Random(7))
    framed = b"".join(
        framed_faults.process(pcm[start : start + 1600])
        for start in range(0, len(pcm), 1600)
    )

    assert framed == whole


def test_a_background_speaker_is_mixed_under_the_caller(monkeypatch):
    _always(monkeypatch)
    persona = _persona(
        background_noise=BackgroundNoiseSettings(
            other_speakers=[
                BackgroundSpeaker(
                    characteristics="A child.", voice="nova", volume=0.5
                )
            ]
        )
    )
    tts = _ToneTTS()
    caller = CallerSpeechTTS(tts, persona)
    caller.use_lines(background_lines=[["Is it my turn yet?"]])

    audio, cost = caller.synthesize(_SENTENCE)
    samples = _samples(audio)

    assert tts.spoken == [_SENTENCE, "Is it my turn yet?"]
    assert tts.voices == [None, "nova"]
    assert max(samples) == _LEVEL + _LEVEL // 2
    assert samples[0] == _LEVEL
    assert cost == pytest.approx(2.0)


def test_the_same_persona_makes_the_same_call_twice():
    persona = _persona(
        speech_patterns=SpeechPatternSettings(
            mid_sentence_pauses="normal",
            slow_to_answer="normal",
            side_speech="normal",
        ),
        line=LineSettings(dropouts="normal", static=0.3),
    )

    def call() -> List[bytes]:
        caller = CallerSpeechTTS(_ToneTTS(), persona)
        return [caller.synthesize(_SENTENCE)[0].get_bytes() for _ in range(5)]

    assert call() == call()


def test_a_streamed_utterance_carries_the_same_audio_as_a_whole_one(
    monkeypatch,
):
    _always(monkeypatch)
    persona = _persona(
        speech_patterns=SpeechPatternSettings(
            mid_sentence_pauses="frequent", side_speech="frequent"
        ),
        line=LineSettings(delay_ms=200, dropouts="frequent"),
    )

    whole, _ = CallerSpeechTTS(_ToneTTS(), persona).synthesize(_SENTENCE)

    async def stream() -> bytes:
        caller = CallerSpeechTTS(_ToneTTS(streaming=True), persona)
        chunks = [
            chunk async for chunk in caller.a_synthesize_stream(_SENTENCE)
        ]
        assert chunks[-1].final
        return b"".join(chunk.get_bytes() for chunk in chunks)

    assert asyncio.run(stream()) == _samples(whole).tobytes()


def test_volume_scales_the_caller_and_nothing_else():
    _, samples, _ = _spoken(_persona(voice=VoiceSettings(volume=0.5)))

    assert set(samples) == {_LEVEL // 2}


def test_pace_is_asked_of_the_tts_model_as_an_instruction():
    persona = _persona(voice=VoiceSettings(pace="slow", instructions="Calm."))

    assert persona.delivery_instructions().startswith(
        "Speak very slowly, leaving a clear pause between phrases. Calm. "
    )


@pytest.mark.parametrize("quality", ["phone", "mobile", "speakerphone"])
def test_every_line_quality_sounds_the_same_whole_or_in_frames(quality):
    from deepeval.voice.phone_line import PhoneLine

    ramp = array(
        "h", [(index * 37) % 20000 - 10000 for index in range(_RATE * 2)]
    ).tobytes()

    whole = PhoneLine(_RATE, quality).process(ramp)
    framed_line = PhoneLine(_RATE, quality)
    framed = b"".join(
        framed_line.process(ramp[start : start + 1600])
        for start in range(0, len(ramp), 1600)
    )

    assert framed == whole
    assert whole != ramp


def test_each_line_quality_changes_the_speech_differently():
    from deepeval.voice.phone_line import PhoneLine

    ramp = array(
        "h", [(index * 37) % 20000 - 10000 for index in range(_RATE)]
    ).tobytes()

    outputs = {
        quality: PhoneLine(_RATE, quality).process(ramp)
        for quality in ("phone", "mobile", "speakerphone")
    }

    assert len(set(outputs.values())) == 3

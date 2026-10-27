import math
import random
from array import array

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import AudioGlitchMetric
from deepeval.metrics.voice._analysis import (
    analyze_audio,
    pitch_and_periodicity,
)
from deepeval.metrics.audio_glitch import AudioGlitchTemplate
from deepeval.metrics.audio_glitch._detectors import (
    _silence_profile,
    _stable_runs,
    _tone_frames,
    detect_audio_glitches,
)
from deepeval.metrics.audio_glitch.audio_glitch import (
    DEFAULT_CLEAN_REASON,
)
from deepeval.metrics.audio_glitch.schema import AudioGlitchReason
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from deepeval.voice.connectors import audio_utils

RATE = 8000


# ------------------------------------------------------------- fixtures


def speech_like(seconds, *, base_hz=200.0, seed=0, noise=20) -> array:
    """A voice-shaped signal: pitch gliding ±30 Hz, harmonics, syllables of
    sound and near-silence, and faint room noise so pauses are never zero."""
    rng = random.Random(seed)
    out = array("h")
    phase = 0.0
    for index in range(int(seconds * RATE)):
        t = index / RATE
        hz = base_hz + 30.0 * math.sin(2 * math.pi * t / 1.0)
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


def to_audio(samples: array, start_time=None) -> Audio:
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(samples.tobytes(), RATE, num_channels=1),
        "audio/wav",
        sampleRate=RATE,
        encoding="wav",
        duration=len(samples) / RATE,
        start_time=start_time,
    )


class FakeReasonModel(DeepEvalBaseLLM):
    def __init__(self):
        self.prompts = []
        super().__init__(model="fake-reason-model")

    def load_model(self):
        return self

    def generate(self, prompt, schema=None, **kwargs):
        self.prompts.append(prompt)
        return AudioGlitchReason(reason="scripted reason")

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self):
        return "fake-reason-model"


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


def _measure(turns, **metric_kwargs):
    model = FakeReasonModel()
    metric = AudioGlitchMetric(model=model, async_mode=False, **metric_kwargs)
    metric.measure(ConversationalTestCase(turns=turns), _show_indicator=False)
    return metric, model


def _events(metric, kind=None):
    events = metric.score_breakdown["events"]
    return [event for event in events if kind is None or event["type"] == kind]


# ---------------------------------------------------------- signal level


@pytest.mark.parametrize("hz", [388.0, 452.0])
def test_pitch_and_periodicity_finds_tones_above_the_voice_range(hz):
    frames = _tone_frames(tone(0.5, hz), RATE)
    assert frames
    assert all(abs(frame.pitch_hz - hz) / hz < 0.02 for frame in frames)
    assert all(frame.periodicity >= 0.95 for frame in frames)


def test_pitch_and_periodicity_is_silent_on_quiet_frames():
    assert pitch_and_periodicity(array("h", [0] * 320), RATE) is None


def test_speech_like_audio_holds_no_tone_length_run():
    runs = _stable_runs(_tone_frames(speech_like(3.0), RATE))
    assert runs
    assert max(run.duration for run in runs) < 0.8


def test_dead_runs_ignore_leading_padding_and_room_noise():
    clip = zeros(0.6) + speech_like(1.0) + speech_like(1.0, seed=1)
    assert _silence_profile(clip, RATE).runs == []
    clip = speech_like(1.0) + zeros(0.8) + speech_like(1.0, seed=1)
    (run,) = _silence_profile(clip, RATE).runs
    assert run.offset == pytest.approx(1.0, abs=0.05)
    assert run.duration == pytest.approx(0.8, abs=0.05)
    assert run.runs_to_end is False


def test_a_trailing_dead_run_needs_a_full_second():
    assert _silence_profile(speech_like(1.0) + zeros(0.7), RATE).runs == []
    (run,) = _silence_profile(speech_like(1.0) + zeros(1.2), RATE).runs
    assert run.runs_to_end is True


def test_pauses_between_words_are_not_dropouts():
    assert analyze_audio(to_audio(spoken_words())).dropout_events == 0


def test_a_short_break_inside_a_word_is_a_dropout():
    clip = spoken_words()
    clip[int(1.1 * RATE) : int(1.2 * RATE)] = zeros(0.1)
    assert analyze_audio(to_audio(clip)).dropout_events == 1


def test_a_break_is_not_a_dropout_when_pauses_are_digital():
    clip = spoken_words(noise=0)
    clip[int(1.1 * RATE) : int(1.2 * RATE)] = zeros(0.1)
    assert analyze_audio(to_audio(clip)).dropout_events == 0


# ----------------------------------------------------------- held tones


def test_a_clean_call_scores_one_without_calling_the_model():
    metric, model = _measure(clean_turns())
    assert metric.score == 1.0
    assert _events(metric) == []
    assert metric.reason == DEFAULT_CLEAN_REASON
    assert model.prompts == []
    assert metric.score_breakdown["turns_with_audio"] == 6
    assert metric.score_breakdown["timeline_available"] is True


def test_a_held_tone_in_the_agents_audio_is_flagged():
    turns = clean_turns()
    turns[3].audio = to_audio(
        speech_like(2.0, base_hz=230.0) + tone(1.5, 388.0) + speech_like(1.0),
        start_time=10.5,
    )
    metric, model = _measure(turns)
    (event,) = _events(metric)
    assert event["type"] == "held_tone"
    assert event["turn"] == 3
    assert event["role"] == "assistant"
    assert event["position"] == "middle"
    assert event["pitch_hz"] == pytest.approx(388.0, rel=0.02)
    assert event["duration_seconds"] == pytest.approx(1.5, abs=0.1)
    assert event["offset_seconds"] == pytest.approx(2.0, abs=0.1)
    assert event["start_time"] == pytest.approx(12.5, abs=0.1)
    assert event["hold_ratio"] >= 2
    assert metric.score == pytest.approx(5 / 6)
    assert metric.reason == "scripted reason"
    assert len(model.prompts) == 1


def test_a_one_second_tone_at_the_end_of_a_turn_is_flagged():
    turns = clean_turns()
    turns[1].audio = to_audio(
        speech_like(2.0, base_hz=230.0) + tone(1.0, 452.0), start_time=3.5
    )
    metric, _ = _measure(turns)
    (event,) = _events(metric, "held_tone")
    assert event["turn"] == 1
    assert event["position"] == "end"
    assert event["duration_seconds"] == pytest.approx(1.0, abs=0.1)


def test_a_held_vowel_with_a_natural_glide_is_not_a_tone():
    turns = clean_turns()
    turns[0].audio = to_audio(
        speech_like(1.0) + vowel(1.0) + speech_like(1.0, seed=3), start_time=0.0
    )
    metric, _ = _measure(turns)
    assert _events(metric) == []


def test_a_synthetic_voice_holding_a_syllable_is_not_a_tone():
    turns = clean_turns()
    turns[1].audio = to_audio(
        speech_like(1.0, base_hz=230.0)
        + synthetic_syllable(1.2)
        + speech_like(1.0, base_hz=230.0, seed=4),
        start_time=3.5,
    )
    metric, _ = _measure(turns)
    assert _events(metric) == []


def test_a_flat_tone_at_the_voices_own_pitch_is_caught_by_its_length():
    turns = clean_turns()
    turns[0].audio = to_audio(
        speech_like(1.0) + tone(2.0, 200.0) + speech_like(1.0, seed=3),
        start_time=0.0,
    )
    metric, _ = _measure(turns)
    (event,) = _events(metric, "held_tone")
    assert event["turn"] == 0
    assert event["pitch_ratio"] < 1.3
    assert event["hold_ratio"] >= 2


def test_a_short_tone_is_not_flagged():
    turns = clean_turns()
    turns[1].audio = to_audio(
        speech_like(1.0, base_hz=230.0)
        + tone(0.5, 388.0)
        + speech_like(1.0, base_hz=230.0, seed=4),
        start_time=3.5,
    )
    metric, _ = _measure(turns)
    assert _events(metric) == []


def test_two_tones_do_not_poison_the_speakers_baseline():
    turns = clean_turns(8)
    for index in (1, 5):
        turns[index].audio = to_audio(
            speech_like(3.0, base_hz=230.0, seed=index) + tone(1.5, 388.0),
            start_time=index * 3.5,
        )
    metric, _ = _measure(turns)
    assert {event["turn"] for event in _events(metric, "held_tone")} == {1, 5}
    assert (
        metric.score_breakdown["baselines"]["assistant"]["median_pitch_hz"]
        < 300
    )


# -------------------------------------------------- dead stream and cuts


def test_dead_stream_in_the_callers_answer():
    turns = clean_turns()
    turns[2].audio = to_audio(
        speech_like(1.0) + zeros(0.7) + speech_like(1.0, seed=5), start_time=7.0
    )
    metric, _ = _measure(turns)
    (event,) = _events(metric)
    assert event["type"] == "dead_stream"
    assert event["role"] == "user"
    assert event["turn"] == 2
    assert event["position"] == "middle"
    assert event["runs_to_end"] is False
    assert event["duration_seconds"] == pytest.approx(0.7, abs=0.05)
    assert event["start_time"] == pytest.approx(8.0, abs=0.05)


def test_zeros_are_not_a_dead_stream_for_a_voice_whose_pauses_are_digital():
    turns = clean_turns()
    for index in (0, 2, 4):
        turns[index].audio = to_audio(
            speech_like(3.0, seed=index, noise=0), start_time=index * 3.5
        )
    turns[2].audio = to_audio(
        speech_like(1.0, noise=0)
        + zeros(0.7)
        + speech_like(1.0, seed=5, noise=0),
        start_time=7.0,
    )
    metric, _ = _measure(turns)
    assert _events(metric) == []
    assert (
        metric.score_breakdown["baselines"]["user"]["digital_silence"] is True
    )
    assert (
        metric.score_breakdown["baselines"]["assistant"]["digital_silence"]
        is False
    )


def _cut_call():
    turns = clean_turns()
    # The caller's clip stops mid-word and the caller speaks again 0.5 s later.
    turns[2].audio = to_audio(
        cut_short(speech_like(3.0), 1.125), start_time=7.0
    )
    turns[3] = Turn(
        role="user",
        content="...and that is what I did.",
        audio=to_audio(speech_like(1.5, seed=6), start_time=8.625),
    )
    return turns


def test_a_stream_cut_is_inferred_when_the_same_speaker_resumes():
    metric, _ = _measure(_cut_call())
    (event,) = _events(metric)
    assert event["type"] == "stream_cut"
    assert event["turn"] == 2
    assert event["role"] == "user"
    assert event["position"] == "end"
    assert event["evidence"] == "abrupt_end"
    assert event["gap_seconds"] == pytest.approx(0.5, abs=0.01)
    assert event["resumed_turn"] == 3


def test_a_stream_cut_is_not_inferred_without_start_times():
    turns = _cut_call()
    turns[3].audio.start_time = None
    metric, _ = _measure(turns)
    assert _events(metric) == []
    assert metric.score_breakdown["timeline_available"] is False


def test_clipping_is_flagged_with_its_span():
    turns = clean_turns()
    turns[1].audio = to_audio(
        speech_like(1.0, base_hz=230.0) + clipped(0.5) + speech_like(1.0),
        start_time=3.5,
    )
    metric, _ = _measure(turns)
    (event,) = _events(metric, "clipping")
    assert event["turn"] == 1
    assert event["fraction"] > 0.01
    assert event["offset_seconds"] == pytest.approx(1.0, abs=0.1)


# ------------------------------------------------------- score and reason


def test_score_is_the_share_of_audio_turns_without_events():
    turns = clean_turns(4)
    turns[1].audio = to_audio(
        speech_like(2.0, base_hz=230.0) + tone(1.5, 388.0), start_time=3.5
    )
    turns[2].audio = to_audio(
        speech_like(1.0) + zeros(0.7) + speech_like(1.0, seed=5), start_time=7.0
    )
    turns.append(Turn(role="user", content="Text only."))
    turns.append(Turn(role="assistant", content="Also text only."))
    metric, _ = _measure(turns)
    assert metric.score_breakdown["turns_with_audio"] == 4
    assert metric.score_breakdown["turns_with_events"] == 2
    assert metric.score == pytest.approx(0.5)


def test_undecodable_audio_counts_against_its_turn():
    turns = clean_turns(4)
    turns[1].audio = Audio.from_bytes(b"not-a-wav", "audio/wav", start_time=3.5)
    metric, _ = _measure(turns)
    (event,) = _events(metric)
    assert event["type"] == "audio_undecodable"
    assert event["turn"] == 1
    assert metric.score == pytest.approx(0.75)


def test_a_call_with_no_audio_is_refused():
    metric = AudioGlitchMetric(model=FakeReasonModel(), async_mode=False)
    with pytest.raises(MissingTestCaseParamsError, match="audio"):
        metric.measure(
            ConversationalTestCase(
                turns=[
                    Turn(role="user", content="Hello"),
                    Turn(role="assistant", content="Hi"),
                ]
            ),
            _show_indicator=False,
        )


def test_the_model_is_shown_the_event_and_its_context():
    turns = clean_turns()
    turns[3].audio = to_audio(
        speech_like(2.0, base_hz=230.0) + tone(1.5, 388.0) + speech_like(1.0),
        start_time=10.5,
    )
    _, model = _measure(turns)
    (prompt,) = model.prompts
    assert "held_tone" in prompt
    assert "Turn 3 spoken by the assistant." in prompt
    assert "Turn 2 spoken by the user." in prompt
    assert "middle" in prompt
    assert "dataBase64" not in prompt


def test_include_reason_false_skips_the_model():
    turns = clean_turns()
    turns[3].audio = to_audio(
        speech_like(2.0, base_hz=230.0) + tone(1.5, 388.0), start_time=10.5
    )
    metric, model = _measure(turns, include_reason=False)
    assert metric.reason is None
    assert model.prompts == []
    assert _events(metric, "held_tone")


def test_strict_mode_zeroes_an_imperfect_score():
    turns = clean_turns()
    turns[3].audio = to_audio(
        speech_like(2.0, base_hz=230.0) + tone(1.5, 388.0), start_time=10.5
    )
    metric, _ = _measure(turns, strict_mode=True)
    assert metric.score == 0.0
    assert metric.success is False


@pytest.mark.asyncio
async def test_sync_and_async_agree():
    turns = clean_turns()
    turns[3].audio = to_audio(
        speech_like(2.0, base_hz=230.0) + tone(1.5, 388.0), start_time=10.5
    )
    test_case = ConversationalTestCase(turns=turns)
    sync_metric = AudioGlitchMetric(model=FakeReasonModel(), async_mode=False)
    sync_metric.measure(test_case, _show_indicator=False)
    async_metric = AudioGlitchMetric(model=FakeReasonModel(), async_mode=True)
    await async_metric.a_measure(test_case, _show_indicator=False)
    assert async_metric.score == sync_metric.score
    assert async_metric.score_breakdown == sync_metric.score_breakdown
    assert async_metric.reason == sync_metric.reason


def test_verbose_logs_contain_baselines_and_events():
    turns = clean_turns()
    turns[3].audio = to_audio(
        speech_like(2.0, base_hz=230.0) + tone(1.5, 388.0), start_time=10.5
    )
    metric, _ = _measure(turns, verbose_mode=True)
    assert "median_pitch_hz" in metric.verbose_logs
    assert "held_tone" in metric.verbose_logs


def test_a_custom_template_reaches_the_model():
    class Hijacked(AudioGlitchTemplate):
        @staticmethod
        def generate_reason(score, events):
            return "HIJACKED"

    turns = clean_turns()
    turns[3].audio = to_audio(
        speech_like(2.0, base_hz=230.0) + tone(1.5, 388.0), start_time=10.5
    )
    _, model = _measure(turns, evaluation_template=Hijacked)
    assert model.prompts == ["HIJACKED"]


def test_detect_reports_both_roles():
    report = detect_audio_glitches(ConversationalTestCase(turns=clean_turns()))
    assert set(report.baselines) == {"user", "assistant"}
    assert report.turns_with_audio == [0, 1, 2, 3, 4, 5]

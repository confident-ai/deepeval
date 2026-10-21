"""Signal-level detection behind `AudioGlitchMetric`.

Each turn's clip is measured for held tones, dead stream, clipping and an
abrupt end; stream cuts are inferred across clips of the same speaker from
their call-relative timing. No model listens to the audio.
"""

from array import array
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Dict, List, Optional, Sequence, Tuple

from deepeval.metrics.voice._analysis import (
    AudioMeasurements,
    analyze_audio,
    dbfs,
    decode_audio,
    frame_samples,
    pitch_and_periodicity,
    rms,
)
from deepeval.test_case import Audio, ConversationalTestCase
from deepeval.voice.connectors import audio_utils
from deepeval.voice.timeline import build_audio_timeline

# Tones are tracked on audio downsampled to this rate: the artifacts of
# interest sit far below its Nyquist frequency and the search is pure Python.
ANALYSIS_SAMPLE_RATE = 8000
TONE_FRAME_MS = 40
TONE_MIN_HZ = 60.0
TONE_MAX_HZ = 1000.0
TONE_PITCH_TOLERANCE = 0.05
# The tracker can land on a sub-harmonic of a held tone for a frame or two.
TONE_PITCH_FOLDS = (1.0, 2.0, 0.5, 3.0, 1 / 3)
TONE_MAX_BRIDGE_FRAMES = 2
TONE_MIN_PERIODICITY = 0.85
# A voice holding a vowel jitters from frame to frame and glides in pitch;
# a synthetic tone does neither. Both are relative to the run's pitch.
TONE_MAX_JITTER = 0.002
TONE_MAX_GLIDE = 0.01
MIN_TONE_SECONDS = 0.8
# A run counts as a held tone when it is far longer than that speaker holds
# any sound, far above the pitch of their voice, or long outright.
TONE_HOLD_RATIO = 3.0
TONE_PITCH_RATIO = 1.3
TONE_ABSOLUTE_SECONDS = 2.5
TYPICAL_HOLD_PERCENTILE = 0.9

DEAD_FRAME_MS = 20
DEAD_MAX_ABS_SAMPLE = 2
MIN_DEAD_SECONDS = 0.5
# TTS engines pad the tail of a clip with digital zeros, so a run that
# reaches the end of the clip needs to be longer before it counts.
DEAD_TRAILING_MIN_SECONDS = 1.0
# A recorded voice leaves room noise in its pauses, so exact zeros mean the
# stream dropped. Some synthetic voices write their pauses as zeros; when a
# speaker's typical pause is at least this much digital, zeros are not
# evidence. A recording's typical pause is never digital at all.
DIGITAL_SILENCE_SHARE = 0.2

STREAM_CUT_MAX_GAP_SECONDS = 15.0
CLIPPING_MIN_FRACTION = 0.01
CLIPPING_SAMPLE = 32760
POSITION_EDGE_FRACTION = 0.2


@dataclass
class ToneFrame:
    index: int
    pitch_hz: float
    periodicity: float
    level_dbfs: float


@dataclass
class StableRun:
    start_frame: int
    frame_count: int
    pitch_hz: float
    periodicity: float
    level_dbfs: float
    jitter: float
    glide: float

    @property
    def offset(self) -> float:
        return self.start_frame * TONE_FRAME_MS / 1000

    @property
    def duration(self) -> float:
        return self.frame_count * TONE_FRAME_MS / 1000


@dataclass
class DeadRun:
    offset: float
    duration: float
    runs_to_end: bool


@dataclass
class SilenceProfile:
    runs: List[DeadRun]
    # Share of each pause (unvoiced stretch) that is exact digital zero.
    pause_dead_fractions: List[float]


@dataclass
class ClipAnalysis:
    turn: int
    role: str
    duration: float
    measurements: AudioMeasurements
    frames: List[ToneFrame]
    stable_runs: List[StableRun]
    silence: SilenceProfile
    clipping_span: Optional[Tuple[float, float]]

    @property
    def dead_runs(self) -> List[DeadRun]:
        return self.silence.runs

    @property
    def trailing_dead_run(self) -> bool:
        return any(run.runs_to_end for run in self.dead_runs)


@dataclass
class SpeakerBaseline:
    role: str
    clips: int
    median_pitch_hz: Optional[float]
    typical_hold_seconds: Optional[float]
    digital_silence: bool


@dataclass
class GlitchReport:
    events: List[Dict]
    turns_with_audio: List[int]
    baselines: Dict[str, Dict]
    timeline_available: bool


def _tone_frames(samples: array, sample_rate: int) -> List[ToneFrame]:
    pcm = samples.tobytes()
    rate = sample_rate
    if sample_rate > ANALYSIS_SAMPLE_RATE:
        pcm = audio_utils.resample_pcm16(pcm, sample_rate, ANALYSIS_SAMPLE_RATE)
        rate = ANALYSIS_SAMPLE_RATE
    data = array("h")
    data.frombytes(pcm)
    frames = []
    for index, frame in enumerate(
        frame_samples(data, rate, frame_ms=TONE_FRAME_MS)
    ):
        pitch = pitch_and_periodicity(
            frame, rate, min_hz=TONE_MIN_HZ, max_hz=TONE_MAX_HZ
        )
        if pitch is None:
            continue
        frames.append(ToneFrame(index, pitch[0], pitch[1], dbfs(rms(frame))))
    return frames


def _same_pitch(pitch: float, reference: float) -> bool:
    return any(
        abs(pitch - reference * factor)
        <= TONE_PITCH_TOLERANCE * reference * factor
        for factor in TONE_PITCH_FOLDS
    )


def _fold(pitch: float, reference: float) -> float:
    return min(
        (abs(pitch / factor - reference), pitch / factor)
        for factor in TONE_PITCH_FOLDS
    )[1]


def _stable_runs(frames: Sequence[ToneFrame]) -> List[StableRun]:
    runs: List[StableRun] = []
    current: List[ToneFrame] = []
    tolerated = 0

    def close() -> None:
        if not current:
            return
        reference = median(frame.pitch_hz for frame in current)
        folded = [_fold(frame.pitch_hz, reference) for frame in current]
        third = max(1, len(folded) // 3)
        runs.append(
            StableRun(
                start_frame=current[0].index,
                frame_count=current[-1].index - current[0].index + 1,
                pitch_hz=median(folded),
                periodicity=median(frame.periodicity for frame in current),
                level_dbfs=mean(frame.level_dbfs for frame in current),
                jitter=(
                    median(
                        abs(later - earlier)
                        for earlier, later in zip(folded, folded[1:])
                    )
                    / reference
                    if len(folded) > 1
                    else 0.0
                ),
                glide=abs(mean(folded[-third:]) - mean(folded[:third]))
                / reference,
            )
        )

    for frame in frames:
        if not current:
            current = [frame]
            tolerated = 0
            continue
        reference = median(item.pitch_hz for item in current)
        missing = frame.index - current[-1].index - 1
        if missing <= TONE_MAX_BRIDGE_FRAMES and _same_pitch(
            frame.pitch_hz, reference
        ):
            current.append(frame)
            tolerated = 0
            continue
        if missing == 0 and tolerated < TONE_MAX_BRIDGE_FRAMES:
            tolerated += 1
            continue
        close()
        current = [frame]
        tolerated = 0
    close()
    return runs


def _silence_profile(samples: array, sample_rate: int) -> SilenceProfile:
    frames = frame_samples(samples, sample_rate, frame_ms=DEAD_FRAME_MS)
    dead = [
        max(frame) <= DEAD_MAX_ABS_SAMPLE and min(frame) >= -DEAD_MAX_ABS_SAMPLE
        for frame in frames
    ]
    voiced = [rms(frame) >= audio_utils.DEFAULT_SILENCE_RMS for frame in frames]
    frame_seconds = DEAD_FRAME_MS / 1000
    runs: List[DeadRun] = []
    pause_dead_fractions: List[float] = []
    index = 0
    heard_speech = False
    while index < len(frames):
        if voiced[index]:
            heard_speech = True
            index += 1
            continue
        start = index
        while index < len(frames) and not voiced[index]:
            index += 1
        pause_dead_fractions.append(sum(dead[start:index]) / (index - start))
        dead_start = start
        while dead_start < index:
            if not dead[dead_start]:
                dead_start += 1
                continue
            dead_end = dead_start
            while dead_end < index and dead[dead_end]:
                dead_end += 1
            length = (dead_end - dead_start) * frame_seconds
            if heard_speech:
                if dead_end >= len(frames):
                    if length >= DEAD_TRAILING_MIN_SECONDS:
                        runs.append(
                            DeadRun(dead_start * frame_seconds, length, True)
                        )
                elif length >= MIN_DEAD_SECONDS and any(voiced[dead_end:]):
                    runs.append(
                        DeadRun(dead_start * frame_seconds, length, False)
                    )
            dead_start = dead_end
    return SilenceProfile(runs, pause_dead_fractions)


def _clipping_span(
    samples: array, sample_rate: int
) -> Optional[Tuple[float, float]]:
    first = last = None
    for index, sample in enumerate(samples):
        if abs(sample) >= CLIPPING_SAMPLE:
            if first is None:
                first = index
            last = index
    if first is None:
        return None
    return first / sample_rate, (last - first + 1) / sample_rate


def analyze_clip(turn: int, role: str, audio: Audio) -> ClipAnalysis:
    samples, sample_rate = decode_audio(audio)
    measurements = analyze_audio(audio)
    frames = _tone_frames(samples, sample_rate)
    return ClipAnalysis(
        turn=turn,
        role=role,
        duration=len(samples) / sample_rate,
        measurements=measurements,
        frames=frames,
        stable_runs=_stable_runs(frames),
        silence=_silence_profile(samples, sample_rate),
        clipping_span=(
            _clipping_span(samples, sample_rate)
            if measurements.clipping_fraction > CLIPPING_MIN_FRACTION
            else None
        ),
    )


def speaker_baselines(
    analyses: Sequence[ClipAnalysis],
) -> Dict[str, SpeakerBaseline]:
    pitches: Dict[str, List[float]] = {}
    holds: Dict[str, List[float]] = {}
    clips: Dict[str, int] = {}
    pauses: Dict[str, List[float]] = {}
    for analysis in analyses:
        pauses.setdefault(analysis.role, []).extend(
            analysis.silence.pause_dead_fractions
        )
        # Frames inside a tone-length run would pull the voice's pitch up.
        held = set()
        for run in analysis.stable_runs:
            if run.duration >= MIN_TONE_SECONDS:
                held.update(
                    range(run.start_frame, run.start_frame + run.frame_count)
                )
        pitches.setdefault(analysis.role, []).extend(
            frame.pitch_hz
            for frame in analysis.frames
            if frame.index not in held
        )
        holds.setdefault(analysis.role, []).extend(
            run.duration for run in analysis.stable_runs if run.frame_count > 1
        )
        clips[analysis.role] = clips.get(analysis.role, 0) + 1
    baselines = {}
    for role in clips:
        role_holds = sorted(holds[role])
        baselines[role] = SpeakerBaseline(
            role=role,
            clips=clips[role],
            median_pitch_hz=median(pitches[role]) if pitches[role] else None,
            typical_hold_seconds=(
                role_holds[int(TYPICAL_HOLD_PERCENTILE * (len(role_holds) - 1))]
                if role_holds
                else None
            ),
            digital_silence=(
                bool(pauses[role])
                and median(pauses[role]) >= DIGITAL_SILENCE_SHARE
            ),
        )
    return baselines


def _position(offset: float, duration: float, clip_duration: float) -> str:
    if clip_duration <= 0 or offset <= POSITION_EDGE_FRACTION * clip_duration:
        return "start"
    if offset + duration >= (1 - POSITION_EDGE_FRACTION) * clip_duration:
        return "end"
    return "middle"


def _event(
    analysis: ClipAnalysis,
    audio: Audio,
    event_type: str,
    offset: float,
    duration: float,
    **extra,
) -> Dict:
    return {
        "type": event_type,
        "turn": analysis.turn,
        "role": analysis.role,
        "offset_seconds": round(offset, 3),
        "duration_seconds": round(duration, 3),
        "start_time": (
            None
            if audio.start_time is None
            else round(audio.start_time + offset, 3)
        ),
        "position": _position(offset, duration, analysis.duration),
        **extra,
    }


def held_tone_events(
    analysis: ClipAnalysis, audio: Audio, baseline: SpeakerBaseline
) -> List[Dict]:
    events = []
    for run in analysis.stable_runs:
        if (
            run.duration < MIN_TONE_SECONDS
            or run.periodicity < TONE_MIN_PERIODICITY
            or run.jitter > TONE_MAX_JITTER
            or run.glide > TONE_MAX_GLIDE
        ):
            continue
        hold_ratio = (
            run.duration / baseline.typical_hold_seconds
            if baseline.typical_hold_seconds
            else None
        )
        pitch_ratio = (
            run.pitch_hz / baseline.median_pitch_hz
            if baseline.median_pitch_hz
            else None
        )
        if not (
            (hold_ratio is not None and hold_ratio >= TONE_HOLD_RATIO)
            or (pitch_ratio is not None and pitch_ratio >= TONE_PITCH_RATIO)
            or run.duration >= TONE_ABSOLUTE_SECONDS
        ):
            continue
        events.append(
            _event(
                analysis,
                audio,
                "held_tone",
                run.offset,
                run.duration,
                pitch_hz=round(run.pitch_hz, 1),
                periodicity=round(run.periodicity, 3),
                level_dbfs=round(run.level_dbfs, 1),
                jitter=round(run.jitter, 4),
                glide=round(run.glide, 4),
                hold_ratio=None if hold_ratio is None else round(hold_ratio, 2),
                pitch_ratio=(
                    None if pitch_ratio is None else round(pitch_ratio, 2)
                ),
            )
        )
    return events


def detect_audio_glitches(test_case: ConversationalTestCase) -> GlitchReport:
    turns = test_case.turns
    analyses: Dict[int, ClipAnalysis] = {}
    events: List[Dict] = []
    turns_with_audio: List[int] = []
    for index, turn in enumerate(turns):
        if turn.audio is None:
            continue
        turns_with_audio.append(index)
        try:
            analyses[index] = analyze_clip(index, turn.role, turn.audio)
        except (TypeError, ValueError) as error:
            events.append(
                {
                    "type": "audio_undecodable",
                    "turn": index,
                    "role": turn.role,
                    "offset_seconds": 0.0,
                    "duration_seconds": 0.0,
                    "start_time": turn.audio.start_time,
                    "position": "start",
                    "reason": str(error),
                }
            )

    baselines = speaker_baselines(list(analyses.values()))
    for index, analysis in analyses.items():
        audio = turns[index].audio
        baseline = baselines[analysis.role]
        events.extend(held_tone_events(analysis, audio, baseline))
        for run in [] if baseline.digital_silence else analysis.dead_runs:
            events.append(
                _event(
                    analysis,
                    audio,
                    "dead_stream",
                    run.offset,
                    run.duration,
                    runs_to_end=run.runs_to_end,
                )
            )
        if analysis.clipping_span is not None:
            offset, duration = analysis.clipping_span
            events.append(
                _event(
                    analysis,
                    audio,
                    "clipping",
                    offset,
                    duration,
                    fraction=round(analysis.measurements.clipping_fraction, 4),
                )
            )

    timeline_available = all(
        turns[index].audio.start_time is not None for index in turns_with_audio
    )
    if timeline_available:
        entries = build_audio_timeline(turns)
        for current, following in zip(entries, entries[1:]):
            analysis = analyses.get(current.turn_index)
            if analysis is None or current.role != following.role:
                continue
            ends_mid_speech = analysis.measurements.ends_abruptly or (
                analysis.trailing_dead_run
                and not baselines[analysis.role].digital_silence
            )
            if not ends_mid_speech:
                continue
            gap = following.start_time - current.end_time
            if not 0 < gap <= STREAM_CUT_MAX_GAP_SECONDS:
                continue
            trailing = [
                event
                for event in events
                if event["type"] == "dead_stream"
                and event["turn"] == current.turn_index
                and event["runs_to_end"]
            ]
            for event in trailing:
                events.remove(event)
            events.append(
                _event(
                    analysis,
                    turns[current.turn_index].audio,
                    "stream_cut",
                    (
                        trailing[0]["offset_seconds"]
                        if trailing
                        else analysis.duration
                    ),
                    0.0,
                    evidence="trailing_dead_run" if trailing else "abrupt_end",
                    gap_seconds=round(gap, 3),
                    resumed_turn=following.turn_index,
                )
            )

    events.sort(key=lambda event: (event["turn"], event["offset_seconds"]))
    return GlitchReport(
        events=events,
        turns_with_audio=turns_with_audio,
        baselines={role: asdict(item) for role, item in baselines.items()},
        timeline_available=timeline_available,
    )

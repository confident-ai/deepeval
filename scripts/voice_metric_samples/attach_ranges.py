import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import deepeval.metrics as M
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from deepeval.voice.connectors import audio_utils

ROOT = Path(__file__).resolve().parents[2]
CALLS = json.loads(
    (ROOT / "docs/public/audio/voice-metrics/calls.json").read_text()
)
SAMPLES_PATH = ROOT / "docs/src/components/AudioSample/samples.json"
SAMPLES = json.loads(SAMPLES_PATH.read_text())
AUDIO = ROOT / "docs/public/audio/voice-metrics"
RATE = 24000

METRICS = {
    "audio-dropout": M.AudioDropoutMetric,
    "audio-clipping": M.AudioClippingMetric,
    "audio-loop": M.AudioLoopMetric,
    "abrupt-cut-off": M.AbruptCutoffMetric,
    "held-tone": M.HeldToneMetric,
    "dead-stream": M.DeadStreamMetric,
    "echo": M.EchoMetric,
    "background-noise": M.BackgroundNoiseMetric,
    "loudness": M.LoudnessMetric,
    "loudness-consistency": M.LoudnessConsistencyMetric,
    "volume-variance": M.VolumeVarianceMetric,
    "pitch-variation": M.PitchVariationMetric,
    "abrupt-pitch-jump": M.AbruptPitchJumpMetric,
    "pitch-consistency": M.PitchConsistencyMetric,
    "speaking-rate": M.SpeakingRateMetric,
    "pauses": M.PausesMetric,
    "response-latency": M.ResponseLatencyMetric,
    "response-delay-consistency": M.ResponseDelayConsistencyMetric,
    "time-to-first-response": M.TimeToFirstResponseMetric,
    "agent-interruption": M.AgentInterruptionMetric,
    "failed-response": M.FailedResponseMetric,
    "silent-reply": M.SilentReplyMetric,
    "unexpected-call-end": M.UnexpectedCallEndMetric,
}


def decode(file, channels):
    raw = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(AUDIO / file),
            "-f",
            "s16le",
            "-ar",
            str(RATE),
            "-ac",
            str(channels),
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype="<i2").reshape(-1, channels)


def as_audio(samples, start):
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(
            np.ascontiguousarray(samples).tobytes(), RATE, num_channels=1
        ),
        "audio/wav",
        sampleRate=RATE,
        encoding="wav",
        duration=len(samples) / RATE,
        start_time=start,
    )


def test_case(entry):
    stereo = entry["stereo"]
    pcm = decode(entry["file"], 2 if stereo else 1)
    turns = []
    for turn in entry["turns"]:
        lane = (1 if turn["role"] == "assistant" else 0) if stereo else 0
        clip = pcm[int(turn["start"] * RATE) : int(turn["end"] * RATE), lane]
        turns.append(
            Turn(
                role=turn["role"],
                content=turn["text"] or "",
                audio=as_audio(clip, turn["start"]),
            )
        )
    recording = None
    if stereo:
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        handle.write(
            audio_utils.pcm16_to_wav_bytes(
                np.ascontiguousarray(pcm).tobytes(), RATE, num_channels=2
            )
        )
        handle.close()
        recording = handle.name
    return ConversationalTestCase(
        turns=turns,
        metadata=entry.get("metadata") or None,
        call_recording_path=recording,
    )


for name, metric_class in METRICS.items():
    entry = CALLS[name]
    metric = metric_class(threshold=None)
    metric.measure(test_case(entry), _show_indicator=False)
    regions = []
    for event in metric.score_breakdown.get("events", []):
        if "start_time" not in event or "duration_seconds" not in event:
            continue
        start = event["start_time"]
        regions.append(
            {
                "start": round(start, 3),
                "end": round(start + event["duration_seconds"], 3),
                "label": event["type"],
            }
        )
    SAMPLES[name]["regions"] = regions
    print(
        f"{name:28} score={metric.score} {[(r['label'], r['start'], r['end']) for r in regions]}"
    )

SAMPLES_PATH.write_text(json.dumps(SAMPLES, separators=(",", ":")) + "\n")

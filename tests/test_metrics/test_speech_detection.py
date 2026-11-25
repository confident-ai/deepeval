import builtins
import hashlib
import shutil
import threading

import numpy as np
import pytest
import requests

from deepeval.metrics import (
    AbruptCutoffMetric,
    AgentInterruptionMetric,
    AgentResponsivenessMetric,
    FailedResponseMetric,
    RepromptMetric,
    ResponseDelayConsistencyMetric,
    ResponseLatencyMetric,
    TimeToFirstResponseMetric,
    TurnTakingNaturalnessMetric,
    UserInterruptionMetric,
    VoiceReliabilityMetric,
)
from deepeval.metrics.voice.utils import call_activity as activity_module
from deepeval.metrics.voice.utils import speech_detection
from deepeval.metrics.voice.utils.call_activity import (
    recording_activity,
    turn_activity,
)
from deepeval.test_case import ConversationalTestCase, Turn
from tests.test_metrics.helpers import (
    recording,
    speech_like,
    stereo_recording,
    timed_call,
    to_audio,
    tone,
    zeros,
)

WINDOW_SECONDS = (
    speech_detection.SPEECH_WINDOW_SAMPLES / speech_detection.SPEECH_MODEL_RATE
)
RECORDING_METRICS = [
    AgentInterruptionMetric,
    ResponseLatencyMetric,
    ResponseDelayConsistencyMetric,
    TimeToFirstResponseMetric,
    TurnTakingNaturalnessMetric,
    AbruptCutoffMetric,
    UserInterruptionMetric,
]
TURN_AUDIO_METRICS = [
    AgentInterruptionMetric,
    ResponseLatencyMetric,
    ResponseDelayConsistencyMetric,
    TimeToFirstResponseMetric,
    TurnTakingNaturalnessMetric,
    UserInterruptionMetric,
    FailedResponseMetric,
    RepromptMetric,
    AgentResponsivenessMetric,
    VoiceReliabilityMetric,
]


@pytest.fixture(autouse=True)
def _fresh_caches():
    activity_module._recording_cache.clear()
    activity_module._turn_audio_cache.clear()
    yield
    activity_module._recording_cache.clear()
    activity_module._turn_audio_cache.clear()


def _model_hears_speech_except(*quiet):
    """A stand-in for the speech model: speech everywhere but `quiet`, a list
    of (start, end) seconds."""

    def probabilities(samples, sample_rate):
        count = int(
            len(samples) * speech_detection.SPEECH_MODEL_RATE / sample_rate
        ) // (speech_detection.SPEECH_WINDOW_SAMPLES)
        return [
            (0.0 if any(a <= i * WINDOW_SECONDS < b for a, b in quiet) else 1.0)
            for i in range(count)
        ]

    return probabilities


def _caller_spans(path, use_speech_detection):
    activity = recording_activity(recording(path), use_speech_detection)
    return [(round(s.start, 1), round(s.end, 1)) for s in activity.caller]


def test_a_quiet_blip_under_the_callers_speech_level_is_not_talking(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        speech_detection, "speech_probabilities", _model_hears_speech_except()
    )
    # Speech near RMS 5,700; the blip at RMS 570 clears the fixed bar of 300
    # but not 15% of the caller's level.
    path = stereo_recording(
        tmp_path / "call.wav",
        agent=[(0.0, tone(1.0, 300))],
        caller=[
            (1.0, tone(2.0, 220, amplitude=8000)),
            (5.0, tone(0.5, 220, amplitude=800)),
        ],
    )
    assert _caller_spans(path, True) == [(1.0, 3.0)]
    assert _caller_spans(path, False) == [(1.0, 3.0), (5.0, 5.5)]


def test_loud_sound_the_model_does_not_hear_as_speech_is_not_talking(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        speech_detection,
        "speech_probabilities",
        _model_hears_speech_except((5.0, 5.6)),
    )
    path = stereo_recording(
        tmp_path / "call.wav",
        agent=[(0.0, tone(1.0, 300))],
        caller=[(1.0, tone(2.0, 220)), (5.0, tone(0.5, 220))],
    )
    assert _caller_spans(path, True) == [(1.0, 3.0)]
    assert _caller_spans(path, False) == [(1.0, 3.0), (5.0, 5.5)]


@pytest.mark.parametrize(
    "use_speech_detection,score", [(False, 0.0), (True, 1.0)]
)
def test_caller_noise_under_the_agent_is_not_an_interruption(
    tmp_path, monkeypatch, use_speech_detection, score
):
    monkeypatch.setattr(
        speech_detection,
        "speech_probabilities",
        _model_hears_speech_except((2.5, 4.5)),
    )
    path = stereo_recording(
        tmp_path / "call.wav",
        agent=[(0.0, speech_like(1.5)), (3.0, speech_like(3.0))],
        caller=[(2.6, tone(1.5, 150))],
    )
    metric = AgentInterruptionMetric(use_speech_detection=use_speech_detection)
    metric.measure(
        ConversationalTestCase(turns=timed_call([0.5]), audio=recording(path)),
        _show_indicator=False,
    )
    assert metric.score == score


@pytest.mark.parametrize("metric_class", RECORDING_METRICS)
@pytest.mark.parametrize("use_speech_detection", [True, False])
def test_each_timing_metric_hands_the_option_to_the_recording_reader(
    tmp_path, monkeypatch, metric_class, use_speech_detection
):
    calls = []

    def fake_speech_frames(samples, sample_rate, levels):
        calls.append(len(levels))
        return [False] * len(levels)

    monkeypatch.setattr(activity_module, "speech_frames", fake_speech_frames)
    path = stereo_recording(
        tmp_path / "call.wav",
        agent=[(0.0, speech_like(1.5)), (4.0, speech_like(1.5))],
        caller=[(2.0, speech_like(1.0))],
    )
    metric = metric_class(use_speech_detection=use_speech_detection)
    metric.measure(
        ConversationalTestCase(turns=timed_call([0.5]), audio=recording(path)),
        _show_indicator=False,
    )
    assert bool(calls) is use_speech_detection


@pytest.mark.parametrize(
    "use_speech_detection,agent_starts_at", [(False, 2.5), (True, 3.5)]
)
def test_each_turns_audio_counts_only_the_speech_the_model_hears(
    monkeypatch, use_speech_detection, agent_starts_at
):
    # Noise opens the reply's audio; the model hears no speech in it.
    monkeypatch.setattr(
        speech_detection,
        "speech_probabilities",
        _model_hears_speech_except((0.0, 1.0)),
    )
    reply = tone(1.0, 150)
    reply.extend(speech_like(2.0))
    turns = timed_call([0.5])
    turns[1].audio = to_audio(reply, start_time=2.5)
    activity = turn_activity(
        ConversationalTestCase(turns=turns), use_speech_detection
    )
    assert activity.agent[0].start == pytest.approx(agent_starts_at, abs=0.05)


@pytest.mark.parametrize(
    "use_speech_detection,caller_heard",
    [(False, (0.0, 4.0)), (True, (1.0, 1.1))],
)
def test_a_short_faint_word_the_model_hears_in_a_turns_audio_counts(
    monkeypatch, use_speech_detection, caller_heard
):
    monkeypatch.setattr(
        speech_detection, "speech_probabilities", _model_hears_speech_except()
    )
    # A faint "yes": quieter and shorter than loudness alone keeps, so
    # without the model nothing is heard and the whole clip counts.
    caller = zeros(1.0)
    caller.extend(tone(0.1, 200, amplitude=280))
    caller.extend(zeros(2.9))
    turns = [
        Turn(role="user", content="Yes", audio=to_audio(caller, 0.0)),
        Turn(
            role="assistant",
            content="Great.",
            audio=to_audio(speech_like(2.0), 6.0),
        ),
    ]
    activity = turn_activity(
        ConversationalTestCase(turns=turns), use_speech_detection
    )
    [heard] = activity.caller
    assert (heard.start, heard.end) == pytest.approx(caller_heard, abs=0.03)


@pytest.mark.parametrize("metric_class", TURN_AUDIO_METRICS)
@pytest.mark.parametrize("use_speech_detection", [True, False])
def test_each_metric_hands_the_option_to_the_turn_audio_reader(
    monkeypatch, metric_class, use_speech_detection
):
    calls = []

    def fake_speech_frames(samples, sample_rate, levels, min_level):
        calls.append(len(levels))
        return [False] * len(levels)

    monkeypatch.setattr(activity_module, "speech_frames", fake_speech_frames)
    metric = metric_class(use_speech_detection=use_speech_detection)
    metric.measure(
        ConversationalTestCase(turns=timed_call([0.5, 0.5])),
        _show_indicator=False,
    )
    assert bool(calls) is use_speech_detection


@pytest.mark.parametrize(
    "metric_class", list(dict.fromkeys(RECORDING_METRICS + TURN_AUDIO_METRICS))
)
def test_speech_detection_is_on_by_default(metric_class):
    assert metric_class().use_speech_detection is True


def test_speech_detection_without_onnxruntime_says_how_to_fix_it(monkeypatch):
    real_import = builtins.__import__

    def no_onnxruntime(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ModuleNotFoundError("No module named 'onnxruntime'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(speech_detection, "_onnxruntime_mod", None)
    monkeypatch.setattr(builtins, "__import__", no_onnxruntime)

    with pytest.raises(ImportError) as error:
        speech_detection.get_onnxruntime()
    assert "onnxruntime is required" in str(error.value)
    assert "use_speech_detection=False" in str(error.value)


class _FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


def _serve_model(monkeypatch, tmp_path, content, expected=b"model"):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        speech_detection,
        "SPEECH_MODEL_SHA256",
        hashlib.sha256(expected).hexdigest(),
    )
    downloads = []

    def fake_get(url, **kwargs):
        downloads.append(url)
        return _FakeResponse(content)

    monkeypatch.setattr(requests, "get", fake_get)
    return downloads


def test_the_speech_model_is_downloaded_once_and_then_reused(
    tmp_path, monkeypatch
):
    downloads = _serve_model(monkeypatch, tmp_path, b"model")
    path = speech_detection.speech_model_path()
    assert path == tmp_path / "deepeval" / "models" / "silero_vad.onnx"
    assert path.read_bytes() == b"model"
    assert speech_detection.speech_model_path() == path
    assert downloads == [speech_detection.SPEECH_MODEL_URL]


def test_a_speech_model_with_the_wrong_checksum_is_not_kept(
    tmp_path, monkeypatch
):
    _serve_model(monkeypatch, tmp_path, b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        speech_detection.speech_model_path()
    models = tmp_path / "deepeval" / "models"
    assert not models.exists() or not any(models.iterdir())


def test_a_failed_download_names_the_ways_out(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def failing_get(url, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", failing_get)
    with pytest.raises(RuntimeError) as error:
        speech_detection.speech_model_path()
    message = str(error.value)
    assert str(tmp_path / "deepeval" / "models" / "silero_vad.onnx") in message
    assert "use_speech_detection=False" in message


class _FakeSession:
    def run(self, outputs, inputs):
        return np.array([[0.7]], dtype=np.float32), inputs["state"]


@pytest.mark.parametrize("sample_rate", [8000, 16000, 24000])
def test_speech_probabilities_cover_the_same_time_at_any_sample_rate(
    monkeypatch, sample_rate
):
    monkeypatch.setattr(speech_detection, "_speech_session", _FakeSession)
    probabilities = speech_detection.speech_probabilities(
        np.zeros(2 * sample_rate, dtype=np.int16), sample_rate
    )
    assert len(probabilities) == int(2 / WINDOW_SECONDS)
    assert probabilities[0] == pytest.approx(0.7)


def _count_readings(monkeypatch):
    readings = []
    real = activity_module._recording_activity

    def counted(wav, use_speech_detection, agent_channel):
        readings.append(use_speech_detection)
        return real(wav, use_speech_detection, agent_channel)

    monkeypatch.setattr(activity_module, "_recording_activity", counted)
    return readings


def _short_call(path):
    return stereo_recording(
        path,
        agent=[(0.0, speech_like(1.5)), (4.0, speech_like(1.5))],
        caller=[(2.0, speech_like(1.0))],
    )


def test_copies_of_the_same_call_are_read_once(tmp_path, monkeypatch):
    readings = _count_readings(monkeypatch)
    first = _short_call(tmp_path / "first.wav")
    copy = str(tmp_path / "copy.wav")
    shutil.copy(first, copy)
    assert recording_activity(recording(first), False) == recording_activity(
        recording(copy), False
    )
    assert readings == [False]
    recording_activity(recording(copy), True)
    assert readings == [False, True]


def test_a_rewritten_file_is_read_again(tmp_path, monkeypatch):
    readings = _count_readings(monkeypatch)
    path = tmp_path / "call.wav"
    stereo_recording(path, agent=[(0.0, speech_like(1.5))])
    recording_activity(recording(path), False)
    _short_call(path)
    recording_activity(recording(path), False)
    assert len(readings) == 2


def test_metrics_reading_the_same_call_at_once_share_one_reading(
    tmp_path, monkeypatch
):
    readings = _count_readings(monkeypatch)
    paths = []
    for index in range(6):
        paths.append(str(tmp_path / f"copy-{index}.wav"))
        _short_call(paths[-1])
    start = threading.Barrier(len(paths))
    results = []

    def read(path):
        start.wait()
        results.append(recording_activity(recording(path), False))

    threads = [threading.Thread(target=read, args=(path,)) for path in paths]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert readings == [False]
    assert len(results) == 6 and all(r == results[0] for r in results)

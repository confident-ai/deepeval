"""Whether a caller is speaking, not just making sound: a loudness bar set
from the caller's own speech level, plus the Silero VAD speech model."""

import hashlib
import os
import statistics
import tempfile
import threading
from pathlib import Path
from typing import Any, List, Sequence

import requests

from deepeval.metrics.voice.utils.analysis import get_numpy, resample
from deepeval.voice.connectors import audio_utils

# Silero VAD v6.2 (MIT), pinned to a commit so the file never changes.
SPEECH_MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    "be95df9152c0d7618fa1edfeb296fc3dae32376f/src/silero_vad/data/"
    "silero_vad.onnx"
)
SPEECH_MODEL_SHA256 = (
    "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
)
SPEECH_MODEL_FILE = "silero_vad.onnx"
SPEECH_MODEL_RATE = 16000
SPEECH_WINDOW_SAMPLES = 512
SPEECH_CONTEXT_SAMPLES = 64
SPEECH_MIN_PROBABILITY = 0.5
# Sound quieter than this share of the caller's typical speech level is a
# blip, even when the model hears speech in it.
SPEECH_MIN_LEVEL_RATIO = 0.15
DOWNLOAD_TIMEOUT = (10, 60)
_DISABLE_HINT = "or pass use_speech_detection=False"

# Lazy import cache
_onnxruntime_mod = None
_onnxruntime_import_error = None
_session = None
_session_lock = threading.Lock()


def get_onnxruntime() -> Any:
    """Return the onnxruntime module, or raise ImportError with root cause."""
    global _onnxruntime_mod, _onnxruntime_import_error
    if _onnxruntime_mod is not None:
        return _onnxruntime_mod
    try:
        import onnxruntime

        _onnxruntime_mod = onnxruntime
        return _onnxruntime_mod
    except Exception as e:
        _onnxruntime_import_error = e
        raise ImportError(
            "onnxruntime is required for speech detection. Install it with "
            f"`pip install onnxruntime`, {_DISABLE_HINT}. Root cause: {e}"
        )


def _cache_root() -> Path:
    """`${XDG_CACHE_HOME:-~/.cache}/deepeval`."""
    base = os.environ.get("XDG_CACHE_HOME") or ""
    # The XDG spec says a relative value is invalid and should be ignored.
    root = Path(base) if os.path.isabs(base) else Path.home() / ".cache"
    return root / "deepeval"


def speech_model_path() -> Path:
    """The speech model on disk, downloaded and checksum-verified the first
    time it is needed."""
    target = _cache_root() / "models" / SPEECH_MODEL_FILE
    if target.is_file():
        return target
    failure = (
        f"Could not download the speech detection model from "
        f"{SPEECH_MODEL_URL}. Place that file at {target}, {_DISABLE_HINT}."
    )
    try:
        response = requests.get(SPEECH_MODEL_URL, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(f"{failure} Root cause: {error}") from error
    if hashlib.sha256(response.content).hexdigest() != SPEECH_MODEL_SHA256:
        raise RuntimeError(f"{failure} Root cause: checksum mismatch.")
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=target.parent, suffix=".part")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(response.content)
        os.chmod(temp, 0o644)
        os.replace(temp, target)
    except BaseException:
        if os.path.exists(temp):
            os.remove(temp)
        raise
    return target


def _speech_session() -> Any:
    global _session
    with _session_lock:
        if _session is None:
            onnxruntime = get_onnxruntime()
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            _session = onnxruntime.InferenceSession(
                str(speech_model_path()),
                options,
                providers=["CPUExecutionProvider"],
            )
    return _session


def speech_probabilities(samples: Any, sample_rate: int) -> List[float]:
    """The model's speech probability for each 32 ms window of `samples`."""
    np = get_numpy()
    audio = resample(samples, sample_rate, SPEECH_MODEL_RATE) / 32768.0
    session = _speech_session()
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(SPEECH_CONTEXT_SAMPLES, dtype=np.float32)
    rate = np.array(SPEECH_MODEL_RATE, dtype=np.int64)
    probabilities: List[float] = []
    for start in range(
        0, len(audio) - SPEECH_WINDOW_SAMPLES + 1, SPEECH_WINDOW_SAMPLES
    ):
        window = np.concatenate(
            [context, audio[start : start + SPEECH_WINDOW_SAMPLES]]
        )
        output, state = session.run(
            None, {"input": window[None, :], "state": state, "sr": rate}
        )
        probabilities.append(float(output[0, 0]))
        context = window[-SPEECH_CONTEXT_SAMPLES:]
    return probabilities


def speech_frames(
    samples: Any,
    sample_rate: int,
    levels: Sequence[float],
    min_level: float = audio_utils.DEFAULT_SILENCE_RMS,
) -> List[bool]:
    """For each 20 ms frame with level `levels[i]`, whether the speaker is
    speaking: loud enough for this speaker AND heard as speech."""
    voiced = [level for level in levels if level >= min_level]
    if not voiced:
        return [False] * len(levels)
    bar = max(min_level, SPEECH_MIN_LEVEL_RATIO * statistics.median(voiced))
    probabilities = speech_probabilities(samples, sample_rate)
    if not probabilities:
        return [False] * len(levels)
    frame_seconds = audio_utils.DEFAULT_FRAME_MS / 1000
    window_seconds = SPEECH_WINDOW_SAMPLES / SPEECH_MODEL_RATE
    frames = []
    for index, level in enumerate(levels):
        window = min(
            int((index + 0.5) * frame_seconds / window_seconds),
            len(probabilities) - 1,
        )
        frames.append(
            level >= bar and probabilities[window] >= SPEECH_MIN_PROBABILITY
        )
    return frames

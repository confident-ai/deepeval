import contextlib
import os
import wave

from deepeval.dataset import ConversationalGolden
from deepeval.simulator import ConversationSimulator
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from deepeval.voice.recording import CallRecorder
from tests.test_core.test_simulator.helpers import (
    StaticSimulatorModel,
    async_static_callback,
)
from tests.test_core.test_voice.helpers import (
    RATE,
    EchoAgent,
    StubSTT,
    StubTTS,
    wav_audio,
)


def _read(path):
    with contextlib.closing(wave.open(path, "rb")) as reader:
        frames = reader.readframes(reader.getnframes())
        return frames, reader.getnchannels(), reader.getframerate()


def test_recorder_places_channels_at_their_offsets(tmp_path):
    recorder = CallRecorder(sample_rate=RATE)
    recorder.add("user", b"\x10\x27" * RATE, RATE, 100.0)
    recorder.add("agent", b"\xf0\xd8" * RATE, RATE, 100.5)
    path = recorder.finish()

    frames, channels, rate = _read(path)
    assert channels == 2
    assert rate == RATE
    assert len(frames) // 4 == int(1.5 * RATE)
    half_second = int(0.5 * RATE)
    right_start = frames[2 : half_second * 4 : 4]
    assert set(right_start) == {0}
    os.unlink(path)


def test_recorder_discard_removes_spools():
    recorder = CallRecorder(sample_rate=RATE)
    recorder.add("user", b"\x10\x27" * 100, RATE, 1.0)
    paths = [spool["path"] for spool in recorder._spools.values()]
    recorder.discard()
    assert all(not os.path.exists(path) for path in paths)
    assert recorder.finish() is None


def test_voice_simulation_records_the_call():
    cases = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=CallbackVoiceConnector(EchoAgent()),
            tts_model=StubTTS(),
            stt_model=StubSTT(),
            output_dir=None,
            combine_audio_files=False,
            record_call=True,
        ),
    ).simulate(
        [ConversationalGolden(scenario="Refund", expected_outcome="Done.")],
        max_user_simulations=1,
    )

    path = cases[0].call_recording_path
    assert path is not None and os.path.exists(path)
    frames, channels, rate = _read(path)
    assert channels == 2
    assert len(frames) > 0
    os.unlink(path)


def test_recording_off_by_default():
    cases = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=CallbackVoiceConnector(EchoAgent()),
            tts_model=StubTTS(),
            stt_model=StubSTT(),
            output_dir=None,
            combine_audio_files=False,
        ),
    ).simulate(
        [ConversationalGolden(scenario="Refund", expected_outcome="Done.")],
        max_user_simulations=1,
    )

    assert cases[0].call_recording_path is None


def test_chunk_pcm_unwraps_audio_chunks():
    import base64

    from deepeval.test_case import AudioChunk
    from deepeval.voice.recording import _chunk_pcm

    pcm = b"\x10\x27" * 240
    chunk = AudioChunk(
        dataBase64=base64.b64encode(pcm).decode("ascii"),
        mimeType="audio/pcm",
        sampleRate=16000,
    )

    unwrapped, rate = _chunk_pcm(chunk, 24000)
    assert unwrapped == pcm
    assert rate == 16000
    assert _chunk_pcm(b"abcd", 24000) == (b"abcd", 24000)

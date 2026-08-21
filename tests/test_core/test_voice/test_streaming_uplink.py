"""Sending the caller's speech while it is still being synthesized."""

import asyncio
from typing import List, Optional, Union

import pytest

from deepeval.dataset import BackgroundNoiseSettings
from deepeval.test_case import Audio, AudioChunk
from deepeval.voice.background import BackgroundMixer, mix_background
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.transports.callback import (
    CallbackVoiceConnector,
)
from deepeval.voice.connectors.transports.base import UplinkStream
from deepeval.voice.connectors.transports.websocket import (
    BaseWebSocketConnector,
    InboundEvent,
)
from deepeval.voice.connectors.types import ConnectorTurn
from deepeval.voice.streaming import collect_pcm_chunks

_RATE = 24000
# 40ms of a constant tone: two whole 20ms wire frames, so nothing is padded.
_FRAME = b"\x10\x27" * 960
_WIRE_FRAME_BYTES = int(_RATE * audio_utils.DEFAULT_FRAME_MS / 1000) * 2


def _chunk(data: bytes, *, final: bool = False) -> AudioChunk:
    return AudioChunk.from_bytes(
        data, "audio/pcm", sampleRate=_RATE, encoding="pcm", final=final
    )


async def _stream(*payloads: bytes, delay: float = 0.0):
    for index, payload in enumerate(payloads):
        if delay and index:
            await asyncio.sleep(delay)
        yield _chunk(payload, final=index == len(payloads) - 1)


class _Wire(BaseWebSocketConnector):
    """A WebSocket connector with the socket replaced by a list."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sent: List[bytes] = []
        self._uplink = UplinkStream()

    async def _open_session(self) -> str:
        return "wss://example.invalid"

    def _encode_outbound(self, pcm: bytes) -> Union[str, bytes]:
        return pcm

    def _decode_inbound(self, raw) -> Optional[InboundEvent]:
        return None

    async def _send(self, message) -> None:
        self.sent.append(message)

    @property
    def bytes_sent(self) -> int:
        return sum(len(message) for message in self.sent)


@pytest.mark.asyncio
async def test_a_transport_that_can_forward_frames_sends_before_synthesis_ends():
    """The point of the exercise: the agent hears the opening words early."""
    import time

    wire = _Wire(trailing_silence_ms=0)
    seen_midway = []

    async def _watch():
        # After the first frame but well before the last one is synthesized.
        await asyncio.sleep(0.05)
        seen_midway.append((wire.bytes_sent, time.perf_counter()))

    watcher = asyncio.create_task(_watch())
    result = await wire.stream_uplink_chunks(
        _stream(_FRAME, _FRAME, _FRAME, delay=0.2)
    )
    await watcher

    sent_midway, midway_at = seen_midway[0]
    assert sent_midway == len(_FRAME)
    assert wire.bytes_sent == len(_FRAME) * 3
    assert result.audio.duration == pytest.approx(0.12)
    # Sending began on the first frame, not once the last one existed.
    assert result.first_frame_at is not None
    assert result.first_frame_at < midway_at


@pytest.mark.asyncio
async def test_speech_is_not_sent_faster_than_it_could_be_spoken():
    """A socket takes an utterance far faster than anyone could say it.

    An agent handed the caller's words that fast — trailing quiet and all —
    answers before the caller could have finished them. The recording then
    shows the reply beginning inside the caller's own turn and the wait before
    it comes out negative, and an agent listening for a pause cuts in early.
    """
    import time

    wire = _Wire(trailing_silence_ms=0)
    # Half a second of speech, synthesized as fast as the loop will allow.
    began = time.perf_counter()
    result = await wire.stream_uplink_chunks(_stream(*[_FRAME] * 12))
    elapsed = time.perf_counter() - began

    assert wire.bytes_sent == len(_FRAME) * 12
    assert result.audio.duration == pytest.approx(0.48)
    assert elapsed == pytest.approx(result.audio.duration, abs=0.1)


@pytest.mark.asyncio
async def test_frames_that_do_not_divide_evenly_are_not_padded_mid_utterance():
    """Silence spliced at every boundary would click and stretch the utterance."""
    wire = _Wire(trailing_silence_ms=0)
    ragged = b"\x10\x27" * 500  # 1000 bytes: a partial wire frame

    result = await wire.stream_uplink_chunks(_stream(ragged, ragged, ragged))

    pcm, _, _ = audio_utils.wav_bytes_to_pcm16(result.audio.get_bytes())
    on_the_wire = b"".join(wire.sent)
    assert pcm == ragged * 3
    # Only the final frame is padded, and only up to the frame size.
    assert on_the_wire.startswith(pcm)
    assert len(on_the_wire) - len(pcm) < _WIRE_FRAME_BYTES


@pytest.mark.asyncio
async def test_trailing_silence_closes_a_full_turn_but_not_a_barge():
    full = _Wire(trailing_silence_ms=100)
    await full.stream_uplink_chunks(_stream(_FRAME), trailing_silence=True)

    barge = _Wire(trailing_silence_ms=100)
    await barge.stream_uplink_chunks(_stream(_FRAME), trailing_silence=False)

    padding = full.bytes_sent - barge.bytes_sent
    assert padding == int(_RATE * 0.1) * 2


@pytest.mark.asyncio
async def test_a_cancelled_uplink_stops_sending_but_keeps_recording():
    """The turn still has to say what the caller said, even if it went nowhere."""
    wire = _Wire(trailing_silence_ms=0)

    async def _cancel_after_first():
        await asyncio.sleep(0.05)
        await wire.stop_uplink()

    canceller = asyncio.create_task(_cancel_after_first())
    result = await wire.stream_uplink_chunks(
        _stream(_FRAME, _FRAME, _FRAME, delay=0.2)
    )
    await canceller

    assert wire.bytes_sent == len(_FRAME)
    pcm, _, _ = audio_utils.wav_bytes_to_pcm16(result.audio.get_bytes())
    assert len(pcm) == len(_FRAME) * 3


@pytest.mark.asyncio
async def test_a_transport_that_needs_the_whole_utterance_still_works():
    """The in-process agent is handed one utterance, so the default buffers."""
    received: List[Audio] = []

    async def agent(audio: Audio) -> ConnectorTurn:
        received.append(audio)
        return ConnectorTurn(audio=audio, transcript="Agent reply")

    connector = CallbackVoiceConnector(agent)
    await connector.connect()
    try:
        result = await connector.stream_uplink_chunks(_stream(_FRAME, _FRAME))
        await asyncio.sleep(0.05)
    finally:
        await connector.disconnect()

    pcm, rate, _ = audio_utils.wav_bytes_to_pcm16(result.audio.get_bytes())
    assert pcm == _FRAME * 2
    assert rate == _RATE
    assert received and received[0].get_bytes() == result.audio.get_bytes()


class _StreamingTTS:
    """Speech that takes a moment to begin and then arrives in frames."""

    def __init__(
        self,
        first_frame_delay: float = 0.1,
        frames: int = 3,
        frame_delay: float = 0.0,
    ):
        self.first_frame_delay = first_frame_delay
        self.frames = frames
        self.frame_delay = frame_delay

    def supports_streaming(self) -> bool:
        return True

    def synthesis_cost(self, text: str) -> float:
        return 0.5

    async def a_synthesize_stream(self, text: str, **kwargs):
        await asyncio.sleep(self.first_frame_delay)
        for index in range(self.frames):
            if self.frame_delay and index:
                await asyncio.sleep(self.frame_delay)
            yield _chunk(_FRAME, final=index == self.frames - 1)


class _BufferedTTS:
    def supports_streaming(self) -> bool:
        return False

    async def a_synthesize(self, text: str, **kwargs):
        return (
            Audio.from_bytes(
                audio_utils.pcm16_to_wav_bytes(_FRAME, _RATE), "audio/wav"
            ),
            0.25,
        )


class _STT:
    truncated_audio_pad_seconds = 0.0

    async def a_transcribe(self, audio, **kwargs):
        return "Agent reply", None


def _voice_simulator(connector, tts):
    from deepeval.simulator import ConversationSimulator
    from deepeval.voice import VoiceConfig
    from tests.test_core.test_simulator.helpers import StaticSimulatorModel

    return ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=connector,
            tts_model=tts,
            stt_model=_STT(),
            output_dir=None,
            combine_audio_files=False,
        ),
    )


@pytest.mark.asyncio
async def test_an_utterance_is_timed_from_its_first_frame_not_from_synthesis():
    """Timing the clip from synthesis would place it before the agent heard it."""
    import time

    wire = _Wire(trailing_silence_ms=0)
    simulator = _voice_simulator(wire, _StreamingTTS(first_frame_delay=0.1))

    began = time.perf_counter()
    audio, first_frame_at = await simulator._send_user_utterance(
        "Where is my order?", None, trailing_silence=False
    )

    assert first_frame_at - began >= 0.1
    assert wire.bytes_sent == len(_FRAME) * 3
    pcm, _, _ = audio_utils.wav_bytes_to_pcm16(audio.get_bytes())
    assert pcm == _FRAME * 3
    assert simulator.tts_cost == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_an_utterance_is_timed_from_when_the_transport_sent_it():
    """A transport that needs the whole utterance cannot have sent it early.

    Synthesis produced a first frame long before this connector handed anything
    over, and timing the clip from that frame would put the caller's voice on
    the call seconds before the agent could hear a word of it.
    """
    import time

    class _Buffered(CallbackVoiceConnector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sent_at: Optional[float] = None

        async def stream_uplink(self, audio, *, trailing_silence=True):
            self.sent_at = time.perf_counter()
            await super().stream_uplink(
                audio, trailing_silence=trailing_silence
            )

    async def agent(audio: Audio) -> ConnectorTurn:
        return ConnectorTurn(audio=audio, transcript="Agent reply")

    connector = _Buffered(agent)
    await connector.connect()
    try:
        simulator = _voice_simulator(
            connector,
            _StreamingTTS(first_frame_delay=0.05, frame_delay=0.1, frames=3),
        )
        _, first_frame_at = await simulator._send_user_utterance(
            "Where is my order?", None, trailing_silence=False
        )
    finally:
        await connector.disconnect()

    assert connector.sent_at is not None
    assert first_frame_at == pytest.approx(connector.sent_at, abs=0.02)


@pytest.mark.asyncio
async def test_a_speech_model_that_cannot_stream_still_sends_whole_utterances():
    wire = _Wire(trailing_silence_ms=0)
    simulator = _voice_simulator(wire, _BufferedTTS())

    audio, _ = await simulator._send_user_utterance(
        "Where is my order?", None, trailing_silence=False
    )

    assert b"".join(wire.sent) == _FRAME
    assert audio.get_bytes() == audio_utils.pcm16_to_wav_bytes(_FRAME, _RATE)
    assert simulator.tts_cost == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_collecting_a_stream_preserves_order_and_rate():
    pcm, rate = await collect_pcm_chunks(_stream(b"\x01\x00", b"\x02\x00"))

    assert pcm == b"\x01\x00\x02\x00"
    assert rate == _RATE


def test_mixing_a_stream_frame_by_frame_matches_mixing_it_whole(tmp_path):
    """Streaming must not change what the caller sounds like."""
    bed = tmp_path / "cafe.wav"
    bed.write_bytes(
        audio_utils.pcm16_to_wav_bytes(b"\x64\x00\x00\x00\x9c\xff" * 250, _RATE)
    )
    settings = BackgroundNoiseSettings(audio=str(bed), volume=0.5)
    speech = _FRAME * 3

    whole = mix_background(
        Audio.from_bytes(
            audio_utils.pcm16_to_wav_bytes(speech, _RATE), "audio/wav"
        ),
        settings,
    )
    whole_pcm, _, _ = audio_utils.wav_bytes_to_pcm16(whole.get_bytes())

    mixer = BackgroundMixer(settings)
    streamed = b"".join(
        mixer.mix_chunk(_chunk(_FRAME)).get_bytes() for _ in range(3)
    )

    assert streamed == whole_pcm
    assert streamed != speech


def test_a_missing_background_file_leaves_the_stream_alone():
    mixer = BackgroundMixer(
        BackgroundNoiseSettings(audio="/nowhere/cafe.wav", volume=0.5)
    )
    chunk = _chunk(_FRAME)

    assert mixer.mix_chunk(chunk).get_bytes() == _FRAME
    assert mixer.enabled is False

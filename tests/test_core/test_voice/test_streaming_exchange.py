import asyncio

from deepeval.dataset import ConversationalGolden
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Audio, AudioChunk
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.types import ConnectorTurn
from deepeval.voice.recording import CallRecorder, RecordingConnector
from tests.test_core.test_simulator.helpers import StaticSimulatorModel
from tests.test_core.test_voice.helpers import (
    RATE,
    EchoAgent,
    StubSTT,
    wav_audio,
)

_FRAME = b"\xe8\x03" * 240


def _chunk(data: bytes, *, final: bool = False) -> AudioChunk:
    return AudioChunk.from_bytes(
        data, "audio/pcm", sampleRate=RATE, encoding="pcm", final=final
    )


async def _stream(*payloads: bytes):
    for index, payload in enumerate(payloads):
        yield _chunk(payload, final=index == len(payloads) - 1)


class _StreamingTTS:
    sample_rate = RATE

    def supports_streaming(self) -> bool:
        return True

    def synthesis_cost(self, text: str) -> float:
        return 0.5

    async def a_synthesize_stream(self, text: str, **kwargs):
        for index in range(3):
            yield _chunk(_FRAME, final=index == 2)


class _SpyConnector(CallbackVoiceConnector):
    def __init__(self, agent, calls, **kwargs):
        super().__init__(agent, **kwargs)
        self.calls = calls

    async def exchange_turn(self, audio: Audio) -> ConnectorTurn:
        self.calls.append("exchange_turn")
        return await super().exchange_turn(audio)

    async def exchange_turn_stream(self, chunks) -> ConnectorTurn:
        self.calls.append("exchange_turn_stream")
        return await super().exchange_turn_stream(chunks)


def test_default_exchange_turn_stream_buffers_the_utterance():
    received = []

    async def agent(audio: Audio) -> ConnectorTurn:
        received.append(audio)
        return ConnectorTurn(audio=wav_audio(), transcript="Agent reply")

    async def run():
        connector = CallbackVoiceConnector(agent)
        async with connector:
            return await connector.exchange_turn_stream(_stream(_FRAME, _FRAME))

    result = asyncio.run(run())
    assert result.transcript == "Agent reply"
    pcm, _, _ = audio_utils.wav_bytes_to_pcm16(received[0].get_bytes())
    assert pcm == _FRAME * 2


def test_half_duplex_streams_when_the_tts_model_can():
    calls = []
    cases = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=_SpyConnector(EchoAgent(), calls),
            tts_model=_StreamingTTS(),
            stt_model=StubSTT(),
            output_dir=None,
            combine_audio_files=False,
        ),
    ).simulate(
        [ConversationalGolden(scenario="Refund", expected_outcome="Done.")],
        max_user_simulations=1,
    )

    assert "exchange_turn_stream" in calls
    turns = cases[0].turns
    user_turns = [turn for turn in turns if turn.role == "user"]
    assert user_turns and user_turns[0].audio is not None
    pcm, _, _ = audio_utils.wav_bytes_to_pcm16(user_turns[0].audio.get_bytes())
    assert pcm == _FRAME * 3
    assert any(turn.role == "assistant" and turn.content for turn in turns)


def test_recording_connector_taps_streamed_exchanges():
    async def run():
        recorder = CallRecorder(sample_rate=RATE)
        connector = RecordingConnector(
            CallbackVoiceConnector(EchoAgent()), recorder
        )
        async with connector:
            await connector.exchange_turn_stream(_stream(_FRAME, _FRAME))
        return recorder

    recorder = asyncio.run(run())
    spooled = {
        channel
        for channel, spool in recorder._spools.items()
        if spool["path"] and spool["file"].tell() > 0
    }
    recorder.discard()
    assert spooled == {"user", "agent"}

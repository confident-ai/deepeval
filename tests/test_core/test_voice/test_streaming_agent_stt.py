import asyncio

from deepeval.dataset import ConversationalGolden
from deepeval.models.stt._stream import BufferedTranscribeMixin
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.conversation_simulator import _AgentTranscriber
from deepeval.test_case import Audio, AudioChunk
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from deepeval.voice.connectors.turn_engine import collect_agent_turn
from deepeval.voice.connectors.types import ConnectorTurn
from tests.test_core.test_simulator.helpers import StaticSimulatorModel
from tests.test_core.test_voice.helpers import RATE, EchoAgent, StubTTS

_FRAME = b"\xe8\x03" * 240
_SILENCE = b"\x00\x00" * 240


def test_collect_agent_turn_feeds_the_frame_sink():
    async def run():
        frames: asyncio.Queue = asyncio.Queue()
        frames.put_nowait(_SILENCE)
        frames.put_nowait(_FRAME)
        frames.put_nowait(_FRAME)
        frames.put_nowait(None)
        seen = []
        pcm, _ = await collect_agent_turn(
            frames,
            sample_rate=RATE,
            end_of_turn_silence_ms=10_000,
            frame_gap_timeout_s=0.2,
            max_turn_timeout_s=2.0,
            frame_sink=lambda chunk, rate: seen.append((chunk, rate)),
        )
        return pcm, seen

    pcm, seen = asyncio.run(run())
    assert pcm == _FRAME * 2
    assert seen == [(_FRAME, RATE), (_FRAME, RATE)]


class _StreamingSTT:
    truncated_audio_pad_seconds = 0.0

    def __init__(self):
        self.batch_calls = 0

    def supports_streaming(self) -> bool:
        return True

    async def a_transcribe(self, audio, **kwargs):
        self.batch_calls += 1
        return "Batch reply", None

    async def a_transcribe_stream(
        self,
        audio_stream,
        *,
        partial_every_seconds: float = 1.0,
        on_cost=None,
        **kwargs,
    ):
        count = 0
        async for _chunk in audio_stream:
            count += 1
            if on_cost is not None:
                on_cost(0.01)
            yield f"partial {count}"


def test_transcriber_keeps_the_last_partial_and_costs():
    async def run():
        stt = _StreamingSTT()
        costs = []
        transcriber = _AgentTranscriber(
            stt, {}, partial_every_seconds=0.8, on_cost=costs.append
        )
        transcriber.sink(_FRAME, RATE)
        transcriber.sink(_FRAME, RATE)
        text = await transcriber.finish()
        return text, costs

    text, costs = asyncio.run(run())
    assert text == "partial 2"
    assert costs == [0.01, 0.01]


class _SinkingConnector(CallbackVoiceConnector):
    async def exchange_turn(self, audio: Audio) -> ConnectorTurn:
        if self._agent_frame_sink is not None:
            self._agent_frame_sink(_FRAME, RATE)
        result = await super().exchange_turn(audio)
        result.transcript = None
        return result


def test_half_duplex_uses_the_streamed_transcript():
    stt = _StreamingSTT()
    simulator = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=_SinkingConnector(EchoAgent()),
            tts_model=StubTTS(),
            stt_model=stt,
            output_dir=None,
            combine_audio_files=False,
        ),
    )
    cases = simulator.simulate(
        [ConversationalGolden(scenario="Refund", expected_outcome="Done.")],
        max_user_simulations=1,
    )

    assistant_turns = [
        turn for turn in cases[0].turns if turn.role == "assistant"
    ]
    assert assistant_turns
    assert all(turn.content.startswith("partial ") for turn in assistant_turns)
    assert stt.batch_calls == 0
    assert simulator.stt_cost > 0


class _BatchSTT(BufferedTranscribeMixin):
    async def a_transcribe(self, audio, language=None, **kwargs):
        return "text so far", 0.5


def test_buffered_mixin_reports_partial_costs():
    async def run():
        async def chunks():
            for index in range(2):
                yield AudioChunk.from_bytes(
                    _FRAME,
                    "audio/pcm",
                    sampleRate=RATE,
                    encoding="pcm",
                    final=index == 1,
                )

        costs = []
        texts = []
        async for text in _BatchSTT().a_transcribe_stream(
            chunks(), partial_every_seconds=0.001, on_cost=costs.append
        ):
            texts.append(text)
        return texts, costs

    texts, costs = asyncio.run(run())
    assert texts == ["text so far"]
    assert costs == [0.5, 0.5]

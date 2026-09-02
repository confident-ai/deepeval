import asyncio
import time

import pytest

from deepeval.dataset import ConversationalGolden
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Audio, Turn
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.transports.base import BaseVoiceConnector
from deepeval.voice.connectors.types import AgentEvent
from deepeval.voice.protocol import VoiceProtocol
from deepeval.voice.duplex import DuplexExchange, _looks_complete
from deepeval.voice.floor_control import FloorController
from tests.test_core.test_simulator.helpers import StaticSimulatorModel
from tests.test_core.test_voice.helpers import (
    RATE,
    EchoAgent,
    StubSTT,
    StubTTS,
)

_SPEECH = b"\xe8\x03" * 2400
_QUIET = b"\x00\x00" * 480


def test_looks_complete_reads_terminal_punctuation():
    assert _looks_complete("I can help with that.")
    assert _looks_complete("How may I help you today?")
    assert not _looks_complete("I can help with,")
    assert not _looks_complete("Let me check that...")
    assert not _looks_complete("")
    assert not _looks_complete(None)


class _SentenceSTT:
    truncated_audio_pad_seconds = 0.0

    def __init__(self, text):
        self.text = text

    def supports_streaming(self) -> bool:
        return False

    async def a_transcribe(self, audio, **kwargs):
        return self.text, None


class _QuietAfterSentence(BaseVoiceConnector):
    """Speaks once, then goes quiet without ever closing its turn."""

    protocol = VoiceProtocol.WEBSOCKET
    end_of_turn_silence_ms = 5000
    max_turn_timeout_s = 10.0
    signals_turn_complete = False
    sample_rate = RATE

    @property
    def recv_sample_rate(self) -> int:
        return self.sample_rate

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def exchange_turn(self, audio):
        raise NotImplementedError

    async def stream_uplink(self, audio, *, trailing_silence: bool = True):
        return None

    async def stop_uplink(self) -> None:
        return None

    async def iter_agent_events(self):
        yield AgentEvent(audio=_SPEECH, received_at=time.perf_counter())
        while True:
            await asyncio.sleep(0.02)
            yield AgentEvent(audio=_QUIET, received_at=time.perf_counter())


def _run_exchange(connector, stt_model):
    exchange = DuplexExchange(
        connector=connector,
        tts_model=object(),
        stt_model=stt_model,
        policy=None,
        floor=FloorController(),
        golden=ConversationalGolden(scenario="Test"),
        language="English",
        a_generate_schema=None,
        call_started_at=time.perf_counter(),
    )

    async def run():
        async with connector:
            return await asyncio.wait_for(
                exchange.run(
                    turns=[Turn(role="user", content="Hello")],
                    sent_at=time.perf_counter(),
                    barges_this_conversation=0,
                ),
                timeout=3,
            )

    return asyncio.run(run())


def test_a_finished_thought_ends_the_turn_before_the_full_window():
    result = _run_exchange(
        _QuietAfterSentence(), _SentenceSTT("I can help with that.")
    )
    assert len(result.turns) == 1
    assert result.turns[0].content == "I can help with that."


def test_an_unfinished_thought_waits_out_the_full_window():
    connector = _QuietAfterSentence()
    connector.end_of_turn_silence_ms = 1200
    connector.max_turn_timeout_s = 4.0
    started = time.perf_counter()
    result = _run_exchange(connector, _SentenceSTT("I can help with,"))
    assert time.perf_counter() - started > 1.0
    assert result.turns[0].content == "I can help with,"


def test_duplex_listen_captures_the_greeting():
    simulator = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=CallbackVoiceConnector(EchoAgent()),
            tts_model=StubTTS(),
            stt_model=_SentenceSTT("Hi, this is Riley. How can I help?"),
            output_dir=None,
            combine_audio_files=False,
        ),
    )
    from deepeval.simulator.conversation_simulator import _VoiceSession

    async def run():
        connector = _QuietAfterSentence()
        session = _VoiceSession(
            connector=connector, persona=None, floor=FloorController()
        )
        turns = []
        async with connector:
            await asyncio.wait_for(
                simulator._voice_duplex_listen(
                    session, turns, ConversationalGolden(scenario="Test")
                ),
                timeout=5,
            )
        return turns

    turns = asyncio.run(run())
    assert len(turns) == 1
    assert turns[0].role == "assistant"
    assert turns[0].content == "Hi, this is Riley. How can I help?"

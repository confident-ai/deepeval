import asyncio
import time

import pytest

from deepeval.dataset import ConversationalGolden
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Audio, Turn
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.types import ConnectorTurn
from deepeval.voice.duplex import DuplexExchange
from deepeval.voice.floor_control import FloorController
from tests.test_core.test_simulator.helpers import StaticSimulatorModel
from tests.test_core.test_voice.helpers import (
    RATE,
    EchoAgent,
    StubSTT,
    StubTTS,
    wav_audio,
)


class _NoJudge:
    def __call__(self, *args, **kwargs):
        raise AssertionError("the judge must not run without a policy")


@pytest.mark.asyncio
async def test_duplex_runs_without_a_policy():
    reply_wav = audio_utils.pcm16_to_wav_bytes(
        b"\xe8\x03" * 2400, sample_rate=RATE
    )
    reply_audio = Audio.from_bytes(reply_wav, "audio/wav")

    async def agent(_audio):
        return ConnectorTurn(audio=reply_audio, transcript="Agent reply")

    connector = CallbackVoiceConnector(agent)
    exchange = DuplexExchange(
        connector=connector,
        tts_model=object(),
        stt_model=StubSTT(),
        policy=None,
        floor=FloorController(),
        golden=ConversationalGolden(scenario="Test"),
        language="English",
        a_generate_schema=_NoJudge(),
        call_started_at=time.perf_counter(),
    )
    input_audio = Audio.from_bytes(reply_wav, "audio/wav")
    turns = [Turn(role="user", content="Hello", audio=input_audio)]

    async with connector:
        await connector.stream_uplink(input_audio)
        result = await asyncio.wait_for(
            exchange.run(
                turns=turns,
                sent_at=time.perf_counter(),
                barges_this_conversation=0,
            ),
            timeout=5,
        )

    assert len(result.turns) == 1
    turn = result.turns[0]
    assert turn.role == "assistant"
    assert turn.content == "Agent reply"
    assert turn.interrupted is None
    assert turn.latency_ms is None or turn.latency_ms >= 0
    assert result.barges == 0


class _FullDuplexConnector(CallbackVoiceConnector):
    def __init__(self, agent, calls, **kwargs):
        super().__init__(agent, **kwargs)
        self.calls = calls

    @property
    def supports_duplex(self) -> bool:
        return True

    async def exchange_turn(self, audio: Audio) -> ConnectorTurn:
        self.calls.append("exchange_turn")
        return await super().exchange_turn(audio)

    async def stream_uplink(self, audio: Audio, **kwargs) -> None:
        self.calls.append("stream_uplink")
        return await super().stream_uplink(audio, **kwargs)


def test_full_duplex_transports_use_the_duplex_path_without_a_policy():
    calls = []
    cases = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=_FullDuplexConnector(EchoAgent(), calls),
            tts_model=StubTTS(),
            stt_model=StubSTT(),
            output_dir=None,
            combine_audio_files=False,
        ),
    ).simulate(
        [ConversationalGolden(scenario="Refund", expected_outcome="Done.")],
        max_user_simulations=1,
    )

    assert "stream_uplink" in calls
    assert "exchange_turn" not in calls
    turns = cases[0].turns
    assistant_turns = [turn for turn in turns if turn.role == "assistant"]
    assert assistant_turns
    assert assistant_turns[0].content == "Agent reply"
    assert all(turn.interrupted is None for turn in assistant_turns)

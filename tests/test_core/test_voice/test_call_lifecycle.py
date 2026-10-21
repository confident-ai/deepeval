import asyncio
import time
from types import SimpleNamespace

import pytest

from deepeval.dataset import ConversationalGolden
from deepeval.simulator import ConversationSimulator
from deepeval.voice import CallbackVoiceConnector, LiveKitConnector, VoiceConfig
from deepeval.voice.connectors.transports.base import iter_downlink
from deepeval.voice.connectors.types import AgentEvent
from deepeval.voice.recording import CallRecorder, RecordingConnector
from tests.test_core.test_simulator.helpers import StaticSimulatorModel
from tests.test_core.test_voice.helpers import (
    RATE,
    EchoAgent,
    StubSTT,
    StubTTS,
)

VOICED = b"\xe8\x03"


def _livekit() -> LiveKitConnector:
    connector = LiveKitConnector(
        url="wss://livekit.test", api_key="key", api_secret="secret"
    )
    connector._out_frames = asyncio.Queue()
    return connector


async def _collect(events):
    return [event async for event in events]


@pytest.mark.asyncio
async def test_downlink_stops_for_every_reader_once_the_call_ends():
    queue = asyncio.Queue()
    queue.put_nowait(AgentEvent(audio=VOICED * 10))
    queue.put_nowait(AgentEvent(turn_complete=True, call_ended=True))

    first = await asyncio.wait_for(_collect(iter_downlink(queue)), timeout=1)
    second = await asyncio.wait_for(_collect(iter_downlink(queue)), timeout=1)

    assert [event.audio for event in first] == [VOICED * 10]
    assert second == []


@pytest.mark.asyncio
async def test_livekit_ends_the_call_only_when_the_agent_leaves():
    connector = _livekit()
    agent = SimpleNamespace(identity="callee")
    connector._agent_participant = agent

    connector._on_participant_disconnected(SimpleNamespace(identity="observer"))
    assert not connector.call_ended

    connector._on_participant_disconnected(agent)
    assert connector.call_ended
    events = await asyncio.wait_for(
        _collect(connector.iter_agent_events()), timeout=1
    )
    assert events == []


@pytest.mark.asyncio
async def test_livekit_ends_the_call_when_the_room_disconnects():
    connector = _livekit()
    connector._on_room_disconnected()
    assert connector.call_ended


@pytest.mark.asyncio
async def test_taking_pending_agent_audio_keeps_the_call_ended_marker():
    connector = _livekit()
    connector._out_frames.put_nowait(AgentEvent(audio=VOICED * 10))
    connector._end_call()

    pending = connector.take_pending_agent_events()
    connector.drain_downlink()

    assert [event.audio for event in pending] == [VOICED * 10]
    events = await asyncio.wait_for(
        _collect(connector.iter_agent_events()), timeout=1
    )
    assert events == []


def test_recording_connector_tapes_pending_agent_audio():
    connector = _livekit()
    connector._out_frames.put_nowait(
        AgentEvent(audio=VOICED * 480, received_at=time.perf_counter())
    )
    recorder = CallRecorder(sample_rate=RATE)

    RecordingConnector(connector, recorder).take_pending_agent_events()

    assert recorder._spools["agent"].samples > 0
    recorder.discard()


def test_recorder_ignores_small_arrival_jitter():
    recorder = CallRecorder(sample_rate=RATE)
    recorder.add("agent", VOICED * RATE, RATE, 100.0)
    recorder.add("agent", VOICED * RATE, RATE, 101.1)
    assert recorder._spools["agent"].samples == 2 * RATE

    recorder.add("agent", VOICED * RATE, RATE, 103.5)
    assert recorder._spools["agent"].samples == int(3.5 * RATE) + RATE
    recorder.discard()


class _DuplexConnector(CallbackVoiceConnector):
    @property
    def supports_duplex(self) -> bool:
        return True


class _HangUpConnector(_DuplexConnector):
    ended = False

    @property
    def call_ended(self) -> bool:
        return self.ended

    async def stream_uplink(self, audio, **kwargs) -> None:
        await super().stream_uplink(audio, **kwargs)
        self.ended = True


class _BacklogConnector(_DuplexConnector):
    takes = 0

    def take_pending_agent_events(self):
        self.takes += 1
        if self.takes != 2:
            return []
        return [
            AgentEvent(
                audio=VOICED * int(RATE * 0.5),
                received_at=time.perf_counter(),
            )
        ]


def _simulate(connector, max_user_simulations: int):
    return ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=connector,
            tts_model=StubTTS(),
            stt_model=StubSTT(),
            output_dir=None,
            combine_audio_files=False,
        ),
    ).simulate(
        [ConversationalGolden(scenario="Refund", expected_outcome="Done.")],
        max_user_simulations=max_user_simulations,
    )[
        0
    ]


def test_simulation_stops_when_the_agent_ends_the_call():
    case = _simulate(_HangUpConnector(EchoAgent()), max_user_simulations=5)

    assert [turn.role for turn in case.turns] == ["user", "assistant"]
    assert case.metadata["Stop reason"] == "The agent ended the call"


def test_simulation_without_a_hang_up_has_no_stop_reason():
    case = _simulate(_DuplexConnector(EchoAgent()), max_user_simulations=2)

    assert "Stop reason" not in case.metadata


def test_agent_speech_heard_before_the_caller_replies_is_its_own_turn():
    case = _simulate(_BacklogConnector(EchoAgent()), max_user_simulations=2)

    roles = [turn.role for turn in case.turns]
    assert roles == ["user", "assistant", "assistant", "user", "assistant"]
    backlog, reply = case.turns[2], case.turns[3]
    assert backlog.audio is not None
    assert backlog.audio.start_time <= reply.audio.start_time

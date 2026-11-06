import asyncio
import time
from types import SimpleNamespace
from typing import Optional

import pytest

from deepeval.dataset import ConversationalGolden
from deepeval.test_case import AudioChunk
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.conversation_simulator import (
    _VoiceSession,
    _continued_agent_turn,
)
from deepeval.test_case import Turn
from deepeval.voice.duplex import _pcm_to_audio
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
        if self.takes != 3:
            return []
        return [
            AgentEvent(turn_complete=True),
            AgentEvent(
                audio=VOICED * int(RATE * 0.5),
                received_at=time.perf_counter(),
            ),
            AgentEvent(transcript="Who am I speaking with?"),
            AgentEvent(turn_complete=True),
        ]


def _simulate(
    connector,
    max_user_simulations: int,
    simulator_model: Optional[StaticSimulatorModel] = None,
):
    return ConversationSimulator(
        simulator_model=simulator_model or StaticSimulatorModel(),
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


def test_the_caller_replies_to_what_the_agent_kept_saying():
    model = StaticSimulatorModel()
    case = _simulate(
        _BacklogConnector(EchoAgent()),
        max_user_simulations=2,
        simulator_model=model,
    )

    roles = [turn.role for turn in case.turns]
    assert roles == ["user", "assistant", "user", "assistant"]
    agent, reply = case.turns[1], case.turns[2]
    assert agent.content.endswith("Who am I speaking with?")
    assert agent.content != "Who am I speaking with?"
    assert reply.content == f"simulated user input {model.user_input_count}"


def test_an_agent_pause_nobody_answered_stays_one_turn():
    rate = 24000
    first = _pcm_to_audio(VOICED * rate, rate)
    first.start_time = 1.0
    rest = _pcm_to_audio(VOICED * rate, rate)
    rest.start_time = 2.5
    opening = Turn(
        role="assistant", content="Welcome.", audio=first, latency_ms=600.0
    )

    turn = _continued_agent_turn(opening, "Who am I speaking with?", rest)

    assert turn.content == "Welcome. Who am I speaking with?"
    assert turn.latency_ms == 600.0
    assert turn.audio.start_time == 1.0
    assert turn.audio.duration == pytest.approx(2.5, abs=0.01)


def _signalling_livekit() -> LiveKitConnector:
    connector = _livekit()
    connector._agent_state = "speaking"
    return connector


@pytest.mark.asyncio
async def test_the_caller_waits_for_the_agent_to_finish_its_turn():
    connector = _signalling_livekit()
    queue = connector._out_frames
    queue.put_nowait(AgentEvent(turn_complete=True))
    queue.put_nowait(AgentEvent(transcript="Welcome to the clinic."))
    queue.put_nowait(
        AgentEvent(audio=VOICED * RATE, received_at=time.perf_counter())
    )
    loop = asyncio.get_running_loop()
    loop.call_later(
        0.1,
        queue.put_nowait,
        AgentEvent(audio=VOICED * RATE, received_at=time.perf_counter()),
    )
    loop.call_later(
        0.2,
        queue.put_nowait,
        AgentEvent(
            transcript="Para espanol, oprima 8. Who am I speaking with?"
        ),
    )
    loop.call_later(0.3, queue.put_nowait, AgentEvent(turn_complete=True))
    simulator = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=lambda: connector,
            tts_model=StubTTS(),
            stt_model=StubSTT(),
            output_dir=None,
            combine_audio_files=False,
        ),
    )
    turns = []

    took = await asyncio.wait_for(
        simulator._take_agent_backlog(
            _VoiceSession(
                connector=connector,
                persona=None,
                tts_model=simulator._voice.tts_model,
            ),
            turns,
            time.perf_counter(),
        ),
        timeout=2,
    )

    assert took
    assert [turn.content for turn in turns] == [
        "Para espanol, oprima 8. Who am I speaking with?"
    ]
    assert len(turns[0].audio.get_bytes()) > 2 * len(VOICED * RATE)
    assert queue.empty()


@pytest.mark.asyncio
async def test_a_stale_end_signal_without_speech_is_dropped():
    connector = _signalling_livekit()
    connector._out_frames.put_nowait(AgentEvent(turn_complete=True))
    simulator = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=lambda: connector,
            tts_model=StubTTS(),
            stt_model=StubSTT(),
            output_dir=None,
            combine_audio_files=False,
        ),
    )
    turns = []

    took = await simulator._take_agent_backlog(
        _VoiceSession(
            connector=connector,
            persona=None,
            tts_model=simulator._voice.tts_model,
        ),
        turns,
        time.perf_counter(),
    )

    assert not took
    assert turns == []
    assert connector._out_frames.empty()


@pytest.mark.asyncio
async def test_hanging_up_wakes_a_reader_waiting_on_the_agent():
    connector = _livekit()
    reader = asyncio.create_task(_collect(connector.iter_agent_events()))
    await asyncio.sleep(0)

    await connector.disconnect()

    assert connector.call_ended
    assert await asyncio.wait_for(reader, timeout=1) == []


@pytest.mark.asyncio
async def test_a_hung_up_call_takes_no_more_caller_audio():
    connector = _livekit()
    await connector.disconnect()

    async def chunks():
        yield AudioChunk.from_bytes(
            VOICED * 240, "audio/pcm", sampleRate=24000, encoding="pcm16"
        )

    result = await connector.stream_uplink_chunks(chunks())

    assert result.first_frame_at is None


class _EndableConnector(_DuplexConnector):
    def __init__(self, agent, line: SimpleNamespace):
        super().__init__(agent)
        self.line = line

    @property
    def call_ended(self) -> bool:
        return self.line.dead


class _HangsUpWhileTheCallerThinks(StaticSimulatorModel):
    def __init__(self, line: SimpleNamespace):
        super().__init__()
        self.line = line

    def generate(self, prompt: str, schema=None):
        result = super().generate(prompt, schema=schema)
        if self.user_input_count == 2:
            self.line.dead = True
        return result


def test_a_line_written_after_the_call_ended_is_never_said():
    line = SimpleNamespace(dead=False)
    case = _simulate(
        _EndableConnector(EchoAgent(), line),
        max_user_simulations=3,
        simulator_model=_HangsUpWhileTheCallerThinks(line),
    )

    assert [turn.role for turn in case.turns] == ["user", "assistant"]
    assert case.metadata["Stop reason"] == "The agent ended the call"

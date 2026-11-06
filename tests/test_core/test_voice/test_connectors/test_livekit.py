import asyncio
from types import SimpleNamespace

import pytest

from deepeval.voice import LiveKitConnector
from deepeval.voice.connectors.providers.livekit import (
    connector as livekit_module,
)
from tests.test_core.test_voice.helpers import install_fake_livekit


class _BlockingReader:
    info = SimpleNamespace(attributes={})

    async def read_all(self):
        await asyncio.Event().wait()


def _connector(**kwargs) -> LiveKitConnector:
    return LiveKitConnector(
        url="wss://livekit.test", api_key="key", api_secret="secret", **kwargs
    )


@pytest.mark.asyncio
async def test_hanging_up_one_call_leaves_another_calls_transcripts(
    monkeypatch,
):
    install_fake_livekit(monkeypatch, livekit_module)
    template = _connector()
    first, second = template.clone(), template.clone()
    await first._join_room()
    await second._join_room()

    first._on_transcript_stream(_BlockingReader(), "agent")
    second._on_transcript_stream(_BlockingReader(), "agent")
    await asyncio.sleep(0)
    [second_task] = list(second._transcript_tasks)

    await first.disconnect()
    await asyncio.sleep(0)

    assert not second_task.cancelled()
    assert second._transcript_tasks == {second_task}
    await second.disconnect()
    await asyncio.sleep(0)
    assert second_task.cancelled()


def test_a_connector_that_creates_its_own_rooms_supports_parallel_calls():
    assert _connector().supports_parallel_calls


@pytest.mark.parametrize(
    "fixed",
    [
        {"room_name": "support-line"},
        {"token": "caller-token"},
        {"room": SimpleNamespace()},
    ],
)
def test_a_connector_tied_to_one_room_does_not_support_parallel_calls(fixed):
    assert not _connector(**fixed).supports_parallel_calls


def _publication(name, kind=1):
    return SimpleNamespace(name=name, track=SimpleNamespace(kind=kind))


def _participant(identity, *publications):
    return SimpleNamespace(
        identity=identity,
        kind=None,
        attributes={},
        track_publications=dict(enumerate(publications)),
    )


@pytest.mark.asyncio
async def test_the_background_audio_track_is_not_taken_for_the_agent_voice(
    monkeypatch,
):
    install_fake_livekit(monkeypatch, livekit_module)
    connector = _connector()
    await connector._join_room()
    attached = []
    connector._attach_agent_track = lambda track, participant: attached.append(
        track
    )
    agent = _participant("agent")
    background = _publication("background_audio")
    voice = _publication("roomio_audio")

    connector._on_track_subscribed(background.track, background, agent)
    connector._on_track_subscribed(voice.track, voice, agent)

    assert attached == [voice.track]


@pytest.mark.asyncio
async def test_an_agent_already_in_the_room_is_heard_on_its_voice_track(
    monkeypatch,
):
    install_fake_livekit(monkeypatch, livekit_module)
    connector = _connector()
    await connector._join_room()
    attached = []
    connector._attach_agent_track = lambda track, participant: attached.append(
        track
    )
    voice = _publication("roomio_audio")
    connector._room.remote_participants = {
        "agent": _participant("agent", _publication("background_audio"), voice)
    }

    connector._adopt_existing_agent_track()

    assert attached == [voice.track]


async def _state_watching_connector(monkeypatch):
    install_fake_livekit(monkeypatch, livekit_module)
    monkeypatch.setattr(livekit_module, "_TURN_END_SETTLE_S", 0.01)
    connector = _connector(transcript_grace_s=0.0)
    await connector._join_room()
    return connector


def _drain(connector):
    events = []
    while not connector._out_frames.empty():
        events.append(connector._out_frames.get_nowait())
    return events


def _turn_ends(connector):
    return [event for event in _drain(connector) if event.turn_complete]


def test_silence_decides_the_turn_until_the_agent_publishes_its_state():
    assert _connector().signals_turn_complete is False


@pytest.mark.asyncio
async def test_the_turn_ends_when_the_agent_goes_back_to_listening(
    monkeypatch,
):
    connector = await _state_watching_connector(monkeypatch)

    for state in ("listening", "speaking", "thinking", "speaking", "listening"):
        connector._on_agent_state(state)
    await asyncio.sleep(0.05)

    assert connector.signals_turn_complete is True
    assert len(_turn_ends(connector)) == 1


@pytest.mark.asyncio
async def test_a_tool_call_pause_does_not_end_the_turn(monkeypatch):
    connector = await _state_watching_connector(monkeypatch)

    connector._on_agent_state("speaking")
    connector._on_agent_state("thinking")
    await asyncio.sleep(0.05)

    assert _turn_ends(connector) == []


@pytest.mark.asyncio
async def test_speaking_again_within_the_settle_window_keeps_the_turn_open(
    monkeypatch,
):
    connector = await _state_watching_connector(monkeypatch)

    connector._on_agent_state("speaking")
    connector._on_agent_state("listening")
    connector._on_agent_state("speaking")
    await asyncio.sleep(0.05)

    assert _turn_ends(connector) == []


@pytest.mark.asyncio
async def test_thinking_without_speaking_ends_no_turn(monkeypatch):
    connector = await _state_watching_connector(monkeypatch)

    connector._on_agent_state("thinking")
    connector._on_agent_state("listening")
    await asyncio.sleep(0.05)

    assert _turn_ends(connector) == []


@pytest.mark.asyncio
async def test_the_room_reports_agent_state_changes(monkeypatch):
    connector = await _state_watching_connector(monkeypatch)
    agent = _participant("agent")
    on_change = connector._room.handlers["participant_attributes_changed"]

    on_change({"lk.agent.state": "speaking"}, agent)
    on_change({"other": "listening"}, agent)
    on_change({"lk.agent.state": "listening"}, agent)
    await asyncio.sleep(0.05)

    assert len(_turn_ends(connector)) == 1


@pytest.mark.asyncio
async def test_the_state_an_agent_joined_with_is_adopted(monkeypatch):
    connector = await _state_watching_connector(monkeypatch)
    connector._agent_participant = SimpleNamespace(
        attributes={"lk.agent.state": "speaking"}
    )

    connector._adopt_agent_state()
    connector._on_agent_state("listening")
    await asyncio.sleep(0.05)

    assert connector.signals_turn_complete is True
    assert len(_turn_ends(connector)) == 1


class _TranscriptReader:
    def __init__(self, text, final=True):
        self.text = text
        self.info = SimpleNamespace(
            attributes={"lk.transcription_final": str(final).lower()}
        )

    async def read_all(self):
        return self.text


def _transcripts(connector):
    return [event.transcript for event in _drain(connector) if event.transcript]


@pytest.mark.asyncio
async def test_every_segment_of_a_reply_stays_in_its_transcript(monkeypatch):
    connector = await _state_watching_connector(monkeypatch)

    connector._on_agent_state("speaking")
    await connector._read_transcript(_TranscriptReader("Sure, let me check."))
    connector._on_agent_state("thinking")
    connector._on_agent_state("speaking")
    await connector._read_transcript(
        _TranscriptReader("You're booked for 5pm.")
    )

    assert _transcripts(connector) == [
        "Sure, let me check.",
        "Sure, let me check. You're booked for 5pm.",
    ]


@pytest.mark.asyncio
async def test_the_next_reply_starts_a_fresh_transcript(monkeypatch):
    connector = await _state_watching_connector(monkeypatch)

    connector._on_agent_state("speaking")
    await connector._read_transcript(_TranscriptReader("Hi, this is Riley."))
    connector._on_agent_state("listening")
    await asyncio.sleep(0.05)
    connector._on_agent_state("thinking")
    connector._on_agent_state("speaking")
    await connector._read_transcript(_TranscriptReader("Your order shipped."))

    assert _transcripts(connector) == [
        "Hi, this is Riley.",
        "Your order shipped.",
    ]


@pytest.mark.asyncio
async def test_without_agent_state_each_segment_replaces_the_last(monkeypatch):
    install_fake_livekit(monkeypatch, livekit_module)
    connector = _connector()
    await connector._join_room()

    await connector._read_transcript(_TranscriptReader("Sure, let me check."))
    await connector._read_transcript(_TranscriptReader("You're booked."))

    assert _transcripts(connector) == ["Sure, let me check.", "You're booked."]


class _SilentSTT:
    truncated_audio_pad_seconds = 0.0

    async def a_transcribe(self, *_args, **_kwargs):
        return "", 0.0


def _audio(connector, seconds, voiced):
    import math
    import struct
    import time

    from deepeval.voice.connectors.types import AgentEvent

    rate = connector.recv_sample_rate
    amplitude = 8000 if voiced else 0
    frame = b"".join(
        struct.pack(
            "<h", int(amplitude * math.sin(2 * math.pi * 440 * n / rate))
        )
        for n in range(rate // 100)
    )
    for _ in range(int(seconds * 100)):
        connector._out_frames.put_nowait(
            AgentEvent(audio=frame, received_at=time.perf_counter())
        )


@pytest.mark.asyncio
async def test_a_reply_with_a_tool_call_pause_is_one_assistant_turn(
    monkeypatch,
):
    import time

    from deepeval.dataset import ConversationalGolden
    from deepeval.voice.duplex import DuplexExchange
    from deepeval.voice.floor_control import FloorController

    connector = await _state_watching_connector(monkeypatch)
    connector._on_agent_state("listening")
    connector._on_agent_state("speaking")
    _audio(connector, 1.0, voiced=True)
    await connector._read_transcript(_TranscriptReader("Sure, let me check."))
    connector._on_agent_state("thinking")
    _audio(connector, 1.5, voiced=False)
    connector._on_agent_state("speaking")
    _audio(connector, 1.0, voiced=True)
    await connector._read_transcript(
        _TranscriptReader("You're booked for 5pm.")
    )
    _audio(connector, 0.3, voiced=False)
    connector._on_agent_state("listening")

    turns = []
    await DuplexExchange(
        connector=connector,
        tts_model=None,
        stt_model=_SilentSTT(),
        policy=None,
        floor=FloorController(),
        golden=ConversationalGolden(scenario="Book an appointment"),
        language="English",
        a_generate_schema=None,
        call_started_at=time.perf_counter(),
    ).run(turns=turns, sent_at=time.perf_counter(), barges_this_conversation=0)

    assert [turn.role for turn in turns] == ["assistant"]
    assert turns[0].content == "Sure, let me check. You're booked for 5pm."


@pytest.mark.asyncio
async def test_the_turn_waits_for_a_transcript_still_in_flight(monkeypatch):
    connector = await _state_watching_connector(monkeypatch)
    connector.transcript_grace_s = 1.0

    connector._on_agent_state("speaking")
    connector._on_agent_state("listening")
    await asyncio.sleep(0.1)
    assert _turn_ends(connector) == []

    await connector._read_transcript(_TranscriptReader("Hi, this is Riley."))
    await asyncio.sleep(0.1)

    events = _drain(connector)
    assert [
        "transcript" if event.transcript else "turn_complete"
        for event in events
    ] == ["transcript", "turn_complete"]


@pytest.mark.asyncio
async def test_the_turn_ends_without_a_transcript_after_the_grace(monkeypatch):
    connector = await _state_watching_connector(monkeypatch)
    connector.transcript_grace_s = 0.05

    connector._on_agent_state("speaking")
    connector._on_agent_state("listening")
    await asyncio.sleep(0.2)

    assert len(_turn_ends(connector)) == 1


class _HeldTranscriptReader:
    def __init__(self, text):
        self.text = text
        self.release = asyncio.Event()
        self.info = SimpleNamespace(
            attributes={"lk.transcription_final": "true"}
        )

    async def read_all(self):
        await self.release.wait()
        return self.text


async def _no_chunks():
    return
    yield


@pytest.mark.asyncio
async def test_the_turn_end_waits_for_a_transcript_still_open(monkeypatch):
    connector = await _state_watching_connector(monkeypatch)
    connector.transcript_grace_s = 1.0
    connector._on_agent_state("speaking")
    await connector._read_transcript(_TranscriptReader("Sure, let me check."))
    still_open = _HeldTranscriptReader("You're booked for 5pm.")
    connector._on_transcript_stream(still_open, "agent")

    connector._on_agent_state("listening")
    await asyncio.sleep(0.1)
    assert _turn_ends(connector) == []

    still_open.release.set()
    await asyncio.sleep(0.1)
    assert _transcripts(connector) == [
        "Sure, let me check. You're booked for 5pm."
    ]


@pytest.mark.asyncio
async def test_a_transcript_finishing_after_its_turn_ended_is_dropped(
    monkeypatch,
):
    connector = await _state_watching_connector(monkeypatch)
    connector._on_agent_state("speaking")
    late = _HeldTranscriptReader("What can I help you with?")
    connector._on_transcript_stream(late, "agent")
    connector._on_agent_state("listening")
    await asyncio.sleep(0.05)
    _drain(connector)

    connector._on_agent_state("speaking")
    late.release.set()
    await asyncio.sleep(0.05)
    await connector._read_transcript(_TranscriptReader("Your order shipped."))

    assert _transcripts(connector) == ["Your order shipped."]


@pytest.mark.asyncio
async def test_without_agent_state_the_caller_speaking_closes_the_agent_turn(
    monkeypatch,
):
    connector = await _state_watching_connector(monkeypatch)
    late = _HeldTranscriptReader("I'm here to help you with anything.")
    connector._on_transcript_stream(late, "agent")

    await connector.stream_uplink_chunks(_no_chunks())
    late.release.set()
    await asyncio.sleep(0.05)

    assert _transcripts(connector) == []
